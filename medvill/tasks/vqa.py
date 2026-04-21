"""VQA-RAD fine-tuning and evaluation."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from typing import Optional

from ..models import MedViLLForVQA
from ..metrics import RunningPerplexity
from ..utils import save_checkpoint, get_logger, log_metrics

logger = get_logger(__name__)


def vqa_accuracy(pred_ids: list[int], gt_ids: list[int]) -> float:
    """Standard VQA accuracy (exact match)."""
    return accuracy_score(gt_ids, pred_ids)


class VQATrainer:
    def __init__(
        self,
        model: MedViLLForVQA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: torch.device,
        label2ans: Optional[list] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.label2ans = label2ans

        total_steps = (len(train_loader) // cfg.training.gradient_accumulation_steps) * cfg.training.epochs
        warmup_steps = int(total_steps * cfg.training.warmup_ratio)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.fp16 and device.type == "cuda")

    def train(self) -> None:
        best_acc = 0.0
        for epoch in range(self.cfg.training.epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            log_metrics(logger, {"epoch": epoch, "train_loss": train_loss, **val_metrics})

            if val_metrics.get("accuracy", 0.0) > best_acc:
                best_acc = val_metrics["accuracy"]
                save_checkpoint(
                    self.model, self.optimizer, epoch, 0,
                    val_metrics, self.cfg.training.output_dir, name="vqa_best"
                )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, steps = 0.0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"[VQA] Epoch {epoch}")):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.amp.autocast("cuda", enabled=self.cfg.training.fp16 and self.device.type == "cuda"):
                out = self.model(**batch)
                loss = out["loss"] / self.cfg.training.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.training.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += out["loss"].item()
            steps += 1

        return total_loss / steps

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        ppl_tracker = RunningPerplexity()

        for batch in tqdm(loader, desc="[VQA] Eval", leave=False):
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = self.model(**tensor_batch)

            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            labels = tensor_batch["labels"].cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        acc = vqa_accuracy(all_preds, all_labels)
        return {"accuracy": acc, "num_samples": len(all_labels)}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[list[str], list[str]]:
        """Return predicted and ground-truth answer strings for generation metrics."""
        if self.label2ans is None:
            raise ValueError("label2ans must be provided for predict()")

        self.model.eval()
        pred_texts, gt_texts = [], []

        for batch in tqdm(loader, desc="[VQA] Predict", leave=False):
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = self.model(**tensor_batch)

            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            labels = tensor_batch["labels"].cpu().tolist()

            pred_texts.extend([self.label2ans[p] for p in preds])
            gt_texts.extend([self.label2ans[l] for l in labels])

        return pred_texts, gt_texts
