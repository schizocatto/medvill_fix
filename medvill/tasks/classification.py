"""Diagnosis classification fine-tuning and evaluation."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import Optional

from ..models import MedViLLForClassification
from ..metrics import compute_classification_metrics
from ..utils import save_checkpoint, get_logger, log_metrics

logger = get_logger(__name__)


class ClassificationTrainer:
    def __init__(
        self,
        model: MedViLLForClassification,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: torch.device,
        label_names: Optional[list] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.label_names = label_names

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
        best_auc = 0.0
        for epoch in range(self.cfg.training.epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            log_metrics(logger, {"epoch": epoch, "train_loss": train_loss, **val_metrics})

            auc = val_metrics.get("auc_macro", 0.0)
            if auc > best_auc:
                best_auc = auc
                save_checkpoint(
                    self.model, self.optimizer, epoch, 0,
                    val_metrics, self.cfg.training.output_dir, name="cls_best"
                )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, steps = 0.0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"[Cls] Epoch {epoch}")):
            batch = {k: v.to(self.device) for k, v in batch.items()}

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
        all_logits, all_labels = [], []

        for batch in tqdm(loader, desc="[Cls] Eval", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            all_logits.append(out["logits"].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)

        return compute_classification_metrics(
            logits, labels,
            multilabel=self.model.multilabel,
            label_names=self.label_names,
        )
