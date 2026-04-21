"""Report generation fine-tuning and evaluation."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import Optional

from ..models import MedViLLForGeneration
from ..metrics import compute_bleu4, RunningPerplexity
from ..utils import save_checkpoint, get_logger, log_metrics

logger = get_logger(__name__)


class GenerationTrainer:
    def __init__(
        self,
        model: MedViLLForGeneration,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        cfg,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device

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
        best_bleu = 0.0
        for epoch in range(self.cfg.training.epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            log_metrics(logger, {"epoch": epoch, "train_loss": train_loss, **val_metrics})

            if val_metrics.get("bleu4", 0.0) > best_bleu:
                best_bleu = val_metrics["bleu4"]
                save_checkpoint(
                    self.model, self.optimizer, epoch, 0,
                    val_metrics, self.cfg.training.output_dir, name="gen_best"
                )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, steps = 0.0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"[Gen] Epoch {epoch}")):
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.amp.autocast("cuda", enabled=self.cfg.training.fp16 and self.device.type == "cuda"):
                out = self.model(**tensor_batch)
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
        """Compute BLEU-4 and perplexity on the val set."""
        self.model.eval()
        ppl_tracker = RunningPerplexity()
        hypotheses, references = [], []

        for batch in tqdm(loader, desc="[Gen] Eval", leave=False):
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            target_texts = batch.get("target_text", [])

            out = self.model(**tensor_batch)

            # Perplexity from teacher-forced logits
            ppl_tracker.update(out["logits"], tensor_batch["lm_labels"])

            # BLEU from greedy generation
            pv = tensor_batch["pixel_values"]
            preds = self.model.generate(pv, self.tokenizer)
            hypotheses.extend(preds)
            references.extend(target_texts)

        bleu_scores = compute_bleu4(hypotheses, references)
        return {"perplexity": ppl_tracker.compute(), **bleu_scores}
