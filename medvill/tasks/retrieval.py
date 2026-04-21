"""Image–report retrieval fine-tuning and evaluation."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import Optional

from ..models import MedViLLForRetrieval
from ..metrics import compute_retrieval_metrics, compute_bleu4, RunningPerplexity
from ..utils import save_checkpoint, get_logger, log_metrics

logger = get_logger(__name__)


class RetrievalTrainer:
    def __init__(
        self,
        model: MedViLLForRetrieval,
        train_loader: DataLoader,
        eval_dataset,
        cfg,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_dataset = eval_dataset
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
        best_r1 = 0.0
        for epoch in range(self.cfg.training.epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate()
            log_metrics(logger, {"epoch": epoch, "train_loss": train_loss, **val_metrics})

            r1 = val_metrics.get("I2T_R@1", 0.0)
            if r1 > best_r1:
                best_r1 = r1
                save_checkpoint(
                    self.model, self.optimizer, epoch, 0,
                    val_metrics, self.cfg.training.output_dir, name="ret_best"
                )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, steps = 0.0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"[Ret] Epoch {epoch}")):
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
    def evaluate(self) -> dict:
        """
        Compute I2T and T2I retrieval metrics on the eval dataset.

        Builds full (N×N) score matrix — suitable for datasets < 5 000 items.
        """
        self.model.eval()
        n = len(self.eval_dataset)

        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

        # Collect all features
        all_pv, all_ids, all_am, all_tti = [], [], [], []
        for batch in tqdm(loader, desc="[Ret] Building index", leave=False):
            all_pv.append(batch["pixel_values"])
            all_ids.append(batch["input_ids"])
            all_am.append(batch["attention_mask"])
            all_tti.append(batch["token_type_ids"])

        all_pv = torch.cat(all_pv).to(self.device)
        all_ids = torch.cat(all_ids).to(self.device)
        all_am = torch.cat(all_am).to(self.device)
        all_tti = torch.cat(all_tti).to(self.device)

        # Build NxN score matrix in chunks
        scores_i2t = np.zeros((n, n), dtype=np.float32)
        chunk = self.cfg.training.batch_size

        for i in range(0, n, chunk):
            pv_i = all_pv[i:i + chunk]
            n_i = pv_i.size(0)
            for j in range(0, n, chunk):
                ids_j = all_ids[j:j + chunk]
                am_j = all_am[j:j + chunk]
                tti_j = all_tti[j:j + chunk]
                n_j = ids_j.size(0)

                pv_exp = pv_i.unsqueeze(1).expand(-1, n_j, -1, -1, -1).reshape(n_i * n_j, *pv_i.shape[1:])
                ids_exp = ids_j.unsqueeze(0).expand(n_i, -1, -1).reshape(n_i * n_j, -1)
                am_exp = am_j.unsqueeze(0).expand(n_i, -1, -1).reshape(n_i * n_j, -1)
                tti_exp = tti_j.unsqueeze(0).expand(n_i, -1, -1).reshape(n_i * n_j, -1)

                out = self.model(ids_exp, am_exp, tti_exp, pv_exp)
                scores_i2t[i:i + n_i, j:j + n_j] = out["score"].reshape(n_i, n_j).cpu().numpy()

        scores_t2i = scores_i2t.T

        i2t = compute_retrieval_metrics(scores_i2t)
        t2i = compute_retrieval_metrics(scores_t2i)

        return {
            **{f"I2T_{k}": v for k, v in i2t.items()},
            **{f"T2I_{k}": v for k, v in t2i.items()},
        }
