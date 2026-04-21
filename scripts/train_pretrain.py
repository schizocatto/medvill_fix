"""Pre-training entry point."""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from medvill.config import MedViLLConfig
from medvill.models import MedViLL
from medvill.data import MedViLLPretrainDataset
from medvill.utils import set_seed, save_checkpoint, get_logger

logger = get_logger(__name__)


def main(args):
    cfg: MedViLLConfig = OmegaConf.merge(
        OmegaConf.structured(MedViLLConfig),
        OmegaConf.load(args.config),
    )
    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)

    train_ds = MedViLLPretrainDataset(
        cfg.data.train_path, tokenizer,
        num_image_embeds=cfg.image.num_image_embeds,
        seq_len=cfg.data.seq_len,
        mlm_prob=cfg.mlm_prob,
        itm_prob=cfg.itm_prob,
        img_size=cfg.image.img_size,
    )
    val_ds = MedViLLPretrainDataset(
        cfg.data.val_path or cfg.data.train_path, tokenizer,
        num_image_embeds=cfg.image.num_image_embeds,
        seq_len=cfg.data.seq_len,
        mlm_prob=0.0,
        itm_prob=0.0,
        img_size=cfg.image.img_size,
    ) if cfg.data.val_path else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
    ) if val_ds else None

    model = MedViLL(cfg.model, cfg.image).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    total_steps = (len(train_loader) // cfg.training.gradient_accumulation_steps) * cfg.training.epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.fp16 and device.type == "cuda")

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss, mlm_total, itm_total, steps = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Pretrain Epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=cfg.training.fp16 and device.type == "cuda"):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    pixel_values=batch["pixel_values"],
                    mlm_labels=batch["mlm_labels"],
                    itm_labels=batch["is_aligned"],
                )
                loss = out["loss"] / cfg.training.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += out["loss"].item()
            mlm_total += (out["mlm_loss"].item() if out["mlm_loss"] else 0)
            itm_total += (out["itm_loss"].item() if out["itm_loss"] else 0)
            steps += 1

            if (step + 1) % cfg.training.logging_steps == 0:
                logger.info(
                    f"  step {step+1} | loss={total_loss/steps:.4f} "
                    f"mlm={mlm_total/steps:.4f} itm={itm_total/steps:.4f}"
                )

        save_checkpoint(model, optimizer, epoch, steps, {"train_loss": total_loss / steps},
                        cfg.training.output_dir, "pretrain")
        logger.info(f"Epoch {epoch} done | avg_loss={total_loss/steps:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    main(parser.parse_args())
