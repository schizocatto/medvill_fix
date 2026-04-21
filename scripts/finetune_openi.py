"""
Unified OpenI fine-tuning script — covers classification, retrieval, and generation.
VQA uses VQA-RAD (not OpenI), so it is handled by scripts/train_vqa.py instead.

Colab T4 rough timing (OpenI ~2 500 train samples, batch_size=16, FP16):
  classification : ~2–3 min / epoch  →  5 epochs ≈ 15 min
  retrieval      : ~3–4 min / epoch  →  5 epochs ≈ 20 min
  generation     : ~4–5 min / epoch  →  5 epochs ≈ 25 min

All config values have sensible defaults so no YAML is needed.
CLI flags override defaults; pass --config to layer a YAML on top.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medvill.config import MedViLLConfig
from medvill.utils import set_seed, load_pretrained_encoder, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults tuned for Colab T4 (16 GB VRAM) + OpenI dataset size
# ---------------------------------------------------------------------------
_T4_DEFAULTS = {
    "classification": dict(batch_size=16, epochs=5,  lr=2e-5, fp16=True, grad_accum=2),
    "retrieval":      dict(batch_size=12, epochs=5,  lr=2e-5, fp16=True, grad_accum=2),
    "generation":     dict(batch_size=8,  epochs=5,  lr=3e-5, fp16=True, grad_accum=4),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfg(args) -> MedViLLConfig:
    cfg = OmegaConf.structured(MedViLLConfig)
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))

    td = _T4_DEFAULTS[args.task]

    overrides = {
        "training": {
            "batch_size":                   args.batch_size  or td["batch_size"],
            "epochs":                        args.epochs      or td["epochs"],
            "lr":                            args.lr          or td["lr"],
            "fp16":                          not args.no_fp16,
            "gradient_accumulation_steps":   args.grad_accum  or td["grad_accum"],
            "output_dir":                    args.output_dir,
        },
        "data": {
            "train_path": args.train_path,
            "val_path":   args.val_path   or args.train_path,
            "test_path":  args.test_path  or args.val_path or args.train_path,
            "num_workers": args.num_workers,
        },
    }
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _infer_label_map(data_path: str) -> dict:
    labels: set[str] = set()
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            raw = rec.get("label", "")
            if isinstance(raw, str):
                labels.update(l.strip() for l in raw.split(",") if l.strip())
    return {l: i for i, l in enumerate(sorted(labels))}


def _eta(start: float, done: int, total: int) -> str:
    elapsed = time.time() - start
    if done == 0:
        return "?"
    remaining = elapsed / done * (total - done)
    m, s = divmod(int(remaining), 60)
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_classification(args, cfg, device):
    from medvill.models import MedViLLForClassification
    from medvill.data import ClassificationDataset
    from medvill.tasks import ClassificationTrainer

    if args.label_map:
        with open(args.label_map) as f:
            label2idx = json.load(f)
    else:
        label2idx = _infer_label_map(cfg.data.train_path)
        logger.info(f"Inferred {len(label2idx)} labels from training data")

    label_names = [k for k, _ in sorted(label2idx.items(), key=lambda x: x[1])]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)

    train_ds = ClassificationDataset(
        cfg.data.train_path, tokenizer, label2idx,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=True,
        drop_img_percent=cfg.data.drop_img_percent,
    )
    val_ds = ClassificationDataset(
        cfg.data.val_path, tokenizer, label2idx,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    model = MedViLLForClassification(cfg.model, cfg.image, num_labels=len(label2idx), multilabel=True)

    if args.pretrained:
        missing, unexpected = load_pretrained_encoder(model, args.pretrained)
        logger.info(f"Pretrained encoder loaded | missing={len(missing)} unexpected={len(unexpected)}")

    _log_plan(cfg, len(train_ds), "classification")
    ClassificationTrainer(model, train_loader, val_loader, cfg, device, label_names).train()


def run_retrieval(args, cfg, device):
    from medvill.models import MedViLLForRetrieval
    from medvill.data import RetrievalDataset, RetrievalEvalDataset
    from medvill.tasks import RetrievalTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)

    train_ds = RetrievalDataset(
        cfg.data.train_path, tokenizer,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=True,
    )
    eval_ds = RetrievalEvalDataset(
        cfg.data.val_path, tokenizer,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = MedViLLForRetrieval(cfg.model, cfg.image)

    if args.pretrained:
        missing, unexpected = load_pretrained_encoder(model, args.pretrained)
        logger.info(f"Pretrained encoder loaded | missing={len(missing)} unexpected={len(unexpected)}")

    _log_plan(cfg, len(train_ds), "retrieval")
    RetrievalTrainer(model, train_loader, eval_ds, cfg, device).train()


def run_generation(args, cfg, device):
    from medvill.models import MedViLLForGeneration
    from medvill.data import ReportGenerationDataset
    from medvill.tasks import GenerationTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)

    train_ds = ReportGenerationDataset(
        cfg.data.train_path, tokenizer,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=True,
    )
    val_ds = ReportGenerationDataset(
        cfg.data.val_path, tokenizer,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    model = MedViLLForGeneration(cfg.model, cfg.image)

    if args.pretrained:
        missing, unexpected = load_pretrained_encoder(model, args.pretrained)
        logger.info(f"Pretrained encoder loaded | missing={len(missing)} unexpected={len(unexpected)}")

    _log_plan(cfg, len(train_ds), "generation")
    GenerationTrainer(model, train_loader, val_loader, tokenizer, cfg, device).train()


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_plan(cfg, n_train: int, task: str):
    bs = cfg.training.batch_size
    accum = cfg.training.gradient_accumulation_steps
    epochs = cfg.training.epochs
    steps_per_epoch = math.ceil(n_train / bs)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / accum)
    total_opt_steps = opt_steps_per_epoch * epochs

    # T4 throughput heuristics (samples/sec at batch_size=16, FP16)
    sps = {"classification": 14, "retrieval": 10, "generation": 7}.get(task, 10)
    est_min = (n_train * epochs) / (sps * 60)

    logger.info(
        f"\n{'='*60}\n"
        f"  Task            : {task}\n"
        f"  Train samples   : {n_train}\n"
        f"  Batch size      : {bs}  (grad accum ×{accum})\n"
        f"  Epochs          : {epochs}\n"
        f"  Steps/epoch     : {steps_per_epoch}  ({opt_steps_per_epoch} optimizer steps)\n"
        f"  Total opt steps : {total_opt_steps}\n"
        f"  FP16            : {cfg.training.fp16}\n"
        f"  Output dir      : {cfg.training.output_dir}\n"
        f"  T4 est. time    : ~{est_min:.0f} min\n"
        f"{'='*60}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune MedViLL on the OpenI dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--task", required=True,
        choices=["classification", "retrieval", "generation"],
        help="Which downstream task to fine-tune. (VQA → use scripts/train_vqa.py)",
    )

    # ---- data paths ----
    p.add_argument("--train_path", required=True, help="Path to OpenI train.jsonl")
    p.add_argument("--val_path",   default=None,  help="Path to OpenI val.jsonl (falls back to train_path)")
    p.add_argument("--test_path",  default=None,  help="Path to OpenI test.jsonl")

    # ---- model ----
    p.add_argument("--pretrained", default=None,
                   help="Path to pre-training .pt checkpoint to warm-start from")
    p.add_argument("--config",     default=None,
                   help="Optional YAML config to layer on top of built-in defaults")

    # ---- classification specific ----
    p.add_argument("--label_map",  default=None,
                   help="JSON {label: idx} (classification only; inferred from data if omitted)")

    # ---- training hyper-params (override built-in T4 defaults) ----
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--grad_accum",  type=int,   default=None,
                   help="Gradient accumulation steps")
    p.add_argument("--no_fp16",     action="store_true",
                   help="Disable mixed-precision (useful on CPU or older GPUs)")
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--output_dir",  default="outputs/finetune_openi",
                   help="Directory to save checkpoints")
    p.add_argument("--seed",        type=int,   default=42)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = _build_cfg(args)
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}  |  Task: {args.task}")

    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)

    dispatch = {
        "classification": run_classification,
        "retrieval":      run_retrieval,
        "generation":     run_generation,
    }
    dispatch[args.task](args, cfg, device)


if __name__ == "__main__":
    main()
