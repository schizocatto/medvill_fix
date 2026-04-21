"""Diagnosis classification fine-tuning entry point."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from medvill.config import MedViLLConfig
from medvill.models import MedViLLForClassification
from medvill.data import ClassificationDataset
from medvill.tasks import ClassificationTrainer
from medvill.utils import set_seed, load_pretrained_encoder, get_logger

logger = get_logger(__name__)


def main(args):
    cfg: MedViLLConfig = OmegaConf.merge(
        OmegaConf.structured(MedViLLConfig),
        OmegaConf.load(args.config),
    )
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label map: load from file or auto-detect
    if args.label_map:
        with open(args.label_map) as f:
            label2idx = json.load(f)
    else:
        label2idx = _infer_label_map(cfg.data.train_path)
        logger.info(f"Inferred {len(label2idx)} labels: {list(label2idx)[:10]}")

    label_names = [k for k, v in sorted(label2idx.items(), key=lambda x: x[1])]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)

    train_ds = ClassificationDataset(
        cfg.data.train_path, tokenizer, label2idx,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=True,
        drop_img_percent=cfg.data.drop_img_percent,
    )
    val_ds = ClassificationDataset(
        cfg.data.val_path or cfg.data.train_path, tokenizer, label2idx,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)

    model = MedViLLForClassification(
        cfg.model, cfg.image,
        num_labels=len(label2idx),
        multilabel=(args.multilabel),
    )

    if args.pretrained:
        missing, unexpected = load_pretrained_encoder(model, args.pretrained)
        logger.info(f"Loaded pretrained encoder | missing={len(missing)} unexpected={len(unexpected)}")

    trainer = ClassificationTrainer(model, train_loader, val_loader, cfg, device, label_names)
    trainer.train()


def _infer_label_map(data_path: str) -> dict:
    import json as _json
    labels = set()
    with open(data_path) as f:
        for line in f:
            rec = _json.loads(line.strip())
            raw = rec.get("label", "")
            if isinstance(raw, str):
                labels.update(l.strip() for l in raw.split(",") if l.strip())
    return {l: i for i, l in enumerate(sorted(labels))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument("--pretrained", default=None, help="Path to pre-training checkpoint")
    parser.add_argument("--label_map", default=None, help="JSON file: {label: idx}")
    parser.add_argument("--multilabel", action="store_true", default=True)
    main(parser.parse_args())
