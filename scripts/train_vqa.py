"""VQA-RAD fine-tuning entry point."""
import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from medvill.config import MedViLLConfig
from medvill.models import MedViLLForVQA
from medvill.data import VQARadDataset
from medvill.tasks import VQATrainer
from medvill.metrics import compute_bleu4
from medvill.utils import set_seed, load_pretrained_encoder, get_logger

logger = get_logger(__name__)


def main(args):
    cfg: MedViLLConfig = OmegaConf.merge(
        OmegaConf.structured(MedViLLConfig),
        OmegaConf.load(args.config),
    )
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)

    # Build/load answer vocabulary from train set first
    train_ds = VQARadDataset(
        args.vqa_dir, tokenizer,
        split="train", organ=cfg.vqa_rad,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=True,
    )
    val_ds = VQARadDataset(
        args.vqa_dir, tokenizer,
        split="test", organ=cfg.vqa_rad,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False,
        # reuse vocab built above
    )
    # Align vocab between splits
    val_ds.ans2label = train_ds.ans2label
    val_ds.label2ans = train_ds.label2ans
    val_ds.num_answers = train_ds.num_answers

    logger.info(f"VQA answer vocab size: {train_ds.num_answers}")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=cfg.data.num_workers)

    model = MedViLLForVQA(cfg.model, cfg.image, num_answers=train_ds.num_answers)

    if args.pretrained:
        missing, unexpected = load_pretrained_encoder(model, args.pretrained)
        logger.info(f"Loaded pretrained encoder | missing={len(missing)} unexpected={len(unexpected)}")

    trainer = VQATrainer(model, train_loader, val_loader, cfg, device, label2ans=train_ds.label2ans)
    trainer.train()

    # --- Post-training: BLEU-4 + Perplexity on test set ---
    logger.info("Computing generation metrics on test set...")
    pred_texts, gt_texts = trainer.predict(val_loader)
    bleu = compute_bleu4(pred_texts, gt_texts)
    logger.info(f"VQA BLEU-4: {bleu['bleu4']:.2f}")
    logger.info(f"Full BLEU scores: {bleu}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vqa.yaml")
    parser.add_argument("--vqa_dir", required=True, help="Path to VQA-RAD dataset directory")
    parser.add_argument("--pretrained", default=None)
    main(parser.parse_args())
