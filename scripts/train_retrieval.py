"""Image–report retrieval fine-tuning entry point."""
import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from medvill.config import MedViLLConfig
from medvill.models import MedViLLForRetrieval
from medvill.data import RetrievalDataset, RetrievalEvalDataset
from medvill.tasks import RetrievalTrainer
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

    train_ds = RetrievalDataset(
        cfg.data.train_path, tokenizer,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=True,
    )
    eval_ds = RetrievalEvalDataset(
        cfg.data.val_path or cfg.data.test_path, tokenizer,
        seq_len=cfg.data.seq_len, img_size=cfg.image.img_size,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True)

    model = MedViLLForRetrieval(cfg.model, cfg.image)

    if args.pretrained:
        missing, unexpected = load_pretrained_encoder(model, args.pretrained)
        logger.info(f"Loaded pretrained encoder | missing={len(missing)} unexpected={len(unexpected)}")

    trainer = RetrievalTrainer(model, train_loader, eval_ds, cfg, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--pretrained", default=None)
    main(parser.parse_args())
