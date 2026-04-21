"""
Unified evaluation script.

Usage examples:
    python scripts/evaluate.py --task classification --checkpoint outputs/cls_best.pt --config configs/classification.yaml --data_path data/test.jsonl
    python scripts/evaluate.py --task retrieval     --checkpoint outputs/ret_best.pt --config configs/retrieval.yaml  --data_path data/test.jsonl
    python scripts/evaluate.py --task vqa           --checkpoint outputs/vqa_best.pt --config configs/vqa.yaml         --vqa_dir data/vqa_rad/
    python scripts/evaluate.py --task generation    --checkpoint outputs/gen_best.pt --config configs/generation.yaml  --data_path data/test.jsonl
"""
import argparse
import json
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

from medvill.config import MedViLLConfig
from medvill.utils import set_seed, load_checkpoint, get_logger, log_metrics
from medvill.metrics import (
    compute_bleu4, RunningPerplexity,
    compute_retrieval_metrics, compute_classification_metrics,
)

logger = get_logger(__name__)


def eval_classification(cfg, checkpoint, data_path, device, label_map=None):
    from medvill.models import MedViLLForClassification
    from medvill.data import ClassificationDataset

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)

    if label_map:
        with open(label_map) as f:
            label2idx = json.load(f)
    else:
        from scripts.train_classification import _infer_label_map
        label2idx = _infer_label_map(data_path)

    label_names = [k for k, v in sorted(label2idx.items(), key=lambda x: x[1])]
    ds = ClassificationDataset(data_path, tokenizer, label2idx,
                               seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = MedViLLForClassification(cfg.model, cfg.image, num_labels=len(label2idx))
    load_checkpoint(checkpoint, model, map_location=str(device))
    model = model.to(device).eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out["logits"].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    metrics = compute_classification_metrics(
        np.concatenate(all_logits), np.concatenate(all_labels),
        label_names=label_names,
    )
    log_metrics(logger, metrics, prefix="[Cls] ")
    return metrics


def eval_retrieval(cfg, checkpoint, data_path, device):
    from medvill.models import MedViLLForRetrieval
    from medvill.data import RetrievalEvalDataset

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)
    ds = RetrievalEvalDataset(data_path, tokenizer, seq_len=cfg.data.seq_len, img_size=cfg.image.img_size)

    model = MedViLLForRetrieval(cfg.model, cfg.image)
    load_checkpoint(checkpoint, model, map_location=str(device))
    model = model.to(device).eval()

    from medvill.tasks.retrieval import RetrievalTrainer
    dummy_loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False)

    # Build NxN score matrix
    n = len(ds)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    all_pv, all_ids, all_am, all_tti = [], [], [], []
    for batch in loader:
        all_pv.append(batch["pixel_values"])
        all_ids.append(batch["input_ids"])
        all_am.append(batch["attention_mask"])
        all_tti.append(batch["token_type_ids"])

    all_pv = torch.cat(all_pv).to(device)
    all_ids = torch.cat(all_ids).to(device)
    all_am = torch.cat(all_am).to(device)
    all_tti = torch.cat(all_tti).to(device)

    scores_i2t = np.zeros((n, n), dtype=np.float32)
    chunk = cfg.training.batch_size
    with torch.no_grad():
        for i in range(0, n, chunk):
            pv_i = all_pv[i:i+chunk]; ni = pv_i.size(0)
            for j in range(0, n, chunk):
                ids_j = all_ids[j:j+chunk]; nj = ids_j.size(0)
                am_j = all_am[j:j+chunk]; tti_j = all_tti[j:j+chunk]
                pv_exp = pv_i.unsqueeze(1).expand(-1, nj, -1, -1, -1).reshape(ni*nj, *pv_i.shape[1:])
                ids_exp = ids_j.unsqueeze(0).expand(ni, -1, -1).reshape(ni*nj, -1)
                am_exp = am_j.unsqueeze(0).expand(ni, -1, -1).reshape(ni*nj, -1)
                tti_exp = tti_j.unsqueeze(0).expand(ni, -1, -1).reshape(ni*nj, -1)
                out = model(ids_exp, am_exp, tti_exp, pv_exp)
                scores_i2t[i:i+ni, j:j+nj] = out["score"].reshape(ni, nj).cpu().numpy()

    i2t = compute_retrieval_metrics(scores_i2t)
    t2i = compute_retrieval_metrics(scores_i2t.T)
    metrics = {**{f"I2T_{k}": v for k, v in i2t.items()}, **{f"T2I_{k}": v for k, v in t2i.items()}}
    log_metrics(logger, metrics, prefix="[Ret] ")
    return metrics


def eval_vqa(cfg, checkpoint, vqa_dir, device):
    from medvill.models import MedViLLForVQA
    from medvill.data import VQARadDataset
    from medvill.metrics import compute_bleu4, RunningPerplexity

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)
    train_ds = VQARadDataset(vqa_dir, tokenizer, split="train", organ=cfg.vqa_rad, seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False)
    test_ds = VQARadDataset(vqa_dir, tokenizer, split="test", organ=cfg.vqa_rad, seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False)
    test_ds.ans2label = train_ds.ans2label
    test_ds.label2ans = train_ds.label2ans

    model = MedViLLForVQA(cfg.model, cfg.image, num_answers=train_ds.num_answers)
    load_checkpoint(checkpoint, model, map_location=str(device))
    model = model.to(device).eval()

    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    all_preds, all_labels, pred_texts, gt_texts = [], [], [], []
    ppl = RunningPerplexity()

    with torch.no_grad():
        for batch in tqdm(loader):
            tb = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**tb)
            preds = out["logits"].argmax(-1).cpu().tolist()
            labels = tb["labels"].cpu().tolist()
            all_preds.extend(preds); all_labels.extend(labels)
            pred_texts.extend([train_ds.label2ans[p] for p in preds])
            gt_texts.extend([train_ds.label2ans[l] for l in labels])

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    bleu = compute_bleu4(pred_texts, gt_texts)
    metrics = {"accuracy": acc, **bleu}
    log_metrics(logger, metrics, prefix="[VQA] ")
    return metrics


def eval_generation(cfg, checkpoint, data_path, device):
    from medvill.models import MedViLLForGeneration
    from medvill.data import ReportGenerationDataset

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)
    ds = ReportGenerationDataset(data_path, tokenizer, seq_len=cfg.data.seq_len, img_size=cfg.image.img_size, train=False)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = MedViLLForGeneration(cfg.model, cfg.image)
    load_checkpoint(checkpoint, model, map_location=str(device))
    model = model.to(device).eval()

    hypotheses, references = [], []
    ppl = RunningPerplexity()

    with torch.no_grad():
        for batch in tqdm(loader):
            target_texts = batch.pop("target_text", [])
            tb = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**tb)
            ppl.update(out["logits"], tb["lm_labels"])
            preds = model.generate(tb["pixel_values"], tokenizer)
            hypotheses.extend(preds)
            references.extend(target_texts)

    bleu = compute_bleu4(hypotheses, references)
    metrics = {"perplexity": ppl.compute(), **bleu}
    log_metrics(logger, metrics, prefix="[Gen] ")
    return metrics


def main(args):
    cfg: MedViLLConfig = OmegaConf.merge(
        OmegaConf.structured(MedViLLConfig),
        OmegaConf.load(args.config),
    )
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "classification":
        eval_classification(cfg, args.checkpoint, args.data_path, device, args.label_map)
    elif args.task == "retrieval":
        eval_retrieval(cfg, args.checkpoint, args.data_path, device)
    elif args.task == "vqa":
        eval_vqa(cfg, args.checkpoint, args.vqa_dir, device)
    elif args.task == "generation":
        eval_generation(cfg, args.checkpoint, args.data_path, device)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["classification", "retrieval", "vqa", "generation"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--vqa_dir", default=None)
    parser.add_argument("--label_map", default=None)
    main(parser.parse_args())
