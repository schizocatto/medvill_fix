"""Retrieval metrics: Recall@K, MRR, and Precision@K."""
from __future__ import annotations
import numpy as np
from typing import List, Union


def recall_at_k(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute Recall@K for a single query.

    Args:
        scores:        1-D array of similarity scores for all gallery items.
        ground_truth:  1-D boolean / int array; 1 = relevant.
        k:             Cutoff rank.
    Returns:
        1.0 if at least one relevant item is in top-k, else 0.0.
    """
    ranked = np.argsort(-scores)[:k]
    return float(ground_truth[ranked].any())


def mean_recall_at_k(
    all_scores: np.ndarray,
    all_gt: np.ndarray,
    k: int,
) -> float:
    """
    Average Recall@K across all queries.

    Args:
        all_scores: (n_queries, n_gallery)
        all_gt:     (n_queries, n_gallery) binary relevance matrix.
    """
    return float(np.mean([recall_at_k(all_scores[i], all_gt[i], k) for i in range(len(all_scores))]))


def mean_reciprocal_rank(
    all_scores: np.ndarray,
    all_gt: np.ndarray,
) -> float:
    """Mean Reciprocal Rank (MRR) across all queries."""
    mrrs = []
    for scores, gt in zip(all_scores, all_gt):
        ranked = np.argsort(-scores)
        for rank, idx in enumerate(ranked, 1):
            if gt[idx]:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)
    return float(np.mean(mrrs))


def compute_retrieval_metrics(
    scores: np.ndarray,
    ks: List[int] = [1, 5, 10],
) -> dict:
    """
    Full retrieval evaluation assuming a *diagonal* ground truth (1 query per gallery item).

    ``scores`` shape: (N, N) where scores[i, j] = similarity of query i to gallery j.
    Ground truth: identity matrix (each query has exactly one correct gallery item).
    """
    n = scores.shape[0]
    gt = np.eye(n, dtype=bool)

    results: dict = {}
    for k in ks:
        results[f"R@{k}"] = mean_recall_at_k(scores, gt, k)
    results["MRR"] = mean_reciprocal_rank(scores, gt)

    # Median rank
    ranks = []
    for i in range(n):
        ranked = np.argsort(-scores[i])
        rank = np.where(ranked == i)[0][0] + 1
        ranks.append(rank)
    results["median_rank"] = float(np.median(ranks))
    results["mean_rank"] = float(np.mean(ranks))

    return results


def scores_from_itm_model(
    model,
    image_dataset,
    text_dataset,
    tokenizer,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Build full (N_img x N_txt) ITM score matrix by evaluating all pairs.

    This is O(N²) and is feasible only for small eval sets (< 1 000 items).
    For large-scale retrieval use dual-encoder embeddings instead.
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    n = len(image_dataset)
    score_matrix = np.zeros((n, n), dtype=np.float32)

    img_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
    txt_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i_batch, img_batch in enumerate(img_loader):
            for j_batch, txt_batch in enumerate(txt_loader):
                pv = img_batch["pixel_values"].to(device)
                ids = txt_batch["input_ids"].to(device)
                am = txt_batch["attention_mask"].to(device)
                tti = txt_batch["token_type_ids"].to(device)

                # Expand to all (img, txt) pairs in this sub-block
                B_i, B_j = pv.size(0), ids.size(0)
                pv_exp = pv.unsqueeze(1).expand(-1, B_j, -1, -1, -1).reshape(B_i * B_j, *pv.shape[1:])
                ids_exp = ids.unsqueeze(0).expand(B_i, -1, -1).reshape(B_i * B_j, -1)
                am_exp = am.unsqueeze(0).expand(B_i, -1, -1).reshape(B_i * B_j, -1)
                tti_exp = tti.unsqueeze(0).expand(B_i, -1, -1).reshape(B_i * B_j, -1)

                out = model(ids_exp, am_exp, tti_exp, pv_exp)
                scores = out["score"].reshape(B_i, B_j).cpu().numpy()

                r_start = i_batch * batch_size
                c_start = j_batch * batch_size
                score_matrix[r_start:r_start + B_i, c_start:c_start + B_j] = scores

    return score_matrix
