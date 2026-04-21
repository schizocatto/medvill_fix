"""Classification metrics: AUC-ROC, F1, Accuracy."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
)
from typing import Optional


def compute_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    multilabel: bool = True,
    threshold: float = 0.5,
    label_names: Optional[list] = None,
) -> dict:
    """
    Compute AUC-ROC, F1, and accuracy.

    Args:
        logits:      (N, C) raw model output (before sigmoid/softmax).
        labels:      (N, C) for multilabel or (N,) for multiclass.
        multilabel:  True → sigmoid + BCE evaluation, False → softmax + CE.
        threshold:   Decision threshold for multilabel F1 / accuracy.
        label_names: Optional class names for per-label reporting.

    Returns:
        dict of aggregate + per-label metrics.
    """
    results: dict = {}

    if multilabel:
        probs = _sigmoid(logits)
        preds = (probs >= threshold).astype(int)

        # Aggregate
        try:
            results["auc_macro"] = roc_auc_score(labels, probs, average="macro")
            results["auc_micro"] = roc_auc_score(labels, probs, average="micro")
            results["ap_macro"] = average_precision_score(labels, probs, average="macro")
        except ValueError:
            results["auc_macro"] = float("nan")
            results["auc_micro"] = float("nan")
            results["ap_macro"] = float("nan")

        results["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        results["f1_micro"] = f1_score(labels, preds, average="micro", zero_division=0)
        results["accuracy"] = accuracy_score(labels, preds)

        # Per-label
        if label_names is not None and labels.ndim == 2:
            for i, name in enumerate(label_names):
                try:
                    results[f"auc_{name}"] = roc_auc_score(labels[:, i], probs[:, i])
                except ValueError:
                    results[f"auc_{name}"] = float("nan")
                results[f"f1_{name}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)

    else:
        probs = _softmax(logits)
        preds = np.argmax(probs, axis=-1)

        try:
            results["auc_macro"] = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            results["auc_macro"] = float("nan")

        results["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        results["accuracy"] = accuracy_score(labels, preds)

    return results


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
