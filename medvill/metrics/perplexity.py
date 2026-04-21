"""Perplexity computation for language-model outputs."""
from __future__ import annotations
import math
import torch
from typing import List, Optional


def perplexity_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from model logits and ground-truth labels.

    Args:
        logits: (B, L, vocab_size) — raw logits (before softmax).
        labels: (B, L)             — token ids; positions with ``ignore_index`` are excluded.
        ignore_index: value used to mark padding / non-target positions.

    Returns:
        Perplexity as a Python float.  Returns ``inf`` if no valid tokens.
    """
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    with torch.no_grad():
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        n_valid = (labels != ignore_index).sum().item()
    if n_valid == 0:
        return float("inf")
    return math.exp(loss.item() / n_valid)


def perplexity_from_loss(avg_loss: float) -> float:
    """Convert average cross-entropy loss to perplexity."""
    return math.exp(avg_loss)


def batch_perplexity(
    logits_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    ignore_index: int = -100,
) -> float:
    """
    Aggregate perplexity over multiple batches without loading all logits at once.

    Each call accumulates total_loss and total_tokens, then returns
    exp(total_loss / total_tokens).
    """
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")

    with torch.no_grad():
        for logits, labels in zip(logits_list, labels_list):
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            n = (labels != ignore_index).sum().item()
            total_loss += loss.item()
            total_tokens += n

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


class RunningPerplexity:
    """Accumulates cross-entropy loss incrementally (memory-efficient)."""

    def __init__(self, ignore_index: int = -100):
        self._total_loss = 0.0
        self._total_tokens = 0
        self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        loss = self._loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        n = (labels != self.ignore_index).sum().item()
        self._total_loss += loss.item()
        self._total_tokens += n

    def compute(self) -> float:
        if self._total_tokens == 0:
            return float("inf")
        return math.exp(self._total_loss / self._total_tokens)

    def reset(self) -> None:
        self._total_loss = 0.0
        self._total_tokens = 0
