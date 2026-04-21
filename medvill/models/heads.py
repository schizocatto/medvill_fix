from __future__ import annotations
import torch
import torch.nn as nn
from transformers import BertConfig


class MLMHead(nn.Module):
    """Masked Language Model prediction head (dense → GELU → LayerNorm → vocab)."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(hidden_states))))


class ITMHead(nn.Module):
    """Image–Text Matching binary classification head."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


class ClassificationHead(nn.Module):
    """Linear head for single- or multi-label diagnosis classification."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(pooled))


class VQAHead(nn.Module):
    """Two-layer MLP head for VQA answer classification."""

    def __init__(self, hidden_size: int, num_answers: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size * 2, num_answers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(pooled))))


class GenerationHead(nn.Module):
    """LM head for token-level generation (seq2seq style)."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.act(self.dense(hidden_states))))
