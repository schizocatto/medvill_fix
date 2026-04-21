"""BLEU-4 computation using sacrebleu."""
from __future__ import annotations
from typing import List
import sacrebleu


def compute_bleu4(
    hypotheses: List[str],
    references: List[str],
    lowercase: bool = True,
) -> dict:
    """
    Compute corpus-level BLEU-4.

    Args:
        hypotheses: List of generated strings.
        references:  List of ground-truth strings (same length).
        lowercase:   Whether to lowercase before scoring.

    Returns:
        dict with keys ``bleu4``, ``bleu1``, ``bleu2``, ``bleu3``,
        ``precision_1``, ..., ``precision_4``, ``brevity_penalty``.
    """
    if lowercase:
        hypotheses = [h.lower() for h in hypotheses]
        references = [r.lower() for r in references]

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "bleu4": result.score,
        "bleu1": result.precisions[0],
        "bleu2": result.precisions[1],
        "bleu3": result.precisions[2],
        "bleu4_precision": result.precisions[3],
        "brevity_penalty": result.bp,
        "ratio": result.sys_len / result.ref_len if result.ref_len > 0 else 0.0,
    }


def compute_sentence_bleu4(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU-4 (used for per-sample logging)."""
    result = sacrebleu.sentence_bleu(hypothesis.lower(), [reference.lower()])
    return result.score
