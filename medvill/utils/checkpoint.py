"""Save / load model checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: dict,
    output_dir: str,
    name: str = "checkpoint",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_epoch{epoch:03d}_step{step}.pt"
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    # Keep a "best" pointer
    (output_dir / f"{name}_latest.pt").unlink(missing_ok=True)
    torch.save({"path": str(path)}, output_dir / f"{name}_latest.pt")
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def load_pretrained_encoder(
    model,
    pretrained_path: str,
    strict: bool = False,
    map_location: str = "cpu",
) -> tuple:
    """Load only encoder weights from a pre-training checkpoint."""
    ckpt = torch.load(pretrained_path, map_location=map_location)
    state = ckpt.get("model_state_dict", ckpt)
    # Filter to encoder keys only
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=strict)
    return missing, unexpected
