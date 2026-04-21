from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING


@dataclass
class ModelConfig:
    bert_model: str = "bert-base-uncased"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    vocab_size: int = 30522


@dataclass
class ImageConfig:
    encoder_type: str = "resnet50"   # resnet50 | vit
    img_size: int = 512
    num_image_embeds: int = 180
    img_embed_pool_type: str = "max"
    img_hidden_sz: int = 2048
    img_channel: int = 3
    patch_size: int = 16             # used when encoder_type == "vit"


@dataclass
class DataConfig:
    train_path: str = MISSING
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    seq_len: int = 253
    max_seq_len: int = 512
    drop_img_percent: float = 0.0
    num_workers: int = 4


@dataclass
class TrainingConfig:
    output_dir: str = "outputs"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 1000
    resume_from: Optional[str] = None


@dataclass
class MedViLLConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Pre-training probabilities
    mlm_prob: float = 0.15
    itm_prob: float = 0.5
    bi_prob: float = 0.5
    s2s_prob: float = 0.5

    # Task selection
    task: str = "pretrain"           # pretrain | classification | retrieval | vqa | generation

    # Task-specific
    num_labels: int = 2
    num_answers: int = 458           # VQA-RAD answer vocab size
    vqa_rad: Optional[str] = None    # chest | head | abd | None (all)
    label_smoothing: float = 0.0
