"""Core MedViLL model and all task-specific wrappers."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Optional, Dict, Any

from .image_encoder import ResNetImageEncoder, PatchEmbedding, ImageBertEmbeddings
from .heads import MLMHead, ITMHead, ClassificationHead, VQAHead, GenerationHead


# ---------------------------------------------------------------------------
# Attention-mask helpers
# ---------------------------------------------------------------------------

def _build_bidirectional_mask(
    batch_size: int, total_len: int, device: torch.device
) -> torch.Tensor:
    """Full bidirectional mask — every token attends to every token."""
    return torch.ones(batch_size, 1, 1, total_len, device=device)


def _build_seq2seq_mask(
    batch_size: int,
    num_image_embeds: int,
    txt_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Image tokens see everything; text tokens see image + left context (causal)."""
    total = num_image_embeds + txt_len
    mask = torch.ones(total, total, device=device)
    txt_causal = torch.tril(torch.ones(txt_len, txt_len, device=device))
    mask[num_image_embeds:, num_image_embeds:] = txt_causal
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, total, total)


def _to_extended_attn_mask(
    mask: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert 0/1 mask to large-negative additive mask expected by BERT encoder."""
    return (1.0 - mask.to(dtype)) * torch.finfo(dtype).min


# ---------------------------------------------------------------------------
# Shared multimodal encoder
# ---------------------------------------------------------------------------

class MedViLLEncoder(nn.Module):
    """
    Combines a CNN/ViT image encoder and BERT text encoder into a single
    joint transformer stack (image tokens prepended to text tokens).
    """

    def __init__(self, model_cfg, image_cfg):
        super().__init__()
        bert_cfg = BertConfig.from_pretrained(
            model_cfg.bert_model,
            hidden_dropout_prob=model_cfg.hidden_dropout_prob,
            attention_probs_dropout_prob=model_cfg.attention_probs_dropout_prob,
        )

        # --- Image encoder ---
        if image_cfg.encoder_type == "resnet50":
            self.img_encoder = ResNetImageEncoder(pool_type=image_cfg.img_embed_pool_type)
            img_hidden_sz = self.img_encoder.out_dim
        elif image_cfg.encoder_type == "vit":
            self.img_encoder = PatchEmbedding(
                img_size=image_cfg.img_size,
                patch_size=image_cfg.patch_size,
                embed_dim=bert_cfg.hidden_size,
            )
            img_hidden_sz = bert_cfg.hidden_size
        else:
            raise ValueError(f"Unknown encoder_type: {image_cfg.encoder_type}")

        self.img_embeddings = ImageBertEmbeddings(
            num_image_embeds=image_cfg.num_image_embeds,
            img_hidden_sz=img_hidden_sz,
            hidden_size=bert_cfg.hidden_size,
            dropout=bert_cfg.hidden_dropout_prob,
        )

        # --- BERT text + shared encoder ---
        bert = BertModel.from_pretrained(model_cfg.bert_model, config=bert_cfg)
        self.txt_embeddings = bert.embeddings
        self.encoder = bert.encoder
        self.pooler = bert.pooler

        self.hidden_size = bert_cfg.hidden_size
        self.num_image_embeds = image_cfg.num_image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        pixel_values: torch.Tensor,
        attn_mask_2d: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, L_txt)
            attention_mask: (B, L_txt)  0/1 padding mask
            token_type_ids: (B, L_txt)
            pixel_values: (B, 3, H, W)
            attn_mask_2d: (B, 1, L_total, L_total) pre-built 2D mask (for seq2seq)
        Returns:
            sequence_output: (B, L_img + L_txt, hidden)
            pooled_output:   (B, hidden)  from CLS
        """
        B = pixel_values.size(0)
        device = pixel_values.device

        # --- Image path ---
        img_feats = self.img_encoder(pixel_values)
        img_type_ids = torch.zeros(
            B, self.num_image_embeds, dtype=torch.long, device=device
        )
        img_emb = self.img_embeddings(img_feats, img_type_ids)  # (B, N_img, H)

        # --- Text path ---
        if token_type_ids is None:
            token_type_ids = torch.ones(
                B, input_ids.size(1), dtype=torch.long, device=device
            )
        txt_emb = self.txt_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        combined = torch.cat([img_emb, txt_emb], dim=1)  # (B, N_img+L_txt, H)

        # --- Build attention mask ---
        if attn_mask_2d is not None:
            ext_mask = _to_extended_attn_mask(attn_mask_2d, combined.dtype)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(B, input_ids.size(1), device=device)
            img_attn = torch.ones(B, self.num_image_embeds, device=device)
            combined_mask = torch.cat([img_attn, attention_mask], dim=1)
            ext_mask = _to_extended_attn_mask(
                combined_mask[:, None, None, :], combined.dtype
            )

        out = self.encoder(combined, attention_mask=ext_mask, return_dict=True)
        seq = out.last_hidden_state
        pooled = self.pooler(seq)
        return seq, pooled


# ---------------------------------------------------------------------------
# Pre-training model
# ---------------------------------------------------------------------------

class MedViLL(nn.Module):
    """MedViLL pre-training with MLM + ITM losses."""

    def __init__(self, model_cfg, image_cfg):
        super().__init__()
        self.encoder = MedViLLEncoder(model_cfg, image_cfg)
        bert_cfg = BertConfig.from_pretrained(model_cfg.bert_model)
        self.mlm_head = MLMHead(bert_cfg)
        self.itm_head = ITMHead(bert_cfg.hidden_size)
        self._criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_image_embeds = image_cfg.num_image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        mlm_labels: Optional[torch.Tensor] = None,
        itm_labels: Optional[torch.Tensor] = None,
        attn_mask_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        seq, pooled = self.encoder(
            input_ids, attention_mask, token_type_ids, pixel_values, attn_mask_2d
        )

        txt_seq = seq[:, self.num_image_embeds:, :]
        mlm_logits = self.mlm_head(txt_seq)
        itm_logits = self.itm_head(pooled)

        mlm_loss = (
            self._criterion(
                mlm_logits.reshape(-1, mlm_logits.size(-1)), mlm_labels.reshape(-1)
            )
            if mlm_labels is not None
            else None
        )
        itm_loss = (
            self._criterion(itm_logits, itm_labels) if itm_labels is not None else None
        )

        loss = None
        if mlm_loss is not None and itm_loss is not None:
            loss = mlm_loss + itm_loss
        elif mlm_loss is not None:
            loss = mlm_loss
        elif itm_loss is not None:
            loss = itm_loss

        return {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "itm_loss": itm_loss,
            "mlm_logits": mlm_logits,
            "itm_logits": itm_logits,
            "sequence_output": seq,
            "pooled_output": pooled,
        }


# ---------------------------------------------------------------------------
# Classification fine-tuning model
# ---------------------------------------------------------------------------

class MedViLLForClassification(nn.Module):
    """Multimodal diagnosis classification (binary or multi-label)."""

    def __init__(self, model_cfg, image_cfg, num_labels: int, multilabel: bool = True):
        super().__init__()
        self.encoder = MedViLLEncoder(model_cfg, image_cfg)
        self.head = ClassificationHead(self.encoder.hidden_size, num_labels)
        self.multilabel = multilabel
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        _, pooled = self.encoder(input_ids, attention_mask, token_type_ids, pixel_values)
        logits = self.head(pooled)

        loss = None
        if labels is not None:
            if self.multilabel:
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits, "pooled": pooled}


# ---------------------------------------------------------------------------
# Retrieval fine-tuning model
# ---------------------------------------------------------------------------

class MedViLLForRetrieval(nn.Module):
    """
    Cross-encoder retrieval: scores an (image, text) pair with the ITM head.
    At inference, all pairs are ranked by the alignment score.
    """

    def __init__(self, model_cfg, image_cfg):
        super().__init__()
        self.encoder = MedViLLEncoder(model_cfg, image_cfg)
        self.itm_head = ITMHead(self.encoder.hidden_size)
        self._criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        _, pooled = self.encoder(input_ids, attention_mask, token_type_ids, pixel_values)
        logits = self.itm_head(pooled)               # (B, 2)
        score = torch.softmax(logits, dim=-1)[:, 1]  # P(aligned)

        loss = None
        if labels is not None:
            loss = self._criterion(logits, labels)

        return {"loss": loss, "logits": logits, "score": score, "pooled": pooled}


# ---------------------------------------------------------------------------
# VQA fine-tuning model
# ---------------------------------------------------------------------------

class MedViLLForVQA(nn.Module):
    """Visual Question Answering: image + question → answer class."""

    def __init__(self, model_cfg, image_cfg, num_answers: int):
        super().__init__()
        self.encoder = MedViLLEncoder(model_cfg, image_cfg)
        self.head = VQAHead(self.encoder.hidden_size, num_answers)
        self._criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        _, pooled = self.encoder(input_ids, attention_mask, token_type_ids, pixel_values)
        logits = self.head(pooled)

        loss = None
        if labels is not None:
            loss = self._criterion(logits, labels)

        return {"loss": loss, "logits": logits, "pooled": pooled}


# ---------------------------------------------------------------------------
# Generation fine-tuning model
# ---------------------------------------------------------------------------

class MedViLLForGeneration(nn.Module):
    """
    Seq2Seq report generation.

    The attention mask is built so that:
    - Image tokens attend to all tokens (bidirectional)
    - Text tokens attend to image tokens + left context (causal)

    This is a single-stack decoder-only approach (no separate encoder/decoder),
    identical to the original MedViLL design.
    """

    def __init__(self, model_cfg, image_cfg):
        super().__init__()
        self.encoder = MedViLLEncoder(model_cfg, image_cfg)
        bert_cfg = BertConfig.from_pretrained(model_cfg.bert_model)
        self.gen_head = GenerationHead(bert_cfg.hidden_size, bert_cfg.vocab_size)
        self._criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_image_embeds = image_cfg.num_image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        lm_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, L = input_ids.shape
        device = input_ids.device

        # Build seq2seq 2D mask
        attn_mask_2d = _build_seq2seq_mask(B, self.num_image_embeds, L, device)

        seq, _ = self.encoder(
            input_ids, attention_mask, token_type_ids, pixel_values, attn_mask_2d
        )

        txt_seq = seq[:, self.num_image_embeds:, :]   # (B, L, H)
        logits = self.gen_head(txt_seq)                 # (B, L, vocab)

        loss = None
        if lm_labels is not None:
            loss = self._criterion(
                logits.reshape(-1, logits.size(-1)), lm_labels.reshape(-1)
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> list[str]:
        """Greedy / temperature-scaled auto-regressive generation."""
        device = pixel_values.device
        B = pixel_values.size(0)
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        generated = torch.full((B, 1), cls_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            attn_mask = torch.ones(B, generated.size(1), device=device)
            tok_types = torch.ones(B, generated.size(1), dtype=torch.long, device=device)

            out = self.forward(
                input_ids=generated,
                attention_mask=attn_mask,
                token_type_ids=tok_types,
                pixel_values=pixel_values,
            )
            next_logits = out["logits"][:, -1, :] / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            finished |= (next_token.squeeze(-1) == sep_id)
            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return texts
