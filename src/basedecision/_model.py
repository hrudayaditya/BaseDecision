"""The BaseDecision network: an encoder backbone and a typed decision head. Inference only."""
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head.

    The network behind every local answer: the encoder reads the packed request, a small
    transformer head refines the hidden states, and a scorer reads one logit per option at that
    option's marker token. Inference only; the training code is not part of this package.
    """

    def __init__(self, encoder: Any, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1) -> None:
        """Build the head around ``encoder`` (a Hugging Face ``ModernBertModel``).

        Args:
            encoder: The backbone; must expose ``config.hidden_size``.
            head_layers: Transformer layers in the head (0 disables it).
            n_act: Outputs of the action head, whose result is not interpreted by this package.
            dropout: Dropout inside the head (inactive in ``eval()`` mode).
        """
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor, marker_pos: Tensor, marker_mask: Tensor, qtype: Tensor, detach_encoder: bool = False) -> tuple[Tensor, Tensor]:
        """Score every option of a padded batch.

        Args:
            input_ids: Token ids, shape ``(batch, length)``.
            attention_mask: 1 for real tokens, 0 for padding, same shape.
            marker_pos: Position of each option's marker token, shape ``(batch, options)``.
            marker_mask: True where ``marker_pos`` holds a real option, same shape.
            qtype: Question-type index per request, shape ``(batch,)``.
            detach_encoder: Stop gradients at the encoder (training only).

        Returns:
            ``(logits, action_logits)``: one logit per option (padding options are -1e4) and the
            action head's output.
        """
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...)
            # has nothing to select for the second slot and raises. The answer
            # is still well-defined: softmax over one logit is 1.0 regardless of
            # its value, so pad the missing second entry with 0.0 - that gives
            # the act head top1 - top2 == 1.0, the same "fully decided" signal
            # it would see for any other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits
