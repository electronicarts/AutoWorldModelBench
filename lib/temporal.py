# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Shared temporal context encoder for D3PM and MaskGIT baselines.

Block-causal Transformer: within-frame bidirectional attention,
cross-frame causal attention. Used as the history encoder for both
discrete world model families.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_block_causal_mask(
    num_frames: int,
    num_entities: int,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    """Build block-causal attention mask for space-time entity sequences.

    Token (frame f, entity e) can attend to (frame f', entity e') iff:
      f' <= f  AND  entity_mask[e'] == True

    Within the same frame: full bidirectional attention.
    Across frames: causal (can only attend to past/current).

    Args:
        num_frames: F (number of frames in sequence)
        num_entities: N (max entities per frame, after padding)
        entity_mask: [B, N] bool — True for valid entities

    Returns:
        mask: [B, F*N, F*N] bool — True means CAN attend (PyTorch convention for additive mask is opposite, but we convert at usage)
    """
    B = entity_mask.shape[0]
    S = num_frames * num_entities
    device = entity_mask.device

    # Frame index for each position in the sequence
    frame_idx = torch.arange(S, device=device) // num_entities  # [S]

    # Temporal causality: frame_idx[i] >= frame_idx[j] means i can attend to j
    temporal_ok = frame_idx.unsqueeze(0) >= frame_idx.unsqueeze(1)  # [S, S]

    # Entity validity: tile entity_mask across frames
    # ent_valid[b, j] = entity_mask[b, j % N]
    ent_valid = entity_mask.repeat(1, num_frames)  # [B, S]

    # Combine: [B, S, S]
    # mask[b, i, j] = temporal_ok[i, j] AND ent_valid[b, j]
    mask = temporal_ok.unsqueeze(0) & ent_valid.unsqueeze(1)

    return mask  # [B, S, S], True = can attend


def make_cross_attn_mask(
    num_target_frames: int,
    num_context_frames: int,
    num_entities: int,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    """Build cross-attention mask: target frame t attends to context frames 0..t.

    Target frame index t (0-based in target seq) corresponds to predicting
    frame t+1 from context frames 0..t.

    Args:
        num_target_frames: W (number of target frames to predict)
        num_context_frames: W+1 (context frames 0..W)
        num_entities: N
        entity_mask: [B, N] bool

    Returns:
        mask: [B, W*N, (W+1)*N] bool — True = can attend
    """
    B = entity_mask.shape[0]
    device = entity_mask.device
    S_tgt = num_target_frames * num_entities
    S_ctx = num_context_frames * num_entities

    # Target token i in target frame t_tgt can attend to context token j in context frame t_ctx
    # iff t_ctx <= t_tgt (0-indexed: target frame 0 predicts frame 1 from context frame 0)
    tgt_frame = torch.arange(S_tgt, device=device) // num_entities  # [S_tgt]
    ctx_frame = torch.arange(S_ctx, device=device) // num_entities  # [S_ctx]

    # Target frame t (0-indexed) should see context frames 0..t
    temporal_ok = tgt_frame.unsqueeze(1) >= ctx_frame.unsqueeze(0)  # [S_tgt, S_ctx]

    # Entity validity in context
    ctx_valid = entity_mask.repeat(1, num_context_frames)  # [B, S_ctx]

    mask = temporal_ok.unsqueeze(0) & ctx_valid.unsqueeze(1)
    return mask  # [B, S_tgt, S_ctx]


def _sanitize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure no row in attention mask is all-False.

    Rows that are entirely False (= all keys masked) would cause softmax
    to produce NaN. For those rows, allow attending to all keys — the output
    is meaningless but gradients stay clean. These positions must be masked
    out in the loss.
    """
    all_masked = ~mask.any(dim=-1, keepdim=True)  # [B, S, 1]
    return mask | all_masked  # all-False rows become all-True


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with self-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        if attn_mask is not None:
            safe_mask = _sanitize_mask(attn_mask)
            num_heads = self.attn.num_heads
            float_mask = torch.zeros(
                safe_mask.shape, dtype=h.dtype, device=h.device
            )
            float_mask.masked_fill_(~safe_mask, float("-inf"))
            float_mask = float_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            float_mask = float_mask.reshape(-1, float_mask.shape[2], float_mask.shape[3])
            h, _ = self.attn(h, h, h, attn_mask=float_mask)
        else:
            h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention from queries to key-value context, plus self-attention among queries."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
        self_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attention: queries attend to context
        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        if cross_mask is not None:
            safe_cm = _sanitize_mask(cross_mask)
            num_heads = self.cross_attn.num_heads
            fm = torch.zeros(safe_cm.shape, dtype=q.dtype, device=q.device)
            fm.masked_fill_(~safe_cm, float("-inf"))
            fm = fm.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(-1, fm.shape[1], fm.shape[2])
            h, _ = self.cross_attn(q, kv, kv, attn_mask=fm)
        else:
            h, _ = self.cross_attn(q, kv, kv)
        queries = queries + h

        # Self-attention among queries (bidirectional within each target frame)
        q2 = self.norm_self(queries)
        if self_mask is not None:
            safe_sm = _sanitize_mask(self_mask)
            num_heads = self.self_attn.num_heads
            fm2 = torch.zeros(safe_sm.shape, dtype=q2.dtype, device=q2.device)
            fm2.masked_fill_(~safe_sm, float("-inf"))
            fm2 = fm2.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(-1, fm2.shape[1], fm2.shape[2])
            h2, _ = self.self_attn(q2, q2, q2, attn_mask=fm2)
        else:
            h2, _ = self.self_attn(q2, q2, q2)
        queries = queries + h2

        # FFN
        queries = queries + self.ff(self.norm_ff(queries))
        return queries


class TemporalContextEncoder(nn.Module):
    """Block-causal Transformer over space-time entity sequences.

    Within each frame: full bidirectional attention among entities.
    Across frames: causal (frame t can attend to frames 0..t).

    Used as shared backbone for D3PM and MaskGIT.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        entity_mask: torch.Tensor,
        num_frames: int,
        num_entities: int,
    ) -> torch.Tensor:
        """Encode entity tokens with block-causal temporal attention.

        Args:
            tokens: [B, F*N, H] — entity tokens laid out as [frame_0_ents, frame_1_ents, ...]
            entity_mask: [B, N] — True for valid entities
            num_frames: F
            num_entities: N

        Returns:
            [B, F*N, H] — contextualized representations
        """
        mask = make_block_causal_mask(num_frames, num_entities, entity_mask)

        x = tokens
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        return self.norm_out(x)
