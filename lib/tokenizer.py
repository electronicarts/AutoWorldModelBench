# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Entity tokenizer for discrete world models (D3PM, MaskGIT).

Converts Format D continuous states to discrete tokens using per-game
adaptive quantization. Handles position + alive + gameplay fields.

State layout: [pos_x, pos_y, alive, vel_x, vel_y, stat_0..stat_13]
Tokens:       [pos_x_bin, pos_y_bin, alive_bin, ...gameplay_bins]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import GameQuantConfig, GAME_QUANT, GAMEPLAY_FIELD_SPECS

# Offset in the state tensor where the 14 gameplay stat fields begin
_GAMEPLAY_STATE_OFFSET = 5


class EntityTokenizer:
    """Uniform-bin quantization of entity states for discrete models.

    Per mutable entity per frame produces (3 + K) tokens:
      [pos_x_bin, pos_y_bin, alive_bin, ...gameplay_field_bins]

    where K = number of active gameplay fields for this game.
    Position bins are uniform over [0, 1]. Alive is binary {0, 1}.
    Gameplay fields use integer rounding or uniform [0, 1] binning
    depending on their type.
    """

    def __init__(self, game_id: str | None = None, config: GameQuantConfig | None = None):
        if config is not None:
            self.config = config
        elif game_id is not None:
            self.config = GAME_QUANT.get(game_id, GameQuantConfig())
        else:
            self.config = GameQuantConfig()

    @property
    def K_x(self) -> int:
        return self.config.K_x

    @property
    def K_y(self) -> int:
        return self.config.K_y

    @property
    def vocab_sizes(self) -> list[int]:
        """[K_x, K_y, 2, ...gameplay vocab sizes]."""
        return self.config.vocab_sizes

    @property
    def tokens_per_entity(self) -> int:
        return 3 + len(self.config.gameplay_fields)

    @property
    def total_vocab(self) -> int:
        """Total vocabulary across all token positions (for offset embedding)."""
        return sum(self.vocab_sizes)

    @property
    def num_gameplay_tokens(self) -> int:
        return len(self.config.gameplay_fields)

    def tokenize(self, states: torch.Tensor) -> torch.Tensor:
        """Quantize continuous states to discrete tokens.

        Args:
            states: [..., state_dim] float with state_dim >= 3.
                    Layout: [pos_x, pos_y, alive, vel_x, vel_y, stat_0..stat_13].

        Returns:
            tokens: [..., 3 + K] long.
        """
        pos_x = states[..., 0]
        pos_y = states[..., 1]
        alive = states[..., 2]

        tok_x = (pos_x * self.K_x).long().clamp(0, self.K_x - 1)
        tok_y = (pos_y * self.K_y).long().clamp(0, self.K_y - 1)
        tok_alive = (alive > 0.5).long()

        parts = [tok_x, tok_y, tok_alive]

        state_dim = states.shape[-1]
        for spec in self.config.gameplay_specs:
            field_idx = _GAMEPLAY_STATE_OFFSET + spec.index
            if field_idx >= state_dim:
                # Gameplay fields not present in state tensor — zero token
                parts.append(torch.zeros_like(tok_alive))
                continue
            val = states[..., field_idx]
            if spec.is_integer:
                tok = val.round().long().clamp(0, spec.vocab_size - 1)
            else:
                tok = (val * spec.vocab_size).long().clamp(0, spec.vocab_size - 1)
            parts.append(tok)

        return torch.stack(parts, dim=-1)

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert discrete tokens back to continuous states (bin centers).

        Args:
            tokens: [..., 3 + K] long.

        Returns:
            states: [..., 3 + K] float. [pos_x, pos_y, alive, ...gameplay].
                    Note: velocity is not reconstructed from tokens.
        """
        pos_x = (tokens[..., 0].float() + 0.5) / self.K_x
        pos_y = (tokens[..., 1].float() + 0.5) / self.K_y
        alive = tokens[..., 2].float()

        parts = [pos_x, pos_y, alive]

        for i, spec in enumerate(self.config.gameplay_specs):
            tok = tokens[..., 3 + i]
            if spec.is_integer:
                val = tok.float()
            else:
                val = (tok.float() + 0.5) / spec.vocab_size
            parts.append(val)

        return torch.stack(parts, dim=-1)

    def soft_detokenize_pos(
        self,
        logits_x: torch.Tensor,
        logits_y: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable position reconstruction from logits (for velocity consistency).

        Args:
            logits_x: [..., K_x] raw logits for pos_x.
            logits_y: [..., K_y] raw logits for pos_y.

        Returns:
            [..., 2] float — expected (pos_x, pos_y) under softmax distribution.
        """
        probs_x = F.softmax(logits_x, dim=-1)
        probs_y = F.softmax(logits_y, dim=-1)
        bins_x = (torch.arange(self.K_x, device=logits_x.device).float() + 0.5) / self.K_x
        bins_y = (torch.arange(self.K_y, device=logits_y.device).float() + 0.5) / self.K_y
        exp_x = (probs_x * bins_x).sum(-1)
        exp_y = (probs_y * bins_y).sum(-1)
        return torch.stack([exp_x, exp_y], dim=-1)

    def embed_tokens(
        self,
        tokens: torch.Tensor,
        embeddings: list[torch.nn.Embedding],
    ) -> torch.Tensor:
        """Embed tokens using per-field embedding tables.

        Args:
            tokens: [..., 3 + K] long.
            embeddings: list of (3 + K) Embedding modules.

        Returns:
            [..., (3+K) * embed_dim] float — concatenated embeddings.
        """
        parts = []
        for i, emb in enumerate(embeddings):
            parts.append(emb(tokens[..., i]))
        return torch.cat(parts, dim=-1)
