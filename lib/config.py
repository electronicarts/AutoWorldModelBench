# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Shared configuration for all baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── Gameplay field definitions (14 unified stat fields, alphabetical) ──

GAMEPLAY_FIELD_NAMES: list[str] = [
    "age", "col", "direction", "hp", "lane_type",
    "on_ground", "piece_type", "reached", "rotation",
    "row", "segment_index", "size_class", "tagged", "tagged_count",
]

NUM_GAMEPLAY_FIELDS: int = 14


@dataclass
class GameplayFieldSpec:
    """Quantization spec for one gameplay field."""
    index: int        # 0-13 position in 14-field array
    vocab_size: int   # number of discrete bins / categories
    is_integer: bool  # True: round to int; False: uniform [0,1] binning


GAMEPLAY_FIELD_SPECS: list[GameplayFieldSpec] = [
    GameplayFieldSpec(0,  32,  False),  # age (continuous, normalized)
    GameplayFieldSpec(1,  16,  True),   # col
    GameplayFieldSpec(2,  4,   True),   # direction
    GameplayFieldSpec(3,  16,  False),  # hp (continuous [0,1])
    GameplayFieldSpec(4,  5,   True),   # lane_type
    GameplayFieldSpec(5,  2,   True),   # on_ground (binary)
    GameplayFieldSpec(6,  7,   True),   # piece_type
    GameplayFieldSpec(7,  2,   True),   # reached (binary)
    GameplayFieldSpec(8,  4,   True),   # rotation
    GameplayFieldSpec(9,  16,  True),   # row
    GameplayFieldSpec(10, 32,  True),   # segment_index
    GameplayFieldSpec(11, 3,   True),   # size_class
    GameplayFieldSpec(12, 2,   True),   # tagged (binary)
    GameplayFieldSpec(13, 16,  True),   # tagged_count
]

# Per-game active gameplay field indices
GAME_GAMEPLAY_FIELDS: dict[str, list[int]] = {
    "asteroids":       [0, 3, 11],    # age, hp, size_class
    "breakout":        [1, 3, 9],     # col, hp, row
    "frogger":         [4, 7],        # lane_type, reached
    "kong":            [],            # none
    "platformer":      [5],           # on_ground
    "pong":            [],            # none
    "racer":           [],            # none
    "snake":           [2, 10],       # direction, segment_index
}


@dataclass
class GameQuantConfig:
    """Per-game quantization settings for discrete models."""
    K_x: int = 256
    K_y: int = 256
    gameplay_fields: list[int] = field(default_factory=list)

    @property
    def vocab_sizes(self) -> list[int]:
        """Vocabulary size per token: [K_x, K_y, 2(alive), ...gameplay]."""
        base = [self.K_x, self.K_y, 2]
        for fi in self.gameplay_fields:
            base.append(GAMEPLAY_FIELD_SPECS[fi].vocab_size)
        return base

    @property
    def gameplay_specs(self) -> list[GameplayFieldSpec]:
        """Specs for active gameplay fields only."""
        return [GAMEPLAY_FIELD_SPECS[fi] for fi in self.gameplay_fields]


# Per-game quantization: matches native coordinate resolution
GAME_QUANT: dict[str, GameQuantConfig] = {
    "asteroids":       GameQuantConfig(K_x=200, K_y=200, gameplay_fields=[0, 3, 11]),
    "breakout":        GameQuantConfig(K_x=160, K_y=200, gameplay_fields=[1, 3, 9]),
    "frogger":         GameQuantConfig(K_x=256, K_y=224, gameplay_fields=[4, 7]),
    "kong":            GameQuantConfig(K_x=160, K_y=120, gameplay_fields=[]),
    "platformer":      GameQuantConfig(K_x=160, K_y=120, gameplay_fields=[5]),
    "pong":            GameQuantConfig(K_x=160, K_y=120, gameplay_fields=[]),
    "racer":           GameQuantConfig(K_x=200, K_y=200, gameplay_fields=[]),
    "snake":           GameQuantConfig(K_x=20,  K_y=20,  gameplay_fields=[2, 10]),
}


@dataclass
class BaselineConfig:
    """Full configuration for a baseline training run."""

    # ── Data ──
    game_id: str = "snake"
    data_dir: Path = Path("data/format_d")
    raw_data_dir: Path = Path("")  # ECS-dategen raw JSONL root
    window_size: int = 8
    batch_size: int = 128

    # ── Training ──
    lr: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    max_steps: int | None = None
    grad_clip: float = 1.0
    warmup_steps: int = 500
    patience: int = 10  # early stopping

    # ── Model ──
    hidden_dim: int = 256
    latent_dim: int = 64       # RSSM stochastic latent
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # ── Discrete model ──
    diffusion_steps: int = 100  # D3PM noise steps
    mask_iterations: int = 8    # MaskGIT decode iterations

    # ── Eval ──
    eval_every_n_steps: int = 500
    eval_episodes: int = 50
    rollout_horizon: int = 100

    # ── System ──
    model_type: str = "rssm"  # rssm | d3pm | maskgit
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    output_dir: Path = Path("outputs")

    @property
    def quant(self) -> GameQuantConfig:
        return GAME_QUANT.get(self.game_id, GameQuantConfig())
