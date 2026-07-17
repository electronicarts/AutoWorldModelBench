# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Parquet episode loading and in-memory batch serving.

Loads self-contained parquet files (one per split per game),
builds windowed training tensors, and serves random batches
from CPU memory with zero disk I/O during training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


# Tensor columns stored as binary in parquet
TENSOR_COLUMNS = [
    "registry", "states", "actions", "action_masks",
    "globals", "global_masks", "rewards", "terminals",
    "mutable_mask", "gameplay_mask", "type_ids", "slot_ids",
]


def _bytes_to_array(data: bytes, shape: list[int], dtype: str) -> np.ndarray:
    """Reconstruct numpy array from raw bytes + shape + dtype metadata."""
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()


def load_episodes_from_parquet(
    parquet_path: Path,
    max_episodes: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Read a parquet file into a list of episode dicts with numpy arrays.

    Args:
        parquet_path: Path to the parquet file.
        max_episodes: If set, subsample to at most this many episodes
            *before* deserializing tensor columns (saves time and memory
            on large parquet files).
        seed: Random seed for reproducible subsampling.

    Each episode dict has:
        game_id, episode_id, num_frames, num_entities,
        registry, states, actions, globals, terminals,
        mutable_mask, gameplay_mask (if present), type_ids, slot_ids, ...
    """
    table = pq.read_table(parquet_path)

    indices = list(range(len(table)))
    if max_episodes is not None and len(table) > max_episodes:
        import random
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_episodes))

    episodes = []
    for i in indices:
        ep = {
            "game_id": table["game_id"][i].as_py(),
            "episode_id": table["episode_id"][i].as_py(),
            "num_frames": table["num_frames"][i].as_py(),
            "num_entities": table["num_entities"][i].as_py(),
        }

        for col in TENSOR_COLUMNS:
            if col not in table.column_names:
                continue
            raw = table[col][i].as_py()
            if raw is None:
                continue
            shape = json.loads(table[f"{col}_shape"][i].as_py())
            dtype = table[f"{col}_dtype"][i].as_py()
            ep[col] = _bytes_to_array(raw, shape, dtype)

        episodes.append(ep)

    return episodes


def build_windowed_tensors(
    episodes: list[dict],
    window_size: int = 8,
    max_entities: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convert episodes to stacked windowed tensors ready for training.

    Sliding windows with stride = window_size // 2.
    Pads entity dimension to max_entities.
    Pre-allocates output arrays to avoid 2x peak memory from np.stack().

    Returns dict with keys: registry, states, target_states, actions,
    globals, mutable_mask, gameplay_mask, entity_mask.
    """
    if not episodes:
        raise ValueError("No episodes to process")

    if max_entities is None:
        max_entities = max(ep["num_entities"] for ep in episodes)

    reg_dim = episodes[0]["registry"].shape[-1]
    state_dim = episodes[0]["states"].shape[-1]
    action_dim = episodes[0]["actions"].shape[-1]
    global_dim = episodes[0]["globals"].shape[-1]

    has_gameplay = "gameplay_mask" in episodes[0] and episodes[0]["gameplay_mask"] is not None
    gm_fields = episodes[0]["gameplay_mask"].shape[-1] if has_gameplay else 14

    stride = max(1, window_size // 2)

    # First pass: count total windows for pre-allocation
    n_windows = 0
    for ep in episodes:
        T = ep["num_frames"]
        n_windows += len(range(0, max(1, T - window_size), stride))

    t0 = time.time()

    # Pre-allocate output arrays (zero-filled)
    registry = np.zeros((n_windows, max_entities, reg_dim), dtype=np.float32)
    states = np.zeros((n_windows, window_size, max_entities, state_dim), dtype=np.float32)
    targets = np.zeros((n_windows, window_size, max_entities, state_dim), dtype=np.float32)
    actions = np.zeros((n_windows, window_size, action_dim), dtype=np.float32)
    globals_ = np.zeros((n_windows, window_size, global_dim), dtype=np.float32)
    terminals = np.zeros((n_windows, window_size), dtype=np.float32)
    mutable = np.zeros((n_windows, max_entities), dtype=np.bool_)
    gameplay = np.zeros((n_windows, max_entities, gm_fields), dtype=np.bool_)
    entity_mask = np.zeros((n_windows, max_entities), dtype=np.bool_)

    # Second pass: fill windows in-place
    idx = 0
    for ep in episodes:
        N = ep["num_entities"]
        T = ep["num_frames"]

        for start in range(0, max(1, T - window_size), stride):
            end = min(start + window_size, T - 1)
            W = end - start

            registry[idx, :N] = ep["registry"]
            states[idx, :W, :N] = ep["states"][start:end]
            targets[idx, :W, :N] = ep["states"][start + 1:end + 1]
            actions[idx, :W] = ep["actions"][start:end]
            globals_[idx, :W] = ep["globals"][start:end]

            if "terminals" in ep and ep["terminals"] is not None:
                t_slice = ep["terminals"][start + 1:end + 1]
                terminals[idx, :len(t_slice)] = t_slice.astype(np.float32)

            mutable[idx, :N] = ep["mutable_mask"]
            if has_gameplay:
                gameplay[idx, :N] = ep["gameplay_mask"]
            entity_mask[idx, :N] = True

            idx += 1

    elapsed = time.time() - t0
    print(f"  Built {n_windows} windows from {len(episodes)} episodes, "
          f"N={max_entities}, {elapsed:.1f}s")

    return {
        "registry": torch.from_numpy(registry),
        "states": torch.from_numpy(states),
        "target_states": torch.from_numpy(targets),
        "actions": torch.from_numpy(actions),
        "globals": torch.from_numpy(globals_),
        "terminals": torch.from_numpy(terminals),
        "mutable_mask": torch.from_numpy(mutable),
        "gameplay_mask": torch.from_numpy(gameplay),
        "entity_mask": torch.from_numpy(entity_mask),
    }


class InMemoryLoader:
    """Serves random batches from pre-loaded tensors. No disk I/O."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = True,
        device: str = "cuda",
    ):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.n = tensors["registry"].shape[0]
        self.dataset = self  # compatibility

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n)
        else:
            indices = torch.arange(self.n)

        for start in range(0, self.n, self.batch_size):
            idx = indices[start:start + self.batch_size]
            batch = {
                k: v[idx].to(self.device, non_blocking=True)
                for k, v in self.tensors.items()
            }
            yield batch
