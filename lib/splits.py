# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Deterministic episode split logic for train/val/test partitions.

Test episodes are defined by a pre-computed manifest (lib/test_manifest.json).
Train/val split uses MD5 hash on non-test episodes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load_test_manifest(manifest_path: str | Path) -> dict:
    """Load the full test manifest JSON."""
    with open(manifest_path) as f:
        return json.load(f)


def get_test_episode_ids(manifest_path: str | Path, game_id: str) -> set[str]:
    """Return the set of test episode IDs for a game."""
    manifest = load_test_manifest(manifest_path)
    return set(manifest.get("games", {}).get(game_id, []))


def episode_is_train(episode_id: str, val_split: float = 0.1) -> bool:
    """MD5 hash logic for train/val split. Applied AFTER test filtering."""
    h = int(hashlib.md5(episode_id.encode()).hexdigest(), 16) % 1000
    return h >= int(val_split * 1000)


def generate_test_manifest(
    data_dir: str | Path,
    output_path: str | Path,
    test_frac: float = 0.1,
) -> dict:
    """Scan Parquet data, select test episodes, write manifest JSON.

    Selection: MD5(episode_id + ":test") % 1000 < test_frac * 1000.
    Uses a ":test" salt so the hash is independent of the train/val split hash.

    Expects data_dir to contain subdirectories per game, each with
    train.parquet (and optionally val.parquet, test.parquet).
    """
    import pyarrow.parquet as pq

    data_dir = Path(data_dir)
    manifest: dict = {
        "version": 1,
        "test_frac": test_frac,
        "hash_salt": ":test",
        "generated": datetime.now(timezone.utc).isoformat(),
        "games": {},
    }

    threshold = int(test_frac * 1000)

    for game_dir in sorted(data_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        game_id = game_dir.name

        # Collect all episode IDs from all parquet files
        episode_ids = set()
        for pq_file in game_dir.glob("*.parquet"):
            table = pq.read_table(pq_file, columns=["episode_id"])
            for i in range(len(table)):
                episode_ids.add(table["episode_id"][i].as_py())

        # Select test episodes via salted MD5 hash
        test_eps = sorted(
            ep for ep in episode_ids
            if int(hashlib.md5((ep + ":test").encode()).hexdigest(), 16) % 1000 < threshold
        )
        manifest["games"][game_id] = test_eps

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
