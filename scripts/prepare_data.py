# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Download data and build tensor caches for AutoWorldBench games.

Full data pipeline: downloads parquet episodes from HuggingFace Hub (if not
already present locally), builds windowed tensors, and saves them to
data/{game}/_cache/.

Usage:
    # Prepare all games:
    python scripts/prepare_data.py

    # Prepare a single game:
    python scripts/prepare_data.py --game asteroids

    # Force rebuild existing caches:
    python scripts/prepare_data.py --force
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT / "lib"))

from loader import load_episodes_from_parquet, build_windowed_tensors

ALL_GAMES = [
    "asteroids", "breakout", "frogger", "kong", "platformer",
    "pong", "racer", "snake",
]

CACHE_BASE = REPO_ROOT / "data"

WINDOW_SIZE = 8
SEED = 42
MAX_TRAIN_EPISODES_CAP = 10000

HF_REPO_ID = "AutoWorldModel/AutoWorldModelBench"


def _download_hf_file(game: str, filename: str, revision: str | None = None) -> Path:
    """Download a file from HuggingFace Hub and return its local cached path."""
    from huggingface_hub import hf_hub_download
    path = f"data/{game}/{filename}"
    return Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=path,
        repo_type="dataset",
        revision=revision,
    ))


def _ensure_parquet_dir(game: str, revision: str | None = None) -> Path:
    """Return a parquet directory for the game, downloading from HF if needed."""
    needed_files = ["meta.json", "train.parquet", "val.parquet"]
    optional_files = ["test.parquet", "scenario.parquet"]

    parquet_dir = REPO_ROOT / "data" / game
    if parquet_dir.is_dir() and all((parquet_dir / f).exists() for f in needed_files):
        return parquet_dir

    rev_label = f", revision={revision}" if revision else ""
    print(f"  Local parquet not found for {game}{rev_label}, downloading from HuggingFace Hub...")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for filename in needed_files:
        if (parquet_dir / filename).exists():
            continue
        hf_path = _download_hf_file(game, filename, revision=revision)
        shutil.copy2(hf_path, parquet_dir / filename)
        print(f"    Downloaded {filename}")

    for filename in optional_files:
        if (parquet_dir / filename).exists():
            continue
        try:
            hf_path = _download_hf_file(game, filename, revision=revision)
            shutil.copy2(hf_path, parquet_dir / filename)
            print(f"    Downloaded {filename}")
        except Exception:
            pass

    return parquet_dir


def _resolve_eval_window_size(eval_window_size: int | None) -> int:
    """Derive eval window size from task configs if not explicitly given."""
    if eval_window_size is not None:
        return eval_window_size
    config_dirs = sorted((REPO_ROOT / "tasks").glob("*/config.json"))
    if config_dirs:
        cfg = json.loads(config_dirs[0].read_text())
        horizons = cfg.get("final_eval_horizons", [1, 10, 20])
        return max(horizons) + WINDOW_SIZE + 1
    return max(20, 0) + WINDOW_SIZE + 1


def prepare_game(
    game: str,
    *,
    force: bool = False,
    eval_window_size: int | None = None,
    revision: str | None = None,
) -> None:
    """Build and save tensor caches for a single game."""
    import torch

    eval_ws = _resolve_eval_window_size(eval_window_size)
    cache_dir = CACHE_BASE / game / "_cache"

    if (
        not force
        and (cache_dir / "train_tensors.pt").exists()
        and (cache_dir / "val_tensors.pt").exists()
    ):
        print(f"Cache already exists for {game} at {cache_dir}, skipping. Use --force to rebuild.")
        return

    parquet_dir = _ensure_parquet_dir(game, revision=revision)

    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = parquet_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    max_entities = meta["max_entities"]
    num_train = meta.get("num_train", 5000)
    max_train_episodes = min(num_train, MAX_TRAIN_EPISODES_CAP)

    print(f"\n{'=' * 50}")
    print(f"Preparing cache for {game} (max_entities={max_entities})")
    print(f"{'=' * 50}")

    for split in ["train", "val"]:
        parquet_path = parquet_dir / f"{split}.parquet"
        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path} not found, skipping {split}")
            continue

        print(f"  Loading {split} from {parquet_path}...")
        if split == "train":
            episodes = load_episodes_from_parquet(
                parquet_path, max_episodes=max_train_episodes, seed=SEED,
            )
        else:
            episodes = load_episodes_from_parquet(parquet_path)
        print(f"  {len(episodes)} episodes loaded")

        tensors = build_windowed_tensors(episodes, WINDOW_SIZE, max_entities)
        print(f"  {tensors['states'].shape[0]} windows (W={WINDOW_SIZE})")

        out_path = cache_dir / f"{split}_tensors.pt"
        torch.save(tensors, out_path)
        print(f"  Saved {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")

        if split == "val" and eval_ws > WINDOW_SIZE:
            long_eps = [ep for ep in episodes if ep["states"].shape[0] >= eval_ws]
            if long_eps:
                eval_tensors = build_windowed_tensors(long_eps, eval_ws, max_entities)
                eval_path = cache_dir / "val_eval_tensors.pt"
                torch.save(eval_tensors, eval_path)
                print(f"  {eval_tensors['states'].shape[0]} eval windows (W={eval_ws}) "
                      f"from {len(long_eps)}/{len(episodes)} episodes")
                print(f"  Saved {eval_path.name} ({eval_path.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"  No episodes >= {eval_ws} frames, skipping eval tensors")

    print(f"Done. Cache for {game} saved to {cache_dir}")


def prepare_games(
    games: list[str],
    *,
    force: bool = False,
    eval_window_size: int | None = None,
    revision: str | None = None,
) -> None:
    """Build caches for a list of games."""
    for game in games:
        prepare_game(game, force=force, eval_window_size=eval_window_size,
                      revision=revision)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build tensor caches for AutoWorldBench games.",
    )
    parser.add_argument("--game", type=str, default=None,
                        help="Prepare a single game (default: all games)")
    parser.add_argument("--eval-window-size", type=int, default=None,
                        help="Window size for eval tensors (default: max horizon + 1 from config)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild existing caches")
    parser.add_argument("--revision", type=str, default=None,
                        help="HuggingFace branch/revision to download from (e.g. 'v2')")
    args = parser.parse_args()

    if args.game:
        if args.game not in ALL_GAMES:
            print(f"Note: {args.game} is not in the standard game list, proceeding anyway.")
        games = [args.game]
    else:
        games = ALL_GAMES

    prepare_games(games, force=args.force,
                  eval_window_size=args.eval_window_size,
                  revision=args.revision)
    print(f"\nAll caches prepared.")


if __name__ == "__main__":
    main()
