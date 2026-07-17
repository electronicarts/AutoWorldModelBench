# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Run evaluation-only on saved checkpoints with longer rollout horizons.

No retraining needed — loads best_model.pt + rebuilds val tensors with
larger window_size to support h=50 rollouts.

Usage:
    python eval_only.py                             # all 32 tasks, horizons 1,5,50
    python eval_only.py --model dreamer d3pm        # subset of models
    python eval_only.py --game snake pong            # subset of games
    python eval_only.py --horizons 1 5 20 50        # custom horizons
    python eval_only.py --window-size 64            # custom window (must be > max horizon)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "lib"))

from loader import load_episodes_from_parquet, build_windowed_tensors
from evaluator import UnifiedEvaluator

GAMES = [
    "asteroids", "breakout", "frogger", "kong", "platformer",
    "pong", "racer", "snake",
]
MODELS = ["dreamer", "ar_transformer", "d3pm", "maskgit"]

# Model class names inside each template
MODEL_CLASSES = {
    "dreamer": "DreamerWorldModel",
    "ar_transformer": "ARTransformer",
    "d3pm": "D3PMWorldModel",
    "maskgit": "MaskGITWorldModel",
}


def load_template_module(model_type: str):
    """Import the template file as a module to access its model class."""
    template_path = REPO_ROOT / "templates" / f"{model_type}.py"
    spec = importlib.util.spec_from_file_location(f"template_{model_type}", template_path)
    mod = importlib.util.module_from_spec(spec)
    # Temporarily set TASK_DIR so path setup in templates doesn't crash
    spec.loader.exec_module(mod)
    return mod


def build_model(model_type: str, registry_dim: int, state_dim: int,
                hidden_dim: int, game_id: str):
    """Construct model from template and return it."""
    mod = load_template_module(model_type)
    cls = getattr(mod, MODEL_CLASSES[model_type])

    if model_type in ("d3pm", "maskgit"):
        from tokenizer import EntityTokenizer
        tokenizer = EntityTokenizer(game_id=game_id)
        return cls(tokenizer=tokenizer, registry_dim=registry_dim,
                   hidden_dim=hidden_dim)
    else:
        return cls(registry_dim=registry_dim, state_dim=state_dim,
                   hidden_dim=hidden_dim)


def eval_task(game: str, model_type: str, window_size: int,
              horizons: list[int], max_batches: int,
              eval_batch_size: int,
              eval_timeout_per_horizon: float | None = None) -> dict:
    """Evaluate a single task. Returns result dict."""
    task_dir = REPO_ROOT / "tasks" / f"{game}_{model_type}"
    ckpt_path = task_dir / "outputs" / "best_model.pt"
    tag = f"{game}/{model_type}"

    if not ckpt_path.exists():
        return {"game": game, "model": model_type, "error": "no checkpoint"}

    config = json.loads((task_dir / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build val tensors with larger window
    data_dir = REPO_ROOT / "data" / game
    meta = json.loads((data_dir / "meta.json").read_text())
    max_entities = meta["max_entities"]

    print(f"  Building val tensors (W={window_size})...", flush=True)
    episodes = load_episodes_from_parquet(data_dir / "val.parquet")
    val_tensors = build_windowed_tensors(episodes, window_size, max_entities)
    n_windows = val_tensors["states"].shape[0]
    print(f"  {n_windows} windows from {len(episodes)} episodes", flush=True)

    if n_windows == 0:
        return {"game": game, "model": model_type, "error": "no valid windows"}

    # Construct model and load checkpoint
    registry_dim = val_tensors["registry"].shape[-1]
    state_dim = val_tensors["states"].shape[-1]
    hidden_dim = config.get("hidden_dim", 256)

    model = build_model(model_type, registry_dim, state_dim, hidden_dim, game)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Run evaluator
    print(f"  Running eval horizons={horizons}...", flush=True)
    evaluator = UnifiedEvaluator(
        model=model, model_type=model_type, val_tensors=val_tensors,
        device=device, max_batches=max_batches, horizons=horizons,
        eval_batch_size=eval_batch_size,
        eval_timeout_per_horizon=eval_timeout_per_horizon,
    )
    eval_metrics = evaluator.evaluate_all()

    return {
        "game": game,
        "model": model_type,
        "eval_metrics": eval_metrics,
        "n_windows": n_windows,
    }


def print_summary(results: list[dict], horizons: list[int]):
    """Print results tables for each horizon."""
    games = []
    models = []
    for r in results:
        if r["game"] not in games:
            games.append(r["game"])
        if r["model"] not in models:
            models.append(r["model"])

    lookup = {(r["game"], r["model"]): r for r in results}

    for h in horizons:
        hkey = f"h{h}"
        print(f"\n{'='*80}")
        print(f"EVAL: {hkey} composite")
        print(f"{'='*80}")

        header = f"{'Game':<18s}"
        for m in models:
            header += f"  {m:>14s}"
        print(header)
        print("─" * (18 + 16 * len(models)))

        for game in games:
            row = f"{game:<18s}"
            for model in models:
                r = lookup.get((game, model))
                if r is None or "error" in r:
                    row += f"  {'—':>14s}"
                else:
                    em = r.get("eval_metrics", {})
                    comp = em.get(hkey, {}).get("composite")
                    if comp is not None:
                        row += f"  {comp:>14.4f}"
                    else:
                        row += f"  {'—':>14s}"
            print(row)
        print("─" * (18 + 16 * len(models)))

        # Per-model averages
        for model in models:
            scores = []
            for g in games:
                r = lookup.get((g, model))
                if r and "error" not in r:
                    comp = r.get("eval_metrics", {}).get(hkey, {}).get("composite")
                    if comp is not None:
                        scores.append(comp)
            if scores:
                print(f"  {model:>14s}: avg={sum(scores)/len(scores):.4f} ({len(scores)}/{len(games)} games)")

    # Errors
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors:
            print(f"  {r['game']}/{r['model']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", nargs="+", default=None)
    parser.add_argument("--model", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 50])
    parser.add_argument("--window-size", type=int, default=None,
                        help="Val window size (default: max_horizon + 1)")
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per task in seconds")
    parser.add_argument("--eval-timeout", type=int, default=120,
                        help="Max seconds per horizon eval (default 120)")
    args = parser.parse_args()

    games = args.game if args.game else GAMES
    models = args.model if args.model else MODELS
    horizons = sorted(args.horizons)
    window_size = args.window_size or (max(horizons) + 1)

    print(f"Eval-only: {len(games)} games × {len(models)} models")
    print(f"Horizons: {horizons}, window_size: {window_size}")

    t0 = time.time()
    results = []
    total = len(games) * len(models)
    i = 0

    for model in models:
        for game in games:
            i += 1
            task_dir = REPO_ROOT / "tasks" / f"{game}_{model}"
            if not task_dir.exists():
                continue

            print(f"\n[{i}/{total}] {game}/{model}", flush=True)
            try:
                result = eval_task(game, model, window_size, horizons,
                                   args.max_batches, args.eval_batch_size,
                                   eval_timeout_per_horizon=args.eval_timeout)
            except Exception as e:
                result = {"game": game, "model": model, "error": str(e)}
                print(f"  ERROR: {e}")

            results.append(result)

            # Metrics summary for this task
            if "eval_metrics" in result:
                em = result["eval_metrics"]
                parts = []
                for h in horizons:
                    comp = em.get(f"h{h}", {}).get("composite")
                    if comp is not None:
                        parts.append(f"h{h}={comp:.4f}")
                print(f"  {' '.join(parts)} ({em.get('eval_time_seconds', 0):.1f}s)")

            # Save incremental
            out_path = REPO_ROOT / "eval_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print_summary(results, horizons)
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to {REPO_ROOT / 'eval_results.json'}")


if __name__ == "__main__":
    main()
