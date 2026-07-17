# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Evaluate base (config_template) and best models on test and scenario data.

For each (game, model, agent) combo, finds the best run from summary.tsv across
all checkpoint jobs, then evaluates both the config_template checkpoint and the
best run checkpoint on:
  1. test set  (data_test/parquet/{game}/test.parquet)
  2. scenarios (data_scnarios/{game}/scenario.parquet), split per scenario name

Agent (claude vs codex) is detected from each trial's agent/ folder
(claude-code.txt vs codex.txt marker).

Outputs slim JSON reports containing only (game, model, agent, run_type,
weighted_composite). Scenario reports are organized per scenario name.

Per-game outputs:
    results/{game}_test.json
    results/{game}_scenario.json
Aggregate outputs (merged across games):
    results/final_report_test.json
    results/final_report_scenario.json

Usage:
    python scripts/evaluate_final.py
    python scripts/evaluate_final.py --game pong --model dreamer
    python scripts/evaluate_final.py --skip-cache   # don't rebuild tensor caches
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lib"))

from loader import load_episodes_from_parquet, build_windowed_tensors
from evaluator import UnifiedEvaluator

ALL_GAMES = [
    "asteroids", "breakout", "frogger", "kong", "platformer",
    "pong", "racer", "snake",
]
ALL_MODELS = ["dreamer", "ar_transformer", "d3pm", "maskgit"]
ALL_AGENTS = ["claude", "codex"]

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATA_TEST_DIR = REPO_ROOT / "data_test" / "parquet"
DATA_SCENARIO_DIR = REPO_ROOT / "data_scnarios"
RESULTS_DIR = REPO_ROOT / "results"

WINDOW_SIZE = 8
EVAL_HORIZONS = [1, 10, 20]
EVAL_WINDOW_SIZE = max(EVAL_HORIZONS) + WINDOW_SIZE + 1  # 29

# Evaluator uses max_batches=16 * batch_size=64 = 1024 windows per horizon.
# Cap loaded windows to avoid pulling multi-GB tensors for no reason.
EVAL_MAX_WINDOWS = 2048

SCENARIO_EID_RE = re.compile(r"^scenario_(.+?)_(data_ep_|case_)")


def detect_agent(trial_dir: Path) -> str | None:
    """Detect agent type from marker file in trial/agent/ folder."""
    agent_dir = trial_dir / "agent"
    if (agent_dir / "claude-code.txt").exists():
        return "claude"
    if (agent_dir / "codex.txt").exists():
        return "codex"
    return None


def scenario_name_of(episode_id: str) -> str | None:
    m = SCENARIO_EID_RE.match(episode_id)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Tensor cache building
# ---------------------------------------------------------------------------

def _save_tensors_cache(tensors: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, cache_path)
    n_windows = tensors["states"].shape[0]
    size_mb = cache_path.stat().st_size / 1e6
    print(f"  Saved {cache_path.name}: {n_windows} windows ({size_mb:.1f} MB)")


def _build_test_cache(
    parquet_path: Path,
    meta_path: Path,
    cache_path: Path,
    *,
    force: bool = False,
) -> None:
    if cache_path.exists() and not force:
        print(f"  Cache exists: {cache_path.name}")
        return

    meta = json.loads(meta_path.read_text())
    max_entities = meta["max_entities"]

    print(f"  Loading {parquet_path.name} ...")
    episodes = load_episodes_from_parquet(parquet_path)
    long_eps = [ep for ep in episodes if ep["states"].shape[0] >= EVAL_WINDOW_SIZE]
    if not long_eps:
        print(f"  WARNING: no episodes >= {EVAL_WINDOW_SIZE} frames, using all {len(episodes)}")
        long_eps = episodes

    print(f"  Building tensors (W={EVAL_WINDOW_SIZE}) from {len(long_eps)}/{len(episodes)} episodes ...")
    tensors = build_windowed_tensors(long_eps, EVAL_WINDOW_SIZE, max_entities)
    _save_tensors_cache(tensors, cache_path)


def _build_scenario_caches(
    parquet_path: Path,
    meta_path: Path,
    cache_dir: Path,
    *,
    force: bool = False,
) -> list[str]:
    """Build one cache per scenario name, return list of scenario names."""
    meta = json.loads(meta_path.read_text())
    max_entities = meta["max_entities"]

    print(f"  Loading {parquet_path.name} ...")
    episodes = load_episodes_from_parquet(parquet_path)

    buckets: dict[str, list[dict]] = {}
    for ep in episodes:
        name = scenario_name_of(ep["episode_id"])
        if name is None:
            continue
        buckets.setdefault(name, []).append(ep)

    scenario_names = sorted(buckets)
    for name in scenario_names:
        cache_path = cache_dir / f"scenario_{name}_eval_tensors.pt"
        if cache_path.exists() and not force:
            print(f"  Cache exists: {cache_path.name}")
            continue
        bucket = buckets[name]
        long_eps = [ep for ep in bucket if ep["states"].shape[0] >= EVAL_WINDOW_SIZE]
        if not long_eps:
            print(f"  [{name}] no episodes >= {EVAL_WINDOW_SIZE} frames, using all {len(bucket)}")
            long_eps = bucket
        print(f"  [{name}] Building tensors from {len(long_eps)}/{len(bucket)} episodes ...")
        tensors = build_windowed_tensors(long_eps, EVAL_WINDOW_SIZE, max_entities)
        _save_tensors_cache(tensors, cache_path)

    return scenario_names


def build_caches_for_game(game: str, *, force: bool = False) -> list[str]:
    """Build test and per-scenario caches for a single game. Return scenario names."""
    print(f"\n=== Caching {game} ===")

    test_meta = DATA_TEST_DIR / game / "meta.json"
    scen_meta = DATA_SCENARIO_DIR / game / "meta.json"

    test_parquet = DATA_TEST_DIR / game / "test.parquet"
    if test_parquet.exists():
        meta = test_meta if test_meta.exists() else scen_meta
        if not meta.exists():
            print(f"  SKIP test: no meta.json found for {game}")
        else:
            test_cache = DATA_TEST_DIR / game / "_cache" / "test_eval_tensors.pt"
            _build_test_cache(test_parquet, meta, test_cache, force=force)
    else:
        print(f"  No test.parquet for {game}")

    scenario_names: list[str] = []
    scen_parquet = DATA_SCENARIO_DIR / game / "scenario.parquet"
    if scen_parquet.exists():
        meta = scen_meta if scen_meta.exists() else test_meta
        if not meta.exists():
            print(f"  SKIP scenario: no meta.json found for {game}")
        else:
            cache_dir = DATA_SCENARIO_DIR / game / "_cache"
            scenario_names = _build_scenario_caches(scen_parquet, meta, cache_dir, force=force)
    else:
        print(f"  No scenario.parquet for {game}")

    return scenario_names


def discover_scenario_names(game: str) -> list[str]:
    """List scenario names available as cached tensor files for a game."""
    cache_dir = DATA_SCENARIO_DIR / game / "_cache"
    if not cache_dir.exists():
        return []
    names = []
    for p in sorted(cache_dir.glob("scenario_*_eval_tensors.pt")):
        name = p.name[len("scenario_"):-len("_eval_tensors.pt")]
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Find best experiments from checkpoints
# ---------------------------------------------------------------------------

def _parse_summary(tsv_path: Path) -> list[dict]:
    rows = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["_composite"] = float(row.get("composite", 0) or 0)
            except (ValueError, TypeError):
                row["_composite"] = 0.0
            rows.append(row)
    return rows


def _split_game_model(task_name: str) -> tuple[str | None, str | None]:
    for m in ALL_MODELS:
        suffix = "_" + m
        if task_name.endswith(suffix):
            return task_name[:-len(suffix)], m
    return None, None


def find_best_experiments(
    games: list[str], models: list[str], agents: list[str],
) -> dict[tuple[str, str, str], dict]:
    """Find best + base experiments per (game, model, agent)."""
    results: dict[tuple[str, str, str], dict] = {}

    for job_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not job_dir.is_dir() or job_dir.name.endswith(".tar.gz"):
            continue
        for ts_dir in sorted(job_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            for trial_dir in sorted(ts_dir.iterdir()):
                result_json = trial_dir / "result.json"
                if not result_json.exists():
                    continue
                rj = json.loads(result_json.read_text())
                task_name = rj.get("task_name", "").replace("ecs-worldbench/", "")
                game, model = _split_game_model(task_name)
                if game is None or game not in games or model not in models:
                    continue

                agent = detect_agent(trial_dir)
                if agent is None or agent not in agents:
                    continue

                task_dir = trial_dir / "artifacts" / "app" / "tasks" / task_name
                summary_path = task_dir / "summary.tsv"
                if not summary_path.exists():
                    continue

                rows = _parse_summary(summary_path)
                if not rows:
                    continue

                exp_base = trial_dir / "artifacts" / "app" / "experiments" / task_name
                config_json = task_dir / "config.json"
                if not config_json.exists():
                    continue

                best_row = max(rows, key=lambda r: r["_composite"])
                best_run_id = best_row.get("run_id", "")
                best_score = best_row["_composite"]
                best_exp_dir = exp_base / best_run_id

                # Base run is either "config_template" (preferred) or "config".
                base_row = next(
                    (r for r in rows if r.get("run_id") in ("config_template", "config")),
                    None,
                )
                base_run_id = base_row.get("run_id") if base_row else None
                base_score = base_row["_composite"] if base_row else None
                base_exp_dir = exp_base / base_run_id if base_row else None

                key = (game, model, agent)
                prev = results.get(key)
                if prev is None or best_score > prev["best_score"]:
                    results[key] = {
                        "best_run_id": best_run_id,
                        "best_score": best_score,
                        "best_exp_dir": best_exp_dir,
                        "base_run_id": base_run_id,
                        "base_score": base_score,
                        "base_exp_dir": base_exp_dir,
                        "config_json": config_json,
                        "task_dir": task_dir,
                        "game": game,
                        "model": model,
                        "agent": agent,
                        "job": job_dir.name,
                    }

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _weighted_composite(metrics: dict) -> float:
    h1 = metrics.get("h1", {}).get("composite", 0.0) or 0.0
    h10 = metrics.get("h10", {}).get("composite", 0.0) or 0.0
    h20 = metrics.get("h20", {}).get("composite", 0.0) or 0.0
    return round(0.1 * h1 + 0.2 * h10 + 0.7 * h20, 6)


def evaluate_checkpoint(
    checkpoint_path: Path,
    train_py_path: Path,
    config: dict,
    eval_tensors: dict[str, torch.Tensor],
    device: str,
) -> dict:
    import importlib.util

    exp_dir_str = str(train_py_path.parent)
    if exp_dir_str not in sys.path:
        sys.path.insert(0, exp_dir_str)

    spec = importlib.util.spec_from_file_location(
        f"train_{checkpoint_path.stem}_{id(checkpoint_path)}",
        str(train_py_path),
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    model = train_mod.build_model(config, eval_tensors, device)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    evaluator = UnifiedEvaluator(
        model=model,
        model_type=config["model_type"],
        val_tensors=eval_tensors,
        device=device,
        max_batches=16,
        horizons=EVAL_HORIZONS,
        eval_batch_size=64,
        eval_timeout_per_horizon=120.0,
        context_len=WINDOW_SIZE,
    )
    t0 = time.time()
    metrics = evaluator.evaluate_all()
    metrics["eval_time_seconds"] = round(time.time() - t0, 2)

    del model, evaluator, state_dict
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def _eval_on_cache(
    exp_dir: Path,
    config: dict,
    cache_path: Path,
    device: str,
) -> float | None:
    """Load cache, evaluate checkpoint, return weighted composite (or None on failure)."""
    checkpoint = exp_dir / "best_model.pt"
    train_py = exp_dir / "train.py"
    if not checkpoint.exists() or not train_py.exists():
        return None
    if not cache_path.exists():
        return None
    try:
        eval_tensors = torch.load(cache_path, map_location="cpu", weights_only=False)
        metrics = evaluate_checkpoint(checkpoint, train_py, config, eval_tensors, device)
        weighted = _weighted_composite(metrics)
        del eval_tensors
        gc.collect()
        torch.cuda.empty_cache()
        return weighted
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def _eval_experiment_subprocess(
    exp_dir: Path,
    config: dict,
    cache_paths: dict[str, Path],
    device: str,
) -> dict[str, float | None]:
    """Evaluate an experiment on many caches in a fresh subprocess.

    Each call starts a new Python interpreter, so per-experiment module state
    (from importing train.py + its sibling modules) is fully reclaimed on exit.

    Args:
        exp_dir: experiment directory containing best_model.pt and train.py.
        config: parsed config dict for build_model.
        cache_paths: {label: path_to_tensor_cache.pt}.
        device: "cuda" or "cpu".

    Returns:
        {label: weighted_composite or None} for each label.
    """
    checkpoint = exp_dir / "best_model.pt"
    train_py = exp_dir / "train.py"
    if not checkpoint.exists() or not train_py.exists():
        return {label: None for label in cache_paths}

    payload = {
        "exp_dir": str(exp_dir),
        "checkpoint": str(checkpoint),
        "train_py": str(train_py),
        "config": config,
        "caches": {label: str(p) for label, p in cache_paths.items()},
        "device": device,
    }

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    # Forward worker's stderr (log lines) to our stderr for visibility
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print(f"    WORKER FAILED rc={result.returncode}")
        return {label: None for label in cache_paths}
    try:
        out = json.loads(result.stdout.strip().splitlines()[-1])
        return {label: out.get(label) for label in cache_paths}
    except Exception as e:
        print(f"    WORKER OUTPUT PARSE ERROR: {e}  stdout-tail={result.stdout[-200:]}")
        return {label: None for label in cache_paths}


def _worker_main() -> None:
    """Subprocess worker: read JSON payload on stdin, evaluate, emit JSON on stdout."""
    payload = json.loads(sys.stdin.read())
    exp_dir = Path(payload["exp_dir"])
    checkpoint = Path(payload["checkpoint"])
    train_py = Path(payload["train_py"])
    config = payload["config"]
    caches = {label: Path(p) for label, p in payload["caches"].items()}
    device = payload["device"]

    out: dict[str, float | None] = {}
    for label, cache_path in caches.items():
        if not cache_path.exists():
            print(f"    [worker] SKIP {label}: cache missing", file=sys.stderr)
            out[label] = None
            continue
        try:
            try:
                loaded = torch.load(
                    cache_path, map_location="cpu", weights_only=False, mmap=True,
                )
            except (TypeError, RuntimeError):
                loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Subsample leading windows — evaluator uses at most
            # max_batches * batch_size windows per horizon.
            n = loaded["states"].shape[0]
            if n > EVAL_MAX_WINDOWS:
                eval_tensors = {
                    k: (v[:EVAL_MAX_WINDOWS].clone() if hasattr(v, "shape") and v.shape and v.shape[0] == n else v)
                    for k, v in loaded.items()
                }
                del loaded
                gc.collect()
            else:
                eval_tensors = loaded
            metrics = evaluate_checkpoint(checkpoint, train_py, config, eval_tensors, device)
            weighted = _weighted_composite(metrics)
            out[label] = weighted
            print(f"    [worker] {label}: composite={weighted}", file=sys.stderr)
            del eval_tensors, metrics
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    [worker] ERROR {label}: {e}", file=sys.stderr)
            out[label] = None

    print(json.dumps(out))


def evaluate_game(
    game: str,
    experiments_for_game: dict[tuple[str, str, str], dict],
    scenario_names: list[str],
    device: str,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Evaluate all (model, agent) experiments for a single game.

    Returns (test_entries, scenario_entries_by_scenario_name).
    """
    test_cache = DATA_TEST_DIR / game / "_cache" / "test_eval_tensors.pt"

    test_entries: list[dict] = []
    scenario_entries: dict[str, list[dict]] = {n: [] for n in scenario_names}

    for (g, model, agent) in sorted(experiments_for_game):
        info = experiments_for_game[(g, model, agent)]
        task_config = json.loads(info["config_json"].read_text())

        print(f"\n--- {game} / {model} / {agent} (job: {info['job']}) ---")
        print(f"  best_run_id={info['best_run_id']}  best_score={info['best_score']:.4f}"
              f"  base_score={info['base_score']}")

        for run_type, exp_dir, run_id in [
            ("best", info["best_exp_dir"], info["best_run_id"]),
            ("base", info["base_exp_dir"], info.get("base_run_id") or "config_template"),
        ]:
            if exp_dir is None or not exp_dir.exists():
                print(f"  [{run_type}] SKIP: exp dir missing")
                continue

            # Prefer per-experiment config.json (colocated with train.py) —
            # the task-level config.json only reflects the most recent run
            # and may not match earlier checkpoints' saved state_dict shapes.
            local_config = exp_dir / "config.json"
            if local_config.exists():
                config = json.loads(local_config.read_text())
            else:
                config = task_config

            # Build batch of caches to evaluate in one subprocess
            batch: dict[str, Path] = {}
            if test_cache.exists():
                batch["__test__"] = test_cache
            for scn in scenario_names:
                cache = DATA_SCENARIO_DIR / game / "_cache" / f"scenario_{scn}_eval_tensors.pt"
                if cache.exists():
                    batch[scn] = cache

            if not batch:
                print(f"  [{run_type}] SKIP: no caches available")
                continue

            print(f"  [{run_type}] subprocess-evaluating {exp_dir.name} on {len(batch)} caches ...",
                  flush=True)
            t0 = time.time()
            results = _eval_experiment_subprocess(exp_dir, config, batch, device)
            print(f"  [{run_type}] done in {time.time() - t0:.1f}s")

            if "__test__" in batch:
                test_entries.append({
                    "model": model,
                    "agent": agent,
                    "run_type": run_type,
                    "run_id": run_id,
                    "weighted_composite": results.get("__test__"),
                })
            for scn in scenario_names:
                scenario_entries[scn].append({
                    "model": model,
                    "agent": agent,
                    "run_type": run_type,
                    "run_id": run_id,
                    "weighted_composite": results.get(scn),
                })

    return test_entries, scenario_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Subprocess worker mode: bypass argparse and read JSON from stdin.
    if "--worker" in sys.argv:
        _worker_main()
        return

    parser = argparse.ArgumentParser(description="Evaluate base and best models on test/scenario data")
    parser.add_argument("--game", type=str, nargs="+", default=None)
    parser.add_argument("--model", type=str, nargs="+", default=None)
    parser.add_argument("--agent", type=str, nargs="+", default=None,
                        choices=ALL_AGENTS)
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    games = args.game or ALL_GAMES
    models = args.model or ALL_MODELS
    agents = args.agent or ALL_AGENTS

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find best experiments once (cheap)
    print("=" * 60)
    print("Discovering best experiments from checkpoints")
    print("=" * 60)
    all_experiments = find_best_experiments(games, models, agents)
    print(f"Found {len(all_experiments)} (game, model, agent) tasks")
    for (g, m, a), info in sorted(all_experiments.items()):
        print(f"  {g:<15} {m:<16} {a:<6}  best={info['best_score']:.4f} ({info['best_run_id']})")

    aggregate_test: dict[str, list[dict]] = {}
    aggregate_scenario: dict[str, dict[str, list[dict]]] = {}

    for game in games:
        print("\n" + "=" * 60)
        print(f"GAME: {game}")
        print("=" * 60)

        if not args.skip_cache:
            build_caches_for_game(game, force=args.force_cache)

        scenario_names = discover_scenario_names(game)
        print(f"  scenarios: {scenario_names}")

        if args.cache_only:
            continue

        exps_for_game = {k: v for k, v in all_experiments.items() if k[0] == game}
        if not exps_for_game:
            print(f"  No experiments for {game}, skipping")
            continue

        test_entries, scenario_entries = evaluate_game(
            game, exps_for_game, scenario_names, args.device,
        )

        # per-game files
        test_out = {"game": game, "results": test_entries}
        scen_out = {"game": game, "scenarios": scenario_entries}

        (results_dir / f"{game}_test.json").write_text(json.dumps(test_out, indent=2))
        (results_dir / f"{game}_scenario.json").write_text(json.dumps(scen_out, indent=2))
        print(f"  Wrote {game}_test.json and {game}_scenario.json")

        aggregate_test[game] = test_entries
        aggregate_scenario[game] = scenario_entries

    if args.cache_only:
        print("\nCache-only mode, done.")
        return

    # Aggregate files
    (results_dir / "final_report_test.json").write_text(
        json.dumps({"games": aggregate_test}, indent=2)
    )
    (results_dir / "final_report_scenario.json").write_text(
        json.dumps({"games": aggregate_scenario}, indent=2)
    )
    print(f"\nWrote final_report_test.json and final_report_scenario.json in {results_dir}")


if __name__ == "__main__":
    main()
