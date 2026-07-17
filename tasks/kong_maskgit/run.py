# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Experiment-tracking wrapper around train.py.

Usage:
    python run.py --config config_template.json       # baseline
    python run.py --config configs/my_experiment.json  # custom
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TASK_DIR = Path(__file__).parent
REPO_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lib"))

from task_utils import (
    load_config,
    get_next_experiment_id,
    setup_experiment_dir,
    append_to_summary,
    update_comparison_plot,
)


def main():
    parser = argparse.ArgumentParser(description="Run experiment with tracking")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    task_name = TASK_DIR.name
    run_id = config.get("run_id") or get_next_experiment_id(
        REPO_ROOT / "experiments" / task_name
    )

    # Setup experiment directory
    exp_dir = setup_experiment_dir(
        REPO_ROOT, task_name, run_id, TASK_DIR / "train.py"
    )

    # Write config to task dir (train.py reads from here)
    config["run_id"] = run_id
    (TASK_DIR / "config.json").write_text(json.dumps(config, indent=2))

    # Also save to experiment dir
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Run training as subprocess for clean isolation
    t0 = time.time()
    status = "success"
    training_result = {}
    try:
        result = subprocess.run(
            [sys.executable, str(TASK_DIR / "train.py")],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("MAX_TRAIN_SECONDS", config.get("max_train_seconds", 600))) * 3,
            cwd=str(TASK_DIR),
        )
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-2000:]}")
            status = "error"

        # Copy outputs to experiment dir
        outputs_dir = TASK_DIR / "outputs"
        if outputs_dir.exists():
            for f in outputs_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, exp_dir / f.name)

        # Load training result
        tr_path = outputs_dir / "training_result.json"
        if tr_path.exists():
            training_result = json.loads(tr_path.read_text())
        elif status != "error":
            status = "error"
            training_result = {"error": "No training_result.json produced"}

    except subprocess.TimeoutExpired:
        status = "timeout"
        training_result = {"error": "Process timed out"}
    except Exception as e:
        status = "error"
        training_result = {"error": str(e)}

    elapsed = time.time() - t0

    # Extract metrics and append to summary.tsv
    eval_metrics = training_result.get("eval_metrics", {})
    h1 = eval_metrics.get("h1", {})
    h10 = eval_metrics.get("h10", {})
    h20 = eval_metrics.get("h20", {})
    h1_comp = h1.get("composite", 0.0) or 0.0
    h10_comp = h10.get("composite", 0.0) or 0.0
    h20_comp = h20.get("composite", 0.0) or 0.0
    weighted = round(0.1 * h1_comp + 0.2 * h10_comp + 0.7 * h20_comp, 6)
    row = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "composite": weighted,
        "h1_composite": h1_comp,
        "h10_composite": h10_comp,
        "h20_composite": h20_comp,
        "position_l1": h1.get("position_l1", ""),
        "alive_f1": h1.get("alive_f1", ""),
        "terminal_f1": h1.get("terminal_f1", ""),
        "val_loss": training_result.get("best_val_loss", ""),
        "train_time_sec": round(elapsed, 1),
        "num_params": training_result.get("num_params", ""),
        "total_steps": training_result.get("total_steps", ""),
        "notes": config.get("notes", ""),
    }
    append_to_summary(TASK_DIR / "summary.tsv", row)
    update_comparison_plot(TASK_DIR / "summary.tsv")

    # Keep the best checkpoint across all runs.
    # The verifier scores outputs/best_model.pt, so it must always
    # contain the overall best model, not just the latest run's.
    best_tracker = TASK_DIR / "outputs" / ".best_score"
    best_model = TASK_DIR / "outputs" / "best_model.pt"
    best_overall = TASK_DIR / "outputs" / "best_model_overall.pt"
    prev_best = -1.0
    if best_tracker.exists():
        try:
            prev_best = float(best_tracker.read_text().strip())
        except (ValueError, OSError):
            prev_best = -1.0

    if weighted > prev_best and best_model.exists():
        shutil.copy2(best_model, best_overall)
        best_tracker.write_text(str(weighted))
        print(f"New best score: {weighted} (prev {prev_best})")
    elif best_overall.exists() and best_model.exists():
        shutil.copy2(best_overall, best_model)
        print(f"Kept previous best ({prev_best}) -- restored best_model.pt")

    # Write results.json to experiment dir
    combined = {**row, **training_result}
    (exp_dir / "results.json").write_text(json.dumps(combined, indent=2, default=str))

    print(f"Run {run_id}: score={weighted} (h1={h1_comp}, h10={h10_comp}, h20={h20_comp}) status={status} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
