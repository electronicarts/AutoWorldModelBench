# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Shared utilities for autoresearch task scaffolds.

Used by each task's run.py and the agent to manage
experiment directories, result tracking, and model loading.
"""

import csv
import importlib.util
import json
import re
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_COLUMNS = [
    "run_id",
    "timestamp",
    "status",
    "composite",
    "position_l1",
    "alive_f1",
    "terminal_f1",
    "val_loss",
    "train_time_sec",
    "num_params",
    "total_steps",
    "notes",
]


def load_config(config_path: str | Path) -> dict:
    """Load a JSON config and validate required fields."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if "run_id" not in config:
        config["run_id"] = config_path.stem
    return config


def get_next_experiment_id(experiments_dir: Path) -> str:
    """Return the next auto-incremented experiment ID like 'exp004'.

    Scans existing directories matching exp\\d+ pattern.
    """
    experiments_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for d in experiments_dir.iterdir():
        if d.is_dir():
            match = re.match(r"exp(\d+)", d.name)
            if match:
                existing.append(int(match.group(1)))
    next_num = max(existing, default=0) + 1
    return f"exp{next_num:03d}"


def setup_experiment_dir(
    repo_root: Path,
    task_name: str,
    run_id: str,
    model_file_src: Path,
) -> Path:
    """Create experiments/{task_name}/{run_id}/ and copy model file.

    Returns the experiment directory path.
    """
    exp_dir = repo_root / "experiments" / task_name / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    dest = exp_dir / model_file_src.name
    if not dest.exists():
        shutil.copy2(model_file_src, dest)
    return exp_dir


def load_model_from_exp(exp_dir: Path, model_type: str) -> types.ModuleType:
    """Import the model module from an experiment directory.

    Uses importlib to avoid sys.path pollution and module cache conflicts.
    """
    # EVB uses train.py as the combined model+training file
    model_file = exp_dir / "train.py"
    if not model_file.exists():
        model_file = exp_dir / f"{model_type}.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found in: {exp_dir}")

    module_name = f"exp_{model_type}_{exp_dir.name}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(model_file),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def append_to_summary(summary_path: Path, row: dict):
    """Append a row to the summary TSV file.

    Creates the file with headers if it doesn't exist.
    """
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_comparison_plot(summary_path: Path, output_path: Path | None = None):
    """Generate a comparison chart from summary.tsv."""
    if not summary_path.exists():
        return

    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        return

    # Deduplicate: keep last entry per run_id
    seen = {}
    for row in rows:
        seen[row["run_id"]] = row
    rows = list(seen.values())

    run_ids = [r["run_id"] for r in rows]
    x = np.arange(len(run_ids))
    metrics = {
        "Composite Score": [float(r.get("composite") or 0) for r in rows],
        "Position L1 (lower)": [float(r.get("position_l1") or 0) for r in rows],
        "Alive F1 (higher)": [float(r.get("alive_f1") or 0) for r in rows],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    task_name = summary_path.parent.name
    fig.suptitle(f"{task_name}: Experiment Progression", fontsize=14, fontweight="bold")

    for ax, (metric_name, values) in zip(axes.flat, metrics.items()):
        ax.plot(x, values, "o-", color="#1f77b4", linewidth=2, markersize=8)
        for i, val in enumerate(values):
            ax.annotate(
                f"{val:.4f}",
                (x[i], values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
            )
        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(run_ids, rotation=35, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path is None:
        output_path = summary_path.parent / "comparison_plot.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
