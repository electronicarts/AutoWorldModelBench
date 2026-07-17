# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Re-copy model template(s) to all task directories.

After editing a template, run this to push changes to all 8 game dirs.

Usage:
    python refresh.py --model dreamer       # refresh dreamer in all 8 games
    python refresh.py --model dreamer d3pm  # refresh two models
    python refresh.py --prepare             # refresh prepare.py in all 32 dirs
    python refresh.py --all                 # refresh everything
"""
import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent
TEMPLATES = REPO_ROOT / "templates"
TASKS = REPO_ROOT / "tasks"

GAMES = [
    "asteroids", "breakout", "frogger", "kong", "platformer",
    "pong", "racer", "snake",
]
MODELS = ["dreamer", "ar_transformer", "d3pm", "maskgit"]


def refresh_model(model: str) -> int:
    src = TEMPLATES / f"{model}.py"
    if not src.exists():
        print(f"  WARNING: {src} not found")
        return 0
    count = 0
    for game in GAMES:
        dst = TASKS / f"{game}_{model}" / "train.py"
        if dst.parent.exists():
            shutil.copy(src, dst)
            count += 1
    print(f"  {model}: updated {count} task dirs")
    return count


def refresh_prepare() -> int:
    src = TEMPLATES / "prepare.py"
    count = 0
    for model in MODELS:
        for game in GAMES:
            dst = TASKS / f"{game}_{model}" / "prepare.py"
            if dst.parent.exists():
                shutil.copy(src, dst)
                count += 1
    print(f"  prepare.py: updated {count} task dirs")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=None,
                        help="Model(s) to refresh")
    parser.add_argument("--prepare", action="store_true",
                        help="Refresh prepare.py in all task dirs")
    parser.add_argument("--all", action="store_true",
                        help="Refresh all models + prepare")
    args = parser.parse_args()

    if args.all:
        for model in MODELS:
            refresh_model(model)
        refresh_prepare()
    elif args.model:
        for model in args.model:
            refresh_model(model)
    elif args.prepare:
        refresh_prepare()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
