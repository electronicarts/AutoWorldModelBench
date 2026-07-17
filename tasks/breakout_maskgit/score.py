# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Independent checkpoint evaluation for AutoWorldBench.

Usage:
    python score.py --checkpoint outputs/best_model.pt
    python score.py --checkpoint outputs/best_model.pt --horizons 1 10 20

Evaluates a model checkpoint against the validation split using
UnifiedEvaluator and prints JSON metrics to stdout.
Called by tests/test.sh for verifier-independent scoring.
"""
import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

# -- path setup --
TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR.parent.parent / "lib"))
sys.path.insert(0, str(TASK_DIR))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model state_dict (.pt)")
    parser.add_argument("--config", default=str(TASK_DIR / "config.json"),
                        help="Path to config.json")
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="Evaluation horizons (default: from config.json)")
    args = parser.parse_args()

    try:
        import torch
        from evaluator import UnifiedEvaluator

        config = json.loads(Path(args.config).read_text())
        horizons = args.horizons or config.get("final_eval_horizons", [1, 10, 20])

        # Resolve cache directory (try config, then /data/{game}/_cache)
        cache_dir = Path(config.get("cache_dir", f"/data/{config['game_id']}/_cache"))
        if not cache_dir.exists():
            cache_dir = Path(f"/data/{config['game_id']}/_cache")

        # Load validation tensors for evaluation
        eval_path = cache_dir / "val_eval_tensors.pt"
        if not eval_path.exists():
            eval_path = cache_dir / "val_tensors.pt"
        if not eval_path.exists():
            print(json.dumps({"error": f"Val tensors not found at {eval_path}"}))
            sys.exit(1)

        eval_tensors = torch.load(eval_path, map_location="cpu", weights_only=False)

        # We also need train tensors for model dimensions (some models need
        # state_dim which may differ from eval tensors layout)
        train_path = cache_dir / "train_tensors.pt"
        if train_path.exists():
            train_tensors = torch.load(train_path, map_location="cpu", weights_only=False)
        else:
            # Fall back to eval tensors for dimension extraction
            train_tensors = eval_tensors

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Import train.py dynamically to get build_model()
        train_py = TASK_DIR / "train.py"
        if not train_py.exists():
            print(json.dumps({"error": "train.py not found in task directory"}))
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("train_mod", str(train_py))
        train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_mod)

        # Build model using the template's build_model() function
        if not hasattr(train_mod, "build_model"):
            print(json.dumps({"error": "No build_model() found in train.py"}))
            sys.exit(1)

        model = train_mod.build_model(config, train_tensors, device)

        # Load checkpoint state dict
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(json.dumps({"error": f"Checkpoint not found: {checkpoint_path}"}))
            sys.exit(1)

        state_dict = torch.load(str(checkpoint_path), map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Run evaluation via UnifiedEvaluator
        evaluator = UnifiedEvaluator(
            model=model,
            model_type=config["model_type"],
            val_tensors=eval_tensors,
            device=device,
            max_batches=16,
            horizons=horizons,
            eval_batch_size=64,
            eval_timeout_per_horizon=60.0,
        )

        t0 = time.time()
        metrics = evaluator.evaluate_all()
        metrics["eval_time_seconds"] = round(time.time() - t0, 2)
        metrics["source"] = "independent_eval"
        metrics["checkpoint"] = str(checkpoint_path)

        # Print JSON to stdout (test.sh captures this)
        print(json.dumps(metrics, indent=2))

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    except Exception:
        # Print error as JSON so the caller can detect failure
        print(json.dumps({
            "error": traceback.format_exc(),
            "source": "score_py_exception",
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
