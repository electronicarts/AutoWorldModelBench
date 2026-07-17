# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Test-set evaluation pipeline: collect predictions and generate reports.

Provides:
  - TestSetEvaluator: runs a checkpoint against held-out test episodes,
    computes metrics via UnifiedEvaluator, optionally saves predictions.
  - PredictionCollector: saves/loads per-sample predictions in NPZ format.
  - ReportGenerator: produces Markdown comparison reports.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from evaluator import UnifiedEvaluator, make_adapter
from loader import load_episodes_from_parquet, build_windowed_tensors
from splits import get_test_episode_ids

_MANIFEST_PATH = Path(__file__).resolve().parent / "test_manifest.json"


# ======================================================================
# TestSetEvaluator
# ======================================================================

class TestSetEvaluator:
    """Run a trained checkpoint against the held-out test set."""

    def __init__(
        self,
        model_type: str,
        checkpoint_path: str | Path,
        config_dict: dict,
        game_id: str,
        data_dir: str | Path,
        device: str = "cuda",
        manifest_path: str | Path | None = None,
    ):
        self.model_type = model_type
        self.checkpoint_path = Path(checkpoint_path)
        self.config_dict = config_dict
        self.game_id = game_id
        self.data_dir = Path(data_dir)
        self.device = device
        self.manifest_path = Path(manifest_path) if manifest_path else _MANIFEST_PATH

    def _build_model(self) -> torch.nn.Module:
        """Reconstruct model from checkpoint using task's train.py module."""
        import importlib.util
        import sys

        # Find the train.py in the task directory or experiment directory
        exp_dir = self.checkpoint_path.parent
        train_file = exp_dir / "train.py"
        if not train_file.exists():
            # Fall back to task directory
            task_name = f"{self.game_id}_{self.model_type}"
            repo_root = Path(__file__).resolve().parent.parent
            train_file = repo_root / "tasks" / task_name / "train.py"
        if not train_file.exists():
            # Fall back to template
            repo_root = Path(__file__).resolve().parent.parent
            train_file = repo_root / "templates" / f"{self.model_type}.py"

        if not train_file.exists():
            raise FileNotFoundError(
                f"Cannot find model source for {self.model_type} "
                f"(checked {exp_dir}, tasks/, templates/)"
            )

        # Dynamic import
        module_name = f"_test_eval_{self.model_type}_{id(self)}"
        spec = importlib.util.spec_from_file_location(module_name, str(train_file))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        # Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)

        # Find the model class and instantiate
        model = self._instantiate_model(mod)
        model.load_state_dict(ckpt, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def _instantiate_model(self, mod) -> torch.nn.Module:
        """Instantiate the correct model class from the imported module."""
        cfg = self.config_dict
        hidden_dim = cfg.get("hidden_dim", 256)

        # Load sample tensors to get dimensions
        cache_dir = Path(cfg.get("cache_dir", self.data_dir / self.game_id / "_cache"))
        val_tensors = torch.load(cache_dir / "val_tensors.pt", weights_only=False)
        registry_dim = val_tensors["registry"].shape[-1]
        state_dim = val_tensors["states"].shape[-1]
        action_dim = val_tensors["actions"].shape[-1]
        global_dim = val_tensors["globals"].shape[-1]

        if self.model_type == "dreamer":
            return mod.DreamerWorldModel(
                registry_dim=registry_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                global_dim=global_dim,
                hidden_dim=hidden_dim,
            )
        elif self.model_type == "ar_transformer":
            return mod.ARTransformerWorldModel(
                registry_dim=registry_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                global_dim=global_dim,
                hidden_dim=hidden_dim,
            )
        elif self.model_type == "d3pm":
            return mod.D3PMWorldModel(
                registry_dim=registry_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                global_dim=global_dim,
                hidden_dim=hidden_dim,
                game_id=self.game_id,
            )
        elif self.model_type == "maskgit":
            return mod.MaskGITWorldModel(
                registry_dim=registry_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                global_dim=global_dim,
                hidden_dim=hidden_dim,
                game_id=self.game_id,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _build_test_tensors(self) -> dict[str, torch.Tensor] | None:
        """Build windowed tensors from test episodes only."""
        game_data_dir = self.data_dir / self.game_id

        # Load all episodes and filter to test set
        test_episode_ids = get_test_episode_ids(self.manifest_path, self.game_id)
        if not test_episode_ids:
            return None

        all_episodes = []
        for pq_file in sorted(game_data_dir.glob("*.parquet")):
            episodes = load_episodes_from_parquet(pq_file)
            for ep in episodes:
                if ep["episode_id"] in test_episode_ids:
                    all_episodes.append(ep)

        if not all_episodes:
            return None

        window_size = self.config_dict.get("window_size", 8)
        return build_windowed_tensors(all_episodes, window_size=window_size)

    def run(
        self,
        save_predictions: bool = False,
        output_dir: str | Path | None = None,
        max_batches: int = 16,
        horizons: list[int] | None = None,
    ) -> dict:
        """Execute test evaluation.

        Returns dict with metrics, metadata, and optionally predictions path.
        """
        t0 = time.time()

        model = self._build_model()
        adapter = make_adapter(model, self.model_type)

        test_tensors = self._build_test_tensors()
        if test_tensors is None:
            return {"error": "No test data found"}

        test_episode_ids = sorted(get_test_episode_ids(self.manifest_path, self.game_id))

        evaluator = UnifiedEvaluator(
            model=model,
            model_type=self.model_type,
            val_tensors=test_tensors,
            device=self.device,
            max_batches=max_batches,
            horizons=horizons or [1, 5, 10, 20],
            eval_batch_size=64,
        )
        metrics = evaluator.evaluate_all()

        # Determine output directory
        if output_dir is None:
            output_dir = self.checkpoint_path.parent
        output_dir = Path(output_dir)

        run_id = self.checkpoint_path.parent.name

        result = {
            "task": f"{self.game_id}_{self.model_type}",
            "run_id": run_id,
            "checkpoint": str(self.checkpoint_path),
            "game_id": self.game_id,
            "model_type": self.model_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_episodes": test_episode_ids,
            "num_test_episodes": len(test_episode_ids),
            "metrics": metrics,
            "eval_time_seconds": round(time.time() - t0, 2),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)

        if save_predictions:
            predictions = self._collect_predictions(adapter, test_tensors)
            pred_path = output_dir / "test_predictions.npz"
            PredictionCollector.save(predictions, pred_path)
            result["predictions_path"] = str(pred_path)

        return result

    @torch.no_grad()
    def _collect_predictions(self, adapter, test_tensors: dict) -> dict:
        """Run model on test tensors, collect per-sample predictions."""
        all_pred_states = []
        all_true_states = []
        all_pred_terminals = []
        all_true_terminals = []
        all_mutable_masks = []
        all_entity_masks = []

        n_samples = test_tensors["states"].shape[0]
        bs = 64
        for start in range(0, min(n_samples, bs * 16), bs):
            batch = {k: v[start:start + bs].to(self.device)
                     for k, v in test_tensors.items()}

            B, W, N, D_s = batch["states"].shape
            adapter.reset()

            # Teacher-forced: predict last step
            for t in range(W - 1):
                state = batch["states"][:, t]
                action = batch["actions"][:, t]
                globals_ = batch["globals"][:, t]
                true_next = batch["target_states"][:, t]

                with torch.amp.autocast("cuda", enabled=self.device == "cuda"):
                    pred_state, pred_terminal = adapter.predict_one_step(
                        batch["registry"], state, action, globals_,
                        batch["entity_mask"],
                    )

                if t == W - 2:
                    for b in range(min(B, batch["states"].shape[0])):
                        all_pred_states.append(pred_state[b].cpu().numpy())
                        all_true_states.append(true_next[b].cpu().numpy())
                        all_pred_terminals.append(
                            pred_terminal[b].cpu().numpy()
                            if pred_terminal.numel() > 0
                            else np.array(0.0, dtype=np.float32)
                        )
                        true_term = batch["terminals"][:, t][b] if "terminals" in batch else 0.0
                        all_true_terminals.append(
                            np.array(float(true_term), dtype=np.float32)
                        )
                        all_mutable_masks.append(batch["mutable_mask"][b].cpu().numpy())
                        all_entity_masks.append(batch["entity_mask"][b].cpu().numpy())

        return {
            "pred_states": all_pred_states,
            "true_states": all_true_states,
            "pred_terminals": all_pred_terminals,
            "true_terminals": all_true_terminals,
            "mutable_masks": all_mutable_masks,
            "entity_masks": all_entity_masks,
            "metadata": json.dumps({
                "game_id": self.game_id,
                "model_type": self.model_type,
                "checkpoint": str(self.checkpoint_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
        }


# ======================================================================
# PredictionCollector
# ======================================================================

class PredictionCollector:
    """Save/load per-sample predictions in NPZ format."""

    @staticmethod
    def save(predictions: dict, output_path: Path) -> None:
        arrays = {}
        for key in ["pred_states", "true_states", "mutable_masks", "entity_masks"]:
            if key in predictions:
                arrays[key] = np.array(predictions[key], dtype=object)
        for key in ["pred_terminals", "true_terminals"]:
            if key in predictions:
                arrays[key] = np.array(predictions[key], dtype=np.float32)
        if "metadata" in predictions:
            arrays["metadata"] = np.array(predictions["metadata"])
        np.savez_compressed(output_path, **arrays)

    @staticmethod
    def load(npz_path: Path) -> dict:
        data = np.load(npz_path, allow_pickle=True)
        result = {}
        for key in data.files:
            result[key] = data[key]
            if key == "metadata":
                result[key] = str(result[key])
        return result


# ======================================================================
# ReportGenerator
# ======================================================================

class ReportGenerator:
    """Generate Markdown evaluation reports from test results."""

    @staticmethod
    def task_report(task_name: str, results: list[dict], output_path: Path) -> str:
        """Generate a per-task comparison report."""
        if not results:
            return f"# Test Report: {task_name}\n\nNo results found.\n"

        lines = [
            f"# Test Evaluation Report: {task_name}",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        r0 = results[0]
        lines.append(f"Test episodes: {r0.get('num_test_episodes', 'N/A')}")
        lines.append("")

        # Sort by h1 composite descending
        sorted_results = sorted(
            results,
            key=lambda r: r.get("metrics", {}).get("h1", {}).get("composite", 0),
            reverse=True,
        )

        lines.append("## Rankings (by h1 composite)")
        lines.append("")
        lines.append("| Rank | Run ID | Composite | Position L1 | Alive F1 | Terminal F1 |")
        lines.append("|------|--------|-----------|-------------|----------|-------------|")
        for i, r in enumerate(sorted_results):
            h1 = r.get("metrics", {}).get("h1", {})
            lines.append(
                f"| {i + 1} | {r['run_id']} "
                f"| {h1.get('composite', 0):.4f} "
                f"| {h1.get('position_l1', 0):.4f} "
                f"| {h1.get('alive_f1', 0):.4f} "
                f"| {h1.get('terminal_f1', 0):.4f} |"
            )
        lines.append("")

        has_h5 = any("h5" in r.get("metrics", {}) for r in results)
        if has_h5:
            lines.append("## Horizon h=5 (open-loop rollout)")
            lines.append("")
            lines.append("| Run ID | Composite | Position L1 | Alive F1 | Terminal F1 |")
            lines.append("|--------|-----------|-------------|----------|-------------|")
            for r in sorted_results:
                h5 = r.get("metrics", {}).get("h5", {})
                if h5:
                    lines.append(
                        f"| {r['run_id']} "
                        f"| {h5.get('composite', 0):.4f} "
                        f"| {h5.get('position_l1', 0):.4f} "
                        f"| {h5.get('alive_f1', 0):.4f} "
                        f"| {h5.get('terminal_f1', 0):.4f} |"
                    )
            lines.append("")

        lines.append("## Metadata")
        lines.append("")
        lines.append("| Run ID | Eval Time (s) | Checkpoint |")
        lines.append("|--------|---------------|------------|")
        for r in sorted_results:
            lines.append(
                f"| {r['run_id']} "
                f"| {r.get('eval_time_seconds', 'N/A')} "
                f"| {Path(r['checkpoint']).name} |"
            )
        lines.append("")

        report = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        return report

    @staticmethod
    def cross_task_report(all_results: dict[str, list[dict]], output_path: Path) -> str:
        """Generate a cross-task leaderboard."""
        lines = [
            "# Cross-Task Test Leaderboard",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        lines.append("## Best Results per Task")
        lines.append("")
        lines.append("| Task | Best Run | h1 Composite | h5 Composite | Position L1 | Alive F1 |")
        lines.append("|------|----------|--------------|--------------|-------------|----------|")

        model_scores: dict[str, list[float]] = {}

        for task_name in sorted(all_results):
            results = all_results[task_name]
            if not results:
                continue
            best = max(
                results,
                key=lambda r: r.get("metrics", {}).get("h1", {}).get("composite", 0),
            )
            h1 = best.get("metrics", {}).get("h1", {})
            h5 = best.get("metrics", {}).get("h5", {})
            h5_comp = f"{h5.get('composite', 0):.4f}" if h5 else "N/A"
            lines.append(
                f"| {task_name} | {best['run_id']} "
                f"| {h1.get('composite', 0):.4f} "
                f"| {h5_comp} "
                f"| {h1.get('position_l1', 0):.4f} "
                f"| {h1.get('alive_f1', 0):.4f} |"
            )

            model_type = best.get("model_type", task_name.rsplit("_", 1)[-1])
            model_scores.setdefault(model_type, []).append(h1.get("composite", 0))

        lines.append("")

        if model_scores:
            lines.append("## By Model Type")
            lines.append("")
            lines.append("| Model | Avg h1 Composite | Games |")
            lines.append("|-------|-----------------|-------|")
            for model, scores in sorted(model_scores.items()):
                avg = sum(scores) / len(scores) if scores else 0
                lines.append(f"| {model} | {avg:.4f} | {len(scores)} |")
            lines.append("")

        report = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        return report
