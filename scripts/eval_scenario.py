# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Evaluate model checkpoints on game scenario test cases and generate PDF report.

For each model checkpoint, loads scenario episodes grouped by scenario name,
runs open-loop rollout from 8-frame history, and reports position L1 at each
rollout step. Produces per-scenario comparison plots and a summary heatmap.

Usage:
    python scripts/eval_scenario.py
    python scripts/eval_scenario.py --game pong
    python scripts/eval_scenario.py --game pong --model dreamer
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "lib"))

from evaluator import make_adapter, position_l1, alive_f1, composite_score
from loader import load_episodes_from_parquet, build_windowed_tensors

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints" / "jobs_01-33-07"
DATA_DIR = REPO_ROOT / "data"
SCENARIO_DIR = REPO_ROOT / "gameScenario"

CONTEXT_LEN = 8


def discover_models() -> list[dict]:
    models = []
    for job_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        task_dirs = [d for d in job_dir.iterdir() if d.is_dir()]
        if not task_dirs:
            continue
        task_dir = task_dirs[0]
        task_name = task_dir.name.split("__")[0]
        config_path = task_dir / "artifacts" / "app" / "tasks" / task_name / "config.json"
        train_py = task_dir / "artifacts" / "app" / "tasks" / task_name / "train.py"
        checkpoint = task_dir / "artifacts" / "app" / "tasks" / task_name / "outputs" / "best_model_overall.pt"
        if not all(p.exists() for p in [config_path, train_py, checkpoint]):
            continue
        config = json.loads(config_path.read_text())
        models.append({
            "task_name": task_name,
            "game_id": config["game_id"],
            "model_type": config["model_type"],
            "config": config,
            "train_py": train_py,
            "checkpoint": checkpoint,
        })
    return models


def load_model(info: dict, tensors: dict, device: str):
    spec = importlib.util.spec_from_file_location("train_mod", str(info["train_py"]))
    train_mod = importlib.util.module_from_spec(spec)
    for lib_path in [
        info["train_py"].parent.parent / "lib",
        info["train_py"].parent.parent.parent / "lib",
        REPO_ROOT / "lib",
    ]:
        if str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))
    spec.loader.exec_module(train_mod)
    model = train_mod.build_model(info["config"], tensors, device)
    state_dict = torch.load(str(info["checkpoint"]), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_scenario_name(episode_id: str) -> str:
    m = re.match(r"scenario_(.+?)_(?:case_\d+|data_ep_\w+)$", episode_id)
    if m:
        return m.group(1)
    m = re.match(r"scenario_(.+?)_\d+$", episode_id)
    if m:
        return m.group(1)
    return episode_id


def group_episodes_by_scenario(parquet_path: Path) -> dict[str, list[dict]]:
    episodes = load_episodes_from_parquet(parquet_path)
    groups = defaultdict(list)
    for ep in episodes:
        scenario = extract_scenario_name(ep["episode_id"])
        groups[scenario].append(ep)
    return dict(groups)


@torch.no_grad()
def evaluate_rollout_per_step(
    model,
    model_type: str,
    tensors: dict[str, torch.Tensor],
    device: str,
    max_horizon: int = 20,
) -> dict[str, list[float]]:
    """Run rollout and return per-step position L1 and composite scores."""
    adapter = make_adapter(model, model_type)
    is_recurrent = model_type == "dreamer"

    B, W, N, D_s = tensors["states"].shape
    mask = tensors["mutable_mask"] & tensors["entity_mask"]
    mutable = tensors["mutable_mask"]
    immutable = ~mutable & tensors["entity_mask"]

    ctx_W = CONTEXT_LEN
    usable_horizon = min(max_horizon, W - ctx_W)
    if usable_horizon <= 0:
        return {"pos_l1": [], "composite": []}

    step_pos_errs = [[] for _ in range(usable_horizon)]
    step_alive_pred = [[] for _ in range(usable_horizon)]
    step_alive_true = [[] for _ in range(usable_horizon)]

    bs = 64
    use_amp = device == "cuda"

    for start in range(0, B, bs):
        batch = {k: v[start:start+bs].to(device) for k, v in tensors.items()}
        b = batch["states"].shape[0]
        b_mask = mask[start:start+bs].to(device)
        b_imm = immutable[start:start+bs].to(device)

        if is_recurrent:
            adapter.reset()
            current_state = batch["states"][:, 0]
            for step in range(ctx_W + usable_horizon):
                action = batch["actions"][:, step]
                globals_ = batch["globals"][:, step]
                if step < ctx_W:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred_state, pred_terminal = adapter.predict_one_step(
                            batch["registry"], batch["states"][:, step],
                            action, globals_, batch["entity_mask"],
                        )
                else:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred_state, pred_terminal = adapter.predict_one_step(
                            batch["registry"], current_state,
                            action, globals_, batch["entity_mask"],
                        )
                    true_state = batch["states"][:, step + 1] if step + 1 < W else batch["states"][:, -1]
                    imm = b_imm.unsqueeze(-1)
                    pred_state = torch.where(imm.expand_as(pred_state), true_state[..., :pred_state.shape[-1]], pred_state)

                    ridx = step - ctx_W
                    if ridx < usable_horizon:
                        true_next = batch["states"][:, step + 1] if step + 1 < W else batch["states"][:, -1]
                        err = (pred_state[..., :2] - true_next[..., :2]).abs().sum(-1)
                        step_pos_errs[ridx].append((err * b_mask.float()).sum().item())
                        step_alive_pred[ridx].append((pred_state[..., 2] > 0.5)[b_mask])
                        step_alive_true[ridx].append((true_next[..., 2] > 0.5)[b_mask])

                    next_full = torch.zeros_like(current_state)
                    next_full[:, :, :2] = pred_state[:, :, :2]
                    next_full[:, :, 2] = pred_state[:, :, 2]
                    gp = pred_state.shape[-1] - 3
                    if gp > 0 and D_s > 5:
                        next_full[:, :, 5:5 + gp] = pred_state[:, :, 3:]
                    current_state = next_full
        else:
            window_states = batch["states"][:, :ctx_W].clone()
            for step in range(usable_horizon):
                act_start = step
                act_end = step + ctx_W + 1
                if act_end > W:
                    break
                step_actions = batch["actions"][:, act_start:act_end]
                step_globals = batch["globals"][:, act_start:act_end]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = adapter.predict_rollout(
                        batch["registry"], window_states,
                        step_actions, step_globals, batch["entity_mask"],
                    )
                if isinstance(preds, tuple):
                    pred_all, _ = preds
                else:
                    pred_all = preds

                pred_state = pred_all[:, 0]
                true_state = batch["states"][:, ctx_W + step]
                imm = b_imm.unsqueeze(-1)
                pred_state = torch.where(imm.expand_as(pred_state), true_state[..., :pred_state.shape[-1]], pred_state)

                true_next = batch["states"][:, ctx_W + step]
                err = (pred_state[..., :2] - true_next[..., :2]).abs().sum(-1)
                step_pos_errs[step].append((err * b_mask.float()).sum().item())
                step_alive_pred[step].append((pred_state[..., 2] > 0.5)[b_mask])
                step_alive_true[step].append((true_next[..., 2] > 0.5)[b_mask])

                new_frame = torch.zeros(b, 1, N, D_s, device=device)
                new_frame[:, 0, :, :2] = pred_state[:, :, :2]
                new_frame[:, 0, :, 2] = pred_state[:, :, 2]
                gp = pred_state.shape[-1] - 3
                if gp > 0 and D_s > 5:
                    new_frame[:, 0, :, 5:5 + gp] = pred_state[:, :, 3:]
                window_states = torch.cat([window_states[:, 1:], new_frame], dim=1)

    n_entities = mask.float().sum().item()
    pos_l1_per_step = []
    comp_per_step = []
    for s in range(usable_horizon):
        if step_pos_errs[s]:
            pl1 = sum(step_pos_errs[s]) / max(n_entities, 1.0)
            pa = torch.cat(step_alive_pred[s]).bool() if step_alive_pred[s] else torch.tensor([])
            ta = torch.cat(step_alive_true[s]).bool() if step_alive_true[s] else torch.tensor([])
            if pa.numel() > 0:
                tp = (pa & ta).sum().float()
                fp = (pa & ~ta).sum().float()
                fn = (~pa & ta).sum().float()
                if tp + fp + fn == 0:
                    af1 = 1.0
                elif tp == 0:
                    af1 = 0.0
                else:
                    prec = tp / (tp + fp).clamp(min=1e-8)
                    rec = tp / (tp + fn).clamp(min=1e-8)
                    af1 = (2 * prec * rec / (prec + rec).clamp(min=1e-8)).item()
            else:
                af1 = 1.0
            comp = composite_score(pl1, af1)
        else:
            pl1 = float("nan")
            comp = float("nan")
        pos_l1_per_step.append(pl1)
        comp_per_step.append(comp)

    return {"pos_l1": pos_l1_per_step, "composite": comp_per_step}


def build_scenario_tensors(episodes: list[dict], max_entities: int) -> dict[str, torch.Tensor]:
    max_frames = max(ep["num_frames"] for ep in episodes)
    return build_windowed_tensors(episodes, max_frames, max_entities)


def generate_pdf_report(all_data: dict, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(str(output_path)) as pdf:
        # --- Title page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.7, "Scenario Evaluation Report", transform=ax.transAxes,
                fontsize=28, fontweight="bold", ha="center", va="center")
        ax.text(0.5, 0.55, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                transform=ax.transAxes, fontsize=14, ha="center", va="center", color="gray")

        summary_lines = []
        for game_id, game_data in sorted(all_data.items()):
            n_scenarios = len(game_data["scenarios"])
            n_models = len(game_data["models"])
            summary_lines.append(f"{game_id}: {n_scenarios} scenarios, {n_models} models")
        ax.text(0.5, 0.35, "\n".join(summary_lines), transform=ax.transAxes,
                fontsize=16, ha="center", va="center", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        for game_id, game_data in sorted(all_data.items()):
            scenarios = game_data["scenarios"]
            models = game_data["models"]
            results = game_data["results"]
            model_names = sorted(models.keys())

            # --- Game summary heatmap: pos_L1 at h10, h16, h20 per scenario per model ---
            scenario_names = sorted(scenarios.keys())
            for horizon_label, horizon_idx in [("h10", 9), ("h16", 15), ("h20", 19)]:
                fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.5 + 2), max(6, len(scenario_names) * 0.5 + 2)))
                matrix = np.full((len(scenario_names), len(model_names)), np.nan)
                for si, sname in enumerate(scenario_names):
                    for mi, mname in enumerate(model_names):
                        key = (sname, mname)
                        if key in results and horizon_idx < len(results[key]["pos_l1"]):
                            matrix[si, mi] = results[key]["pos_l1"][horizon_idx]

                im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(len(scenario_names)))
                ax.set_yticklabels([s.replace("_", " ") for s in scenario_names], fontsize=8)
                for si in range(len(scenario_names)):
                    for mi in range(len(model_names)):
                        val = matrix[si, mi]
                        if not np.isnan(val):
                            color = "white" if val > 0.25 else "black"
                            ax.text(mi, si, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)
                ax.set_title(f"{game_id.upper()} — Position L1 at {horizon_label}", fontsize=14, fontweight="bold")
                fig.colorbar(im, ax=ax, label="Position L1 (lower is better)")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # --- Per-scenario rollout curves ---
            for sname in scenario_names:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                for mname in model_names:
                    key = (sname, mname)
                    if key not in results:
                        continue
                    steps = list(range(1, len(results[key]["pos_l1"]) + 1))
                    ax1.plot(steps, results[key]["pos_l1"], label=mname, marker=".", markersize=3, linewidth=1.5)
                    ax2.plot(steps, results[key]["composite"], label=mname, marker=".", markersize=3, linewidth=1.5)

                ax1.set_xlabel("Rollout Step")
                ax1.set_ylabel("Position L1")
                ax1.set_title(f"Position L1 — {sname.replace('_', ' ')}")
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(1, 20)

                ax2.set_xlabel("Rollout Step")
                ax2.set_ylabel("Composite Score")
                ax2.set_title(f"Composite — {sname.replace('_', ' ')}")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(1, 20)
                ax2.set_ylim(0, 1)

                fig.suptitle(f"{game_id.upper()} / {sname.replace('_', ' ')}", fontsize=13, fontweight="bold")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # --- Model comparison bar chart at h10/h20 ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for idx, (horizon_label, horizon_idx) in enumerate([("h10", 9), ("h20", 19)]):
                ax = axes[idx]
                x = np.arange(len(scenario_names))
                width = 0.8 / max(len(model_names), 1)
                for mi, mname in enumerate(model_names):
                    vals = []
                    for sname in scenario_names:
                        key = (sname, mname)
                        if key in results and horizon_idx < len(results[key]["pos_l1"]):
                            vals.append(results[key]["pos_l1"][horizon_idx])
                        else:
                            vals.append(0)
                    ax.bar(x + mi * width - 0.4 + width / 2, vals, width, label=mname)
                ax.set_xticks(x)
                ax.set_xticklabels([s.replace("_", " ")[:20] for s in scenario_names],
                                   rotation=60, ha="right", fontsize=7)
                ax.set_ylabel("Position L1")
                ax.set_title(f"{game_id.upper()} — {horizon_label}")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on game scenarios with PDF report")
    parser.add_argument("--game", "-g", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--max-horizon", type=int, default=20)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = discover_models()
    if args.game:
        models = [m for m in models if m["game_id"] == args.game]
    if args.model:
        models = [m for m in models if m["model_type"] == args.model]

    print(f"Found {len(models)} model(s)")

    games = sorted(set(m["game_id"] for m in models))
    all_data = {}

    for game_id in games:
        parquet_path = DATA_DIR / game_id / "scenario.parquet"
        if not parquet_path.exists():
            print(f"No scenario parquet for {game_id}, skipping")
            continue

        meta_path = DATA_DIR / game_id / "meta.json"
        meta = json.loads(meta_path.read_text())
        max_entities = meta["max_entities"]

        scenario_groups = group_episodes_by_scenario(parquet_path)
        print(f"\n{game_id}: {len(scenario_groups)} scenarios, "
              f"{sum(len(v) for v in scenario_groups.values())} total episodes")

        game_models = [m for m in models if m["game_id"] == game_id]
        game_results = {}

        for info in game_models:
            print(f"\n  Loading {info['task_name']}...")
            sample_tensors = build_scenario_tensors(
                next(iter(scenario_groups.values())), max_entities
            )
            model = load_model(info, sample_tensors, device)

            for scenario_name, episodes in sorted(scenario_groups.items()):
                tensors = build_scenario_tensors(episodes, max_entities)
                result = evaluate_rollout_per_step(
                    model, info["model_type"], tensors, device,
                    max_horizon=args.max_horizon,
                )
                game_results[(scenario_name, info["model_type"])] = result

                h10 = result["pos_l1"][9] if len(result["pos_l1"]) > 9 else float("nan")
                h20 = result["pos_l1"][19] if len(result["pos_l1"]) > 19 else float("nan")
                print(f"    {scenario_name:<45} h10={h10:.4f}  h20={h20:.4f}")

            del model
            torch.cuda.empty_cache()

        all_data[game_id] = {
            "scenarios": scenario_groups,
            "models": {m["model_type"]: m for m in game_models},
            "results": game_results,
        }

    output_path = Path(args.output) if args.output else REPO_ROOT / "scenario_report.pdf"
    generate_pdf_report(all_data, output_path)

    json_path = output_path.with_suffix(".json")
    json_results = {}
    for game_id, gd in all_data.items():
        json_results[game_id] = {}
        for (sname, mtype), res in gd["results"].items():
            json_results[game_id][f"{sname}/{mtype}"] = res
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()
