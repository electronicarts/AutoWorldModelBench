# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Generate all 32 game x model task directories in Harbor format.

Creates tasks/{game}_{model}/ with:
  - task.toml              (Harbor task config)
  - instruction.md         (agent instructions)
  - environment/Dockerfile (per-task Docker layer on autoworldbench-base)
  - environment/docker-compose.yaml (GPU, data mounts, auth overrides)
  - tests/test.sh          (verifier: extracts best composite from summary.tsv)
  - train.py               (copied from templates/{model}.py)

  - config.json            (game x model hyperparameters)
  - config_template.json   (copy of config.json, for agent to modify)
  - run.py                 (experiment-tracking wrapper around train.py)
  - score.py               (sandboxed scoring API wrapper)

Usage:
    python setup_tasks.py              # all 32 tasks
    python setup_tasks.py --model d3pm # only 8 D3PM tasks
    python setup_tasks.py --game snake # only 4 snake tasks
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES = REPO_ROOT / "templates"
TASKS = REPO_ROOT / "tasks"
DATA = REPO_ROOT / "data"

GAMES = [
    "asteroids", "breakout", "frogger", "kong", "platformer",
    "pong", "racer", "snake",
]
MODELS = ["dreamer", "ar_transformer", "d3pm", "maskgit"]

# Wall-clock training budget (seconds). At runtime this is controlled by the
# MAX_TRAIN_SECONDS environment variable set by Harbor. The value here is only
# used as a default when generating config.json for local development.
MAX_TRAIN_SECONDS = int(os.environ.get("MAX_TRAIN_SECONDS", 600))

# Evaluation horizons. Single source of truth — prepare_data.py reads this to
# compute EVAL_WINDOW_SIZE, and config.json / score.py get these values.
EVAL_HORIZONS = [1, 10, 20]

# Batch sizes: model x max_entity_count ranges (from empirical H100 profiling)
BATCH_SIZES = {
    "dreamer":          {(0, 10): 4096, (10, 20): 2048, (20, 30): 1024, (30, 999): 512},
    "ar_transformer":   {(0, 10): 2048, (10, 20): 512,  (20, 30): 256,  (30, 55): 192, (55, 999): 128},
    "d3pm":             {(0, 10): 2048, (10, 20): 512,  (20, 30): 256,  (30, 55): 192, (55, 999): 128},
    "maskgit":          {(0, 10): 2048, (10, 20): 512,  (20, 30): 256,  (30, 55): 192, (55, 999): 128},
}


def get_batch_size(model: str, max_entities: int) -> int:
    for (lo, hi), bs in BATCH_SIZES[model].items():
        if lo <= max_entities < hi:
            return bs
    return 64


def _get_gameplay_fields(game: str) -> list[str]:
    """Get active gameplay field names for a game."""
    try:
        from lib.config import GAME_GAMEPLAY_FIELDS, GAMEPLAY_FIELD_NAMES
        indices = GAME_GAMEPLAY_FIELDS.get(game, [])
        return [GAMEPLAY_FIELD_NAMES[i] for i in indices]
    except ImportError:
        return []


# ---------------------------------------------------------------------------
# Harbor format generators
# ---------------------------------------------------------------------------

def _generate_task_toml(task_dir: Path, game: str, model: str) -> None:
    """Generate task.toml -- Harbor task configuration."""
    hard_games = {"asteroids", "frogger", "kong"}
    easy_games = {"pong", "snake", "racer"}
    if game in hard_games:
        difficulty = "hard"
    elif game in easy_games:
        difficulty = "easy"
    else:
        difficulty = "medium"

    tags = [game, model]
    discrete_grid = {"snake", "frogger"}
    continuous_physics = {"pong", "breakout", "asteroids", "racer", "kong"}
    if game in discrete_grid:
        tags.append("discrete-grid")
    elif game in continuous_physics:
        tags.append("continuous-physics")
    if model in ("dreamer", "ar_transformer"):
        tags.append("continuous-model")
    else:
        tags.append("discrete-model")

    keywords_str = ", ".join(f'"{t}"' for t in [*tags, "world-model", "autoworldbench"])

    content = dedent(f"""\
        version = "1.1"

        [task]
        name = "autoworldbench/{game}_{model}"
        description = "Maximize composite score for {game} using {model} world model"
        authors = [{{name = "AutoWorldBench"}}]
        keywords = [{keywords_str}]

        [metadata]
        game_id = "{game}"
        model_type = "{model}"
        difficulty = "{difficulty}"
        category = "world_model_research"

        [verifier]
        timeout_sec = 300.0
        user = "root"

        [agent]
        timeout_sec = 28800.0
        user = "agent"

        [environment]
        build_timeout_sec = 600.0
        cpus = 4
        memory_mb = 235520
        storage_mb = 51200
        gpus = 0
        allow_internet = true

        [environment.env]
        TASK_ID = "{game}_{model}"
        GAME_ID = "{game}"
        MODEL_TYPE = "{model}"
    """)
    (task_dir / "task.toml").write_text(content)


def _generate_instruction_md(task_dir: Path, game: str, model: str, config: dict, max_N: int) -> None:
    """Generate instruction.md -- agent instructions (Harbor format)."""
    gameplay_fields = _get_gameplay_fields(game)
    fields_str = ", ".join(gameplay_fields) if gameplay_fields else "none"
    task_name = f"{game}_{model}"
    game_title = game.replace('_', ' ').title()
    window_size = config.get('window_size', 8)

    SP = " " * 8  # match dedent indentation of the outer template
    if model == "dreamer":
        model_desc = (
            f"The model maintains a learned latent state that encodes the game history, and\n"
            f"{SP}predicts the **next frame** by advancing this state forward. By repeating this\n"
            f"{SP}process, it should serve as an **interactive forward model** of the game that\n"
            f"{SP}can be rolled forward in latent space and compared against the true simulator."
        )
    else:
        model_desc = (
            f"The model receives a short history of past frames and must predict the **next\n"
            f"{SP}frame**. By repeating this process autoregressively, it should serve as an\n"
            f"{SP}**interactive forward model** of the game that can be stepped forward and\n"
            f"{SP}compared against the true simulator."
        )
    model_label = model.replace('_', ' ')

    content = dedent(f"""\
        # {game_title} -- {model_label.title()} World Model

        ## Mission

        Treat the {model_label} as a **starting-point world model** for {game_title}.

        {model_desc}

        Your job is to improve the baseline so that predicted rollouts remain faithful to
        real gameplay, especially over longer horizons where small errors compound.

        ## Success Criterion

        The **only objective that matters** is the **final score** produced by the evaluator from a completed run.

        Per-horizon composite:

        `0.9 * (1 - position_l1) + 0.1 * alive_f1`

        Final score:

        `0.1 * h1 + 0.2 * h10 + 0.7 * h20`

        This means:
        - **90% of the final score comes from h10 and h20**.
        - **Position accuracy dominates** each per-horizon composite.
        - Single-step quality matters only insofar as it improves long-horizon rollout fidelity.

        ### Critical interpretation

        - `val_loss`, training loss, wall-clock time, and throughput are **diagnostics only**.
        - They are **not optimization targets**.
        - A lower `val_loss` does **not** count as an improvement unless the **final score** improves.
        - A shorter training run does **not** count as an improvement unless the **final score** improves in a controlled comparison.

        When making decisions, optimize for **h10/h20 rollout stability**, not for one-step validation behavior.

        ## Data

        - Game: {game} ({max_N} max entities per frame)
        - Active gameplay fields: {fields_str}
        - Shared cache: `/data/{game}/_cache/`
        - State: 5 core fields (pos_x, pos_y, alive, vel_x, vel_y) + 14 gameplay fields

        ## Constraints

        - Per-experiment training budget: **{MAX_TRAIN_SECONDS}s**
        - Window size: **{window_size}** frames
        - Evaluation horizons: **h=1, h=10, h=20**

        ## Ground Rules

        ### Files you may modify
        You may ONLY modify:
        - `train.py` — model architecture, optimizer, hyperparameters, training loop
        - `config.json`
        - `configs/*` — create new configs here
        - `experiments/*` — experiment outputs, logs, notes

        Everything else is READ-ONLY. Do not modify, overwrite, or shadow:
        - `score.py`
        - `run.py`
        - `evaluator.py`
        - anything under `/opt/autoworldbench/lib/`
        - `instruction.md`
        - `/data/`

        Do not create new Python files that shadow existing modules.
        The evaluation harness is the ground-truth metric.

        ## First Action: Establish the Baseline

        **Check `summary.tsv` first.** If it already has data rows, skip the baseline
        and continue iterating from the latest result — you are resuming a prior session.

        If `summary.tsv` does not exist or has no data rows, your first run must be the
        unmodified baseline:

        ```bash
        python run.py --config config_template.json
        ```

        Do not modify any code before this first run.
        All future experiments must be judged against this baseline.

        ## Running Experiments

        ```bash
        python run.py --config config_template.json
        python run.py --config configs/my_experiment.json
        ```

        Each run trains, evaluates, saves results under:

        `experiments/{task_name}/<run_id>/`

        After each run, write `EXPERIMENT.md` in that run directory describing:
        - hypothesis
        - exact change made
        - why the change should help h10/h20 rollout fidelity
        - final score and key horizon results
        - what you learned

        ## Training Budget

        Per-experiment training budget:
        - **{MAX_TRAIN_SECONDS} seconds maximum** from `config.json max_train_seconds`

        - For any **serious, non-failing experiment**, use the **full allotted training budget**.
        - Do **not** introduce early stopping as a default strategy.
        - Do **not** shorten training just to run more experiments.
        - Do **not** prefer earlier checkpoints because they have lower `val_loss`.

        Early stopping is only allowed if the explicit hypothesis is that less training
        improves the **final score**, tested as a controlled experiment against a full-budget run.

        ## Research Priorities

        Prioritize changes that are plausibly causal for **long-horizon rollout fidelity**.

        Good directions include:
        - training objectives that better match open-loop rollout behavior
        - methods that reduce compounding error across steps
        - architecture changes that improve temporal consistency and entity dynamics
        - losses or curricula that emphasize h10/h20 behavior
        - better handling of velocity, persistence, alive/dead transitions, and structured state evolution
        - training strategies that improve robustness under autoregressive rollout

        When choosing between ideas, prefer the one with the stronger causal connection to:
        - lower position drift over long horizons
        - more stable open-loop rollouts
        - better entity persistence and transition modeling

        ## Experiment Validity

        A run only counts as evidence if:
        1. It uses the standard evaluation harness.
        2. It completes evaluation and produces a **final score**.
        3. It is compared fairly against prior runs.
        4. Its claimed improvement is based on **final score**, not on proxy metrics.

        Never prefer an experiment because it has a better `val_loss` if its **final score** is worse or unproven.

        Keep comparisons fair — unless explicitly testing a specific variable, keep fixed:
        training budget, stopping policy, evaluation procedure, data source.

        ## Failure Handling

        **Simple bugs** (typo, missing import, shape mismatch): fix and re-run.

        **Fundamental failures** (OOM, NaN loss, non-convergence): log the failure and
        move on to a different idea. Do not keep retrying without a new hypothesis.

        ## How You Work: Iterative Research Loop

        After every experiment, follow this exact cycle:
        1. Read the **final score** from the completed run.
        2. What happened specifically at **h10** and **h20**?
        3. Did position error improve, worsen, or shift across horizons?
        4. What concrete failure mode should the **next experiment** target?
        5. Create **one** new config based on this analysis.
        6. Run that experiment immediately.

        Each experiment must be informed by the one before it. This is iterative research.
        Decide the next experiment ONLY after seeing the previous result.
        After completing a run, immediately start the next one.

        ## One-Sentence Rule

        When in doubt, choose the action that is **most likely to improve the final score
        on long-horizon rollouts**, not the action that merely lowers `val_loss`,
        shortens training, or increases experiment throughput.
    """)
    (task_dir / "instruction.md").write_text(content)


def _generate_environment_dockerfile(task_dir: Path, game: str, model: str) -> None:
    """Generate environment/Dockerfile -- thin layer on autoworldbench-base."""
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    task_name = f"{game}_{model}"
    content = dedent(f"""\
        FROM autoworldbench-base:latest

        # Nest task files so run.py's REPO_ROOT (TASK_DIR.parent.parent) = /app/ (writable)
        COPY --chown=agent:agent train.py run.py config.json config_template.json score.py /app/tasks/{task_name}/

        WORKDIR /app/tasks/{task_name}

        # Inherited from autoworldbench-base; stated explicitly so the layer runs non-root.
        USER agent
    """)
    (env_dir / "Dockerfile").write_text(content)


def _generate_environment_compose(task_dir: Path, game: str, model: str) -> None:
    """Generate environment/docker-compose.yaml -- GPU, mounts, auth overrides."""
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    task_name = f"{game}_{model}"
    task_path = f"/app/tasks/{task_name}"
    content = dedent(f"""\
        services:
          main:
            runtime: nvidia
            build:
              context: ..
              dockerfile: environment/Dockerfile
              args:
                HOST_UID: ${{HOST_UID:-1007}}
                HOST_GID: ${{HOST_GID:-1007}}
            shm_size: 8g
            environment:
              - PYTHONPATH=/opt/autoworldbench/lib:/opt/autoworldbench:{task_path}
              - TASK_ID={task_name}
              - GAME_ID={game}
              - MODEL_TYPE={model}
              - MAX_TRAIN_SECONDS={MAX_TRAIN_SECONDS}
            volumes:
              - ${{HOME}}/.aws:/home/agent/.aws:ro
              - ${{HOME}}/.claude:/home/agent/.claude:ro
    """)
    (env_dir / "docker-compose.yaml").write_text(content)


def _generate_test_sh(task_dir: Path) -> None:
    """Generate tests/test.sh -- verifier that independently evaluates the checkpoint."""
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    task_name = task_dir.name
    content = dedent("""\
        #!/bin/bash
        set -e
        cd /app/tasks/__TASK_NAME__

        mkdir -p /logs/verifier

        # ---------------------------------------------------------------------------
        # Strategy: independently evaluate the agent's best checkpoint via score.py.
        # Fall back to self-reported summary.tsv only if score.py fails.
        # ---------------------------------------------------------------------------

        SCORE_OK=0

        # Prefer best_model_overall.pt (survives agent timeouts) over best_model.pt
        # which may have been overwritten by a later, worse run's trainer.
        BEST_CKPT=""
        if [ -f outputs/best_model_overall.pt ]; then
            BEST_CKPT=outputs/best_model_overall.pt
        elif [ -f outputs/best_model.pt ]; then
            BEST_CKPT=outputs/best_model.pt
        fi

        if [ -n "$BEST_CKPT" ]; then
            echo "Found $BEST_CKPT -- running independent evaluation ..."
            SCORE_JSON=$(python score.py --checkpoint "$BEST_CKPT" --horizons 1 10 20 2>/tmp/score_stderr) && SCORE_OK=1 || true

            if [ "$SCORE_OK" = "1" ] && [ -n "$SCORE_JSON" ]; then
                # Weighted multi-horizon score: 0.1*h1 + 0.2*h10 + 0.7*h20
                WEIGHTED_SCORE=$(echo "$SCORE_JSON" | python3 -c "
        import sys, json
        d = json.load(sys.stdin)
        h1 = d.get('h1', {}).get('composite', 0.0)
        h10 = d.get('h10', {}).get('composite', 0.0)
        h20 = d.get('h20', {}).get('composite', 0.0)
        score = 0.1 * h1 + 0.2 * h10 + 0.7 * h20
        print(round(score, 6))
        " 2>/dev/null) || WEIGHTED_SCORE=""

                if [ -n "$WEIGHTED_SCORE" ] && [ "$WEIGHTED_SCORE" != "None" ]; then
                    echo "$WEIGHTED_SCORE" > /logs/verifier/reward.txt
                    echo "$SCORE_JSON" > /logs/verifier/rewards.json
                    echo "Weighted score (0.1*h1 + 0.2*h10 + 0.7*h20): $WEIGHTED_SCORE"
                    exit 0
                else
                    echo "WARNING: could not parse weighted score from score.py output"
                    SCORE_OK=0
                fi
            else
                echo "WARNING: score.py failed or produced no output"
                [ -f /tmp/score_stderr ] && cat /tmp/score_stderr
                SCORE_OK=0
            fi
        else
            echo "No best checkpoint found -- skipping independent eval"
        fi

        # ---------------------------------------------------------------------------
        # Fallback: read self-reported composite from summary.tsv
        # ---------------------------------------------------------------------------
        echo "Falling back to summary.tsv ..."

        if [ ! -f summary.tsv ] || [ $(wc -l < summary.tsv) -le 1 ]; then
            echo "0.0" > /logs/verifier/reward.txt
            echo '{"composite": 0.0, "source": "no_experiments"}' > /logs/verifier/rewards.json
            echo "No experiments found in summary.tsv"
            exit 0
        fi

        BEST_COMPOSITE=$(tail -n +2 summary.tsv | awk -F'\\t' '{print $4}' | grep -v '^$' | sort -rn | head -1)

        if [ -z "$BEST_COMPOSITE" ]; then
            echo "0.0" > /logs/verifier/reward.txt
            echo '{"composite": 0.0, "source": "no_composite_in_tsv"}' > /logs/verifier/rewards.json
            echo "No composite score found"
            exit 0
        fi

        NUM_EXPERIMENTS=$(tail -n +2 summary.tsv | wc -l)

        echo "$BEST_COMPOSITE" > /logs/verifier/reward.txt
        cat > /logs/verifier/rewards.json << ENDJSON
        {
            "composite": $BEST_COMPOSITE,
            "num_experiments": $NUM_EXPERIMENTS,
            "source": "summary_tsv_fallback"
        }
        ENDJSON

        echo "Best composite (self-reported): $BEST_COMPOSITE (from $NUM_EXPERIMENTS experiments)"
    """).replace("__TASK_NAME__", task_name)
    (tests_dir / "test.sh").write_text(content)


def _generate_score_py(task_dir: Path, game: str, model: str) -> None:
    """Generate score.py -- independent checkpoint evaluation for the verifier."""
    content = dedent('''\
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

        # -- path setup: works inside Docker container where TASK_DIR=/app --
        TASK_DIR = Path(__file__).parent

        # Add library paths (container layout: /opt/autoworldbench/lib)
        sys.path.insert(0, "/opt/autoworldbench/lib")
        sys.path.insert(0, "/opt/autoworldbench")
        # Also support running from repo checkout
        sys.path.insert(0, str(TASK_DIR.parent.parent / "lib"))
        sys.path.insert(0, str(TASK_DIR.parent / "lib"))
        # Add TASK_DIR itself so train.py can import its siblings (e.g. tokenizer)
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
    ''')
    (task_dir / "score.py").write_text(content)


def _generate_run_py(task_dir: Path) -> None:
    """Generate run.py -- experiment-tracking wrapper around train.py."""
    content = dedent('''\
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
    ''')
    (task_dir / "run.py").write_text(content)


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------

def create_task(game: str, model: str) -> None:
    task_dir = TASKS / f"{game}_{model}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Copy model-specific train.py only if it doesn't already exist
    src = TEMPLATES / f"{model}.py"
    if not src.exists():
        print(f"  WARNING: template {src} not found, skipping {game}_{model}")
        return
    dst = task_dir / "train.py"
    if not dst.exists():
        shutil.copy(src, dst)

    # Read max_entities from game meta if available
    meta_path = DATA / game / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        max_N = meta["max_entities"]
    else:
        max_N = 20  # safe default

    meta_eps = meta.get("num_train", 10000) if meta_path.exists() else 10000
    max_train_episodes = min(meta_eps, 10000)

    # Write config.json
    config = {
        "game_id": game,
        "model_type": model,
        "cache_dir": f"/data/{game}/_cache",
        "hidden_dim": 256,
        "batch_size": get_batch_size(model, max_N),
        "max_steps": 10000,
        "max_train_seconds": MAX_TRAIN_SECONDS,
        "max_train_episodes": max_train_episodes,
        "eval_every": 250,
        "patience": 10,
        "lr": 3e-4,
        "window_size": 8,
        "seed": 42,
        "final_eval_horizons": EVAL_HORIZONS,
    }
    (task_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Generate config_template.json (copy of config.json for agent to modify)
    (task_dir / "config_template.json").write_text(json.dumps(config, indent=2))

    # Harbor format files
    _generate_task_toml(task_dir, game, model)
    _generate_instruction_md(task_dir, game, model, config, max_N)
    _generate_environment_dockerfile(task_dir, game, model)
    _generate_environment_compose(task_dir, game, model)
    _generate_test_sh(task_dir)

    # Scoring and run wrappers
    _generate_score_py(task_dir, game, model)
    _generate_run_py(task_dir)


def _find_harbor_package_dir() -> Path | None:
    """Locate the installed harbor package directory."""
    try:
        import harbor
        return Path(harbor.__file__).parent
    except ImportError:
        return None


def patch_harbor():
    """Apply runtime patches to the installed Harbor package.

    Patches (idempotent, safe to re-run):
      a) Strip cgroup resource limits from the base docker-compose.
      b) Run the log-directory chmod as root (user=0) so bind-mounted host dirs
         created by the Docker daemon are accessible to the non-root agent user.
    """
    harbor_dir = _find_harbor_package_dir()
    if harbor_dir is None:
        print("  WARNING: harbor package not found, skipping patches")
        return

    patched = []

    # (a) Strip deploy.resources from base compose for compatibility
    compose_base = harbor_dir / "environments" / "docker" / "docker-compose-base.yaml"
    if compose_base.exists():
        desired = dedent("""\
            services:
              main:
                volumes:
                  - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
                  - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}
                  - ${HOST_ARTIFACTS_PATH}:${ENV_ARTIFACTS_PATH}
        """)
        if compose_base.read_text() != desired:
            compose_base.write_text(desired)
            patched.append("strip cgroup resource limits from base compose")

    # (b) chmod as root so bind-mounted dirs are writable by agent
    docker_py = harbor_dir / "environments" / "docker" / "docker.py"
    if docker_py.exists():
        src = docker_py.read_text()
        old = (
            'f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"\n'
            '        )'
        )
        new = (
            'f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}",\n'
            '            user=0,\n'
            '        )'
        )
        if old in src:
            docker_py.write_text(src.replace(old, new))
            patched.append("chmod log dirs as root (user=0)")

    # (c) Make codex install skip nvm when codex is already on PATH
    # (d) Copy staged config to writable location so hooks can be appended
    # (e) Inject stop hook config (features + hooks.Stop) into config.toml
    # (f) Wrap codex exec in a restart loop with continuation context
    codex_py = harbor_dir / "agents" / "installed" / "codex.py"
    if codex_py.exists():
        src = codex_py.read_text()

        # (c) skip nvm install
        old_install = (
            '        # Install codex (as default user)\n'
            '        version_spec = f"@{self._version}" if self._version else "@latest"\n'
            '        await self.exec_as_agent(\n'
            '            environment,\n'
            '            command=(\n'
            '                "set -euo pipefail; "\n'
        )
        new_install = (
            '        # Install codex (as default user) — skip if already on PATH\n'
            '        version_spec = f"@{self._version}" if self._version else "@latest"\n'
            '        await self.exec_as_agent(\n'
            '            environment,\n'
            '            command=(\n'
            '                "set -euo pipefail; command -v codex &>/dev/null && { codex --version; exit 0; }; "\n'
        )
        if old_install in src:
            src = src.replace(old_install, new_install)
            patched.append("skip codex nvm install when already on PATH")

        # (d) Config staging + exec loop: replace the codex exec block with
        #     a loop wrapper that restarts codex when it stops.
        #     The loop script itself is agents/codex_loop.py, mounted at
        #     /tmp/codex-loop.py by orchestrate.py / run_harbor.sh.
        old_setup_fence = (
            '        if setup_command.strip():\n'
            '            await self.exec_as_agent(\n'
            '                environment,\n'
            '                command=setup_command,\n'
            '                env=env,\n'
            '            )\n'
            '        try:\n'
            '            await self.exec_as_agent(\n'
            '                environment,\n'
            '                command=(\n'
            '                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "\n'
            '                    "codex exec "\n'
            '                    "--dangerously-bypass-approvals-and-sandbox "\n'
            '                    "--skip-git-repo-check "\n'
            '                    f"--model {model} "\n'
            '                    "--json "\n'
            '                    "--enable unified_exec "\n'
            '                    f"{cli_flags_arg}"\n'
            '                    "-- "\n'
            '                    f"{escaped_instruction} "\n'
            '                    f"2>&1 </dev/null | tee {\n'
            '                        EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME\n'
            '                    }"\n'
            '                ),\n'
            '                env=env,\n'
            '            )'
        )
        new_setup_fence = (
            '        # --- ECS patch: config staging + exec loop ---\n'
            '        _agent_dir = EnvironmentPaths.agent_dir.as_posix()\n'
            '        _output_path = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()\n'
            '        import base64 as _b64\n'
            '        _b64_instruction = _b64.b64encode(instruction.encode()).decode()\n'
            '        setup_command += (\n'
            '            f\'\\ncp /tmp/codex-config-staged.toml "{_agent_dir}/config.toml" 2>/dev/null || true\\n\'\n'
            '            f\'echo "{_b64_instruction}" | base64 -d > /tmp/codex-instruction.txt\\n\'\n'
            '        )\n'
            '\n'
            '        if setup_command.strip():\n'
            '            await self.exec_as_agent(\n'
            '                environment,\n'
            '                command=setup_command,\n'
            '                env=env,\n'
            '            )\n'
            '        try:\n'
            '            await self.exec_as_agent(\n'
            '                environment,\n'
            '                command=(\n'
            '                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "\n'
            '                    "python3 /tmp/codex-loop.py "\n'
            '                    f"--model {model} "\n'
            '                    f"--output {_output_path} "\n'
            '                    "--instruction /tmp/codex-instruction.txt "\n'
            '                    f"--flags \'{cli_flags_arg.strip()}\'"\n'
            '                ),\n'
            '                env=env,\n'
            '            )'
        )
        if old_setup_fence in src:
            src = src.replace(old_setup_fence, new_setup_fence)
            patched.append("codex exec loop with config staging")

        codex_py.write_text(src)

    if patched:
        print(f"  Harbor patches applied: {'; '.join(patched)}")
    else:
        print("  Harbor patches already applied")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    games = [args.game] if args.game else GAMES
    models = [args.model] if args.model else MODELS

    patch_harbor()

    TASKS.mkdir(exist_ok=True)
    total = len(games) * len(models)
    print(f"Creating {total} task directories (Harbor format)...")

    for model in models:
        for game in games:
            create_task(game, model)
            print(f"  tasks/{game}_{model}/")

    print(f"\nDone. {total} task directories created in {TASKS}/")
    print("Build base image: docker build -t autoworldbench-base -f docker/Dockerfile.harbor .")
    print("Run a task: harbor run --path ./tasks/pong_dreamer --agent claude-code")


if __name__ == "__main__":
    main()
