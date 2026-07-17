# Platformer -- Maskgit World Model

## Mission

Treat the maskgit as a **starting-point world model** for Platformer.

The model receives a short history of past frames and must predict the **next
frame**. By repeating this process autoregressively, it should serve as an
**interactive forward model** of the game that can be stepped forward and
compared against the true simulator.

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

- Game: platformer (24 max entities per frame)
- Active gameplay fields: none
- Shared cache: `/data/platformer/_cache/`
- State: 5 core fields (pos_x, pos_y, alive, vel_x, vel_y) + 14 gameplay fields

## Constraints

- Per-experiment training budget: **600s**
- Window size: **8** frames
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

`experiments/platformer_maskgit/<run_id>/`

After each run, write `EXPERIMENT.md` in that run directory describing:
- hypothesis
- exact change made
- why the change should help h10/h20 rollout fidelity
- final score and key horizon results
- what you learned

## Training Budget

Per-experiment training budget:
- **600 seconds maximum** from `config.json max_train_seconds`

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
