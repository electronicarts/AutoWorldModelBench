# AutoWorldModel-Bench

[![Paper](https://img.shields.io/badge/arXiv-2608.11216-b31b1b.svg)](https://arxiv.org/abs/2608.11216)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://electronicarts.github.io/AutoWorldModelBench/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-AutoWorldModelBench-yellow.svg)](https://huggingface.co/datasets/AutoWorldModel/AutoWorldModelBench)

A closed-loop benchmark framework for evaluating frontier coding agents on open-ended world-model research. Agents autonomously improve starter models under a fixed compute budget across 8 game environments — measuring research capability rather than engineering-to-spec task completion.

📄 **Paper:** [AutoWorldModel-Bench: A State-Centric Benchmark for Automated World-Model Research](https://arxiv.org/abs/2608.11216)

## Starter Models

| Model | Type | Architecture |
|-------|------|-------------|
| **Dreamer** | Continuous, recurrent | DreamerV3-style RSSM with discrete categorical latent (32x32) |
| **AR-Transformer** | Continuous, attention | Autoregressive Transformer with temporal context encoder |
| **D3PM** | Discrete, diffusion | Discrete denoising diffusion with temporal cross-attention |
| **MaskGIT** | Discrete, masked | Masked generative Transformer with iterative parallel decoding |

## Games

| Game | Max Entities | Gameplay Fields |
|------|-------------|-----------------|
| asteroids | 20 | age, hp, size_class |
| breakout | 52 | col, hp, row |
| frogger | 28 | lane_type, reached |
| kong | 16 | — |
| platformer | 24 | on_ground |
| pong | 5 | — |
| racer | 6 | — |
| snake | 48 | direction, segment_index |

## Quick Start

```bash
# Install (requires uv: https://docs.astral.sh/uv/)
uv sync

# Prepare data (downloads from HuggingFace Hub)
python scripts/prepare_data.py --game pong

# Run a single task (train + evaluate)
cd tasks/pong_dreamer
python run.py --config config_template.json
```

> **Note:** `config_template.json` uses container paths (`/data/...`). For local runs,
> create a config with `"cache_dir": "../../data/pong/_cache"` or set up a symlink.

## Dataset

Data is hosted on HuggingFace: [`AutoWorldModel/AutoWorldModelBench`](https://huggingface.co/datasets/AutoWorldModel/AutoWorldModelBench)

`prepare_data.py` automatically downloads and caches parquet files. Each game contains:
- `train.parquet` — 10,000 training episodes
- `val.parquet` — 3,000 validation episodes
- `test.parquet` — 3,000 test episodes
- `scenario.parquet` — 3,000 scenario-based test episodes
- `meta.json` — game metadata (max_entities, dimensions)

## Repository Structure

```
├── lib/                    # Shared infrastructure
│   ├── trainer.py          # Training loop (step-based, AMP, early stopping)
│   ├── loader.py           # Parquet loading + windowed tensor construction
│   ├── evaluator.py        # UnifiedEvaluator (position L1, alive F1, composite)
│   ├── temporal.py         # TemporalContextEncoder, CrossAttentionBlock
│   ├── tokenizer.py        # EntityTokenizer for discrete models
│   ├── task_utils.py       # Experiment tracking utilities
│   └── config.py           # Per-game quantization configs
│
├── templates/              # Source-of-truth model implementations
│   ├── dreamer.py
│   ├── ar_transformer.py
│   ├── d3pm.py
│   └── maskgit.py
│
├── tasks/                  # 32 task directories (8 games x 4 models)
│   └── {game}_{model}/
│       ├── train.py        # Model training (copied from templates/)
│       ├── run.py          # Experiment-tracking wrapper
│       ├── score.py        # Standalone evaluation
│       ├── config_template.json  # Default hyperparameters
│       ├── instruction.md  # Agent instructions
│       ├── task.toml       # Harbor task metadata
│       ├── environment/    # Dockerfile + docker-compose for Harbor
│       └── tests/test.sh   # Verifier script
│
├── scripts/
│   ├── orchestrate.py      # GPU-parallel Harbor orchestrator
│   ├── prepare_data.py     # Download + cache data from HuggingFace
│   ├── setup_tasks.py      # (Re)generate task directories
│   ├── refresh.py          # Sync templates to task directories
│   └── preflight.py        # Environment validation
│
├── docker/                 # Docker images for Harbor
│   ├── Dockerfile.harbor   # Base image (PyTorch, Claude Code, Codex)
│   └── Dockerfile.base     # Minimal base image
│
├── agents/                 # Agent configurations
│   ├── claude-code.json    # Claude Code agent config
│   └── codex.toml          # Codex agent config
│
├── run_harbor.sh           # Single-task Harbor runner
└── agent.py                # LLM-driven hyperparameter optimization agent
```

## Evaluation Metrics

Models are evaluated on multi-step open-loop rollouts at horizons {1, 10, 20}:

- **Position L1**: Mean absolute error on entity (x, y) positions (lower is better)
- **Alive F1**: F1 score on entity alive/dead classification
- **Composite**: Weighted combination `0.9 * (1 - pos_l1) + 0.1 * alive_f1` (higher is better)

## Harbor (Agent Benchmarking)

To run tasks with an AI coding agent via [Harbor](https://github.com/harbor-framework/harbor):

```bash
# Single task
./run_harbor.sh --task pong_dreamer --agent claude-code

# All models for a game
./run_harbor.sh --game pong --agent claude-code

# All 32 tasks with GPU-parallel orchestration
python scripts/orchestrate.py --all --agent claude-code --gpus 0,1,2,3
```

Supported agents: `claude-code`, `codex`

See [docs/SETUP.md](docs/SETUP.md) for Docker setup and agent authentication.

## Editing Models

Edit the template, then sync to task directories:

```bash
vim templates/dreamer.py
python scripts/refresh.py --model dreamer   # updates 8 task dirs
python scripts/refresh.py --all             # updates all 32 dirs
```

## Data Format

Each episode is stored as a row in parquet with binary tensor columns:
- **registry**: `[N, 34]` — static entity attributes (type, archetype, slot)
- **states**: `[T, N, 23]` — dynamic entity state (position, velocity, gameplay fields)
- **actions**: `[T, 7]` — per-frame player actions
- **globals**: `[T, 17]` — global game state features
- **mutable_mask**: `[N]` — which entities to predict
- **gameplay_mask**: `[N, 14]` — per-entity active gameplay fields

Where `T` = episode length, `N` = max entities for the game.

## Contributing

Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor. You can sign [here](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*).

## Citation

If you use AutoWorldModel-Bench in your research, please cite our paper:

```bibtex
@article{autoworldmodelbench,
  title   = {AutoWorldModel-Bench: A State-Centric Benchmark for Automated World-Model Research},
  journal = {arXiv preprint arXiv:2608.11216},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.11216}
}
```

