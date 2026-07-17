#!/bin/bash
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

# Convenience wrapper for running AutoWorldBench tasks with Harbor.
#
# Usage:
#   Single task:
#     ./run_harbor.sh --task pong_dreamer --agent claude-code
#     ./run_harbor.sh --task pong_dreamer --agent codex
#
#   All models for a game:
#     ./run_harbor.sh --game pong --agent claude-code
#
#   All games for a model:
#     ./run_harbor.sh --model dreamer --agent claude-code
#
#   All 32 tasks:
#     ./run_harbor.sh --all --agent claude-code
#
#   Use AWS Bedrock instead of Anthropic API:
#     ./run_harbor.sh --task pong_dreamer --agent claude-code --bedrock
#
#   Extra flags passed through to harbor:
#     ./run_harbor.sh --task pong_dreamer --agent claude-code -- --timeout-multiplier 0.5 -n 2
#
# Prerequisites:
#   docker build -t autoworldbench-base --build-arg HOST_UID=$(id -u) \
#       --build-arg HOST_GID=$(id -g) -f docker/Dockerfile.harbor .

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env (exports HF_TOKEN, API keys, etc.)
[[ -f "${SCRIPT_DIR}/.env" ]] && set -a && source "${SCRIPT_DIR}/.env" && set +a

# Activate venv if harbor is not on PATH
if ! command -v harbor &>/dev/null && [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

DATA_DIR="${SCRIPT_DIR}/data"
TASKS_DIR="${SCRIPT_DIR}/tasks"

# Defaults
AGENT="claude-code"
TASK=""
GAME=""
MODEL=""
RUN_ALL=false
USE_BEDROCK=false
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task|-t)   TASK="$2"; shift 2 ;;
        --game|-g)   GAME="$2"; shift 2 ;;
        --model|-m)  MODEL="$2"; shift 2 ;;
        --agent|-a)  AGENT="$2"; shift 2 ;;
        --all)       RUN_ALL=true; shift ;;
        --bedrock)   USE_BEDROCK=true; shift ;;
        --)          shift; EXTRA_ARGS=("$@"); break ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Build Harbor --path and --include-task-name flags
HARBOR_PATH=""
INCLUDE_FLAGS=()

if [[ -n "$TASK" ]]; then
    # Single task: point directly at the task dir
    HARBOR_PATH="${TASKS_DIR}/${TASK}"
    if [[ ! -d "$HARBOR_PATH" ]]; then
        echo "Error: Task directory not found: $HARBOR_PATH"
        echo "Available tasks:"
        ls "$TASKS_DIR" | head -10
        echo "  ... ($(ls "$TASKS_DIR" | wc -l) total)"
        exit 1
    fi
elif [[ -n "$GAME" && -n "$MODEL" ]]; then
    # Specific game + model = single task
    TASK="${GAME}_${MODEL}"
    HARBOR_PATH="${TASKS_DIR}/${TASK}"
    if [[ ! -d "$HARBOR_PATH" ]]; then
        echo "Error: Task directory not found: $HARBOR_PATH"
        exit 1
    fi
elif [[ -n "$GAME" ]]; then
    # All models for a game: point at tasks/ with include filter
    HARBOR_PATH="$TASKS_DIR"
    INCLUDE_FLAGS+=(-i "${GAME}_*")
    echo "Running all models for game: $GAME"
elif [[ -n "$MODEL" ]]; then
    # All games for a model: point at tasks/ with include filter
    HARBOR_PATH="$TASKS_DIR"
    INCLUDE_FLAGS+=(-i "*_${MODEL}")
    echo "Running all games for model: $MODEL"
elif [[ "$RUN_ALL" == true ]]; then
    # All 32 tasks
    HARBOR_PATH="$TASKS_DIR"
    echo "Running all 32 tasks"
else
    echo "Usage: run_harbor.sh [--task TASK | --game GAME | --model MODEL | --all] [--agent AGENT] [-- extra flags]"
    echo ""
    echo "Options:"
    echo "  --task, -t TASK    Run a single task (e.g. pong_dreamer)"
    echo "  --game, -g GAME    Run all 4 models for a game (e.g. pong)"
    echo "  --model, -m MODEL  Run all 8 games for a model (e.g. dreamer)"
    echo "  --all              Run all 32 tasks"
    echo "  --agent, -a AGENT  Agent to use (default: claude-code)"
    echo "                     Options: claude-code, codex"
    echo "  --bedrock          Use AWS Bedrock instead of Anthropic API (claude-code only)"
    echo ""
    echo "Games:  asteroids, breakout, frogger, kong, platformer, pong, racer, snake"
    echo "Models: dreamer, ar_transformer, d3pm, maskgit"
    echo ""
    echo "Examples:"
    echo "  ./run_harbor.sh --task pong_dreamer --agent claude-code"
    echo "  ./run_harbor.sh --game pong --agent codex"
    echo "  ./run_harbor.sh --model dreamer --agent codex"
    echo "  ./run_harbor.sh --all --agent claude-code -- -n 4"
    exit 1
fi

# Verify data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Data directory not found at $DATA_DIR"
    echo "Run: python setup_tasks.py && python tasks/*/prepare.py"
    exit 1
fi

echo "Agent: $AGENT"
echo "Path:  $HARBOR_PATH"
[[ ${#INCLUDE_FLAGS[@]} -gt 0 ]] && echo "Filter: ${INCLUDE_FLAGS[*]}"
echo ""

# Pre-build tensor caches for the game(s) being run
echo "Preparing data caches..."
if [[ -n "$TASK" ]]; then
    # Single task: extract game_id from its config.json
    _GAME_ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['game_id'])" \
        "${TASKS_DIR}/${TASK}/config.json" 2>/dev/null || echo "")
    if [[ -n "$_GAME_ID" ]]; then
        python scripts/prepare_data.py --game "$_GAME_ID"
    fi
elif [[ -n "$GAME" ]]; then
    python scripts/prepare_data.py --game "$GAME"
elif [[ "$RUN_ALL" == true ]]; then
    python scripts/prepare_data.py
elif [[ -n "$MODEL" ]]; then
    python scripts/prepare_data.py
fi
echo ""

# Build mounts JSON: only mount _cache directories to prevent test data leakage
MOUNTS="["
FIRST=true
for game_dir in "${DATA_DIR}"/*/; do
    cache_dir="${game_dir}_cache"
    if [[ -d "$cache_dir" ]]; then
        game_name=$(basename "$game_dir")
        [[ "$FIRST" == true ]] || MOUNTS+=","
        MOUNTS+="{\"type\":\"bind\",\"source\":\"${cache_dir}\",\"target\":\"/data/${game_name}/_cache\",\"read_only\":true}"
        FIRST=false
    fi
done
MOUNTS+="]"

AGENT_FLAGS=()

case "$AGENT" in
    claude-code)
        if [[ "$USE_BEDROCK" == true ]]; then
            # AWS Bedrock mode
            AGENT_FLAGS+=(--ae CLAUDE_CODE_USE_BEDROCK=1)
            AGENT_FLAGS+=(--ae AWS_REGION=${AWS_REGION:-us-east-1})
            if [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
                AGENT_FLAGS+=(--ae "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}")
                AGENT_FLAGS+=(--ae "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}")
                AGENT_FLAGS+=(--ae "AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}")
            else
                AGENT_FLAGS+=(--ae AWS_PROFILE=${AWS_PROFILE:-default})
            fi
        else
            # Standard Anthropic API mode (default)
            if [[ -n "${ANTHROPIC_BASE_URL:-}" ]]; then
                AGENT_FLAGS+=(--ae "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}")
            fi
            if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
                AGENT_FLAGS+=(--ae "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
            else
                echo "Warning: ANTHROPIC_API_KEY not set. Set it in .env or export it." >&2
            fi
        fi
        ;;
    codex)
        MODEL=$(python3 -c "import tomllib; print(tomllib.load(open('${SCRIPT_DIR}/agents/codex.toml','rb'))['model'])")
        AGENT_FLAGS+=(-m "$MODEL")
        MOUNTS="${MOUNTS%]},{\"type\":\"bind\",\"source\":\"${SCRIPT_DIR}/agents/codex.toml\",\"target\":\"/tmp/codex-config-staged.toml\",\"read_only\":true},{\"type\":\"bind\",\"source\":\"${SCRIPT_DIR}/agents/codex_loop.py\",\"target\":\"/tmp/codex-loop.py\",\"read_only\":true}]"
        if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
            AGENT_FLAGS+=(--ae "OPENAI_BASE_URL=${OPENAI_BASE_URL}")
        fi
        if [[ -n "${OPENAI_API_KEY:-}" ]]; then
            AGENT_FLAGS+=(--ae "OPENAI_API_KEY=${OPENAI_API_KEY}")
        elif [[ -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
            AGENT_FLAGS+=(--ae "OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}")
        else
            echo "Warning: OPENAI_API_KEY not set" >&2
        fi
        ;;
esac

harbor run \
    --path "$HARBOR_PATH" \
    --agent "$AGENT" \
    --mounts-json "$MOUNTS" \
    --artifact /app/experiments \
    --artifact /app/outputs \
    --artifact /app/summary.tsv \
    --artifact /app/train.py \
    --artifact /app/config.json \
    -n 1 \
    "${AGENT_FLAGS[@]}" \
    "${INCLUDE_FLAGS[@]}" \
    "${EXTRA_ARGS[@]}"
