#!/usr/bin/env bash
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PATCHES_DIR="$REPO_ROOT/patches"
HARBOR_DIR="$REPO_ROOT/.venv/lib/python3.12/site-packages/harbor"

if [ ! -d "$HARBOR_DIR" ]; then
    echo "ERROR: Harbor not installed at $HARBOR_DIR"
    echo "Run 'uv sync' first."
    exit 1
fi

echo "Applying Harbor patches..."

cp "$PATCHES_DIR/harbor_codex.py" \
   "$HARBOR_DIR/agents/installed/codex.py"
echo "  [1/3] codex.py — ECS exec loop + base64 instruction + auth fixes"

cp "$PATCHES_DIR/harbor_docker.py" \
   "$HARBOR_DIR/environments/docker/docker.py"
echo "  [2/3] docker.py — chmod user=0 for bind-mount permissions"

cp "$PATCHES_DIR/harbor_docker_compose_base.yaml" \
   "$HARBOR_DIR/environments/docker/docker-compose-base.yaml"
echo "  [3/3] docker-compose-base.yaml — stripped deploy.resources"

echo "Done. All Harbor patches applied."
