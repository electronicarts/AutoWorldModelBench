# Environment Setup

## Local Development (no Docker)

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Or with pip:
pip install -e .

# Prepare data for a game (downloads from HuggingFace)
python scripts/prepare_data.py --game pong

# Train a single model
cd tasks/pong_dreamer
python train.py
```

## Harbor (Agent Benchmarking)

Harbor runs each task inside a Docker container with an AI coding agent.
Follow **all** steps below for a fresh machine.

### 0. Install Harbor

```bash
uv sync --extra harbor
```

This installs Harbor and all dependencies into `.venv/`.

### 1. Agent Authentication

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

| Agent | Required Variable | Optional |
|-------|-------------------|----------|
| `claude-code` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` (custom gateway) |
| `claude-code --bedrock` | `AWS_PROFILE` + `AWS_REGION` | — |
| `codex` | `OPENAI_API_KEY` (or `AZURE_OPENAI_API_KEY`) | `OPENAI_BASE_URL` (Azure/proxy) |

For **AWS Bedrock** mode, configure credentials:

```ini
# ~/.aws/config
[profile default]
sso_session = my-sso
sso_account_id = <your-account-id>
sso_role_name = <role-with-bedrock-access>
region = us-east-1

[sso-session my-sso]
sso_start_url = https://<your-org>.awsapps.com/start/#
sso_region = us-east-1
sso_registration_scopes = sso:account:access
```

Then: `aws sso login`

The container runs as a non-root `agent` user (UID 1007). If `~/.aws` files are
owner-only (common when running as root), fix permissions:

```bash
chmod 644 ~/.aws/config
chmod -R o+rX ~/.aws/sso ~/.aws/cli
```

### 2. Build the Base Docker Image

Every per-task Dockerfile starts with `FROM autoworldbench-base:latest`:

```bash
docker build -t autoworldbench-base \
    --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) \
    -f docker/Dockerfile.harbor .
```

### 3. NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Apply Harbor Patches

Harbor requires patches for Codex exec loop support and bind-mount permissions:

```bash
bash scripts/apply_harbor_patches.sh
```

**Re-run after every `uv sync` or Harbor upgrade.**

| Patch file | Purpose |
|-----------|---------|
| `harbor_codex.py` | Codex exec loop, base64 instruction encoding, auth fixes |
| `harbor_docker.py` | `chmod` as root so agent user can write to bind-mounted dirs |
| `harbor_docker_compose_base.yaml` | Strip `deploy.resources` for compatibility |

### 5. Validate

```bash
# Network from containers:
docker run --rm alpine sh -c "nslookup archive.ubuntu.com"

# GPU access:
docker run --rm --runtime=nvidia alpine sh -c "echo GPU ok"

# Base image exists:
docker images | grep autoworldbench-base

# End-to-end test:
./run_harbor.sh --task pong_dreamer --agent claude-code
```

### 6. Run

```bash
# Single task
./run_harbor.sh --task pong_dreamer --agent claude-code

# All models for one game
./run_harbor.sh --game pong --agent claude-code

# GPU-parallel orchestration (all 32 tasks)
python scripts/orchestrate.py --all --agent claude-code --gpus 0,1,2,3

# Dry run (show what would execute)
python scripts/orchestrate.py --all --agent claude-code --dry-run
```

### Preflight Check

Validate the full environment before running:

```bash
python scripts/preflight.py
python scripts/preflight.py --fix   # auto-fix: prepare data, apply patches
```
