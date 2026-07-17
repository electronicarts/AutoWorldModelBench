# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Preflight validation for AutoWorldBench GPU-parallel Harbor runs.

Validates Docker, GPUs, auth, base image, data caches, and connectivity.
Exit 0 if all blockers pass, exit 1 otherwise.

Usage:
    python scripts/preflight.py                      # all checks
    python scripts/preflight.py --fix                 # auto-fix: prepare data, apply Harbor patches
    python scripts/preflight.py --skip-connectivity   # skip Bedrock/Azure API tests
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "data"
TASKS_DIR = REPO_ROOT / "tasks"

ALL_GAMES = [
    "asteroids", "breakout", "frogger", "kong", "platformer",
    "pong", "racer", "snake",
]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class CheckResult:
    name: str
    passed: bool
    blocker: bool
    message: str


@dataclass
class PreflightReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)
        icon = f"{GREEN}✓{RESET}" if result.passed else (
            f"{RED}✗{RESET}" if result.blocker else f"{YELLOW}⚠{RESET}"
        )
        label = "FAIL" if not result.passed and result.blocker else (
            "WARN" if not result.passed else "OK"
        )
        print(f"  {icon} [{label}] {result.name}: {result.message}")

    @property
    def has_blockers(self) -> bool:
        return any(not r.passed and r.blocker for r in self.results)

    def summary(self) -> None:
        passed = sum(1 for r in self.results if r.passed)
        warnings = sum(1 for r in self.results if not r.passed and not r.blocker)
        blocked = sum(1 for r in self.results if not r.passed and r.blocker)
        total = len(self.results)
        print(f"\n{BOLD}Summary:{RESET} {passed}/{total} passed, "
              f"{warnings} warnings, {blocked} blockers")
        if self.has_blockers:
            print(f"{RED}Preflight FAILED — fix blockers above before running.{RESET}")
        else:
            print(f"{GREEN}Preflight PASSED{RESET}")


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _load_env() -> dict[str, str]:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export").strip()
                v = v.strip().strip('"').strip("'")
                env[k] = v
    return env


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_docker_daemon(report: PreflightReport) -> None:
    try:
        r = _run(["docker", "info"])
        if r.returncode == 0:
            report.add(CheckResult("Docker daemon", True, True, "running"))
        else:
            report.add(CheckResult("Docker daemon", False, True,
                                   f"not responding: {r.stderr.strip()[:120]}"))
    except FileNotFoundError:
        report.add(CheckResult("Docker daemon", False, True, "docker not found on PATH"))
    except subprocess.TimeoutExpired:
        report.add(CheckResult("Docker daemon", False, True, "docker info timed out"))



def check_nvidia_runtime(report: PreflightReport) -> None:
    try:
        r = _run(["docker", "run", "--rm", "--runtime=nvidia",
                  "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi"], timeout=60)
        if r.returncode == 0:
            report.add(CheckResult("NVIDIA runtime", True, True, "nvidia-smi OK in container"))
        else:
            report.add(CheckResult("NVIDIA runtime", False, True,
                                   f"failed: {r.stderr.strip()[:120]}"))
    except FileNotFoundError:
        report.add(CheckResult("NVIDIA runtime", False, True, "docker not found"))
    except subprocess.TimeoutExpired:
        report.add(CheckResult("NVIDIA runtime", False, True, "timed out (60s)"))


def check_gpus_visible(report: PreflightReport) -> None:
    try:
        r = _run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        if r.returncode == 0:
            gpu_ids = [int(x.strip()) for x in r.stdout.strip().split("\n") if x.strip()]
            count = len(gpu_ids)
            if count == 0:
                report.add(CheckResult("GPUs visible", False, True, "0 GPUs detected"))
            else:
                report.add(CheckResult("GPUs visible", True, True,
                                       f"{count} GPUs: {gpu_ids}"))
        else:
            report.add(CheckResult("GPUs visible", False, True,
                                   f"nvidia-smi failed: {r.stderr.strip()[:120]}"))
    except FileNotFoundError:
        report.add(CheckResult("GPUs visible", False, True, "nvidia-smi not found"))
    except subprocess.TimeoutExpired:
        report.add(CheckResult("GPUs visible", False, True, "nvidia-smi timed out"))


def check_base_image(report: PreflightReport) -> None:
    try:
        r = _run(["docker", "images", "-q", "autoworldbench-base:latest"])
        if r.returncode == 0 and r.stdout.strip():
            report.add(CheckResult("Base image", True, True,
                                   "autoworldbench-base:latest exists"))
        else:
            report.add(CheckResult("Base image", False, True,
                                   "autoworldbench-base:latest not found — "
                                   "run: docker build -t autoworldbench-base -f docker/Dockerfile.harbor ."))
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        report.add(CheckResult("Base image", False, True, str(e)))


def check_env_file(report: PreflightReport) -> dict[str, str]:
    env = _load_env()
    if not ENV_FILE.exists():
        report.add(CheckResult(".env file", False, True, f"{ENV_FILE} not found"))
        return env
    missing = []
    for key in ["HF_TOKEN"]:
        if not env.get(key):
            missing.append(key)
    if missing:
        report.add(CheckResult(".env file", False, True,
                               f"missing keys: {', '.join(missing)}"))
    else:
        report.add(CheckResult(".env file", True, True,
                               f"found with {len(env)} keys"))
    return env


def check_aws_sso(report: PreflightReport) -> None:
    try:
        r = _run(["aws", "sts", "get-caller-identity", "--profile", "claude"], timeout=15)
        if r.returncode == 0:
            identity = json.loads(r.stdout)
            report.add(CheckResult("AWS SSO session", True, True,
                                   f"account={identity.get('Account', '?')}"))
        else:
            report.add(CheckResult("AWS SSO session", False, True,
                                   f"expired or invalid — run: aws sso login --profile claude"))
    except FileNotFoundError:
        report.add(CheckResult("AWS SSO session", False, True, "aws CLI not found"))
    except subprocess.TimeoutExpired:
        report.add(CheckResult("AWS SSO session", False, True, "timed out (15s)"))


def check_claude_in_image(report: PreflightReport) -> None:
    try:
        r = _run(["docker", "run", "--rm", "autoworldbench-base:latest",
                  "claude", "--version"], timeout=30)
        if r.returncode == 0:
            version = r.stdout.strip().split("\n")[0]
            report.add(CheckResult("Claude CLI in image", True, True, version))
        else:
            report.add(CheckResult("Claude CLI in image", False, True,
                                   f"failed: {r.stderr.strip()[:120]}"))
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        report.add(CheckResult("Claude CLI in image", False, True, str(e)))


def check_codex_in_image(report: PreflightReport) -> None:
    try:
        r = _run(["docker", "run", "--rm", "autoworldbench-base:latest",
                  "codex", "--version"], timeout=30)
        if r.returncode == 0:
            version = r.stdout.strip().split("\n")[0]
            report.add(CheckResult("Codex CLI in image", True, True, version))
        else:
            report.add(CheckResult("Codex CLI in image", False, True,
                                   f"failed: {r.stderr.strip()[:120]}"))
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        report.add(CheckResult("Codex CLI in image", False, True, str(e)))


def check_bedrock_connectivity(report: PreflightReport, env: dict[str, str]) -> None:
    raw_model = env.get("ANTHROPIC_MODEL", "us.anthropic.claude-opus-4-6-v1")
    # Strip gateway prefix (e.g., "seed-research/bedrock/us.anthropic...") to get Bedrock model ID
    model_id = raw_model.split("/")[-1] if "/" in raw_model else raw_model
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    })
    try:
        r = _run(["aws", "bedrock-runtime", "invoke-model",
                  "--profile", "claude", "--region", "us-east-1",
                  "--model-id", model_id,
                  "--content-type", "application/json",
                  "--accept", "application/json",
                  "--body", body, "/dev/stdout"], timeout=30)
        if r.returncode == 0:
            report.add(CheckResult("Bedrock connectivity", True, True,
                                   f"invoke-model OK ({model_id})"))
        else:
            report.add(CheckResult("Bedrock connectivity", False, True,
                                   f"failed: {r.stderr.strip()[:120]}"))
    except FileNotFoundError:
        report.add(CheckResult("Bedrock connectivity", False, True, "aws CLI not found"))
    except subprocess.TimeoutExpired:
        report.add(CheckResult("Bedrock connectivity", False, True, "timed out (30s)"))


def check_openai_connectivity(report: PreflightReport, env: dict[str, str]) -> None:
    api_key = env.get("OPENAI_API_KEY", "") or env.get("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        report.add(CheckResult("OpenAI connectivity", False, False,
                               "no OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env (needed for codex agent)"))
        return
    # Azure keys use a different endpoint; skip connectivity check for Azure-only setups
    if not env.get("OPENAI_API_KEY") and env.get("AZURE_OPENAI_API_KEY"):
        report.add(CheckResult("OpenAI connectivity", True, False,
                               "AZURE_OPENAI_API_KEY set (skipping endpoint probe)"))
        return
    base_url = env.get("OPENAI_BASE_URL", "https://api.openai.com")
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        urllib.request.urlopen(req, timeout=10)
        report.add(CheckResult("OpenAI connectivity", True, False, "API reachable"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            report.add(CheckResult("OpenAI connectivity", False, False,
                                   f"auth error: HTTP {e.code}"))
        else:
            report.add(CheckResult("OpenAI connectivity", True, False,
                                   f"endpoint reachable (HTTP {e.code})"))
    except Exception as e:
        report.add(CheckResult("OpenAI connectivity", False, False, str(e)[:120]))


def check_harbor(report: PreflightReport) -> None:
    try:
        r = _run(["harbor", "--version"])
        if r.returncode == 0:
            report.add(CheckResult("Harbor CLI", True, True,
                                   r.stdout.strip().split("\n")[0]))
        else:
            report.add(CheckResult("Harbor CLI", False, True,
                                   "harbor --version failed"))
    except FileNotFoundError:
        report.add(CheckResult("Harbor CLI", False, True,
                               "harbor not found on PATH — activate .venv?"))
    except subprocess.TimeoutExpired:
        report.add(CheckResult("Harbor CLI", False, True, "timed out"))


def check_harbor_patches(report: PreflightReport, fix: bool) -> None:
    try:
        import harbor
        harbor_dir = Path(harbor.__file__).parent
    except ImportError:
        report.add(CheckResult("Harbor patches", False, False, "harbor not importable"))
        return

    issues = []
    compose_base = harbor_dir / "environments" / "docker" / "docker-compose-base.yaml"
    if compose_base.exists() and "deploy" in compose_base.read_text():
        issues.append("compose-base has deploy.resources (needs stripping)")

    docker_py = harbor_dir / "environments" / "docker" / "docker.py"
    if docker_py.exists():
        src = docker_py.read_text()
        if "chmod 777" in src and "user=0" not in src:
            issues.append("chmod not running as root")

    if issues:
        if fix:
            r = _run([sys.executable, str(REPO_ROOT / "scripts" / "setup_tasks.py")])
            if r.returncode == 0:
                report.add(CheckResult("Harbor patches", True, False,
                                       f"auto-fixed: {'; '.join(issues)}"))
            else:
                report.add(CheckResult("Harbor patches", False, False,
                                       f"setup_tasks.py failed: {r.stderr.strip()[:120]}"))
        else:
            report.add(CheckResult("Harbor patches", False, False,
                                   f"{'; '.join(issues)} — use --fix or run setup_tasks.py"))
    else:
        report.add(CheckResult("Harbor patches", True, False, "applied"))


def check_data_caches(report: PreflightReport, fix: bool) -> None:
    missing = []
    for game in ALL_GAMES:
        cache = DATA_DIR / game / "_cache" / "train_tensors.pt"
        if not cache.exists():
            missing.append(game)
    if missing:
        if fix:
            print(f"\n  Running prepare_data.py for {len(missing)} games...")
            r = _run([sys.executable, str(REPO_ROOT / "scripts" / "prepare_data.py")],
                     timeout=600)
            if r.returncode == 0:
                report.add(CheckResult("Data caches", True, False,
                                       f"built for {len(missing)} games"))
            else:
                report.add(CheckResult("Data caches", False, False,
                                       f"prepare_data.py failed: {r.stderr.strip()[:120]}"))
        else:
            report.add(CheckResult("Data caches", False, False,
                                   f"missing for {len(missing)} games: "
                                   f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''} "
                                   f"— use --fix or run prepare_data.py"))
    else:
        report.add(CheckResult("Data caches", True, False,
                               f"all {len(ALL_GAMES)} games cached"))


def check_disk_space(report: PreflightReport) -> None:
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 10:
        report.add(CheckResult("Disk space", False, True,
                               f"{free_gb:.1f} GB free (need >= 10 GB)"))
    elif free_gb < 50:
        report.add(CheckResult("Disk space", False, False,
                               f"{free_gb:.1f} GB free (recommend >= 50 GB)"))
    else:
        report.add(CheckResult("Disk space", True, False, f"{free_gb:.1f} GB free"))


def check_container_dns(report: PreflightReport) -> None:
    try:
        r = _run(["docker", "run", "--rm", "alpine",
                  "sh", "-c", "nslookup archive.ubuntu.com"], timeout=30)
        if r.returncode == 0:
            report.add(CheckResult("Container DNS", True, False, "resolves OK"))
        else:
            report.add(CheckResult("Container DNS", False, False,
                                   f"failed: {r.stderr.strip()[:120]}"))
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        report.add(CheckResult("Container DNS", False, False, str(e)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_preflight(fix: bool = False, skip_connectivity: bool = False) -> bool:
    print(f"\n{BOLD}AutoWorldBench Preflight Checks{RESET}\n")

    report = PreflightReport()

    check_docker_daemon(report)
    check_nvidia_runtime(report)
    check_gpus_visible(report)
    check_base_image(report)
    env = check_env_file(report)
    check_aws_sso(report)
    check_claude_in_image(report)
    check_codex_in_image(report)
    if not skip_connectivity:
        check_bedrock_connectivity(report, env)
        check_openai_connectivity(report, env)
    else:
        print(f"  {YELLOW}⊘{RESET} [SKIP] Bedrock connectivity (--skip-connectivity)")
        print(f"  {YELLOW}⊘{RESET} [SKIP] Azure OpenAI connectivity (--skip-connectivity)")
    check_harbor(report)
    check_harbor_patches(report, fix)
    check_data_caches(report, fix)
    check_disk_space(report)
    check_container_dns(report)

    report.summary()
    return not report.has_blockers


def main():
    parser = argparse.ArgumentParser(
        description="Preflight validation for GPU-parallel Harbor runs.",
    )
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix: build data caches, apply Harbor patches")
    parser.add_argument("--skip-connectivity", action="store_true",
                        help="Skip Bedrock/Azure API connectivity tests")
    args = parser.parse_args()

    ok = run_preflight(fix=args.fix, skip_connectivity=args.skip_connectivity)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
