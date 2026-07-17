# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""GPU-parallel Harbor task orchestrator.

Runs Harbor tasks with explicit GPU pinning (NVIDIA_VISIBLE_DEVICES).
Manages a pool of GPUs with greedy backfill: when a task finishes and
frees a GPU, the next queued task launches immediately.

Usage:
    python scripts/orchestrate.py --all --agent claude-code
    python scripts/orchestrate.py --all --agent claude-code --bedrock
    python scripts/orchestrate.py --model dreamer --agent codex --preflight
    python scripts/orchestrate.py --all --agent claude-code --gpus 0,1,2,3
    python scripts/orchestrate.py --all --agent claude-code --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"
DATA_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ENV_FILE = REPO_ROOT / ".env"

GAMES = [
    "asteroids", "breakout", "frogger", "kong", "platformer",
    "pong", "racer", "snake",
]
MODELS = ["dreamer", "ar_transformer", "d3pm", "maskgit"]

AGENT_CONFIG_FILES = {
    "claude-code": ("agents/claude-code.json", "/home/agent/.claude/settings.json"),
    "codex": ("agents/codex.toml", "/tmp/codex-config-staged.toml"),
}

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


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


def _detect_gpus() -> list[int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return [int(x.strip()) for x in r.stdout.strip().split("\n") if x.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


# ---------------------------------------------------------------------------
# Task list building
# ---------------------------------------------------------------------------

def build_task_list(
    task: str | None,
    game: list[str] | None,
    model: str | None,
    run_all: bool,
) -> list[str]:
    if task:
        task_dir = TASKS_DIR / task
        if not task_dir.is_dir():
            print(f"{RED}Error: task directory not found: {task_dir}{RESET}")
            sys.exit(1)
        return [task]

    if game and len(game) == 1 and model:
        name = f"{game[0]}_{model}"
        if not (TASKS_DIR / name).is_dir():
            print(f"{RED}Error: task directory not found: {TASKS_DIR / name}{RESET}")
            sys.exit(1)
        return [name]

    tasks = []
    filter_games = game if game else GAMES
    filter_models = [model] if model else MODELS

    if not run_all and not game and not model:
        print(f"{RED}Error: specify --task, --game, --model, or --all{RESET}")
        sys.exit(1)

    for g in filter_games:
        if not (DATA_DIR / g / "_cache" / "train_tensors.pt").exists():
            continue
        for m in filter_models:
            name = f"{g}_{m}"
            if (TASKS_DIR / name).is_dir():
                tasks.append(name)
    return tasks


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(tasks: list[str]) -> None:
    resolved_games = set()
    for t in tasks:
        config_path = TASKS_DIR / t / "config.json"
        if not config_path.exists():
            print(f"{RED}Error: config.json not found for task {t}{RESET}")
            sys.exit(1)
        cfg = json.loads(config_path.read_text())
        resolved_games.add(cfg["game_id"])

    for game in sorted(resolved_games):
        cache = DATA_DIR / game / "_cache" / "train_tensors.pt"
        if cache.exists():
            print(f"  Cache exists for {game}, skipping.")
            continue
        print(f"{RED}  No cached data for {game} — run scripts/prepare_data.py manually first.{RESET}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# GPU Pool + Scheduler
# ---------------------------------------------------------------------------

@dataclass
class TaskRun:
    task_name: str
    gpu_id: int
    process: subprocess.Popen
    log_file: Path
    log_fh: object
    start_time: float


@dataclass
class TaskResult:
    task_name: str
    gpu_id: int
    returncode: int
    start_time: float
    end_time: float
    log_file: Path

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time

    @property
    def status(self) -> str:
        return "ok" if self.returncode == 0 else "fail"


class Orchestrator:
    def __init__(
        self,
        tasks: list[str],
        agent: str,
        gpu_ids: list[int],
        env_vars: dict[str, str],
        log_dir: Path,
        artifacts_dir: Path,
        train_seconds: int | None = None,
        agent_timeout: int | None = None,
        force_build: bool = False,
        use_bedrock: bool = False,
    ):
        self.queue = list(tasks)
        self.agent = agent
        self.free_gpus: set[int] = set(gpu_ids)
        self.busy: dict[int, TaskRun] = {}
        self.results: list[TaskResult] = []
        self.env_vars = env_vars
        self.log_dir = log_dir
        self.artifacts_dir = artifacts_dir
        self.train_seconds = train_seconds
        self.agent_timeout = agent_timeout
        self.force_build = force_build
        self.use_bedrock = use_bedrock
        self.total = len(tasks)
        self._shutdown = False

    def _read_codex_config(self) -> dict:
        import tomllib
        config_path = REPO_ROOT / "agents" / "codex.toml"
        if config_path.exists():
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        return {}

    def _inject_gpu_env(self, task_name: str, gpu_id: int) -> None:
        compose_path = TASKS_DIR / task_name / "environment" / "docker-compose.yaml"
        if not compose_path.exists():
            return
        text = compose_path.read_text()
        lines = text.splitlines()
        new_lines = [l for l in lines if "CUDA_VISIBLE_DEVICES=" not in l]
        for i, l in enumerate(new_lines):
            if "MAX_TRAIN_SECONDS=" in l:
                indent = l[:len(l) - len(l.lstrip())]
                new_lines.insert(i + 1, f"{indent}- CUDA_VISIBLE_DEVICES={gpu_id}")
                break
        compose_path.write_text("\n".join(new_lines) + "\n")

    def _build_harbor_cmd(self, task_name: str, gpu_id: int) -> list[str]:
        self._inject_gpu_env(task_name, gpu_id)

        config_path = TASKS_DIR / task_name / "config.json"
        game_id = json.loads(config_path.read_text())["game_id"]
        cache_host = DATA_DIR / game_id / "_cache"

        mount_list = [
            {
                "type": "bind",
                "source": str(cache_host),
                "target": f"/data/{game_id}/_cache",
                "read_only": True,
            },
        ]

        agent_cfg = AGENT_CONFIG_FILES.get(self.agent)
        if agent_cfg:
            source, target = agent_cfg
            mount_list.append({
                "type": "bind",
                "source": str(REPO_ROOT / source),
                "target": target,
                "read_only": True,
            })

        if self.agent == "codex":
            loop_script = REPO_ROOT / "agents" / "codex_loop.py"
            if loop_script.exists():
                mount_list.append({
                    "type": "bind",
                    "source": str(loop_script),
                    "target": "/tmp/codex-loop.py",
                    "read_only": True,
                })


        mounts = json.dumps(mount_list)

        cmd = [
            "harbor", "run",
            "--path", str(TASKS_DIR / task_name),
            "--agent", self.agent,
            "--ae", f"NVIDIA_VISIBLE_DEVICES={gpu_id}",
            "--ae", f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "--mounts-json", mounts,
            "--artifact", "/app",
            "-n", "1",
        ]

        if self.agent == "claude-code":
            if self.use_bedrock:
                # AWS Bedrock mode
                cmd += ["--ae", "CLAUDE_CODE_USE_BEDROCK=1",
                         "--ae", f"AWS_REGION={os.environ.get('AWS_REGION', 'us-east-1')}"]
                if os.environ.get("AWS_SESSION_TOKEN"):
                    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
                        cmd += ["--ae", f"{var}={os.environ[var]}"]
                else:
                    cmd += ["--ae", f"AWS_PROFILE={os.environ.get('AWS_PROFILE', 'default')}"]
            else:
                # Standard Anthropic API mode (default)
                if self.env_vars.get("ANTHROPIC_BASE_URL"):
                    cmd += ["--ae", f"ANTHROPIC_BASE_URL={self.env_vars['ANTHROPIC_BASE_URL']}"]
                if self.env_vars.get("ANTHROPIC_API_KEY"):
                    cmd += ["--ae", f"ANTHROPIC_API_KEY={self.env_vars['ANTHROPIC_API_KEY']}"]
        elif self.agent == "codex":
            codex_cfg = self._read_codex_config()
            if codex_cfg.get("model"):
                cmd += ["-m", codex_cfg["model"]]
            api_key = self.env_vars.get("OPENAI_API_KEY") or self.env_vars.get("AZURE_OPENAI_API_KEY")
            if api_key:
                cmd += ["--ae", f"OPENAI_API_KEY={api_key}"]
            if self.env_vars.get("OPENAI_BASE_URL"):
                cmd += ["--ae", f"OPENAI_BASE_URL={self.env_vars['OPENAI_BASE_URL']}"]

        if self.env_vars.get("HF_TOKEN"):
            cmd += ["--ae", f"HF_TOKEN={self.env_vars['HF_TOKEN']}"]

        if self.train_seconds is not None:
            cmd += ["--ae", f"MAX_TRAIN_SECONDS={self.train_seconds}"]
        if self.agent_timeout is not None:
            # task.toml has timeout_sec=28800; compute multiplier
            base_timeout = 28800
            multiplier = self.agent_timeout / base_timeout
            cmd += ["--agent-timeout-multiplier", str(multiplier)]

        if self.force_build:
            cmd += ["--force-build"]

        return cmd

    def _launch(self, task_name: str, gpu_id: int) -> None:
        cmd = self._build_harbor_cmd(task_name, gpu_id)
        log_path = self.log_dir / f"{task_name}.log"
        log_fh = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env={**os.environ, **self.env_vars},
            )
        except Exception:
            log_fh.close()
            raise
        self.free_gpus.discard(gpu_id)
        self.busy[gpu_id] = TaskRun(
            task_name=task_name,
            gpu_id=gpu_id,
            process=proc,
            log_file=log_path,
            log_fh=log_fh,
            start_time=time.time(),
        )

    def _poll(self) -> None:
        finished = []
        for gpu_id, run in self.busy.items():
            rc = run.process.poll()
            if rc is not None:
                finished.append((gpu_id, rc))

        for gpu_id, rc in finished:
            run = self.busy.pop(gpu_id)
            run.log_fh.close()
            self.free_gpus.add(gpu_id)
            self.results.append(TaskResult(
                task_name=run.task_name,
                gpu_id=gpu_id,
                returncode=rc,
                start_time=run.start_time,
                end_time=time.time(),
                log_file=run.log_file,
            ))
            status = f"{GREEN}ok{RESET}" if rc == 0 else f"{RED}FAIL(rc={rc}){RESET}"
            elapsed = time.time() - run.start_time
            print(f"  Finished {run.task_name} on GPU {gpu_id}: "
                  f"{status} ({elapsed / 60:.1f}m)")

    def _status_line(self) -> str:
        done = len(self.results)
        failed = sum(1 for r in self.results if r.returncode != 0)
        running_parts = [f"{r.task_name}(GPU{r.gpu_id})" for r in self.busy.values()]
        running_str = " ".join(running_parts) if running_parts else "none"
        return (f"\r  [{done + len(self.busy)}/{self.total}] "
                f"Running: {running_str} | "
                f"Done: {done} | Failed: {failed}   ")

    def _shutdown_all(self) -> None:
        if not self.busy:
            return
        print(f"\n{YELLOW}Shutting down {len(self.busy)} running tasks...{RESET}")
        for gpu_id, run in self.busy.items():
            try:
                run.process.terminate()
            except OSError:
                pass

        deadline = time.time() + 30
        while self.busy and time.time() < deadline:
            self._poll()
            if self.busy:
                time.sleep(1)

        for gpu_id, run in list(self.busy.items()):
            try:
                run.process.kill()
            except OSError:
                pass
            run.log_fh.close()
            self.results.append(TaskResult(
                task_name=run.task_name,
                gpu_id=gpu_id,
                returncode=-9,
                start_time=run.start_time,
                end_time=time.time(),
                log_file=run.log_file,
            ))
        self.busy.clear()

    def run(self) -> list[TaskResult]:
        original_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigint(sig, frame):
            self._shutdown = True
            print(f"\n{YELLOW}Ctrl+C received — stopping after current tasks...{RESET}")

        signal.signal(signal.SIGINT, _handle_sigint)

        try:
            while self.queue or self.busy:
                if self._shutdown:
                    self._shutdown_all()
                    break

                while self.queue and self.free_gpus:
                    gpu_id = min(self.free_gpus)
                    task_name = self.queue.pop(0)
                    print(f"  Launching {task_name} on GPU {gpu_id}")
                    self._launch(task_name, gpu_id)
                    time.sleep(2)

                self._poll()
                sys.stdout.write(self._status_line())
                sys.stdout.flush()

                if self.busy:
                    time.sleep(2)
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        print()
        return self.results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(results: list[TaskResult], log_dir: Path) -> None:
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(results),
        "passed": sum(1 for r in results if r.returncode == 0),
        "failed": sum(1 for r in results if r.returncode != 0),
        "tasks": [
            {
                "task": r.task_name,
                "gpu_id": r.gpu_id,
                "status": r.status,
                "returncode": r.returncode,
                "elapsed_minutes": round(r.elapsed / 60, 1),
                "log": str(r.log_file),
            }
            for r in results
        ],
    }
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")


def print_results_table(results: list[TaskResult]) -> None:
    print(f"\n{BOLD}{'Task':<35} {'GPU':>4} {'Status':>8} {'Time':>8}{RESET}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.task_name):
        status = f"{GREEN}ok{RESET}" if r.returncode == 0 else f"{RED}FAIL{RESET}"
        elapsed = f"{r.elapsed / 60:.1f}m"
        print(f"  {r.task_name:<33} {r.gpu_id:>4} {status:>17} {elapsed:>8}")

    passed = sum(1 for r in results if r.returncode == 0)
    failed = sum(1 for r in results if r.returncode != 0)
    total_time = sum(r.elapsed for r in results) / 60
    print("-" * 60)
    print(f"  {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}, "
          f"{total_time:.1f}m total task-time")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU-parallel Harbor task orchestrator.",
    )
    parser.add_argument("--task", "-t", help="Run a single task")
    parser.add_argument("--game", "-g", nargs="+", help="Run all models for one or more games")
    parser.add_argument("--model", "-m", help="Run all games for a model")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="Run all 32 tasks")
    parser.add_argument("--agent", "-a", required=True,
                        help="Agent: claude-code, codex")
    parser.add_argument("--gpus", help="Comma-separated GPU IDs (default: auto-detect)")
parser.add_argument("--train-seconds", type=int, default=None,
                        help="Override MAX_TRAIN_SECONDS per experiment (default: 600)")
    parser.add_argument("--agent-timeout", type=int, default=None,
                        help="Agent timeout in seconds (default: 28800 from task.toml)")
    parser.add_argument("--preflight", action="store_true",
                        help="Run preflight checks before starting")
    parser.add_argument("--force-build", action="store_true",
                        help="Force rebuild per-task Docker images")
    parser.add_argument("--bedrock", action="store_true", dest="use_bedrock",
                        help="Use AWS Bedrock instead of Anthropic API (claude-code only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    # Load environment
    env_vars = _load_env()
    for k, v in env_vars.items():
        os.environ.setdefault(k, v)

    # Agent-specific early validation
    if args.agent == "codex" and not (env_vars.get("OPENAI_API_KEY") or env_vars.get("AZURE_OPENAI_API_KEY")):
        print(f"{RED}Error: OPENAI_API_KEY required in .env for codex agent{RESET}")
        sys.exit(1)
    if args.agent == "claude-code":
        if args.use_bedrock:
            # Bedrock mode: validate AWS credentials
            try:
                r = subprocess.run(
                    ["aws", "sts", "get-caller-identity",
                     "--profile", os.environ.get("AWS_PROFILE", "default")],
                    capture_output=True, text=True, timeout=15,
                )
                if r.returncode != 0:
                    print(f"{RED}Error: AWS SSO session expired — "
                          f"run: aws sso login{RESET}")
                    sys.exit(1)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(f"{RED}Error: aws CLI not available or timed out{RESET}")
                sys.exit(1)
        else:
            # Standard API mode: validate API key
            if not env_vars.get("ANTHROPIC_API_KEY"):
                print(f"{YELLOW}Warning: ANTHROPIC_API_KEY not set in .env{RESET}")

    # Preflight
    if args.preflight:
        print(f"{BOLD}Running preflight checks...{RESET}\n")
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "preflight.py")],
        )
        if r.returncode != 0:
            print(f"\n{RED}Preflight failed — fix issues above before running.{RESET}")
            sys.exit(1)
        print()

    # Detect GPUs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = _detect_gpus()
    if not gpu_ids:
        print(f"{RED}Error: no GPUs detected. Use --gpus to specify manually.{RESET}")
        sys.exit(1)

    # Build task list
    tasks = build_task_list(args.task, args.game, args.model, args.run_all)
    if not tasks:
        print(f"{RED}Error: no matching tasks found.{RESET}")
        sys.exit(1)

    print(f"{BOLD}AutoWorldBench GPU-Parallel Orchestrator{RESET}")
    print(f"  Agent:  {args.agent}")
    if args.agent == "claude-code":
        print(f"  Auth:   {'AWS Bedrock' if args.use_bedrock else 'Anthropic API'}")
    print(f"  GPUs:   {gpu_ids} ({len(gpu_ids)} available)")
    print(f"  Tasks:  {len(tasks)}")
    if args.train_seconds is not None:
        print(f"  Train:  {args.train_seconds}s per experiment")
    if args.agent_timeout is not None:
        print(f"  Timeout: {args.agent_timeout}s agent timeout")
    print()

    if args.dry_run:
        print(f"{BOLD}Dry run — tasks that would execute:{RESET}")
        for i, t in enumerate(tasks):
            gpu = gpu_ids[i % len(gpu_ids)]
            print(f"  {i + 1:3}. {t} → GPU {gpu}")
        print(f"\n{len(tasks)} tasks across {len(gpu_ids)} GPUs "
              f"(~{len(tasks) // len(gpu_ids)} batches)")
        return

    # Prepare data
    print("Preparing data caches...")
    prepare_data(tasks)

    # Create log and artifacts directories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = ARTIFACTS_DIR / timestamp
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs:      {log_dir}")
    print(f"Artifacts: {artifacts_dir}\n")

    # Run
    orch = Orchestrator(
        tasks=tasks,
        agent=args.agent,
        gpu_ids=gpu_ids,
        env_vars=env_vars,
        log_dir=log_dir,
        artifacts_dir=artifacts_dir,
        train_seconds=args.train_seconds,
        agent_timeout=args.agent_timeout,
        force_build=args.force_build,
        use_bedrock=args.use_bedrock,
    )
    results = orch.run()

    # Summary
    write_summary(results, log_dir)
    print_results_table(results)


if __name__ == "__main__":
    main()
