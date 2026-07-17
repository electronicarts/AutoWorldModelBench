# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Autoresearch Agent -- LLM-driven hyperparameter optimization for world models.

Discovers task scaffolds under tasks/, uses an LLM to plan experiments,
executes training via each task's run.py, analyzes results, and iterates.

Usage:
    python agent.py --task pong_dreamer              # optimize one task
    python agent.py --task pong_dreamer --max-iterations 10
    python agent.py --task pong_dreamer --preflight-only
    python agent.py --task pong_dreamer --plan-only
    python agent.py --list-tasks                     # show all tasks
    python agent.py --filter-game pong               # all pong tasks
    python agent.py --filter-model dreamer           # all dreamer tasks
"""

import csv
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
from anthropic import AnthropicBedrock


REPO_ROOT = Path(__file__).resolve().parent


# =============================================================================
# TASK DISCOVERY
# =============================================================================

def discover_tasks(tasks_dir: str = "tasks") -> List[str]:
    """Scan tasks/ for valid task directories (must have run.py and agent.md)."""
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        return []
    return sorted([
        d.name for d in tasks_path.iterdir()
        if d.is_dir() and (d / "run.py").exists()
        and (d / "agent.md").exists()
    ])


def load_task_plan(task_name: str) -> str:
    """Read agent.md for the given task."""
    agents_path = Path("tasks") / task_name / "agent.md"
    if agents_path.exists():
        return agents_path.read_text()
    raise FileNotFoundError(f"agent.md not found for task '{task_name}'")


def load_task_metadata(task_name: str) -> Dict:
    """Load task.json metadata."""
    path = Path("tasks") / task_name / "task.json"
    if not path.exists():
        return {"task_id": task_name}
    return json.loads(path.read_text())


def load_task_config_template(task_name: str) -> Dict:
    """Load task's config template."""
    path = Path("tasks") / task_name / "config_template.json"
    if not path.exists():
        # Fall back to config.json
        path = Path("tasks") / task_name / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"config_template.json not found for task '{task_name}'")
    return json.loads(path.read_text())


def scan_task_completed_runs(task_name: str) -> Dict[str, Dict]:
    """Scan for completed runs from summary.tsv and experiment directories."""
    completed = {}
    task_dir = Path("tasks") / task_name

    # Try summary.tsv first (fast path)
    summary_path = task_dir / "summary.tsv"
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rid = row.get("run_id", "")
                if rid:
                    parsed = {}
                    for k, v in row.items():
                        if v == "" or v is None:
                            parsed[k] = None
                        else:
                            try:
                                parsed[k] = float(v) if "." in v else int(v)
                            except (ValueError, TypeError):
                                parsed[k] = v
                    completed[rid] = parsed

    # Also scan experiments/ for runs not in summary.tsv
    exp_root = REPO_ROOT / "experiments" / task_name
    if exp_root.exists():
        for run_dir in exp_root.iterdir():
            if not run_dir.is_dir():
                continue
            rid = run_dir.name
            if rid in completed:
                continue
            results_path = run_dir / "results.json"
            if results_path.exists():
                try:
                    results = json.loads(results_path.read_text())
                    completed[results.get("run_id", rid)] = results
                except (json.JSONDecodeError, KeyError):
                    pass

    return completed


# =============================================================================
# PREFLIGHT -- system checks before any experiment runs
# =============================================================================

class SystemInfo:
    """Hardware and environment snapshot."""

    def __init__(self):
        self.device: str = "cpu"
        self.gpu_name: str = ""
        self.gpu_memory_mb: int = 0
        self.cpu_count: int = os.cpu_count() or 1
        self.ram_mb: int = 0
        self.disk_free_mb: int = 0
        self.python_version: str = platform.python_version()
        self.packages: Dict[str, str] = {}
        self.missing_packages: List[str] = []
        self.data_status: Dict[str, bool] = {}
        self.issues: List[str] = []
        self.warnings: List[str] = []

    @property
    def ready(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "SYSTEM PREFLIGHT",
            "=" * 60,
            f"Python:       {self.python_version}",
            f"Device:       {self.device}",
        ]
        if self.gpu_name:
            lines.append(f"GPU:          {self.gpu_name} ({self.gpu_memory_mb} MB)")
        lines += [
            f"CPU cores:    {self.cpu_count}",
            f"RAM:          {self.ram_mb} MB",
            f"Disk free:    {self.disk_free_mb} MB",
        ]

        if self.missing_packages:
            lines.append(f"\nMissing packages: {', '.join(self.missing_packages)}")

        data_lines = []
        for name, ok in self.data_status.items():
            status = "OK" if ok else "MISSING"
            data_lines.append(f"  {name}: {status}")
        if data_lines:
            lines.append("\nData:")
            lines.extend(data_lines)

        if self.issues:
            lines.append(f"\nBLOCKING ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")

        status = "READY" if self.ready else "NOT READY"
        lines.append(f"\nStatus: {status}")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_preflight(task_name: Optional[str] = None) -> SystemInfo:
    """Run all preflight checks and return a SystemInfo snapshot."""
    info = SystemInfo()

    # 1. Packages
    required_packages = ["torch", "numpy", "pyarrow"]
    for pkg_name in required_packages:
        try:
            mod = importlib.import_module(pkg_name)
            version = getattr(mod, "__version__", "unknown")
            info.packages[pkg_name] = version
        except ImportError:
            info.missing_packages.append(pkg_name)

    if info.missing_packages:
        info.issues.append(
            f"Missing packages: {', '.join(info.missing_packages)}. "
            f"Run: pip install -e ."
        )

    # 2. GPU / Device
    try:
        import torch
        if torch.cuda.is_available():
            info.device = "cuda"
            info.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.gpu_memory_mb = props.total_memory // (1024 * 1024)
        else:
            info.device = "cpu"
            info.warnings.append("No CUDA GPU detected -- training will be slow")
    except ImportError:
        info.issues.append("PyTorch not installed")

    # 3. RAM
    try:
        import psutil
        info.ram_mb = psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info.ram_mb = int(line.split()[1]) // 1024
                        break
        except FileNotFoundError:
            info.warnings.append("Cannot detect RAM")

    # 4. Disk space
    try:
        usage = shutil.disk_usage(".")
        info.disk_free_mb = usage.free // (1024 * 1024)
        if info.disk_free_mb < 500:
            info.issues.append(f"Low disk space: {info.disk_free_mb} MB free")
        elif info.disk_free_mb < 2000:
            info.warnings.append(f"Disk space is low: {info.disk_free_mb} MB free")
    except OSError:
        info.warnings.append("Cannot check disk space")

    # 5. Data availability (check cache directory from task config)
    if task_name:
        try:
            config = load_task_config_template(task_name)
            cache_dir = config.get("cache_dir", "")
            game_id = config.get("game_id", "")
            if cache_dir:
                cache_path = Path(cache_dir)
                cache_ok = cache_path.exists() and (cache_path / "train_tensors.pt").exists()
                info.data_status[f"cache ({game_id})"] = cache_ok
                if not cache_ok:
                    # Check if raw data exists for preparation
                    data_dir = REPO_ROOT / "data" / game_id
                    raw_ok = data_dir.exists() and any(data_dir.glob("*.parquet"))
                    info.data_status[f"raw data ({game_id})"] = raw_ok
                    if not raw_ok:
                        info.issues.append(
                            f"No data for {game_id}. Need data/{game_id}/*.parquet "
                            f"or prepared cache at {cache_dir}"
                        )
                    else:
                        info.warnings.append(
                            f"Cache not built for {game_id}. "
                            f"Run: python run_all.py --game {game_id} --prepare-only"
                        )
        except FileNotFoundError:
            info.warnings.append(f"No config found for task '{task_name}'")

    return info


# =============================================================================
# GIT HELPERS
# =============================================================================

SWEEP_BRANCH_PREFIX = "sweep/"


def _git(args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args, capture_output=True, text=True, timeout=timeout,
    )


def git_available() -> bool:
    try:
        r = _git(["rev-parse", "--is-inside-work-tree"])
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_git_hash() -> Optional[str]:
    try:
        r = _git(["rev-parse", "HEAD"])
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_current_branch() -> Optional[str]:
    try:
        r = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def git_commit_run(task_name: str, run_id: str) -> Optional[str]:
    """Commit a run's config and results."""
    if not git_available():
        return None
    try:
        task_dir = Path("tasks") / task_name
        files = []
        summary_tsv = task_dir / "summary.tsv"
        if summary_tsv.exists():
            files.append(str(summary_tsv))
        exp_dir = REPO_ROOT / "experiments" / task_name / run_id
        for name in ["results.json", "config.json", "EXPERIMENT.md"]:
            f = exp_dir / name
            if f.exists():
                files.append(str(f))
        if files:
            _git(["add"] + files)
        r = _git(["diff", "--cached", "--quiet"])
        if r.returncode == 0:
            return get_git_hash()
        _git(["commit", "-m", f"[agent] {task_name}: run {run_id}"])
        return get_git_hash()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def git_create_sweep_branch(sweep_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not git_available():
        return None, None
    branch_name = f"{SWEEP_BRANCH_PREFIX}{sweep_name}"
    try:
        _git(["checkout", "-b", branch_name])
        return branch_name, get_git_hash()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None


def git_push_sweep_branch() -> bool:
    if not git_available():
        return False
    try:
        branch = get_current_branch()
        if not branch or not branch.startswith(SWEEP_BRANCH_PREFIX):
            return False
        r = _git(["push", "-u", "origin", branch], timeout=60)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# =============================================================================
# AGENT -- LLM-driven optimization
# =============================================================================

class AutoresearchAgent:
    """Task-based hyperparameter optimization agent.

    Reads agent.md to understand the task, uses an LLM to plan experiments,
    executes training via the task's run.py, analyzes results, and iterates.
    """

    def __init__(
        self,
        task: str,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.task = task
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = AnthropicBedrock()

        self.model = model
        self.system_info: Optional[SystemInfo] = None
        self.task_plan: str = ""
        self.task_metadata: Dict = {}
        self.config_template: Dict = {}
        self.completed: Dict[str, Dict] = {}

    # -- LLM helpers --

    def _ask_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _ask_llm_json(self, prompt: str, max_tokens: int = 4000) -> Dict:
        raw = self._ask_llm(prompt, max_tokens)
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        return json.loads(raw.strip())

    # -- Preflight --

    def preflight(self) -> SystemInfo:
        self.system_info = run_preflight(self.task)
        return self.system_info

    # -- Planning --

    def plan_experiments(self) -> List[Dict]:
        """Use the LLM to read agent.md and generate experiment configs."""
        self.task_plan = load_task_plan(self.task)
        self.task_metadata = load_task_metadata(self.task)
        self.config_template = load_task_config_template(self.task)
        self.completed = scan_task_completed_runs(self.task)

        completed_summary = ""
        if self.completed:
            lines = []
            for rid, res in self.completed.items():
                composite = res.get("composite", "N/A")
                val_loss = res.get("val_loss", "N/A")
                pos_l1 = res.get("position_l1", "N/A")
                alive_f1_val = res.get("alive_f1", "N/A")
                status = res.get("status", "?")
                train_time = res.get("train_time_sec", "?")
                n_params = res.get("num_params", "?")
                lines.append(
                    f"  {rid}: composite={composite}, pos_l1={pos_l1}, "
                    f"alive_f1={alive_f1_val}, val_loss={val_loss}, "
                    f"status={status}, time={train_time}s, params={n_params}"
                )
            completed_summary = "\n".join(lines)
        else:
            completed_summary = "  (none -- first run)"

        prompt = f"""You are a world model researcher. Your goal is to improve the model's
composite score through iterative experimentation.

# Task description (agent.md)
{self.task_plan}

# Baseline config template
```json
{json.dumps(self.config_template, indent=2)}
```

# Prior experiment results
{completed_summary}

# System info
- Device: {self.system_info.device if self.system_info else 'unknown'}
- GPU: {self.system_info.gpu_name if self.system_info else 'unknown'}
- GPU memory: {self.system_info.gpu_memory_mb if self.system_info else 0} MB

# Instructions

Study the task description and prior results above. Decide what to try next --
you may adjust hyperparameters, propose architectural changes, modify the loss
function, or try anything else you think will improve the composite score.

For each experiment:
1. Generate a FULL config JSON based on the template (not a diff)
2. Each config MUST have a unique "run_id" (descriptive, e.g., "exp002_wider_model")
3. Briefly explain your reasoning in a "notes" field

If this is the first run, start with the baseline config to establish a reference score.
Propose 1-3 experiments per batch. Quality over quantity -- each should test a clear hypothesis.
If you believe no further improvement is likely, return [].

Return ONLY a JSON array of config objects, no other text.
"""
        try:
            configs = self._ask_llm_json(prompt)
            if not isinstance(configs, list):
                configs = [configs]
            return configs
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: LLM planning failed ({e}), falling back to baseline")
            if "baseline" not in self.completed:
                baseline = dict(self.config_template)
                baseline["run_id"] = "baseline"
                return [baseline]
            return []

    # -- Execution --

    def run_experiment(self, config: Dict) -> Dict:
        """Write config and execute the task's run.py."""
        run_id = config.get("run_id", "unnamed")
        task_dir = Path("tasks") / self.task

        # Write config to task's configs/ directory
        config_path = task_dir / "configs" / f"{run_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))

        print(f"\n{'=' * 60}")
        print(f"TASK: {self.task} | RUN: {run_id}")
        print(f"Config: {config_path}")
        print(f"{'=' * 60}")

        timeout = self.task_metadata.get("timeout_seconds", 600)

        try:
            result = subprocess.run(
                [sys.executable, str(task_dir / "run.py"), "--config", str(config_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout + result.stderr
            output_lines = output.strip().split("\n")
            for line in output_lines[-20:]:
                print(f"  {line}")

            # Load results from experiment dir
            results_path = REPO_ROOT / "experiments" / self.task / run_id / "results.json"
            if results_path.exists():
                return json.loads(results_path.read_text())

            if result.returncode != 0:
                print(f"Run failed (exit code {result.returncode})")
                return {"run_id": run_id, "status": "failed",
                        "error": output[-2000:]}

            return {"run_id": run_id, "status": "failed",
                    "error": "No results.json produced"}

        except subprocess.TimeoutExpired:
            print(f"Run timed out after {timeout}s")
            results_path = REPO_ROOT / "experiments" / self.task / run_id / "results.json"
            if results_path.exists():
                try:
                    results = json.loads(results_path.read_text())
                    results["status"] = "timeout_partial"
                    return results
                except (json.JSONDecodeError, OSError):
                    pass
            return {"run_id": run_id, "status": "timeout"}

        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"run_id": run_id, "status": "error", "error": str(e)}

    # -- Analysis --

    def analyze_results(self) -> Dict:
        """Use the LLM to analyze all completed runs and decide next steps."""
        self.completed = scan_task_completed_runs(self.task)

        results_summary = json.dumps(
            {rid: {k: v for k, v in res.items() if k not in ("error",)}
             for rid, res in self.completed.items()},
            indent=2,
        )

        prompt = f"""You are a world model researcher analyzing experiment results.

# Task description
{self.task_plan}

# All completed runs
{results_summary}

# Instructions

Analyze the results so far:

1. Which run has the best composite score (higher is better)?
2. What patterns do you see? Which changes helped, which didn't?
3. Is there still room for improvement, or are we hitting diminishing returns?
4. Should we continue experimenting or stop?

Return JSON:
{{
  "decision": "continue" or "stop",
  "best_run_id": "...",
  "best_composite": ...,
  "summary": "2-3 sentence analysis"
}}
"""
        try:
            return self._ask_llm_json(prompt)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: LLM analysis failed ({e})")
            return {"decision": "stop", "summary": "Analysis failed, stopping."}

    # -- Reporting --

    def generate_report(self):
        """Generate a summary report for the task's optimization runs."""
        self.completed = scan_task_completed_runs(self.task)
        if not self.completed:
            print("No completed runs to report on.")
            return

        report_dir = REPO_ROOT / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        sorted_runs = sorted(
            self.completed.items(),
            key=lambda x: -(x[1].get("composite", 0) or 0),
        )

        lines = [
            f"# {self.task} -- Optimization Report",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total runs: {len(sorted_runs)}",
            "",
            "## Ranked Results (by composite score)",
            "",
            "| Rank | Run ID | Composite | Pos L1 | Alive F1 | Val Loss | Params | Time (s) | Status |",
            "|------|--------|-----------|--------|----------|----------|--------|----------|--------|",
        ]

        for i, (rid, res) in enumerate(sorted_runs, 1):
            composite = res.get("composite", "N/A")
            if isinstance(composite, (int, float)):
                composite = f"{composite:.4f}"
            pos_l1 = res.get("position_l1", "N/A")
            if isinstance(pos_l1, (int, float)):
                pos_l1 = f"{pos_l1:.4f}"
            alive_f1_val = res.get("alive_f1", "N/A")
            if isinstance(alive_f1_val, (int, float)):
                alive_f1_val = f"{alive_f1_val:.4f}"
            val_loss = res.get("val_loss", "N/A")
            if isinstance(val_loss, (int, float)):
                val_loss = f"{val_loss:.4f}"
            n_params = res.get("num_params", "N/A")
            if isinstance(n_params, int):
                n_params = f"{n_params:,}"
            train_time = res.get("train_time_sec", "N/A")
            status = res.get("status", "?")
            lines.append(
                f"| {i} | {rid} | {composite} | {pos_l1} | "
                f"{alive_f1_val} | {val_loss} | {n_params} | {train_time} | {status} |"
            )

        if sorted_runs:
            best_rid, best_res = sorted_runs[0]
            lines += ["", f"## Best Run: {best_rid}", "", "```json"]
            config_path = Path("tasks") / self.task / "configs" / f"{best_rid}.json"
            if config_path.exists():
                lines.append(config_path.read_text())
            else:
                lines.append(json.dumps(best_res, indent=2))
            lines.append("```")

        report_path = report_dir / f"{self.task}_report.md"
        report_path.write_text("\n".join(lines))
        print(f"Report saved to {report_path}")

        if sorted_runs:
            best_rid = sorted_runs[0][0]
            best_config_path = report_dir / f"{self.task}_best_config.json"
            config_path = Path("tasks") / self.task / "configs" / f"{best_rid}.json"
            if config_path.exists():
                shutil.copy2(config_path, best_config_path)
                print(f"Best config saved to {best_config_path}")

    # -- Main loop --

    def run(self, max_iterations: int = 20):
        """Main optimization loop."""
        start_time = time.time()

        # Phase 1: Preflight
        print(f"\nPhase 1: Preflight checks for task '{self.task}'")
        info = self.preflight()
        print(info.summary())

        if not info.ready:
            print("\nAborting: preflight checks failed.")
            return

        # Phase 2: Plan
        print("\nPhase 2: Planning experiments")
        experiments = self.plan_experiments()

        if not experiments:
            print("No new experiments to run.")
            self.generate_report()
            return

        print(f"\nPlanned {len(experiments)} experiment(s):")
        for i, cfg in enumerate(experiments, 1):
            rid = cfg.get("run_id", "?")
            notes = cfg.get("notes", "")
            print(f"  {i}. {rid}: {notes[:80]}")

        # Phase 3: Execute & Iterate
        print(f"\nPhase 3: Running experiments (max {max_iterations} iterations)")

        sweep_name = f"{self.task}_{datetime.now().strftime('%Y-%m-%d')}"
        branch_name, _ = git_create_sweep_branch(sweep_name)
        if branch_name:
            print(f"  Created sweep branch: {branch_name}")

        iteration = 0
        all_results = []

        while iteration < max_iterations and experiments:
            for config in experiments:
                iteration += 1
                if iteration > max_iterations:
                    print(f"\nReached max iterations ({max_iterations}), stopping.")
                    break

                result = self.run_experiment(config)
                all_results.append(result)

                rid = result.get("run_id", "?")
                status = result.get("status", "?")
                composite = result.get("composite", "N/A")
                if isinstance(composite, float):
                    composite = f"{composite:.4f}"
                print(f"\n  Result: {rid} -- status={status}, composite={composite}")

                git_commit_run(self.task, rid)

            if iteration >= max_iterations:
                break

            # Analyze & decide
            print("\nPhase 4: Analyzing results")
            analysis = self.analyze_results()
            print(f"  Analysis: {analysis.get('summary', '')}")
            print(f"  Decision: {analysis.get('decision', '?')}")

            if analysis.get("decision") == "stop":
                print("  Optimization converged. Stopping.")
                break

            print("\n  Planning next batch...")
            experiments = self.plan_experiments()
            if not experiments:
                print("  No more experiments to run.")
                break
            print(f"  Planned {len(experiments)} more experiment(s)")

        # Phase 5: Report
        print("\nPhase 5: Generating report")
        self.generate_report()

        if branch_name:
            pushed = git_push_sweep_branch()
            if pushed:
                print("  Pushed sweep branch to origin.")

        # Summary
        elapsed = time.time() - start_time
        self.completed = scan_task_completed_runs(self.task)

        print(f"\n{'=' * 60}")
        print(f"OPTIMIZATION COMPLETE -- {self.task}")
        print(f"{'=' * 60}")
        print(f"Total runs: {len(self.completed)}")
        print(f"Total time: {int(elapsed)}s ({int(elapsed) // 60}m {int(elapsed) % 60}s)")

        if self.completed:
            best_rid = max(
                self.completed,
                key=lambda k: self.completed[k].get("composite", 0) or 0,
            )
            best = self.completed[best_rid]
            print(f"Best run: {best_rid}")
            print(f"  Composite:   {best.get('composite', 'N/A')}")
            print(f"  Position L1: {best.get('position_l1', 'N/A')}")
            print(f"  Alive F1:    {best.get('alive_f1', 'N/A')}")
            print(f"  Val Loss:    {best.get('val_loss', 'N/A')}")
            print(f"  Params:      {best.get('num_params', 'N/A')}")
            print(f"  Time:        {best.get('train_time_sec', 'N/A')}s")

        print(f"\nReport: reports/{self.task}_report.md")
        print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autoresearch agent -- LLM-driven world model optimization. "
                    "Run: python agent.py --task pong_dreamer"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task to optimize (e.g., pong_dreamer, asteroids_d3pm)",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List available tasks and exit",
    )
    parser.add_argument(
        "--filter-game", type=str, default=None,
        help="Filter tasks by game (e.g., pong, snake)",
    )
    parser.add_argument(
        "--filter-model", type=str, default=None,
        help="Filter tasks by model (e.g., dreamer, ar_transformer)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=20,
        help="Maximum number of experiment runs (default: 20)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="LLM model to use for planning/analysis",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Only run preflight checks, then exit",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Run preflight + planning, show experiments without executing",
    )

    args = parser.parse_args()

    # List tasks
    if args.list_tasks or (not args.task and (args.filter_game or args.filter_model)):
        tasks = discover_tasks()
        if args.filter_game:
            tasks = [t for t in tasks if args.filter_game in t.rsplit("_", 1)[0]]
        if args.filter_model:
            tasks = [t for t in tasks if t.rsplit("_", 1)[-1] == args.filter_model
                     or t.endswith("_" + args.filter_model)]

        if tasks:
            print("Available tasks:")
            for t in tasks:
                metadata = load_task_metadata(t)
                game = metadata.get("game_id", "")
                model_type = metadata.get("model_type", "")
                print(f"  {t} (game={game}, model={model_type})")
        else:
            print("No tasks found matching filters.")
        sys.exit(0)

    if args.list_tasks:
        sys.exit(0)

    # Validate task
    if not args.task:
        tasks = discover_tasks()
        print("Error: --task is required.")
        if tasks:
            print(f"Available tasks: {', '.join(tasks[:10])}{'...' if len(tasks) > 10 else ''}")
        sys.exit(1)

    available_tasks = discover_tasks()
    if args.task not in available_tasks:
        print(f"Error: Task '{args.task}' not found.")
        if available_tasks:
            print(f"Available tasks: {', '.join(available_tasks[:10])}...")
        sys.exit(1)

    # Preflight only
    if args.preflight_only:
        info = run_preflight(args.task)
        print(info.summary())
        sys.exit(0 if info.ready else 1)

    # Create agent
    agent = AutoresearchAgent(
        task=args.task,
        api_key=args.api_key,
        model=args.model,
    )

    # Plan only
    if args.plan_only:
        info = agent.preflight()
        print(info.summary())
        if not info.ready:
            sys.exit(1)
        experiments = agent.plan_experiments()
        if experiments:
            print(f"\nPlanned {len(experiments)} experiment(s):")
            for i, cfg in enumerate(experiments, 1):
                print(f"\n--- Experiment {i}: {cfg.get('run_id', '?')} ---")
                print(json.dumps(cfg, indent=2))
        else:
            print("\nNo new experiments to run.")
        sys.exit(0)

    # Full run
    agent.run(max_iterations=args.max_iterations)
