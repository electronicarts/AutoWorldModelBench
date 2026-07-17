# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#!/usr/bin/env python3
"""Exec loop wrapper for Codex agent.

Runs `codex exec` in an infinite loop. When Codex stops (voluntarily or
due to error), restarts it with continuation context from summary.tsv.
Only killed by Harbor's agent timeout.
"""
import argparse
import os
import subprocess
import sys
import time


def build_prompt(instruction_file: str, run_num: int) -> str:
    instruction = open(instruction_file).read()
    tsv = "summary.tsv"
    if not os.path.exists(tsv):
        return instruction
    lines = open(tsv).readlines()
    n = len(lines) - 1
    if n <= 0:
        return instruction
    rows = [l.strip().split("\t") for l in lines[1:] if l.strip()]
    if not rows:
        return instruction
    best = max(rows, key=lambda r: float(r[3]) if len(r) > 3 and r[3] else 0)
    latest = rows[-1]
    header = (
        f"CONTINUATION (restart #{run_num}): {n} experiments completed.\n"
        f"Best so far: {best[0]} score={best[3]}\n"
        f"Latest: {latest[0]} score={latest[3]}\n\n"
        f"Analyze the LATEST result, form a NEW hypothesis, and run the "
        f"NEXT single experiment immediately. Do NOT re-run baseline.\n\n"
    )
    return header + instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--flags", default="")
    args = parser.parse_args()

    run_num = 1
    while True:
        prompt = build_prompt(args.instruction, run_num)
        cmd = [
            "codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model", args.model,
            "--json",
            "--enable", "unified_exec",
        ]
        if args.flags:
            cmd.extend(args.flags.split())
        cmd.extend(["--", prompt])

        print(f"--- CODEX EXEC #{run_num} STARTING ---", flush=True)
        with open(args.output, "a") as log:
            log.write(f"\n--- CODEX EXEC #{run_num} STARTING ---\n")
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            log.write(f"\n--- CODEX EXEC #{run_num} ENDED (exit={proc.returncode}) ---\n")
        print(f"--- CODEX EXEC #{run_num} ENDED (exit={proc.returncode}) ---", flush=True)

        run_num += 1
        time.sleep(2)


if __name__ == "__main__":
    main()
