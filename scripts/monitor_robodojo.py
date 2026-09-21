#!/usr/bin/env python3
"""Monitor running RoboDojo evaluation workers.

Examples:
  python scripts/monitor_robodojo.py
  python scripts/monitor_robodojo.py --watch --interval 5
  python scripts/monitor_robodojo.py --root third_party/RoboDojo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
ARG_RE = {
    "task": re.compile(r"--task_name\s+(\S+)"),
    "gpu": re.compile(r"--device_id\s+(\d+)"),
    "seed": re.compile(r"--seed\s+(\d+)"),
    "run_id": re.compile(r"--run-id\s+(\S+)"),
}


@dataclass
class Worker:
    pid: int
    task: str = "?"
    gpu: str = "?"
    seed: str = "?"
    run_id: str = "?"
    step: str = "-"
    status: str = "RUNNING"
    elapsed: str = "-"
    log: str = "-"


def processes() -> dict[int, tuple[int, str]]:
    out = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,args="], text=True, errors="replace"
    )
    result: dict[int, tuple[int, str]] = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            try:
                result[int(parts[0])] = (int(parts[1]), parts[2])
            except ValueError:
                continue
    return result


def ancestors(pid: int, procs: dict[int, tuple[int, str]]) -> list[str]:
    commands: list[str] = []
    seen: set[int] = set()
    while pid in procs and pid not in seen:
        seen.add(pid)
        parent, command = procs[pid]
        commands.append(command)
        pid = parent
    return commands


def match(pattern: re.Pattern[str], text: str, default: str = "?") -> str:
    found = pattern.search(text)
    return found.group(1) if found else default


def latest_step(log_path: Path) -> tuple[str, str]:
    if not log_path.is_file():
        return "-", "-"
    try:
        # Logs can contain control characters and may be large; the tail is enough.
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 128 * 1024))
            text = handle.read().decode(errors="replace")
    except OSError:
        return "-", "-"
    steps = STEP_RE.findall(text)
    if not steps:
        return "-", "-"
    current, total = steps[-1]
    return f"{current}/{total}", time.strftime("%H:%M:%S", time.localtime(log_path.stat().st_mtime))


def result_status(results_root: Path, run_id: str) -> tuple[str, str]:
    summary = results_root / f"{run_id}.json"
    if not summary.is_file():
        return "RUNNING", "-"
    try:
        data = json.loads(summary.read_text())
        row = (data.get("results") or [{}])[0]
        return str(row.get("status", "DONE")), str(row.get("elapsed_sec", "-")) + "s"
    except (OSError, ValueError, IndexError, TypeError):
        return "DONE", "-"


def discover(root: Path) -> list[Worker]:
    procs = processes()
    workers: list[Worker] = []
    marker = "src/eval_client/main.py --task_name"
    for pid, (_, command) in procs.items():
        if marker not in command:
            continue
        context = "\n".join([command, *ancestors(pid, procs)])
        worker = Worker(
            pid=pid,
            task=match(ARG_RE["task"], command),
            gpu=match(ARG_RE["gpu"], command),
            seed=match(ARG_RE["seed"], command),
            run_id=match(ARG_RE["run_id"], context),
        )
        if worker.run_id != "?":
            log_path = root / "smoke_results" / worker.run_id / "logs" / f"{worker.task}.log"
            worker.log = str(log_path)
            worker.step, _ = latest_step(log_path)
            worker.status, worker.elapsed = result_status(root / "smoke_results", worker.run_id)
        workers.append(worker)

    # Include recently completed summaries when no live process is found for them.
    known = {worker.run_id for worker in workers}
    for summary in sorted((root / "smoke_results").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if summary.stem in known or summary.stem.startswith("20"):
            continue
        try:
            data = json.loads(summary.read_text())
            row = (data.get("results") or [{}])[0]
            task = str(row.get("task", "?"))
            result_path = str(row.get("result_path", ""))
            gpu_match = re.search(r"gpu(\d+)", summary.stem)
            seed_match = re.search(r"arx_x5/(\d+)_", result_path)
            step = "-"
            workers.append(Worker(
                pid=0,
                task=task,
                gpu=gpu_match.group(1) if gpu_match else "-",
                seed=seed_match.group(1) if seed_match else "-",
                run_id=summary.stem,
                step=step,
                status=str(row.get("status", "DONE")),
                elapsed=str(row.get("elapsed_sec", "-")) + "s",
                log=str(row.get("log_path", result_path)),
            ))
        except (OSError, ValueError, IndexError, TypeError):
            continue
    return sorted(workers, key=lambda item: (item.gpu == "?", item.gpu, item.pid))


def render(workers: list[Worker], root: Path) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"RoboDojo monitor  {now}  root={root}"]
    headers = ["GPU", "PID", "Task", "Seed", "Step", "Status", "Elapsed", "Run ID"]
    rows = [[w.gpu, str(w.pid), w.task, w.seed, w.step, w.status, w.elapsed, w.run_id] for w in workers]
    if not rows:
        lines.append("No RoboDojo evaluation process found.")
        return "\n".join(lines)
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines.append("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers)) for _ in [0]) for row in rows)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("third_party/RoboDojo"), help="RoboDojo checkout")
    parser.add_argument("--watch", action="store_true", help="refresh until interrupted")
    parser.add_argument("--interval", type=float, default=5.0, help="refresh interval in seconds")
    parser.add_argument("--running-only", action="store_true", help="hide completed result summaries")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        while True:
            if args.watch:
                print("\033[2J\033[H", end="")
            workers = discover(root)
            if args.running_only:
                workers = [worker for worker in workers if worker.pid]
            print(render(workers, root), flush=True)
            if not args.watch:
                return 0
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
