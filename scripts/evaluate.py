#!/usr/bin/env python3
"""Run a resumable evaluation queue through the existing policy CLI."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AGENTS = {"gpt": "codex", "codex": "codex", "claude": "claude"}
OUTCOMES = ("success", "failed", "give_up", "interrupted", "unreviewed")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def slug(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in value):
        raise argparse.ArgumentTypeError("Use lowercase letters, digits, hyphens or underscores")
    return value


def request_path(condition):
    return ROOT / condition["request"] if condition.get("request") else None


def fingerprint(path):
    """Bind a trial to the actual text and image bytes, independent of host paths."""
    request = read_json(path)
    if not request:
        raise ValueError(f"Missing request: {path}")
    digest = hashlib.sha256(path.read_bytes())
    for part in request.get("content", []):
        if not isinstance(part, dict):
            continue
        if "video" in part:
            raise ValueError(f"Freeze video keyframes once before evaluation: {path}")
        image = path.parent / part["image"]
        digest.update(hashlib.sha256(image.read_bytes()).digest())
    return digest.hexdigest()


def historical_count(condition, agent, history):
    if history == "none" or (history == "reviewed" and condition.get("history_needs_review")):
        return 0
    return len(condition.get("historical", {}).get(agent, []))


def selected_conditions(plan, tasks, conditions):
    available_tasks = {t["id"] for t in plan["tasks"]}
    available_conditions = {c["id"] for t in plan["tasks"] for c in t["conditions"]}
    if set(tasks) - available_tasks or set(conditions) - available_conditions:
        raise ValueError("Unknown task/condition; use --list to see available IDs")
    return [(t, c) for t in plan["tasks"] if not tasks or t["id"] in tasks
            for c in t["conditions"] if not conditions or c["id"] in conditions]


def pending_slots(pairs, state, repetitions, agent, history):
    completed = {(r["task"], r["condition"], r["repetition"]) for r in state.get("trials", [])
                 if r.get("human_outcome") in {"success", "failed"}}
    slots = [(t, c, i) for t, c in pairs
            for i in range(min(repetitions, historical_count(c, agent, history)) + 1, repetitions + 1)
            if (t["id"], c["id"], i) not in completed]
    task_order = list(dict.fromkeys(t["id"] for t, _ in pairs))
    # Rotate condition order between repetitions, identically on both machines.
    def order(slot):
        task, condition, repetition = slot
        ids = [c["id"] for t, c in pairs if t["id"] == task["id"]]
        return task_order.index(task["id"]), repetition, (ids.index(condition["id"]) - repetition + 1) % len(ids)
    return sorted(slots, key=order)


def notify(message):
    print("\a\n" + message, flush=True)
    executable = shutil.which("notify-send")
    if executable:
        env = os.environ.copy()
        bus = Path(f"/run/user/{os.getuid()}/bus")
        if bus.exists():
            env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}")
        try:
            subprocess.run([executable, "--urgency=critical", "GPT Policy evaluation", message],
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass  # The terminal notification and reset gate remain available.


def wait_for_policy(command):
    """The foreground child receives Ctrl+C once and owns homing/finalization."""
    interrupted = False

    def remember_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\nWaiting for policy cleanup. Ctrl+C again cancels its homing.", flush=True)

    previous = signal.signal(signal.SIGINT, remember_interrupt)
    try:
        # Keep the terminal process group: terminal SIGINT reaches the child.
        # Do not use subprocess.run's KeyboardInterrupt cleanup, which kills it.
        child = subprocess.Popen(command, cwd=ROOT)
        return child.wait(), interrupted
    finally:
        signal.signal(signal.SIGINT, previous)


def find_recording(prefix):
    paths = [p for p in [prefix, *(prefix.with_name(prefix.name + "_" + s) for s in OUTCOMES)] if p.is_dir()]
    if len(paths) > 1:
        raise ValueError(f"Ambiguous recording directory: {prefix}")
    return paths[0] if paths else None


def refresh_result(trial):
    recording = find_recording(ROOT / trial["run_prefix"])
    trial["run_dir"] = str(recording.relative_to(ROOT)) if recording else None
    status = read_json(recording / "status.json") if recording else {}
    usage = read_json(recording / "usage.json") if recording else {}
    trial["model_outcome"] = status.get("model_outcome", status.get("outcome", status.get("task_status", status.get("state"))))
    trial["recorded_human_outcome"] = status.get("human_outcome")
    trial["finalization_error"] = status.get("error")
    trial["elapsed_s"] = status.get("elapsed_s")
    trial["calls"] = usage.get("calls")
    trial["tokens"] = usage.get("tokens", {}).get("total_tokens")
    trial["estimated_cost_usd"] = usage.get("estimated_cost_usd")
    trial["usage_complete"] = usage.get("token_totals_complete")
    trial["decisions"] = None
    if recording and (recording / "events.jsonl").is_file():
        try:
            with (recording / "events.jsonl").open() as stream:
                trial["decisions"] = sum(json.loads(line).get("event") == "model_decision" for line in stream if line.strip())
        except json.JSONDecodeError:
            trial["metrics_error"] = "Incomplete events.jsonl; decision count is unknown"


def rate_trial(trial, criterion, state, state_path, ask=None):
    ask = ask or input
    refresh_result(trial)
    trial["phase"] = "needs_review"
    save_json(state_path, state)
    notify(f"{trial['task']} / {trial['condition']} / trial {trial['repetition']} finished. Please inspect and reset the scene.")
    print(f"Recording: {trial.get('run_dir')}\nModel outcome: {trial.get('model_outcome')}\n验收标准：{criterion}")
    if trial.get("finalization_error"):
        print(f"收尾报错：{trial['finalization_error']}\n请处理现场状态后再确认下一轮复位。")
    if trial.get("recorded_human_outcome") in {"success", "failed"}:
        trial.update(human_outcome=trial["recorded_human_outcome"],
                     rated_at=datetime.now().astimezone().isoformat(), phase="rated")
        save_json(state_path, state)
        print(f"已采用本次运行的人工验收结果：{trial['human_outcome']}")
        return True
    while True:
        try:
            answer = ask("人工验收 [s 成功 / f 失败 / q 退出、稍后验收]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer == "q":
            return False
        if answer in {"s", "f"}:
            trial.update(human_outcome="success" if answer == "s" else "failed",
                         rated_at=datetime.now().astimezone().isoformat(), phase="rated")
            save_json(state_path, state)
            return True


def export_results(state, path):
    """Rates use human-rated new trials; historical credits stay separate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["task", "condition", "repetition", "human_outcome", "model_outcome", "run_dir",
               "decisions", "elapsed_s", "calls", "tokens", "estimated_cost_usd", "usage_complete",
               "returncode", "code_commit", "request_sha256"]
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state.get("trials", []))
    summary = []
    for key in sorted({(r["task"], r["condition"]) for r in state.get("trials", [])}):
        rows = [r for r in state["trials"] if (r["task"], r["condition"]) == key]
        rated = [r for r in rows if r.get("human_outcome") in {"success", "failed"}]
        item = {"task": key[0], "condition": key[1], "rated": len(rated), "pending_review": len(rows)-len(rated),
                "successes": sum(r["human_outcome"] == "success" for r in rated)}
        item["success_rate"] = item["successes"] / len(rated) if rated else None
        for field in ("decisions", "elapsed_s", "calls", "tokens", "estimated_cost_usd"):
            values = [r[field] for r in rated if r.get(field) is not None]
            item["total_" + field] = sum(values) if values else None
            item["mean_" + field] = sum(values) / len(values) if values else None
            item[field + "_known_runs"] = len(values)
        summary.append(item)
    save_json(path.with_name(path.stem + "-summary.json"), summary)
    return summary


def report_progress(args, plan, state, path):
    """Refresh the local report after each review, including the whole campaign."""
    summary = {(r["task"], r["condition"]): r for r in export_results(state, path)}
    pairs = selected_conditions(plan, [], [])
    agent = state.get("agent", AGENTS[args.agent])
    repetitions = state.get("repetitions", args.repetitions)
    history = state.get("history", args.history)
    pending = pending_slots(pairs, state, repetitions, agent, history)
    rows = []
    for task, condition in pairs:
        key = task["id"], condition["id"]
        row = {"task": key[0], "condition": key[1], "title": task["title"], "rated": 0,
               "successes": 0, "pending_review": 0, "success_rate": None, **summary.get(key, {})}
        row["historical"] = min(repetitions, historical_count(condition, agent, history))
        row["remaining"] = sum((t["id"], c["id"]) == key for t, c, _ in pending)
        row["status"] = ("DONE" if not row["remaining"] else "BLOCKED" if not condition.get("request")
                         else "REVIEW" if row["pending_review"] else "TO DO")
        rows.append(row)
    rated = [r for r in state.get("trials", []) if r.get("human_outcome") in {"success", "failed"}]
    totals = {"rated": len(rated), "successes": sum(r["human_outcome"] == "success" for r in rated),
              "pending_review": sum(r["pending_review"] for r in rows), "remaining": len(pending),
              "blocked_remaining": sum(r["remaining"] for r in rows if r["status"] == "BLOCKED")}
    totals["failed"] = totals["rated"] - totals["successes"]
    totals["success_rate"] = totals["successes"] / len(rated) if rated else None
    metrics = []
    for field, label, pattern in (("elapsed_s", "Task time", "{:.1f}s"), ("calls", "Calls", "{:,.0f}"),
                                  ("tokens", "Tokens", "{:,.0f}"), ("estimated_cost_usd", "API estimate", "${:.2f}")):
        values = [r[field] for r in rated if r.get(field) is not None]
        totals[field] = sum(values) if values else None
        totals[field + "_known_runs"] = len(values)
        metrics.append(f"{label}: {pattern.format(sum(values)) if values else 'unknown'} ({len(values)}/{len(rated)} known)")
    state["progress"] = {"updated_at": datetime.now().astimezone().isoformat(), "totals": totals, "conditions": rows}
    save_json(path, state)
    print(f"\nEVALUATION PROGRESS · {args.machine} · {args.agent} · {args.campaign}")
    print("History=历史名额；Rated=本批次已验收；Left=剩余；Human success=人工成功率")
    print(f"{'Task':18} {'Condition':19} {'History':>7} {'Rated':>5} {'Left':>5} {'Human success':>16}  Status")
    for row in rows:
        success = f"{row['successes']}/{row['rated']} ({row['success_rate']:.0%})" if row["rated"] else "—"
        print(f"{row['task']:18} {row['condition']:19} {row['historical']:7} {row['rated']:5} {row['remaining']:5} {success:>16}  {row['status']}")
    print(f"新批次：已验收 {totals['rated']} 次（成功 {totals['successes']} / 失败 {totals['failed']}），待验收 {totals['pending_review']} 次。")
    print(f"剩余 {totals['remaining']} 次，其中 {totals['blocked_remaining']} 次尚未配置。历史名额不混入新批次成功率。")
    print(" | ".join(metrics))
    print(f"报告已更新：{path}（以及同名 CSV、*-summary.json）", flush=True)


def live_policy_pids():
    found = []
    if not Path("/proc").exists():
        return found
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(Path(a).name in {"gpt-policy", "claude-policy", "kimi-policy"} for a in args[:2]) or args[1:3] == ["-m", "gpt_policy.main"]:
            found.append(int(proc.name))
    return found


def run_queue(args, plan, pairs, state_path):
    sys.path.insert(0, str(ROOT / "src"))
    from gpt_policy.input import resolve_run_input
    from gpt_policy.settings import load_settings, settings_path
    if live_policy_pids():
        raise ValueError("A policy process is still active; finish its cleanup first")
    if not sys.stdin.isatty():
        raise ValueError("Evaluation needs a foreground terminal for inspection and scene-reset confirmation")
    agent = AGENTS[args.agent]
    config_path = (Path(args.config).expanduser().resolve() if getattr(args, "config", None)
                   else settings_path(machine=args.machine))
    settings = load_settings(config_path)
    config_bytes = json.dumps(settings, sort_keys=True).encode()
    config_bytes += (Path(settings["agent_config_dir"]) / "agents" / f"{agent}.json").read_bytes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip():
        raise ValueError("Commit source changes before starting an evaluation campaign")
    state = read_json(state_path)
    header = {"schema_version": 1, "campaign": args.campaign, "machine": args.machine,
              "agent": agent, "history": args.history, "repetitions": args.repetitions,
              "code_commit": commit, "settings_sha256": hashlib.sha256(config_bytes).hexdigest()}
    if state.get("trials") and any(state.get(k) != v for k, v in header.items()):
        raise ValueError("Campaign settings/code changed; use a new --campaign name to keep results comparable")
    state = state if state.get("trials") else {**header, "trials": []}
    save_json(state_path, state)
    all_pairs = {(t["id"], c["id"]): (t, c) for t in plan["tasks"] for c in t["conditions"]}
    for previous in state["trials"]:
        t, c = all_pairs[previous["task"], previous["condition"]]
        if fingerprint(request_path(c)) != previous["request_sha256"]:
            raise ValueError("A previously evaluated input changed; use a new campaign")
        if previous.get("phase") != "rated":
            reviewed = rate_trial(previous, c.get("criterion", t["criterion"]), state, state_path)
            report_progress(args, plan, state, state_path)
            if not reviewed:
                return
    # Validate configuration before asking someone to arrange a scene.
    subprocess.run([sys.executable, "-m", "gpt_policy.main", "--check", "--config", str(config_path),
                    "--agent", agent], cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
    group = "gpt" if agent == "codex" else agent
    try:
        for task, condition, repetition in pending_slots(pairs, state, args.repetitions, agent, args.history):
            request = request_path(condition)
            if not request:
                print(f"SKIPPED (not configured): {task['id']} / {condition['id']} — {task.get('blocked', '')}")
                continue
            digest = fingerprint(request)
            instruction = resolve_run_input(None, request, None).instruction
            print(f"\n{task['title']} · {condition['id']} · {repetition}/{args.repetitions}")
            print(f"Task instruction:\n{instruction}\n")
            print(f"摆放：{task['reset']}\n验收：{condition.get('criterion', task['criterion'])}")
            if task.get("interactive"):
                print("此任务需要人类在执行过程中参与，请留在现场。")
            answer = input("确认以上英文指令无误、现场摆好、人员离开运动区域后按回车开始；q 退出: ").strip().lower()
            if answer == "q":
                break
            if answer:
                print("未确认指令与复位，本轮不启动。")
                break
            if fingerprint(request) != digest:
                raise ValueError("Task input changed during confirmation; review it before starting")
            if live_policy_pids():
                raise ValueError("Another policy is active")
            name = f"eval-{task['id']}-{condition['id']}-r{repetition:02d}"
            prefix = Path("var/runs") / group / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{name}"
            trial = {"task": task["id"], "condition": condition["id"], "repetition": repetition,
                     "request_sha256": digest, "request": os.path.relpath(request, ROOT), "code_commit": commit,
                     "run_prefix": str(prefix), "started_at": datetime.now().astimezone().isoformat(), "phase": "started"}
            state["trials"].append(trial)
            save_json(state_path, state)  # A restart must never silently rerun this trial.
            config = ROOT / ".runtime" / f"eval-{args.machine}.json"
            save_json(config, {"extends": str(config_path), "agent": agent,
                              "runtime": {"record_dir": str(ROOT / prefix), "task_name": name, "max_decisions": 100}})
            started = time.monotonic()
            try:
                returncode, interrupted = wait_for_policy([sys.executable, "-m", "gpt_policy.main", "--config", str(config),
                                                          "--agent", agent, "--input-json", str(request)])
                trial.update(returncode=returncode, interrupted=interrupted)
            finally:
                trial["wall_s"] = time.monotonic() - started
                trial["phase"] = "needs_review"
                save_json(state_path, state)
            reviewed = rate_trial(trial, condition.get("criterion", task["criterion"]), state, state_path)
            report_progress(args, plan, state, state_path)
            if not reviewed:
                break
            if interrupted:
                print("Evaluation paused after interruption. Run the same command to resume.")
                break
    except (EOFError, KeyboardInterrupt):
        print("\nEvaluation paused.")
    finally:
        export_results(state, state_path)
    remaining = len(pending_slots(pairs, state, args.repetitions, agent, args.history))
    print(f"\nQueue remaining: {remaining}. Results: {state_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--machine", type=slug, help="Local profile name under configs/machines/")
    target.add_argument("--config", type=Path, help="Explicit machine configuration")
    parser.add_argument("--agent", choices=tuple(AGENTS), required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/evaluation.example.json")
    parser.add_argument("--campaign", type=slug, default="evaluation")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--history", choices=("reviewed", "documented", "none"), default="none",
                        help="none: all new trials; documented: credit linked history; reviewed: exclude history flagged for review")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Start physical tasks, with a manual reset gate before each trial")
    mode.add_argument("--list", "--dry-run", action="store_true", help="Show queue and validate inputs without hardware/model calls (default)")
    mode.add_argument("--status", action="store_true", help="Export results without starting tasks")
    args = parser.parse_args()
    if args.config is not None:
        args.config = args.config.expanduser().resolve()
        args.machine = slug(args.config.stem.lower().replace(".", "-"))
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    plan = read_json(args.plan)
    if not plan:
        parser.error(f"Missing plan: {args.plan}")
    pairs = selected_conditions(plan, args.task, args.condition)
    group = "gpt" if AGENTS[args.agent] == "codex" else args.agent
    state_path = ROOT / "var/evaluations" / args.campaign / f"{args.machine}-{group}.json"
    state = read_json(state_path)
    if args.status:
        if not state:
            print("No trials recorded for this campaign.")
            return
        report_progress(args, plan, state, state_path)
        return
    if not args.run:
        pending = pending_slots(pairs, state, args.repetitions, AGENTS[args.agent], args.history)
        for task, condition in pairs:
            count = sum(t["id"] == task["id"] and c["id"] == condition["id"] for t, c, _ in pending)
            path = request_path(condition)
            availability = fingerprint(path)[:12] if path else "NOT CONFIGURED"
            print(f"{task['id']:18} {condition['id']:16} remaining={count} input={availability}")
        print(f"Total remaining: {len(pending)}. No hardware or model calls. Add --run to execute.")
        return
    lock_path = ROOT / ".runtime/evaluation.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluation queue is already running on this machine")
        run_queue(args, plan, pairs, state_path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(1)
