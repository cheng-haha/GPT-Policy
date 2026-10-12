#!/usr/bin/env python3
"""Archive selected old attempts and publish their replacements at the original case paths.

The new campaign owns its own immutable freeze and process queue. The old root
becomes a current-results view, not a resumable campaign. Never choose by score.
"""
import argparse
import copy
import errno
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gpt_policy.simulation.robodojo.campaign import campaign_lock, live_pid, report
from gpt_policy.simulation.robodojo.io import sha256, write_json


def retry_stale(operation):
    # CPFS can invalidate a handle while a state file is atomically replaced.
    # Reopen the path; never replay a robot action or mask malformed data.
    for delay in (0.1, 0.2, 0.5, 1.0, None):
        try:
            return operation()
        except OSError as error:
            if error.errno != errno.ESTALE or delay is None:
                raise
            time.sleep(delay)


def read(path):
    return json.loads(retry_stale(lambda: Path(path).read_text()))


def archive(manifest):
    current = Path(manifest["current"])
    history = Path(manifest["archive"])
    replacement = Path(manifest["replacement"])
    state = read(current / "state.json")
    jobs = {j["id"]: j for j in state["jobs"]}
    entries = manifest["cases"]
    if len({e["original_id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate original cases")
    if len({e["replacement_id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate replacement cases")
    # Validate the entire transaction before moving any old evidence.
    if history.exists():
        raise FileExistsError("Archive already exists; do not repeat archive")
    previous_view = read(current / "CURRENT_VIEW.json") if (current / "CURRENT_VIEW.json").exists() else None
    if previous_view and previous_view["replacement_campaign"] == str(replacement):
        raise ValueError("Replacement campaign must be new")
    for job in jobs.values():
        if job["status"] not in ("finished", "interrupted"):
            raise ValueError("Original campaign is not finished")
        if live_pid(job.get("pid")):
            raise ValueError("An original worker PID is still alive")
    for entry in entries:
        job = jobs[entry["original_id"]]
        folder = current / job["id"]
        if job["spec"] != entry["spec"] or (folder.is_symlink() and previous_view is None):
            raise ValueError("Case identity or original directory mismatch")
        if sha256(folder / "result.json") != entry["original_result_sha256"]:
            raise ValueError("Original result changed since selection")
        result = read(folder / "result.json")
        legacy = result.get("reason") == "model_give_up" or (
            result.get("status") == "runtime_error"
            and "Codex input is too large" in result.get("error", "")
        )
        retry_status = entry.get("authorized_retry_status")
        explicit_infrastructure_retry = (
            retry_status in ("runtime_error", "invalid_native_layout")
            and result.get("status") == retry_status
            and not result.get("valid_for_success_rate")
        )
        if not legacy and not explicit_infrastructure_retry:
            raise ValueError("Selected case is not an authorized interruption")
        pid = folder / "sim/pid.json"
        if (pid.exists() and not (folder / "sim/cleanup.json").exists()
                and live_pid(read(pid)["pid"])):
            raise ValueError("An original simulator is still alive")
    history.mkdir(parents=True)
    write_json(history / "replacement_manifest.json", manifest)
    # Keep a full original state/report/freeze before replacing current summaries.
    for name in ("state.json", "report.json", "freeze.json", "monitor.json",
                 "CURRENT_VIEW.json", "CURRENT_RESULTS.md"):
        p = current / name
        if p.exists():
            shutil.copy2(p, history / name)
    for entry in entries:
        name = entry["original_id"]
        folder = current / name
        if folder.is_symlink():
            # Keep the frozen prior campaign intact, including relative aliases.
            (history / name).symlink_to(folder.resolve(strict=True), target_is_directory=True)
            folder.unlink()
        else:
            folder.rename(history / name)
        for suffix in (".job.json", ".log"):
            p = current / (name + suffix)
            if p.exists():
                p.rename(history / p.name)
        (current / name).symlink_to(replacement / entry["replacement_id"], target_is_directory=True)
    # A mixed current view must never claim to be the original frozen campaign.
    if (current / "freeze.json").exists():
        (current / "freeze.json").unlink()
    if (current / "monitor.json").exists():
        (current / "monitor.json").unlink()
    write_json(current / "CURRENT_VIEW.json", {
        "type": "latest_selected_attempts", "mixed_protocol": True,
        "not_a_resumable_campaign": True,
        "original_freeze": (
            previous_view["original_freeze"] if previous_view else str(history / "freeze.json")
        ),
        "replacement_campaign": str(replacement), "archive": str(history),
        "selection_rule": "Latest authorized attempt regardless of outcome; never best-of.",
        "manifest": manifest,
    })
    publish(manifest)


def publish(manifest):
    current, history = Path(manifest["current"]), Path(manifest["archive"])
    replacement = Path(manifest["replacement"])
    if read(current / "CURRENT_VIEW.json")["manifest"] != manifest:
        raise ValueError("Current view is bound to a different replacement manifest")
    state = copy.deepcopy(read(history / "state.json"))
    new_state_path = replacement / "state.json"
    new = read(new_state_path) if new_state_path.exists() else {"jobs": []}
    by_id = {j["id"]: j for j in new["jobs"]}
    selected = {e["original_id"]: e for e in manifest["cases"]}
    for i, old in enumerate(state["jobs"]):
        entry = selected.get(old["id"])
        if not entry:
            continue
        source = by_id.get(entry["replacement_id"], {
            "id": entry["replacement_id"], "spec": entry["spec"], "status": "pending",
        })
        if source["spec"] != entry["spec"]:
            raise ValueError("Replacement case identity mismatch")
        # Clearing the old result while pending prevents stale failures being counted.
        job = copy.deepcopy(source)
        job.update(id=old["id"], current_attempt=entry["attempt_number"],
                   current_output=str(replacement / entry["replacement_id"]),
                   archived_attempts=entry["prior_attempts"])
        state["jobs"][i] = job
    state.update(dispatch_stopped=new.get("dispatch_stopped"), mixed_protocol=True,
                 replacement_campaign=str(replacement), updated_at_s=time.time())
    write_json(current / "state.json", state)
    summary = report(state)
    summary.update(mixed_protocol=True, selection_rule="latest authorized attempt, not best-of",
                   original_archive=str(history), replacement_campaign=str(replacement))
    write_json(current / "report.json", summary)
    lines = [
        "# Current RoboDojo results", "",
        "Selected legacy interruptions are replaced by the latest authorized attempt.",
        "This is a mixed-version current-results view, not one homogeneous frozen evaluation.",
        "Pending/running/invalid attempts do not contribute to native score/success denominators.", "",
        "| Task | Native success | Native failure | Incomplete/invalid | Running | Pending | Mean score (valid only) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in dict.fromkeys(j["spec"]["task"] for j in state["jobs"]):
        sub = report({"jobs": [j for j in state["jobs"] if j["spec"]["task"] == task]})
        mean = sub["mean_score_among_valid"]
        lines.append(f"| {task} | {sub['native_successes']} | {sub['native_failures']} | "
                     f"{sub['incomplete_or_invalid']} | {sub['running']} | {sub['pending']} | "
                     f"{'—' if mean is None else f'{mean:.4f}'} |")
    lines += ["", f"Historical archive: {history}", f"Replacement campaign: {replacement}", ""]
    temp = current / "CURRENT_RESULTS.md.tmp"
    temp.write_text("\n".join(lines))
    temp.replace(current / "CURRENT_RESULTS.md")
    return summary


def verify_history(manifest):
    selected = {e["original_id"] for e in manifest["cases"]}
    checked = {}
    for name, expected in manifest["all_original_result_sha256"].items():
        root = Path(manifest["archive"] if name in selected else manifest["current"])
        actual = sha256(root / name / "result.json")
        if actual != expected:
            raise ValueError(f"Historical/untouched result changed: {name}")
        checked[name] = actual
    return {"all_original_results_preserved": True, "result_sha256": checked}


def run(manifest_path):
    manifest = read(manifest_path)
    replacement = Path(manifest["replacement"])
    metadata = Path(manifest["metadata"])
    if replacement.exists():
        raise FileExistsError("Replacement campaign already exists; never launch it twice")
    if not (Path(manifest["current"]) / "CURRENT_VIEW.json").exists():
        raise ValueError("Archive the selected old attempts first")
    if read(metadata / "suite.json")["episodes"] != [e["spec"] for e in manifest["cases"]]:
        raise ValueError("Suite differs from the authorized replacement manifest")
    write_json(metadata / "history_integrity_before_launch.json", verify_history(manifest))
    argv = [sys.executable, "-u", "-m", "gpt_policy.simulation.robodojo.cli",
            "--config", str(metadata / "config.json"), "--suite", str(metadata / "suite.json"),
            "--gpus", ",".join(map(str, manifest.get("gpus", range(8)))),
            "--output", str(replacement)]
    with (metadata / "campaign.log").open("x") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(metadata / "controller.json", {"pid": process.pid, "argv": argv})
        return follow(manifest, process)


def controller_alive(controller):
    pid = controller["pid"]
    if not live_pid(pid):
        return False
    try:
        actual = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip(chr(0)).split(chr(0))
    except FileNotFoundError:
        return False
    if actual != controller["argv"]:
        raise RuntimeError("Controller PID was reused; refusing to follow another process")
    return True


def follow(manifest, process=None):
    """Reattach only the publisher to the existing controller; never launch workers."""
    metadata = Path(manifest["metadata"])
    controller = read(metadata / "controller.json")
    with campaign_lock(metadata):
        while process.poll() is None if process is not None else controller_alive(controller):
            try:
                retry_stale(lambda: publish(manifest))
            except OSError as error:
                if error.errno != errno.ESTALE:
                    raise
                print(f"Transient CPFS read/write failure; publisher will retry: {error}", flush=True)
            time.sleep(10)
        summary = retry_stale(lambda: publish(manifest))
        write_json(metadata / "history_integrity.json", retry_stale(lambda: verify_history(manifest)))
        # A reattached observer cannot recover another parent's wait status.
        return_code = process.returncode if process is not None else None
        if return_code is not None:
            (metadata / "exit_code").write_text(str(return_code) + chr(10))
        all_finished = summary["running"] == 0 and summary["pending"] == 0
        write_json(metadata / "completion.json", {
            "exit_code": return_code, "exit_code_known": process is not None,
            "finished_at_s": time.time(), "all_cases_finished": all_finished,
            "replacement_report": str(Path(manifest["replacement"]) / "report.json"),
            "current_summary": {k: v for k, v in summary.items() if k != "episodes"},
        })
        return (return_code or 0) if all_finished else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive", "publish", "run", "follow"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        raise SystemExit(run(args.manifest))
    if args.action == "follow":
        raise SystemExit(follow(read(args.manifest)))
    result = globals()[args.action](read(args.manifest))
    if result is not None:
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
