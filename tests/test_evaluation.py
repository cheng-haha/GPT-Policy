"""Evaluation gates and resumption use fake policies, never hardware."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluation", ROOT / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)




def test_repetition_rotates_condition_order_and_preserves_failed_trials():
    t = {"id": "example"}
    pairs = [(t, {"id": c}) for c in ("a", "b", "c")]
    slots = evaluation.pending_slots(pairs, {}, 3, "codex", "none")
    assert [c["id"] for _, c, _ in slots] == list("abcbcac ab".replace(" ", ""))
    state = {"trials": [{"task": "example", "condition": "a", "repetition": 1, "human_outcome": "failed"}]}
    assert len(evaluation.pending_slots(pairs, state, 3, "codex", "none")) == 8




def test_model_done_is_not_a_human_success_and_missing_usage_stays_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    directory = tmp_path / "var/runs/gpt/trial_success"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({"outcome": "success", "elapsed_s": 9}))
    (directory / "events.jsonl").write_text('{"event":"model_decision"}\n')
    trial = {"task": "test", "condition": "none", "repetition": 1, "run_prefix": "var/runs/gpt/trial"}
    state = {"trials": [trial]}
    path = tmp_path / "results/state.json"
    monkeypatch.setattr(evaluation, "notify", lambda message: None)
    assert evaluation.rate_trial(trial, "must actually insert", state, path, ask=lambda _: "f")
    assert trial["model_outcome"] == "success"
    assert trial["human_outcome"] == "failed"
    evaluation.export_results(state, path)
    summary = evaluation.read_json(path.with_name("state-summary.json"))[0]
    assert summary["success_rate"] == 0
    assert summary["mean_estimated_cost_usd"] is None
    assert summary["estimated_cost_usd_known_runs"] == 0


def test_unrated_crashed_trial_is_preserved_for_review(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "notify", lambda _: None)
    trial = {"task": "test", "condition": "none", "repetition": 1, "run_prefix": "var/runs/gpt/trial", "phase": "started"}
    path = tmp_path / "state.json"
    assert not evaluation.rate_trial(trial, "verify", {"trials": [trial]}, path, ask=lambda _: "q")
    assert evaluation.read_json(path)["trials"][0]["phase"] == "needs_review"
    assert "human_outcome" not in trial


@pytest.mark.parametrize("human", ["success", "failed", None])
def test_policy_human_label_is_reused_without_confusing_model_result(tmp_path, monkeypatch, human):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "notify", lambda _: None)
    directory = tmp_path / f"trial_{human or 'unreviewed'}"
    directory.mkdir()
    (directory / "status.json").write_text(json.dumps({
        "outcome": human or "unreviewed", "model_outcome": "success", "human_outcome": human,
    }))
    trial = {"task": "test", "condition": "none", "repetition": 1, "run_prefix": "trial"}

    def ask(_):
        assert human is None, "Already rated in the policy terminal"
        return "f"

    assert evaluation.rate_trial(trial, "inspect", {"trials": [trial]}, tmp_path / "state.json", ask=ask)
    assert trial["model_outcome"] == "success"
    assert trial["human_outcome"] == (human or "failed")
    assert trial["phase"] == "rated"


def test_ambiguous_recordings_cannot_be_attributed_to_a_trial(tmp_path):
    for suffix in ("success", "failed"):
        (tmp_path / f"trial_{suffix}").mkdir()
    with pytest.raises(ValueError, match="Ambiguous"):
        evaluation.find_recording(tmp_path / "trial")


def test_fingerprint_covers_reference_image_bytes(tmp_path):
    (tmp_path / "image.jpg").write_bytes(b"one")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"instruction": "test", "content": ["test", {"image": "image.jpg"}]}))
    before = evaluation.fingerprint(request)
    (tmp_path / "image.jpg").write_bytes(b"two")
    assert evaluation.fingerprint(request) != before


@pytest.fixture
def fake_queue(tmp_path, monkeypatch):
    import gpt_policy.settings
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "live_policy_pids", lambda: [])
    monkeypatch.setattr(evaluation, "notify", lambda _: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    configs = tmp_path / "configs"
    (configs / "agents").mkdir(parents=True)
    (configs / "agents/codex.json").write_text('{"model":"test"}')
    monkeypatch.setattr(gpt_policy.settings, "load_settings", lambda *args: {"machine": "lab-example", "agent_config_dir": str(configs)})
    monkeypatch.setattr(gpt_policy.settings, "settings_path", lambda **kwargs: configs / "machine.json")
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: "abc" if "rev-parse" in args else b"")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    for c in ("a", "b"):
        (tmp_path / f"{c}.json").write_text(json.dumps({"instruction": c, "content": [c]}))
    task = {"id": "test", "title": "Test", "reset": "reset", "criterion": "inspect",
            "conditions": [{"id": c, "request": f"{c}.json"} for c in ("a", "b")]}
    plan = {"tasks": [task]}
    pairs = evaluation.selected_conditions(plan, [], [])
    state_path = tmp_path / "results/state.json"
    args = SimpleNamespace(machine="lab-example", agent="gpt", campaign="test", repetitions=2, history="none")
    calls = []

    def fake_policy(command):
        calls.append(command)
        config = evaluation.read_json(Path(command[command.index("--config") + 1]))
        prefix = Path(config["runtime"]["record_dir"])
        recording = prefix.with_name(prefix.name + "_success")
        recording.mkdir(parents=True)
        (recording / "status.json").write_text('{"outcome":"success","state":"completed"}')
        (recording / "usage.json").write_text('{"calls":1,"tokens":{"total_tokens":15},"estimated_cost_usd":0.02}')
        return 0, False

    monkeypatch.setattr(evaluation, "wait_for_policy", fake_policy)
    return args, plan, pairs, state_path, calls


def test_queue_waits_for_reset_and_resumes_without_repeating_failed_trial(fake_queue, monkeypatch):
    args, plan, pairs, path, calls = fake_queue
    answers = iter(["", "f", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    evaluation.run_queue(args, plan, pairs, path)
    state = evaluation.read_json(path)
    assert len(calls) == 1
    assert state["trials"][0]["condition"] == "a"
    assert state["trials"][0]["human_outcome"] == "failed"
    answers = iter(["", "s", "q"])
    evaluation.run_queue(args, plan, pairs, path)
    state = evaluation.read_json(path)
    assert len(calls) == 2
    assert [t["condition"] for t in state["trials"]] == ["a", "b"]
    assert all("--input-json" in command for command in calls)
    assert all("--prepare-only" not in command for command in calls)


def test_progress_is_updated_before_next_confirmation_and_includes_unselected_tasks(fake_queue, monkeypatch, capsys):
    args, plan, pairs, path, calls = fake_queue
    plan["tasks"].append({"id": "exploration", "title": "Explore", "conditions": [{"id": "history", "request": None}]})
    answers = iter(["", "f", "q"])

    def answer(prompt):
        value = next(answers)
        if value == "q":
            report = evaluation.read_json(path)["progress"]
            assert report["totals"]["rated"] == 1
            assert report["totals"]["success_rate"] == 0
            assert report["totals"]["remaining"] == 5
            assert report["totals"]["blocked_remaining"] == 2
            assert report["totals"]["elapsed_s"] is None
            assert report["totals"]["estimated_cost_usd"] == 0.02
            assert report["conditions"][0]["remaining"] == 1
            assert report["conditions"][-1]["status"] == "BLOCKED"
            summary = evaluation.read_json(path.with_name("state-summary.json"))
            assert summary[0]["success_rate"] == 0
            assert path.with_suffix(".csv").exists()
            output = capsys.readouterr().out
            assert "EVALUATION PROGRESS" in output and "exploration" in output
            assert "0/1 (0%)" in output and "Task time: unknown" in output
        return value

    monkeypatch.setattr("builtins.input", answer)
    evaluation.run_queue(args, plan, pairs, path)
    assert len(calls) == 1


def test_status_uses_saved_campaign_settings_and_keeps_history_out_of_success_rate(tmp_path):
    args = SimpleNamespace(machine="lab-example", agent="gpt", campaign="test", repetitions=3, history="reviewed")
    plan = {"tasks": [{"id": "test", "title": "Test", "conditions": [
        {"id": "a", "request": "a.json", "historical": {"codex": [{}, {}]}}]}]}
    state = {"repetitions": 2, "history": "none", "agent": "codex", "trials": [
        {"task": "test", "condition": "a", "human_outcome": "failed", "repetition": 1},
        {"task": "test", "condition": "a", "phase": "needs_review", "repetition": 2}]}
    path = tmp_path / "state.json"
    evaluation.report_progress(args, plan, state, path)
    row = evaluation.read_json(path)["progress"]["conditions"][0]
    assert row["historical"] == 0 and row["remaining"] == 1 and row["status"] == "REVIEW"
    assert row["rated"] == 1 and row["success_rate"] == 0
    state.update(history="documented", repetitions=3)
    state["trials"] = [{"task": "test", "condition": "a", "human_outcome": "failed", "repetition": 3}]
    evaluation.report_progress(args, plan, state, path)
    row = state["progress"]["conditions"][0]
    assert row["historical"] == 2 and row["remaining"] == 0 and row["status"] == "DONE"
    assert row["rated"] == 1 and row["success_rate"] == 0


def test_resume_rates_an_unreviewed_trial_before_any_new_motion(fake_queue, monkeypatch):
    args, plan, pairs, path, calls = fake_queue
    answers = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    evaluation.run_queue(args, plan, pairs, path)
    assert evaluation.read_json(path)["trials"][0]["phase"] == "needs_review"
    answers = iter(["f", "q"])
    evaluation.run_queue(args, plan, pairs, path)
    assert len(calls) == 1
    assert evaluation.read_json(path)["trials"][0]["phase"] == "rated"


def test_quit_at_first_reset_gate_starts_no_policy(fake_queue, monkeypatch, capsys):
    args, plan, pairs, path, calls = fake_queue
    instruction = "Pick up the red towel with the robot's right hand.\n" + "Match the demonstrated grasp. " * 40
    (evaluation.ROOT / "a.json").write_text(json.dumps({"instruction": instruction, "content": [instruction]}))

    def confirm(prompt):
        assert instruction.strip() in capsys.readouterr().out
        assert "确认以上英文指令" in prompt
        assert calls == []
        return "q"

    monkeypatch.setattr("builtins.input", confirm)
    evaluation.run_queue(args, plan, pairs, path)
    assert calls == []
    assert evaluation.read_json(path)["trials"] == []


def test_resume_rejects_changed_reference_input(fake_queue, monkeypatch):
    args, plan, pairs, path, calls = fake_queue
    answers = iter(["", "s", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    evaluation.run_queue(args, plan, pairs, path)
    (evaluation.ROOT / "a.json").write_text('{"instruction":"changed","content":["changed"]}')
    with pytest.raises(ValueError, match="input changed"):
        evaluation.run_queue(args, plan, pairs, path)
    assert len(calls) == 1


def test_empty_campaign_can_use_new_code_but_recorded_trials_remain_pinned(fake_queue, monkeypatch):
    args, plan, pairs, path, calls = fake_queue
    answers = iter(["q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    evaluation.run_queue(args, plan, pairs, path)
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: "new" if "rev-parse" in args else b"")
    answers = iter(["", "f", "q"])
    evaluation.run_queue(args, plan, pairs, path)
    assert evaluation.read_json(path)["code_commit"] == "new"
    assert len(calls) == 1
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: "next" if "rev-parse" in args else b"")
    with pytest.raises(ValueError, match="settings/code changed"):
        evaluation.run_queue(args, plan, pairs, path)
    assert len(calls) == 1


@pytest.mark.skipif(os.name != "posix", reason="terminal process groups require POSIX")
def test_ctrl_c_reaches_child_once_and_parent_waits_for_finalization(tmp_path):
    ready, saved = tmp_path / "ready", tmp_path / "saved"
    child = f"""import pathlib,signal,time,sys
def stop(*args):
 time.sleep(.3)
 pathlib.Path({str(saved)!r}).write_text('finalized')
 sys.exit(0)
signal.signal(signal.SIGINT,stop)
pathlib.Path({str(ready)!r}).write_text('ready')
while True: time.sleep(.02)
"""
    parent = f"""import importlib.util,sys
spec=importlib.util.spec_from_file_location('evaluation',{str(ROOT/'scripts/evaluate.py')!r})
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
code,interrupted=m.wait_for_policy([sys.executable,'-c',{child!r}])
assert code==0 and interrupted
"""
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 8
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, (stdout, stderr)
        assert saved.read_text() == "finalized"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)  # Only this test's synthetic process group.
            process.wait()


def test_explicit_config_is_passed_to_preflight_and_trial(fake_queue, tmp_path, monkeypatch):
    args, plan, pairs, path, calls = fake_queue
    args.config = tmp_path / "machine.local.json"
    checks = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: checks.append(command))
    answers = iter(["", "f", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    evaluation.run_queue(args, plan, pairs, path)
    assert checks[0][checks[0].index("--config") + 1] == str(args.config)
    generated = Path(calls[0][calls[0].index("--config") + 1])
    assert evaluation.read_json(generated)["extends"] == str(args.config)


def test_dry_run_accepts_local_override_config_without_opening_hardware(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", str(tmp_path / "machine.local.json"),
                                    "--agent", "gpt", "--dry-run"])
    monkeypatch.setattr(evaluation, "run_queue", lambda *_: pytest.fail("dry run opened queue"))
    evaluation.main()
    assert "Total remaining: 3. No hardware or model calls." in capsys.readouterr().out
