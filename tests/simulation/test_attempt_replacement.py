"""Replacement must preserve evidence and never pick the best outcome."""
import errno
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "replace_attempts", Path(__file__).resolve().parents[2] / "scripts/replace_robodojo_attempts.py"
)
replacement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replacement)


def setup_case(tmp_path):
    current, archive, new = (tmp_path / n for n in ("current", "archive", "new"))
    current.mkdir()
    jobs, entries = [], []
    for i in range(2):
        name = f"{i:03d}_task"
        folder = current / name
        folder.mkdir()
        result = {"complete": False, "valid_for_success_rate": False,
                  "native_success": None, "native_score": None, "reason": "model_give_up"}
        (folder / "result.json").write_text(json.dumps(result))
        (folder / "video.mp4").write_bytes(b"old-video")
        spec = {"task": f"task{i}", "seed": 0, "layout": 0}
        jobs.append({"id": name, "status": "finished", "spec": spec, "result": result})
        if i == 0:
            entries.append({"original_id": name, "replacement_id": "000_replacement", "spec": spec,
                            "attempt_number": 2, "prior_attempts": [{"output": str(archive / name)}],
                            "original_result_sha256": replacement.sha256(folder / "result.json")})
    replacement.write_json(current / "state.json", {"jobs": jobs})
    replacement.write_json(current / "freeze.json", {"immutable": "old"})
    manifest = {"current": str(current), "archive": str(archive), "replacement": str(new), "cases": entries}
    return manifest, current, archive, new


def test_archive_preserves_old_video_and_untouched_case(tmp_path):
    m, current, history, _ = setup_case(tmp_path)
    old_result = (current / "000_task/result.json").read_bytes()
    unchanged = (current / "001_task/result.json").read_bytes()
    replacement.archive(m)
    assert (history / "000_task/result.json").read_bytes() == old_result
    assert (history / "000_task/video.mp4").read_bytes() == b"old-video"
    assert (current / "000_task").is_symlink()
    assert (current / "001_task/result.json").read_bytes() == unchanged
    assert not (current / "freeze.json").exists()
    assert replacement.read(history / "freeze.json") == {"immutable": "old"}
    job = replacement.read(current / "state.json")["jobs"][0]
    assert job["status"] == "pending" and "result" not in job
    with pytest.raises(FileExistsError):
        replacement.archive(m)


@pytest.mark.parametrize("success,complete", [(True, True), (False, True), (None, False)])
def test_latest_outcome_always_replaces_and_invalid_not_scored(tmp_path, success, complete):
    m, current, history, new = setup_case(tmp_path)
    replacement.archive(m)
    new.mkdir()
    result = {"complete": complete, "valid_for_success_rate": complete,
              "native_success": success, "native_score": 1 if success else 0}
    replacement.write_json(new / "state.json", {"jobs": [{
        "id": "000_replacement", "spec": m["cases"][0]["spec"],
        "status": "finished", "result": result,
    }]})
    out = replacement.publish(m)
    assert replacement.read(current / "state.json")["jobs"][0]["result"] == result
    assert out["valid_native_episodes"] == int(complete)
    assert out["native_successes"] == int(success is True)
    assert out["final_success_rate"] is None
    assert out["mixed_protocol"] is True
    assert replacement.read(history / "000_task/result.json")["reason"] == "model_give_up"


def test_changed_original_refuses_before_archiving(tmp_path):
    m, current, history, _ = setup_case(tmp_path)
    (current / "000_task/result.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        replacement.archive(m)
    assert not history.exists()
    assert (current / "000_task").is_dir() and not (current / "000_task").is_symlink()


def test_archived_image_does_not_resolve_to_replacement_frame(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "audit_run", Path(__file__).resolve().parents[2] / "scripts/audit_robodojo_run.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    old = tmp_path / "current/001_case"
    archived = tmp_path / "history/001_case"
    relative = Path("sim/observations/000000/cam_high.png")
    for root, data in [(old, b"replacement frame"), (archived, b"historical frame")]:
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes(data)
    resolved = audit.recorded_image_path(archived, old / relative, old)
    assert resolved.read_bytes() == b"historical frame"
    external = tmp_path / "unrelated.png"
    assert audit.recorded_image_path(archived, external, old) == external


def test_stale_handle_reopens_without_masking_other_errors(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    p.write_text('{"generation": 2}')
    original = Path.read_text
    calls = []
    def flaky(path, *args, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            raise OSError(errno.ESTALE, "Stale file handle")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", flaky)
    monkeypatch.setattr(replacement.time, "sleep", lambda _: None)
    assert replacement.read(p) == {"generation": 2}
    assert len(calls) == 3
    with pytest.raises(PermissionError):
        replacement.retry_stale(lambda: (_ for _ in ()).throw(PermissionError("denied")))
    p.write_text("{invalid")
    with pytest.raises(json.JSONDecodeError):
        replacement.read(p)


def test_follow_does_not_launch_or_claim_completion_while_controller_lives(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    replacement.write_json(metadata / "controller.json", {"pid": 123, "argv": ["known"]})
    m = {"metadata": str(metadata), "replacement": str(tmp_path / "run")}
    live = iter([True, True, False])
    monkeypatch.setattr(replacement, "controller_alive", lambda _: next(live))
    monkeypatch.setattr(replacement.time, "sleep", lambda _: None)
    monkeypatch.setattr(replacement, "verify_history", lambda _: {"preserved": True})
    monkeypatch.setattr(replacement.subprocess, "Popen",
                        lambda *_a, **_k: pytest.fail("Must not relaunch the campaign"))
    events = []
    def publish(_):
        assert not (metadata / "completion.json").exists()
        events.append("publish")
        return {"running": 0, "pending": 0}
    monkeypatch.setattr(replacement, "publish", publish)
    assert replacement.follow(m) == 0
    assert events == ["publish"] * 3
    result = replacement.read(metadata / "completion.json")
    assert result["all_cases_finished"] is True
    assert result["exit_code"] is None and result["exit_code_known"] is False


def test_controller_identity_uses_native_nul_separated_arguments(monkeypatch):
    monkeypatch.setattr(replacement, "live_pid", lambda _: True)
    monkeypatch.setattr(Path, "read_bytes",
                        lambda _: b"python" + bytes(1) + b"-m" + bytes(1) + b"runner" + bytes(1))
    controller = {"pid": 123, "argv": ["python", "-m", "runner"]}
    assert replacement.controller_alive(controller) is True
    with pytest.raises(RuntimeError, match="reused"):
        replacement.controller_alive({"pid": 123, "argv": ["python", "-m", "different"]})


@pytest.mark.parametrize("status", ["runtime_error", "invalid_native_layout"])
def test_second_archive_preserves_prior_campaign_and_generations(tmp_path, status):
    import copy
    import os

    m, current, history, previous = setup_case(tmp_path)
    replacement.archive(m)
    folder = previous / "000_replacement"
    folder.mkdir(parents=True)
    result = {"complete": False, "valid_for_success_rate": False,
              "native_success": None, "status": status, "reason": status}
    replacement.write_json(folder / "result.json", result)
    (folder / "video.mp4").write_bytes(b"second-attempt-video")
    replacement.write_json(previous / "state.json", {"jobs": [{
        "id": "000_replacement", "spec": m["cases"][0]["spec"],
        "status": "finished", "result": result,
    }]})
    replacement.publish(m)
    # Relative aliases must still point to the historical recording after moving.
    alias = current / "000_task"
    alias.unlink()
    alias.symlink_to(os.path.relpath(folder, current), target_is_directory=True)
    prior_view = replacement.read(current / "CURRENT_VIEW.json")
    new = copy.deepcopy(m)
    new.update(archive=str(tmp_path / "archive2"), replacement=str(tmp_path / "next"))
    new["cases"][0].update(
        original_result_sha256=replacement.sha256(folder / "result.json"),
        authorized_retry_status=status, attempt_number=3,
    )
    replacement.archive(new)
    archived = Path(new["archive"])
    assert (archived / "000_task/video.mp4").read_bytes() == b"second-attempt-video"
    assert (folder / "video.mp4").read_bytes() == b"second-attempt-video"
    assert (history / "000_task/video.mp4").read_bytes() == b"old-video"
    assert replacement.read(archived / "CURRENT_VIEW.json") == prior_view
    assert replacement.read(current / "CURRENT_VIEW.json")["original_freeze"] == str(history / "freeze.json")
    assert replacement.read(current / "state.json")["jobs"][0]["status"] == "pending"


def test_infrastructure_retry_cannot_select_valid_native_failure(tmp_path):
    m, current, history, _ = setup_case(tmp_path)
    folder = current / "000_task"
    replacement.write_json(folder / "result.json", {
        "status": "native_completed", "reason": "native_terminal",
        "complete": True, "valid_for_success_rate": True, "native_success": False,
    })
    m["cases"][0].update(authorized_retry_status="runtime_error",
                         original_result_sha256=replacement.sha256(folder / "result.json"))
    with pytest.raises(ValueError, match="authorized interruption"):
        replacement.archive(m)
    assert not history.exists()
