"""Freeze, no-replay resume, quota dispatch stop and GPU slot contracts."""

import json
from pathlib import Path

import pytest

from gpt_policy.simulation.robodojo import campaign
from gpt_policy.simulation.robodojo.io import write_json


def cases(n=5):
    return [{"task": "test", "seed": 0, "layout": i} for i in range(n)]


@pytest.fixture
def queue_backend(monkeypatch):
    calls = []
    active = set()
    state = {"quota": False, "peak": 0}

    def check(config, task, seed, layout):
        return {
            "context": {"sha256": "fixed"},
            "case": {"case_id": f"test_{layout}"},
            "source": config.get("version", 1),
        }

    class Process:
        def __init__(self, command, **kwargs):
            job = json.loads(Path(command[-1]).read_text())
            assert job["gpu"] not in active
            active.add(job["gpu"])
            state["peak"] = max(state["peak"], len(active))
            self.gpu = job["gpu"]
            self.job = job
            self.pid = 900000 + len(calls)
            self.returncode = None
            calls.append(job)
            folder = Path(job["output"])
            folder.mkdir()
            write_json(
                folder / "result.json",
                {
                    "complete": not state["quota"],
                    "valid_for_success_rate": not state["quota"],
                    "native_success": None if state["quota"] else False,
                    "status": "quota_error" if state["quota"] else "native_terminal",
                },
            )

        def poll(self):
            if self.returncode is None:
                active.remove(self.gpu)
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(campaign, "check", check)
    monkeypatch.setattr(
        campaign, "gpu_inventory", lambda gpu: {"uuid": str(gpu), "processes": []}
    )
    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    monkeypatch.setattr(campaign, "live_pid", lambda _: False)
    return calls, state


def test_each_gpu_one_worker_and_resume_does_not_repeat_failures(
    tmp_path, queue_backend
):
    calls, stats = queue_backend
    output = tmp_path / "campaign"
    report = campaign.run_campaign({}, cases(), output, [0, 1])
    assert stats["peak"] == 2
    assert len(calls) == 5
    assert report["valid_native_episodes"] == 5
    assert report["native_failures"] == 5 and report["success_rate_among_valid"] == 0
    campaign.run_campaign({}, cases(), output, [1], resume=True)
    assert len(calls) == 5


def test_quota_stops_pending_dispatch_and_resume_does_not_reset_it(
    tmp_path, queue_backend
):
    calls, stats = queue_backend
    stats["quota"] = True
    output = tmp_path / "campaign"
    report = campaign.run_campaign({}, cases(), output, [0])
    assert len(calls) == 1 and report["pending"] == 4
    assert report["dispatch_stopped"] == "quota_error"
    assert report["native_successes"] == 0 and report["valid_native_episodes"] == 0
    campaign.run_campaign({}, cases(), output, [0], resume=True)
    assert len(calls) == 1


def test_frozen_config_or_source_change_refuses_resume(tmp_path, queue_backend):
    output = tmp_path / "campaign"
    campaign.run_campaign({}, cases(1), output, [0])
    with pytest.raises(ValueError, match="Frozen"):
        campaign.run_campaign({"version": 2}, cases(1), output, [0], resume=True)


def test_interrupted_attempt_is_never_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "live_pid", lambda _: False)
    state = {
        "jobs": [
            {"id": "attempted", "status": "running", "pid": 123},
            {"id": "unstarted", "status": "pending"},
        ]
    }
    campaign.reconcile(tmp_path, state)
    assert state["jobs"][0]["status"] == "interrupted"
    assert state["jobs"][0]["result"]["native_success"] is None
    assert state["jobs"][1]["status"] == "pending"


def test_live_worker_or_orphan_simulator_refuses_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "live_pid", lambda pid: pid == 123)
    with pytest.raises(RuntimeError, match="worker"):
        campaign.reconcile(
            tmp_path, {"jobs": [{"id": "a", "status": "running", "pid": 123}]}
        )
    write_json(tmp_path / "a/sim/pid.json", {"pid": 123})
    with pytest.raises(RuntimeError, match="simulator"):
        campaign.reconcile(
            tmp_path, {"jobs": [{"id": "a", "status": "running", "pid": 456}]}
        )


def test_campaign_exclusive_lock(tmp_path):
    with (
        campaign.campaign_lock(tmp_path),
        pytest.raises(RuntimeError, match="owns"),
        campaign.campaign_lock(tmp_path),
    ):
        pytest.fail("Lock must not be acquired twice")


@pytest.mark.parametrize("gpus", [[], [0, 0], [-1], list(range(9))])
def test_bad_gpu_sets_fail_before_output_creation(tmp_path, gpus):
    with pytest.raises(ValueError):
        campaign.run_campaign({}, cases(), tmp_path / "out", gpus)
    assert not (tmp_path / "out").exists()


def test_documentation_commit_may_change_but_runtime_bytes_cannot():
    old = {
        "evidence": [
            {"source": {"commit": "a", "diff": "", "python_sha256": {"a.py": "one"}}}
        ]
    }
    documented = {
        "evidence": [
            {
                "source": {
                    "commit": "b",
                    "diff": "progress",
                    "python_sha256": {"a.py": "one"},
                }
            }
        ]
    }
    edited = {
        "evidence": [
            {"source": {"commit": "b", "diff": "", "python_sha256": {"a.py": "two"}}}
        ]
    }
    assert campaign.fingerprint(old) == campaign.fingerprint(documented)
    assert campaign.fingerprint(old) != campaign.fingerprint(edited)


def test_busy_slot_waits_without_consuming_attempt(
    tmp_path, queue_backend, monkeypatch
):
    calls, _ = queue_backend
    probes = []

    def inventory(gpu):
        probes.append(gpu)
        return {
            "uuid": "busy",
            "processes": [{"uuid": "busy", "pid": 7}] if len(probes) == 1 else [],
        }

    monkeypatch.setattr(campaign, "gpu_inventory", inventory)
    result = campaign.run_campaign({}, cases(1), tmp_path / "campaign", [0])
    assert len(calls) == 1 and len(probes) == 2
    assert result["attempted"] == 1 and result["native_failures"] == 1


def test_report_separates_running_invalid_and_native_results():
    def ended(success, score, **fields):
        return {"status": "finished", "result": {
            "complete": True, "valid_for_success_rate": True,
            "native_success": success, "native_score": score, **fields,
        }}

    state = {"jobs": [
        ended(True, 1.0), ended(False, 0.4),
        ended(True, 1.0, complete=False),
        ended(None, None), ended(True, 1.0, valid_for_success_rate=1),
        {"status": "running"}, {"status": "pending"},
    ]}
    result = campaign.report(state)
    assert result["valid_native_episodes"] == 2
    assert result["native_successes"] == result["native_failures"] == 1
    assert result["success_rate_among_valid"] == 0.5
    assert result["mean_score_among_valid"] == 0.7
    assert result["finished"] == 5 and result["incomplete_or_invalid"] == 3
    assert result["running"] == result["pending"] == 1
    assert result["final_success_rate"] is result["final_mean_score"] is None
    final = campaign.report({"jobs": state["jobs"][:2]})
    assert final["final_success_rate"] == 0.5 and final["final_mean_score"] == 0.7


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf")])
def test_report_never_invents_a_missing_or_invalid_final_score(score):
    result = campaign.report({"jobs": [{"status": "finished", "result": {
        "complete": True, "valid_for_success_rate": True,
        "native_success": True, "native_score": score,
    }}]})
    assert result["final_success_rate"] == 1.0
    assert result["mean_score_among_valid"] is result["final_mean_score"] is None
    assert result["scored_valid"] == 0
