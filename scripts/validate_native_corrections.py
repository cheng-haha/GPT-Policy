#!/usr/bin/env python3
"""Validate corrective feedback and reconnects on Isaac without remote model calls."""

import argparse
import importlib.util
import json
import runpy
import sys
from pathlib import Path

from gpt_policy.simulation.robodojo.config import check, load_config
from gpt_policy.simulation.robodojo.io import write_json
from gpt_policy.simulation.robodojo.runner import run_episode


def current_observation(request):
    values = []
    for item in request["params"]["input"]:
        if item["type"] != "text":
            continue
        try:
            value = json.loads(item["text"])
        except ValueError:
            continue
        if isinstance(value, dict) and "observation_id" in value:
            values.append(value)
    return values[-1]


def validate(config_path, output, gpu, task):
    root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = runpy.run_path(
        str(root / "tests/simulation/test_model_reconnection.py")
    )["FAKE_SERVER"]
    spec = importlib.util.spec_from_file_location(
        "recording_audit", root / "scripts/audit_robodojo_run.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    summary = []
    for failure in ("give_up", "bad_request", "disconnect"):
        tools = output / (failure + "_fixture")
        tools.mkdir()
        executable = tools / "fake-codex"
        executable.write_text("#!" + sys.executable + "\n" + fixture)
        executable.chmod(0o700)
        (tools / "failure").write_text(failure)
        # Caps apply to engineering fixtures only, never to a policy campaign.
        config.update(
            codex_bin=str(executable),
            model="engineering-fault-injector",
            max_decisions=3 if failure == "give_up" else 2,
        )
        evidence = check(config, task, 0, 0)
        if evidence["source"]["status"]:
            raise RuntimeError("Commit the candidate before native validation")
        evidence["engineering_only"] = {
            "remote_model_called": False,
            "failure_injection": failure,
            "not_a_policy_success_rate_experiment": True,
            "purpose": "Keep one simulator through a difficulty report or model reconnect.",
        }
        run = output / failure
        result = run_episode(config, evidence, run, gpu)
        expected_decisions = 3 if failure == "give_up" else 2
        expected_recoveries = 0 if failure == "give_up" else 1
        assert result["decisions"] == expected_decisions, result
        assert result["provider_recoveries"] == expected_recoveries, result
        assert result["native_control_steps"] == 2, result
        assert result["reason"] == "decision_budget", result
        assert result["complete"] is False, result
        assert result["valid_for_success_rate"] is False, result
        assert result["native_success"] is None, result
        assert result["native_score"] is None, result
        assert result["usage"]["calls"] == 3, result
        assert result["usage"]["failed_calls"] == expected_recoveries, result
        actions = [
            json.loads(line)
            for line in (run / "sim/actions.jsonl").read_text().splitlines()
        ]
        assert [action["step"] for action in actions] == [1, 2]
        requests = [
            json.loads(line)
            for line in (tools / "requests.jsonl").read_text().splitlines()
        ]
        assert len(requests) == 3
        snapshots = [current_observation(request) for request in requests]
        assert [snapshot["step_id"] for snapshot in snapshots] == [0, 1, 1]
        assert len({snapshot["observation_id"].rsplit(":", 1)[0] for snapshot in snapshots}) == 1
        initial_images = [
            item for item in requests[0]["params"]["input"] if item["type"] == "image"
        ]
        assert len(initial_images) == 51
        assert requests[0]["pid"] == requests[1]["pid"]
        if failure == "give_up":
            assert requests[1]["pid"] == requests[2]["pid"]
            feedback = snapshots[2]["previous_result"]
            assert feedback["termination_rejected"]
            assert feedback["native_episode_still_running"]
            assert "bounded corrective action" in feedback["instruction"]
            for key in ("arms", "observation_id", "step_id"):
                assert snapshots[1][key] == snapshots[2][key]
        else:
            assert (requests[1]["pid"] != requests[2]["pid"]) is (failure == "disconnect")
            assert snapshots[1] == snapshots[2]
            retry_images = [
                item for item in requests[2]["params"]["input"] if item["type"] == "image"
            ]
            assert len(retry_images) == 3
            assert requests[1]["params"] == requests[2]["params"]
        assert len({request["params"]["threadId"] for request in requests}) == 1
        integrity = audit.review(run, run / "integrity_review")
        assert integrity["integrity_passed"], integrity["engineering_integrity_checks"]
        item = {
            "failure": failure,
            "passed": True,
            "source_commit": evidence["source"]["commit"],
            "remote_model_called": False,
            "native_controls": 2,
            "provider_recoveries": expected_recoveries,
            "model_attempts": 3,
            "native_episode_id": result["episode_id"],
            "model_process_ids": [request["pid"] for request in requests],
            "same_pending_observation_identity": True,
            "same_native_thread_id": requests[0]["params"]["threadId"],
            "no_action_replay": True,
            "integrity_passed": True,
            "run": str(run),
        }
        summary.append(item)
        write_json(output / "summary.json", summary)
        print(json.dumps(item), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", default="organize_table")
    args = parser.parse_args()
    validate(args.config, args.output, args.gpu, args.task)


if __name__ == "__main__":
    main()
