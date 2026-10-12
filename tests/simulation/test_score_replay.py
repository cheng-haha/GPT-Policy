"""Recorded actions must remain immutable and auditable during offline replay."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gpt_policy.simulation.robodojo.replay import compare_observation, recorded_actions


def source_files(root, commands):
    (root / "sim").mkdir()
    (root / "result.json").write_text(json.dumps({"native_control_steps": len(commands)}))
    (root / "sim/actions.jsonl").write_text(
        "".join(json.dumps({"step": i, "command": command}) + chr(10)
                for i, command in enumerate(commands, 1))
    )


def test_recorded_commands_are_not_replanned_or_rewritten(tmp_path):
    command = np.linspace(-0.2, 0.2, 14).tolist()
    command[6], command[13] = 0, 1
    source_files(tmp_path, [command])
    before = (tmp_path / "sim/actions.jsonl").read_bytes()
    _, rows = recorded_actions(tmp_path)
    assert rows == [{"step": 1, "command": command}]
    assert (tmp_path / "sim/actions.jsonl").read_bytes() == before


@pytest.mark.parametrize("command", [[0] * 13, [float("nan")] * 14, [2] * 14])
def test_malformed_recorded_controls_are_rejected(tmp_path, command):
    source_files(tmp_path, [command])
    with pytest.raises(ValueError):
        recorded_actions(tmp_path)


def test_missing_commands_are_not_silently_ignored(tmp_path):
    source_files(tmp_path, [[0] * 14])
    (tmp_path / "result.json").write_text('{"native_control_steps": 2}')
    with pytest.raises(ValueError, match="count mismatch"):
        recorded_actions(tmp_path)


def test_image_and_robot_comparison_detects_drift(tmp_path):
    directory = tmp_path / "sim/observations/000000"
    directory.mkdir(parents=True)
    arms = {arm: {"joint_pos": [0] * 6, "link6_pose_xyzquat": [0, 0, 1, 0, 0, 0, 1],
                  "gripper": 1} for arm in ("left", "right")}
    (directory / "state.json").write_text(json.dumps({"arms": arms}))
    images = {name: np.zeros((2, 2, 3), dtype=np.uint8)
              for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    for name, rgb in images.items():
        Image.fromarray(rgb).save(directory / f"{name}.png")
    observation = {"arms": arms, "images": images}
    assert compare_observation(observation, tmp_path, 0)["images"]["cam_high"]["exact"]
    arms["left"]["link6_pose_xyzquat"][0] = 0.1
    images["cam_high"][:] = 7
    result = compare_observation(observation, tmp_path, 0)
    assert result["max_link6_position_error_m"] == 0.1
    assert result["images"]["cam_high"]["mae_255"] == 7
    assert not result["images"]["cam_high"]["exact"]


def test_nine_case_mean_is_not_confused_with_task_mean(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts/summarize_robodojo_score_replay.py"
    spec = importlib.util.spec_from_file_location("replay_summary_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tasks = []
    # Nine replays across three tasks, plus their previously successful cases.
    for task, missing, partial in [("organize_table", 4, 25),
                                   ("pack_objects_into_box", 3, 10),
                                   ("build_tower", 2, 30)]:
        cases = []
        for layout in range(5):
            if layout < missing:
                directory = tmp_path / f"{task}__g0__l{layout}" / "sim"
                directory.mkdir(parents=True)
                result = {
                    "replayed_all_controls": True, "source_artifacts_unchanged": True,
                    "reconstructed_score_100": partial, "controls_applied": 10,
                    "max_link6_position_error_m": 0, "max_joint_error_rad": 0,
                    "max_gripper_error": 0, "all_rgb_exact": False, "image_mae_255": {},
                }
                (directory / "replay-result.json").write_text(json.dumps(result))
                cases.append({"layout": layout, "basis": "missing_original_process_state",
                              "score_100": None})
            else:
                cases.append({"layout": layout, "basis": "stored_normalized_native_score",
                              "score_100": 100})
        tasks.append({"task": task, "name": task, "cases": cases})
    report = module.summarize({"tasks": tasks}, tmp_path)
    assert report["mean_score_of_nine_replays_100"] == pytest.approx(190 / 9)
    assert [t["reconstruction_aggregate_mean_score_100"] for t in report["tasks"]] == [40, 46, 72]
    assert report["counted_cases"] == 15
    assert report["episode_weighted_mean_score_100"] == pytest.approx(790 / 15)
    assert report["exact_original_score_claim"] is False
    broken = tmp_path / "build_tower__g0__l0/sim/replay-result.json"
    result = json.loads(broken.read_text())
    result["replayed_all_controls"] = False
    broken.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="terminated early"):
        module.summarize({"tasks": tasks}, tmp_path)



def test_native_horizon_replays_only_prefix_and_preserves_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gpt_policy.simulation.robodojo import replay

    source, output = tmp_path / "source", tmp_path / "replay"
    source.mkdir()
    output.mkdir()
    source_files(source, [[0] * 14] * 3)
    case = {"task": "fixture"}
    (source / "result.json").write_text(json.dumps({"native_control_steps": 3, "case": case}))
    for name, data in [("resolved_config.json", {}), ("case.json", case),
                       ("loaded_layers.json", [])]:
        (source / "sim" / name).write_text(json.dumps(data))
        (output / name).write_text(json.dumps(data))
    before = (source / "sim/actions.jsonl").read_bytes()
    env = SimpleNamespace(
        step_lim=2, end_flag=[False], take_action_cnt=[0],
        reward_manager=SimpleNamespace(score_completed_count=[0], score_meta=[{}],
                                       score_achieved=[[]], final_score_completed_count=[0]))
    def take_action(command):
        env.take_action_cnt[0] += 1
        env.end_flag[0] = env.take_action_cnt[0] == env.step_lim
    env.take_action = take_action
    session = SimpleNamespace(
        output=output, case=case, env=env, step_id=0, obs={},
        reset=lambda: None, observe=lambda: None,
        finish=lambda reason: {"native_success": False},
        budget=SimpleNamespace(expired=lambda: False, evidence={"control_step_limit": "native"}))
    monkeypatch.setattr(replay, "native_process_score_100", lambda env: 25)
    monkeypatch.setattr(replay, "compare_observation", lambda *args: {
        "max_link6_position_error_m": 0, "max_joint_error_rad": 0, "max_gripper_error": 0,
        "images": {c: {"mae_255": 0, "exact": True} for c in replay.CAMERAS}})
    replay.replay_recorded_episode(session, source)
    result = json.loads((output / "replay-result.json").read_text())
    assert result["controls_applied"] == result["controls_requested"] == 2
    assert result["controls_expected"] == 3
    assert result["replayed_requested_prefix"] is True
    assert result["replayed_all_controls"] is False
    assert len((output / "actions.jsonl").read_text().splitlines()) == 2
    assert result["reconstructed_score_100"] == 25
    assert (source / "sim/actions.jsonl").read_bytes() == before
