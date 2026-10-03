from pathlib import Path

import pytest

from gpt_policy.settings import runtime_config


def test_decision_budget_defaults_to_100_and_accepts_explicit_limit():
    assert runtime_config({}).max_decisions == 100
    assert runtime_config({"runtime": {"max_decisions": 5}}).max_decisions == 5


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, float("inf"), float("nan")])
def test_decision_budget_rejects_invalid_limits(value):
    with pytest.raises(ValueError, match="max_decisions"):
        runtime_config({"runtime": {"max_decisions": value}})


def test_runtime_config_reads_unified_runtime_section() -> None:
    value = runtime_config(
        {
            "runtime": {
                "robot_model": "X5",
                "interface": "can2",
                "right_interface": None,
                "gripper_open_readout": -2.5,
                "camera_width": 800,
                "camera_height": 600,
                "convert_camera_images_to_jpeg": True,
                "camera_jpeg_quality": 82,
                "trajectory_hz": 50,
                "task_name": "pick_red",
                "record_dir": "var/custom-run",
                "camera_overrides": ["left=/dev/video2"],
                "input_json": "requests/task.json",
            }
        },
        base_dir=Path("/tmp/arx-config"),
    )

    assert value.right_interface == ""
    assert value.task_name == "pick-red"
    assert value.record_dir == Path("var/custom-run")
    assert value.camera_overrides == ["left=/dev/video2"]
    assert value.input_json == Path("/tmp/arx-config/requests/task.json").resolve()
    assert value.convert_camera_images_to_jpeg is True
    assert value.camera_jpeg_quality == 82


def test_runtime_config_accepts_top_level_input_json() -> None:
    value = runtime_config(
        {"input_json": "request.json"},
        base_dir=Path("/tmp/arx-config"),
    )

    assert value.input_json == Path("/tmp/arx-config/request.json").resolve()


def test_runtime_config_rejects_invalid_task_name() -> None:
    with pytest.raises(ValueError, match="task_name"):
        runtime_config({"runtime": {"task_name": "机械臂任务"}})


def test_runtime_config_validates_camera_jpeg_options() -> None:
    with pytest.raises(ValueError, match="convert_camera_images_to_jpeg"):
        runtime_config({"runtime": {"convert_camera_images_to_jpeg": "yes"}})
    with pytest.raises(ValueError, match="camera_jpeg_quality"):
        runtime_config({"runtime": {"camera_jpeg_quality": 96}})


@pytest.mark.parametrize("name", ["model", "effort", "codex_bin"])
def test_runtime_config_rejects_agent_specific_fields(name: str) -> None:
    with pytest.raises(ValueError, match="configs/agents"):
        runtime_config({"runtime": {name: "legacy"}})
