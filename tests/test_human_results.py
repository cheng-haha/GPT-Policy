"""Final human labels use fake devices and never request a model or motion."""

import json
from io import StringIO
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

import gpt_policy.main as app
from gpt_policy.harness.config import AgentConfig
from gpt_policy.recording.trace import RunRecorder
from gpt_policy.runtime.console import RunConsole


@pytest.mark.parametrize("ending", ["done", "give_up", "budget", "home_error"])
@pytest.mark.parametrize("vote", ["s", "f", None, "ctrl_c"])
@pytest.mark.parametrize("task", ["insert plug", "unscrew and remove the bottle cap"])
def test_final_label_after_shutdown_is_human_owned(tmp_path, monkeypatch, ending, vote, task):
    timeline = []
    root = tmp_path / "run"
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"instruction": task, "content": [task]}))
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--input-json", str(request)])
    monkeypatch.setattr(app, "load_settings", lambda *_: {
        "runtime": {"right_interface": "", "record_dir": str(root), "max_decisions": 1,
                    "input_json": str(request)},
    })
    monkeypatch.setattr(app, "agent_config", lambda *_: AgentConfig(model="test"))
    monkeypatch.setattr(app, "preflight_agent", lambda *_: None)
    monkeypatch.setattr(app, "warm_up_cameras", lambda *_: {})
    monkeypatch.setattr(app, "camera_defaults", lambda *_: [])
    monkeypatch.setattr(app, "observation", lambda *_: "{}")
    monkeypatch.setattr(app, "PixelLocalizer", lambda *_: SimpleNamespace())

    def home():
        timeline.append("home")
        if ending == "home_error":
            raise RuntimeError("home failed")
        return {"home": {"settle": {"settled": True}}}

    robot = SimpleNamespace(dof=6, state=lambda: {}, return_home=home,
                            cancel=lambda: timeline.append("cancel"),
                            close=lambda: timeline.append("robot_closed"))
    cameras = SimpleNamespace(describe=lambda *_: [], close=lambda: timeline.append("cameras_closed"))
    video = SimpleNamespace(details={}, snapshot=lambda **_: {},
                            stop=lambda: timeline.append("video_stopped") or {"finalized": True})
    agent = SimpleNamespace(start=Mock(), close=lambda: timeline.append("agent_closed"),
                            decide=Mock(return_value={
                                "name": {"budget": "invalid_tool", "home_error": "done"}.get(ending, ending),
                                "arguments": {"summary": "model says finished"},
                            }))
    monkeypatch.setattr(app, "ArxRobot", lambda *_: robot)
    monkeypatch.setattr(app, "CameraSet", lambda *_: cameras)
    monkeypatch.setattr(app, "RunVideo", lambda *_: video)
    monkeypatch.setattr(app, "create_agent", lambda *_: agent)

    output = StringIO()
    ui = RunConsole(Console(file=output, width=120, no_color=True))
    monkeypatch.setattr(app, "RunConsole", lambda: ui)
    monkeypatch.setattr("sys.stdin.isatty", lambda: vote is not None)

    def answer(*_, **__):
        assert {"video_stopped", "cameras_closed", "robot_closed", "agent_closed"} <= set(timeline)
        assert (root / "transcript.json").exists()
        assert not list(tmp_path.glob("run_*"))
        timeline.append("vote")
        if vote == "ctrl_c":
            signal.raise_signal(signal.SIGINT)
        return vote

    ui.console.input = Mock(side_effect=answer)
    if ending == "home_error":
        with pytest.raises(RuntimeError, match="home failed"):
            app.main()
    else:
        app.main()

    expected = {"s": "success", "f": "failed", None: "unreviewed", "ctrl_c": "unreviewed"}[vote]
    saved = root.with_name(f"run_{expected}")
    status = json.loads((saved / "status.json").read_text())
    events = [json.loads(line) for line in (saved / "events.jsonl").read_text().splitlines()]
    transcript = json.loads((saved / "transcript.json").read_text())
    assert status["human_outcome"] == (expected if vote in {"s", "f"} else None)
    assert status["outcome"] == expected
    assert status["model_outcome"] == {"done": "success", "give_up": "give_up", "budget": None, "home_error": "success"}[ending]
    assert status["state"] == {"done": "completed", "give_up": "give_up", "budget": "budget_exhausted", "home_error": "failed"}[ending]
    assert sum(e["event"] == "human_evaluation" for e in events) == int(vote in {"s", "f"})
    assert sum(t.get("content", {}).get("event") == "human_evaluation" for t in transcript if isinstance(t.get("content"), dict)) == int(vote in {"s", "f"})
    assert events[-1]["event"] == "run_finished"
    assert timeline.count("home") == 1
    agent.decide.assert_called_once()
    agent.start.assert_called_once()
    context = agent.start.call_args.args[0]
    protocol = json.loads((saved / "protocol.json").read_text())
    assert protocol["base_instructions"] == context.instructions
    assert protocol["control_prompt_profile"] == "default"
    if vote is None:
        ui.console.input.assert_not_called()
    if vote != "s":
        assert "TASK · SUCCESS" not in output.getvalue()
    if ending != "home_error":
        assert "FINALIZATION" not in output.getvalue()


def test_empty_and_invalid_input_do_not_default_to_success(monkeypatch):
    ui = RunConsole(Console(file=StringIO()))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ui.console.input = Mock(side_effect=["", "yes", "f"])
    assert ui.task_result() == "failed"
    assert ui.console.input.call_count == 3


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt, OSError])
def test_missing_human_answer_remains_unreviewed(monkeypatch, error):
    ui = RunConsole(Console(file=StringIO()))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ui.console.input = Mock(side_effect=error)
    assert ui.task_result() is None


def test_human_wait_is_excluded_from_execution_duration(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("gpt_policy.recording.trace.time.time", lambda: clock[0])
    recorder = RunRecorder(tmp_path / "run", {})
    recorder.write("terminal", {"name": "done"})
    clock[0] = 20.0
    recorder.write("execution_finished", {"status": "completed"})
    clock[0] = 100.0
    recorder.write("human_evaluation", {"outcome": "failed"})
    saved = recorder.close()
    status = json.loads((saved / "status.json").read_text())
    assert status["elapsed_s"] == 10.0
    assert status["model_outcome"] == "success"
    assert status["human_outcome"] == status["outcome"] == "failed"


def test_invalid_human_label_cannot_be_archived_as_success(tmp_path):
    recorder = RunRecorder(tmp_path / "run", {})
    recorder.write("terminal", {"name": "done"})
    with pytest.raises(ValueError, match="Human outcome"):
        recorder.write("human_evaluation", {"outcome": "completed"})
    saved = recorder.close()
    assert saved.name == "run_unreviewed"
