import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.config import AgentConfig, named_agent_config
from gpt_policy.harness.errors import AgentOverloadedError
from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.providers.codex import CodexSession
from gpt_policy.input.manifest import ImagePart, TextPart


class Native:
    _input = staticmethod(CodexAppServer._input)
    convert_camera_images_to_jpeg = False
    camera_jpeg_quality = 85
    thread_id = "thread-0"

    def __init__(self, *_):
        self.calls = []
        self.starts = []

    def start_thread(self, *args): self.starts.append(args)

    def refresh_thread(self, *args):
        self.thread_id = f"thread-{len(self.starts)}"
        self.start_thread(*args)

    def decide(self, observation, schema, images, content, **kwargs):
        self.calls.append({"observation": observation, **kwargs})
        wire = {"name": "move_to", "arguments": {"note": f"decision-{len(self.calls)}"}}
        return {**wire, "_wire": wire}

    def close(self): pass


@pytest.mark.parametrize("window,refreshes", [(8, [8, 15]), (3, list(range(3, 16, 2))), (None, [])])
def test_refresh_keeps_demo_all_text_feedback_and_decisions_but_only_two_live_groups(tmp_path, window, refreshes):
    demo_image = tmp_path / "demo.jpg"
    demo_image.write_bytes(b"historical")
    context = AgentContext("original instructions", [{"tool": "move_to"}], {"type": "object"})
    with patch("gpt_policy.harness.providers.codex.CodexAppServer", Native):
        session = CodexSession(AgentConfig(live_image_window=window), "test", False, 85)
    session.start(context)
    static = (TextPart("historical demonstration"), ImagePart(demo_image, "demo"))
    for step in range(16):
        images = {"top": CapturedImage("top", f"live-{step}".encode(), "image/jpeg", 1, 1, step)}
        session.decide(AgentTurn(f"observation-{step}; tool-result-{step-1}", images, static if step == 0 else None))
        call = session.client.calls[-1]
        assert ("replay" in call) == (step in refreshes)
        if step in refreshes:
            replay = call["replay"]
            text = "\n".join(i["text"] for i in replay if i["type"] == "text")
            for previous in range(step):
                assert f"observation-{previous};" in text
                assert f"decision-{previous+1}" in text
            assert f"tool-result-{step-2}" in text
            assert "not a command" in text and "historical demonstration" in text
            urls = [i["url"] for i in replay if i["type"] == "image"]
            assert len(urls) == 2  # one demo plus one previous live group; current is sent separately
            assert images["top"].data_url() not in urls
            previous_image = CapturedImage("top", f"live-{step-1}".encode(), "image/jpeg", 1, 1, 0)
            assert previous_image.data_url() in urls
            assert session.last_context_refresh["retained_live_groups"] == 2
            assert session.last_context_refresh["decisions"] == step + 1
    assert len(session.client.calls) == 16
    assert len(session.client.starts) == 1 + len(refreshes)
    assert all(args == (context.instructions, context.tools) for args in session.client.starts)
    assert sum(bool(t.images) for t, _ in session._history) == 1


@pytest.mark.parametrize("window", [2, True, 3.5, "8"])
def test_invalid_live_window_rejected_by_configuration(tmp_path, window):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents/codex.json").write_text(json.dumps({"live_image_window": window}))
    with pytest.raises(ValueError, match="integer >= 3"):
        named_agent_config("codex", tmp_path)


def test_native_replay_is_size_checked_before_turn_start():
    from test_codex_regression import client
    native = client()
    with pytest.raises(ValueError, match="too large"):
        native.decide("current", {}, replay=[{"type": "text", "text": "x" * 1048576}])
    native._request.assert_not_called()


def test_context_timing_separates_replay_refresh_and_client_and_resets(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("gpt_policy.harness.providers.codex.time.perf_counter", lambda: clock[0])
    with patch("gpt_policy.harness.providers.codex.CodexAppServer", Native):
        session = CodexSession(AgentConfig(live_image_window=3), "test", False, 85)
    session.start(AgentContext("instructions", [], {}))
    original_input, original_refresh, original_decide = (
        session.client._input, session.client.refresh_thread, session.client.decide,
    )

    def inputs(*args):
        clock[0] += 2
        return original_input(*args)

    def refresh(*args):
        clock[0] += 3
        return original_refresh(*args)

    def decide(*args, **kwargs):
        clock[0] += 5
        session.client.last_request_timing = {"input_prepare_s": 1.0, "request_elapsed_s": 4.0}
        return original_decide(*args, **kwargs)

    session.client._input = inputs
    session.client.refresh_thread = refresh
    session.client.decide = decide
    for step in range(5):
        images = {"top": CapturedImage("top", b"frame", "image/jpeg", 1, 1, step)}
        session.decide(AgentTurn(f"observation-{step}", images))
        assert session.last_decision_timing == {
            "context_replay_s": 6.0 if step == 3 else 0.0,
            "thread_refresh_s": 3.0 if step == 3 else 0.0,
            "client_decide_s": 5.0, "input_prepare_s": 1.0, "request_elapsed_s": 4.0,
        }


@pytest.mark.parametrize('previous_steps', [0, 1, 3])
@pytest.mark.parametrize('window', [None, 3])
def test_overload_rebuilds_only_successful_history_with_fresh_observation(previous_steps, window):
    with patch('gpt_policy.harness.providers.codex.CodexAppServer', Native):
        session = CodexSession(AgentConfig(live_image_window=window), 'test', False, 85)
    session.start(AgentContext('same instructions', [], {}))
    static = (TextPart('historical demonstration'),)
    for step in range(previous_steps):
        session.decide(AgentTurn(f'success-{step}', content=static if step == 0 else None))
    history = list(session._history)
    decide = session.client.decide

    def overloaded(*args, **kwargs):
        decide(*args, **kwargs)
        raise AgentOverloadedError('at capacity', provider='codex', code='serverOverloaded')

    session.client.decide = overloaded
    for i in range(2):
        with pytest.raises(AgentOverloadedError):
            session.decide(AgentTurn(f'failed-{i}', content=static if not previous_steps else None))
        assert session._history == history
    session.client.decide = decide
    session.decide(AgentTurn('fresh live observation', content=static if not previous_steps else None))
    assert len(session._history) == previous_steps + 1
    assert len(session.client.starts) == 3
    replay = session.client.calls[-1]['replay']
    text = '\n'.join(item['text'] for item in replay if item['type']=='text')
    assert 'failed-0' not in text and 'failed-1' not in text
    assert session.client.calls[-1]['observation'] == 'fresh live observation'
    for step in range(previous_steps):
        assert f'success-{step}' in text
    if previous_steps:
        assert text.count('historical demonstration') == 1
    assert all(args == ('same instructions', []) for args in session.client.starts)
    session.decide(AgentTurn('next observation'))
    assert 'replay' not in session.client.calls[-1]
