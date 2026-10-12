"""Model recovery must never replay already executed host/robot commands."""

import io
import queue
from unittest.mock import Mock

import pytest

from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.codex_errors import codex_failure
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentRecoverableError, AgentUsageLimitError
from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.providers import codex as provider

COMPACT_BAD_REQUEST = 'Error running remote compact task: {"detail":"Bad Request"}'


@pytest.mark.parametrize(
    "info,message",
    [
        ("other", '{"detail":"Bad Request"}'),
        ("badRequest", "Bad Request"),
        ("other", COMPACT_BAD_REQUEST),
        ("rateLimitExceeded", "rate limit"),
        ("internalServerError", "backend error"),
        ({"httpConnectionFailed": {"httpStatusCode": 503}}, "unavailable"),
        ({"responseStreamDisconnected": {"httpStatusCode": None}}, "connection lost"),
        (
            {"responseStreamDisconnected": {"httpStatusCode": 200}},
            "stream lost after HTTP success",
        ),
        ({"responseTooManyFailedAttempts": {"httpStatusCode": 502}}, "gateway error"),
    ],
)
def test_recoverable_protocol_errors(info, message):
    assert isinstance(
        codex_failure({"codexErrorInfo": info, "message": message}),
        AgentRecoverableError,
    )


@pytest.mark.parametrize(
    "error",
    [
        {"codexErrorInfo": "badRequest", "message": "unsupported parameter: schema"},
        {"code": -32602, "message": "Bad Request"},
        {
            "codexErrorInfo": "other",
            "message": "Bad Request",
            "additionalDetails": "invalid model",
        },
        {"codexErrorInfo": "unauthorized", "message": "Bad Request"},
        {"codexErrorInfo": "contextWindowExceeded", "message": "Bad Request"},
        {"codexErrorInfo": "misalignmentPolicyViolation", "message": "Bad Request"},
        {
            "codexErrorInfo": "other",
            "message": "Bad Request",
            "misalignment": {"errorType": "policy"},
        },
        {
            "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 403}},
            "message": "forbidden",
        },
        {
            "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 400}},
            "message": "invalid request",
        },
        {"codexErrorInfo": "contextWindowExceeded", "message": COMPACT_BAD_REQUEST},
        {"codexErrorInfo": "unauthorized", "message": COMPACT_BAD_REQUEST},
        {
            "codexErrorInfo": "other", "message": COMPACT_BAD_REQUEST,
            "additionalDetails": "invalid model",
        },
        {
            "codexErrorInfo": "other",
            "message": 'Error running remote compact task: {"detail":"unsupported model"}',
        },
        {"message": "unknown error"},
    ],
)
def test_input_credentials_and_policy_errors_are_not_retried(error):
    assert not isinstance(codex_failure(error), AgentRecoverableError)


@pytest.mark.parametrize("code", ["usageLimitExceeded", "sessionBudgetExceeded"])
def test_quota_never_reconnects_or_resets_budget(code):
    assert isinstance(codex_failure({"codexErrorInfo": code}), AgentUsageLimitError)


def test_failed_turn_discards_even_a_complete_looking_action():
    client = object.__new__(CodexAppServer)
    client.model, client.effort, client.thread_id = "test", "medium", "thread"
    client.convert_camera_images_to_jpeg, client.camera_jpeg_quality = False, 85
    client._request = Mock(return_value={"turn": {"id": "turn"}})
    events = iter(
        [
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn",
                    "item": {
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": '{"name":"move_eef_chunk","arguments":{}}',
                    },
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread",
                    "turn": {
                        "id": "turn",
                        "status": "failed",
                        "error": {
                            "codexErrorInfo": "other",
                            "message": '{"detail":"Bad Request"}',
                        },
                    },
                },
            },
        ]
    )
    client._receive = lambda: next(events)
    with pytest.raises(AgentRecoverableError):
        client.decide("current", {})


def test_clean_app_server_eof_is_observable_not_an_infinite_wait():
    client = object.__new__(CodexAppServer)
    client._events = queue.Queue()
    client._read_events(io.StringIO(""))
    with pytest.raises(AgentRecoverableError, match="stdout closed"):
        client._receive()


def test_broken_pipe_reconnects_but_corrupt_local_protocol_does_not():
    client = object.__new__(CodexAppServer)
    client.process = object()
    client._stdin = Mock()
    client._stdin.write.side_effect = BrokenPipeError("pipe closed")
    with pytest.raises(AgentRecoverableError):
        client._send({})
    client._events = queue.Queue()
    client._read_events(io.StringIO("not JSON\n"))
    with pytest.raises(RuntimeError) as error:
        client._receive()
    assert not isinstance(error.value, AgentRecoverableError)


def fake_native(monkeypatch, outcomes):
    clients, requests, waits = [], [], []

    class Native:
        def __init__(self, *args, **kwargs):
            self.closed, self.thread_id = False, None
            self.constructor = (args, kwargs)
            self.starts, self.resumes = [], []
            clients.append(self)

        def start_thread(self, instructions, tools):
            self.starts.append((instructions, tools))
            self.thread_id = "original-thread"

        def resume_thread(self, thread_id):
            self.resumes.append(thread_id)
            self.thread_id = thread_id

        def close(self):
            self.closed = True

        def decide(self, observation, schema, images, content):
            assert not self.closed
            requests.append((self.thread_id, observation, images, content))
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            wire = {"name": "done", "arguments": {"note": outcome}}
            return {**wire, "_wire": wire}

    monkeypatch.setattr(provider, "CodexAppServer", Native)
    monkeypatch.setattr(provider, "wait_with_health", waits.append)
    return clients, requests, waits


def retry_error():
    return codex_failure({"codexErrorInfo": "other", "message": '{"detail":"Bad Request"}'})


def disconnect():
    return AgentRecoverableError("pipe lost", provider="codex", code="appServerDisconnected")


@pytest.mark.parametrize("prior_decisions", [0, 3])
@pytest.mark.parametrize("message", ['{"detail":"Bad Request"}', COMPACT_BAD_REQUEST])
def test_bad_request_retries_current_input_in_the_original_thread(
    monkeypatch, prior_decisions, message,
):
    from gpt_policy.input.manifest import TextPart

    error = codex_failure({"codexErrorInfo": "other", "message": message})
    outcomes = ["ok"] * prior_decisions + [error, error, "recovered", "next"]
    clients, requests, waits = fake_native(monkeypatch, outcomes)
    session = provider.CodexSession(AgentConfig(decision_only=True), "test", False, 85)
    session.start(AgentContext("same instructions", [], {}))
    demo = (TextPart("fixed ICL"),)
    for i in range(prior_decisions):
        session.decide(AgentTurn(f"ack {i}", content=demo if i == 0 else None))
    turn = AgentTurn("pending snapshot", content=demo if prior_decisions == 0 else None)
    session.decide(turn)
    assert len(clients) == 1 and not clients[0].closed
    assert len(clients[0].starts) == 1 and not clients[0].resumes
    assert requests[-1] == requests[-2] == requests[-3]
    assert waits == [5, 15] and len(session.last_recovery) == 2
    assert all(x["strategy"] == "same_thread" for x in session.last_recovery)
    assert not hasattr(session, "_history")
    session.decide(AgentTurn("next snapshot"))
    assert session.last_recovery == []


def test_process_loss_resumes_same_id_without_resending_expert_context(monkeypatch):
    from gpt_policy.input.manifest import TextPart

    clients, requests, waits = fake_native(monkeypatch, ["ok", disconnect(), "recovered"])
    session = provider.CodexSession(AgentConfig(decision_only=True), "test", False, 85)
    session.start(AgentContext("same instructions", [], {}))
    session.decide(AgentTurn("past", content=(TextPart("fixed ICL"),)))
    session.decide(AgentTurn("pending snapshot"))
    assert len(clients) == 2 and clients[0].closed
    assert clients[1].starts == [] and clients[1].resumes == ["original-thread"]
    assert requests[-1] == requests[-2] == ("original-thread", "pending snapshot", None, None)
    assert waits == [5] and session.last_recovery[0]["strategy"] == "native_resume"
    assert all(c.constructor[1]["decision_only"] for c in clients)


def test_persistent_failure_has_finite_retries(monkeypatch):
    clients, requests, waits = fake_native(monkeypatch, [retry_error() for _ in range(4)])
    session = provider.CodexSession(AgentConfig(), "test", False, 85)
    session.start(AgentContext("same", [], {}))
    with pytest.raises(AgentRecoverableError):
        session.decide(AgentTurn("same observation"))
    assert len(requests) == 4 and waits == [5, 15, 30]
    assert len(clients) == 1 and clients[0].thread_id == "original-thread"
    assert session.last_recovery[-1]["retry_scheduled"] is False


@pytest.mark.parametrize("error", [
    AgentUsageLimitError("quota", provider="codex", code="usageLimitExceeded"),
    KeyboardInterrupt(),
])
def test_quota_and_manual_interrupt_do_not_trigger_reconnection(monkeypatch, error):
    clients, _requests, waits = fake_native(monkeypatch, [error])
    session = provider.CodexSession(AgentConfig(), "test", False, 85)
    session.start(AgentContext("same", [], {}))
    with pytest.raises(type(error)):
        session.decide(AgentTurn("current"))
    assert len(clients) == 1 and waits == []


@pytest.mark.parametrize("method", ["turn/start", "thread/resume", "initialize"])
def test_rpc_errors_use_the_same_recovery_classification(method):
    client = object.__new__(CodexAppServer)
    client._next_id = 0
    client._send = Mock()
    client._receive = lambda: {
        "id": 1,
        "error": {"codexErrorInfo": "other", "message": '{"detail":"Bad Request"}'},
    }
    with pytest.raises(AgentRecoverableError):
        client._request(method, {})


@pytest.mark.parametrize("failure", [retry_error(), RuntimeError("native thread unavailable")])
def test_resume_failure_never_falls_back_to_a_new_conversation(monkeypatch, failure):
    clients, requests, waits = fake_native(monkeypatch, [disconnect(), "recovered"])
    session = provider.CodexSession(AgentConfig(), "test", False, 85)
    session.start(AgentContext("unchanged", [], {}))
    factory = session._new_client

    def new_client():
        client = factory()
        if len(clients) == 2:
            client.resume_thread = Mock(side_effect=failure)
        return client

    session._new_client = new_client
    if isinstance(failure, AgentRecoverableError):
        session.decide(AgentTurn("pending"))
        assert waits == [5, 15] and len(requests) == 2
        assert clients[-1].resumes == ["original-thread"]
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            session.decide(AgentTurn("pending"))
        assert waits == [5] and len(requests) == 1
    assert all(not c.starts for c in clients[1:])
    assert clients[1].closed


def test_backoff_honors_original_host_deadline(monkeypatch):
    from gpt_policy.harness.waiting import monitor_health, wait_with_health
    from gpt_policy.simulation.robodojo.budget import EpisodeWallTimeExceeded

    clients, requests, _waits = fake_native(monkeypatch, [retry_error()])
    session = provider.CodexSession(AgentConfig(), "test", False, 85)
    session.start(AgentContext("unchanged", [], {}))
    monkeypatch.setattr(provider, "wait_with_health", wait_with_health)

    def health():
        if requests:
            raise EpisodeWallTimeExceeded("original episode deadline")

    with monitor_health(health), pytest.raises(EpisodeWallTimeExceeded):
        session.decide(AgentTurn("current"))
    assert len(clients) == 1 and len(requests) == 1

def test_same_thread_retry_ignores_late_messages_from_failed_turn():
    import json

    client = object.__new__(CodexAppServer)
    client.model, client.effort, client.thread_id = "test", "medium", "same-thread"
    client.convert_camera_images_to_jpeg, client.camera_jpeg_quality = False, 85
    client._request = Mock(side_effect=[
        {"turn": {"id": "failed-turn"}}, {"turn": {"id": "retry-turn"}},
    ])

    def answer(turn, note):
        return {"method": "item/completed", "params": {
            "threadId": "same-thread", "turnId": turn,
            "item": {"type": "agentMessage", "phase": "final_answer",
                     "text": json.dumps({"name": "move_eef_chunk", "arguments": {"note": note}})},
        }}

    def completed(turn, status):
        return {"method": "turn/completed", "params": {
            "threadId": "same-thread", "turn": {"id": turn, "status": status,
                "error": {"codexErrorInfo": "other", "message": "Bad Request"}},
        }}

    events = iter([
        answer("failed-turn", "discard partial action"),
        completed("failed-turn", "failed"),
        answer("failed-turn", "discard delayed old action"),
        completed("failed-turn", "completed"),
        answer("retry-turn", "accept current completed action"),
        completed("retry-turn", "completed"),
    ])
    client._receive = lambda: next(events)
    with pytest.raises(AgentRecoverableError):
        client.decide("unchanged pending observation", {})
    decision = client.decide("unchanged pending observation", {})
    assert decision["arguments"]["note"] == "accept current completed action"
    assert client.thread_id == "same-thread"
    assert len(client._request.call_args_list) == 2
    assert client._request.call_args_list[0] == client._request.call_args_list[1]
