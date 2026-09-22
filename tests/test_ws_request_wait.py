import asyncio
import sys
from pathlib import Path

import pytest


XPOLICYLAB = Path(__file__).resolve().parents[1] / "third_party" / "XPolicyLab"
if str(XPOLICYLAB) not in sys.path:
    sys.path.insert(0, str(XPOLICYLAB))

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame


class _Socket:
    async def send(self, _data, text=None):
        return None


def _reply(request_id: str) -> Frame:
    return Frame(
        message_type=MessageType.HEARTBEAT_ACK,
        request_id=request_id,
        evaluation_id="eval",
        payload={"ok": True},
    )


def test_request_warns_but_waits_without_hard_deadline(capsys):
    async def run():
        client = PolicyEvalClient(
            PolicyEvalClientConfig(
                url="ws://localhost:1",
                evaluation_id="eval",
                request_warning_s=0.01,
            )
        )
        client._ws = _Socket()

        async def finish_request():
            await asyncio.sleep(0.03)
            request_id = next(iter(client._pending))
            client._pending[request_id].set_result(_reply(request_id))

        asyncio.create_task(finish_request())
        return await client.request(MessageType.HEARTBEAT, {})

    response = asyncio.run(run())
    assert response.message_type is MessageType.HEARTBEAT_ACK
    assert "continuing to wait" in capsys.readouterr().out


def test_legacy_request_deadline_is_only_a_warning_threshold(capsys):
    async def run():
        client = PolicyEvalClient(
            PolicyEvalClientConfig(
                url="ws://localhost:1",
                evaluation_id="eval",
                request_timeout_s=0.01,
                request_warning_s=None,
            )
        )
        client._ws = _Socket()

        async def finish_request():
            await asyncio.sleep(0.03)
            request_id = next(iter(client._pending))
            client._pending[request_id].set_result(_reply(request_id))

        asyncio.create_task(finish_request())
        return await client.request(MessageType.HEARTBEAT, {})

    response = asyncio.run(run())
    assert response.message_type is MessageType.HEARTBEAT_ACK
    assert "continuing to wait" in capsys.readouterr().out


def test_hello_handshake_timeout_is_also_only_a_warning(capsys):
    async def run():
        client = PolicyEvalClient(
            PolicyEvalClientConfig(
                url="ws://localhost:1",
                evaluation_id="eval",
                handshake_timeout_s=0.01,
            )
        )
        client._ws = _Socket()

        async def finish_request():
            await asyncio.sleep(0.03)
            request_id = next(iter(client._pending))
            client._pending[request_id].set_result(
                Frame(
                    message_type=MessageType.HELLO_ACK,
                    request_id=request_id,
                    evaluation_id="eval",
                    payload={"ok": True},
                )
            )

        asyncio.create_task(finish_request())
        return await client.hello()

    response = asyncio.run(run())
    assert response.message_type is MessageType.HELLO_ACK
    assert "continuing to wait" in capsys.readouterr().out


def test_request_timeout_can_be_unbounded():
    config = PolicyEvalClientConfig(url="ws://localhost:1", evaluation_id="eval")
    assert config.request_timeout_s is None
    assert config.request_warning_s == 120.0
