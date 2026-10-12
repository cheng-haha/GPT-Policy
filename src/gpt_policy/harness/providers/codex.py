"""One native Codex conversation per robot run, including retries."""

import json
import logging
import time
from functools import partial

from ..codex import CodexAppServer
from ..config import AgentConfig
from ..errors import AgentRecoverableError
from ..models import AgentContext, AgentDecision, AgentTurn
from ..waiting import wait_with_health

_LOG = logging.getLogger(__name__)
RECOVERY_DELAYS_S = (5, 15, 30)


class CodexSession:
    def __init__(
        self, config: AgentConfig, model: str, convert_images: bool, jpeg_quality: int
    ) -> None:
        if config.live_image_window is not None:
            raise ValueError("live_image_window is disabled; must be null")
        self._new_client = partial(
            CodexAppServer,
            model, config.executable, config.effort, convert_images, jpeg_quality,
            **({"decision_only": True} if config.decision_only else {}),
        )
        self.client = self._new_client()
        self.context: AgentContext | None = None
        self.thread_id: str | None = None
        self.last_recovery = []
        self.last_context_refresh = None
        self.last_decision_timing = None

    def start(self, context: AgentContext) -> None:
        self.client.start_thread(context.instructions, context.tools)
        self.thread_id = self.client.thread_id
        self.context = context

    def decide(self, turn: AgentTurn) -> AgentDecision:
        """Retry the pending decision in the same thread without replaying actions."""
        if self.context is None or self.thread_id is None:
            raise RuntimeError("Agent session has not been started")
        self.last_recovery = []
        self.last_decision_timing = {}
        started = time.perf_counter()
        try:
            for attempt in range(len(RECOVERY_DELAYS_S) + 1):
                try:
                    return self._decide_once(turn)
                except AgentRecoverableError as error:
                    disconnected = error.code == "appServerDisconnected"
                    if disconnected and self.client is not None:
                        self.client.close()
                        self.client = None
                    delay = RECOVERY_DELAYS_S[attempt] if attempt < len(RECOVERY_DELAYS_S) else None
                    record = {
                        "attempt": attempt + 1, "code": error.code,
                        "error": str(error), "thread_id": self.thread_id,
                        "strategy": "native_resume" if self.client is None else "same_thread",
                        "delay_s": delay, "retry_scheduled": delay is not None,
                        "action_replayed": False,
                    }
                    self.last_recovery.append(record)
                    _LOG.warning("Model decision retry: %s", json.dumps(record, ensure_ascii=False))
                    if delay is None:
                        raise
                    wait_with_health(delay)
        finally:
            self.last_decision_timing["recovery_failures"] = len(self.last_recovery)
            self.last_decision_timing["total_decide_s"] = time.perf_counter() - started

    def _decide_once(self, turn: AgentTurn) -> AgentDecision:
        if self.client is None:
            self.client = self._new_client()
            try:
                # Native persistence restores this exact thread. Never substitute
                # a new conversation or reconstruct its history in the host.
                self.client.resume_thread(self.thread_id)
            except BaseException:
                self.client.close()
                self.client = None
                raise
        started = time.perf_counter()
        try:
            return self.client.decide(
                turn.observation, self.context.output_schema, turn.images, turn.content,
            )
        finally:
            self.last_decision_timing["client_decide_s"] = time.perf_counter() - started
            timing = getattr(self.client, "last_request_timing", None)
            if isinstance(timing, dict):
                self.last_decision_timing.update(timing)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
