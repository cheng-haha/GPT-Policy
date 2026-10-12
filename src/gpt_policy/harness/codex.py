"""Small client for Codex's native app-server harness."""

from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ..hardware.camera import CapturedImage
from ..input.manifest import ContentPart
from .codex_errors import codex_failure
from .errors import AgentRecoverableError
from .input_content import to_app_server_items, validate_input_size
from .usage import model_call
from .waiting import receive_event


class CodexAppServer:
    """Keep one native Codex process and one conversation for one robot run."""

    def __init__(
        self,
        model: str,
        codex_bin: str = "codex",
        effort: str = "high",
        convert_camera_images_to_jpeg: bool = False,
        camera_jpeg_quality: int = 85,
        *,
        decision_only: bool = False,
    ) -> None:
        executable = shutil.which(codex_bin)
        if executable is None:
            raise RuntimeError(f"找不到 Codex 可执行文件: {codex_bin}")
        if not 1 <= camera_jpeg_quality <= 95:
            raise ValueError("camera_jpeg_quality 必须在 1 到 95 之间")
        self.model = model
        self.decision_only = decision_only
        self.decision_only_overrides = {
            "web_search": "disabled", "project_doc_max_bytes": 0,
            **{f"features.{name}": False for name in (
                "shell_tool", "unified_exec", "view_image", "browser_use",
                "browser_use_external", "computer_use", "apps", "plugins",
                "multi_agent", "code_mode", "code_mode_host", "skill_search",
                "workspace_dependencies", "hooks", "memories", "goals",
            )},
        }
        self.effort = effort
        self.convert_camera_images_to_jpeg = convert_camera_images_to_jpeg
        self.camera_jpeg_quality = camera_jpeg_quality
        self.project_root = Path(__file__).resolve().parents[3]
        command = [executable]
        if decision_only:
            for key, value in self.decision_only_overrides.items():
                command.extend(["-c", key + "=" + json.dumps(value)])
        command.append("app-server")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self._stdin = self.process.stdin
        self._events: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self._next_id = 0
        self.thread_id: str | None = None
        threading.Thread(target=self._read_events, args=(self.process.stdout,), daemon=True).start()
        init = {"clientInfo": {"name": "gpt-policy", "version": "0.1.0"}}
        if decision_only:
            # Required by the installed protocol for environments=[].
            init["capabilities"] = {"experimentalApi": True}
        try:
            self._request("initialize", init)
            self._send({"method": "initialized", "params": {}})
        except BaseException:
            self.close()
            raise

    def start_thread(self, instructions: str, tools: list[dict[str, Any]] | None = None) -> None:
        """Start a native Codex thread with a GPT-Policy-shaped tool catalog."""
        if tools:
            instructions = instructions + "\n\nRobot tool catalog:\n" + json.dumps(
                tools, ensure_ascii=False
            )
        settings = self._thread_settings()
        if getattr(self, "decision_only", False):
            settings.update(environments=[], selectedCapabilityRoots=[], dynamicTools=[])
        result = self._request(
            "thread/start",
            {
                **settings,
                "ephemeral": False,
                "serviceName": "gpt-policy",
                "baseInstructions": instructions,
            },
        )
        self.thread_id = str(result["thread"]["id"])

    def resume_thread(self, thread_id: str) -> None:
        """Restore the same native conversation after losing the local process."""
        result = self._request(
            "thread/resume",
            {**self._thread_settings(), "threadId": thread_id, "excludeTurns": True},
        )
        if result["thread"]["id"] != thread_id:
            raise RuntimeError("Codex resumed a different thread; refusing to continue")
        self.thread_id = thread_id

    def _thread_settings(self) -> dict[str, Any]:
        settings = {
            "model": self.model,
            "cwd": str(self.project_root),
            "approvalPolicy": "never",
            "sandbox": "read-only",
        }
        if getattr(self, "decision_only", False):
            overrides = dict(self.decision_only_overrides)
            # Disable inherited MCP entries by name without copying credentials.
            effective = self._request("config/read", {"includeLayers": False})
            for name in effective.get("config", {}).get("mcp_servers", {}):
                overrides[f"mcp_servers.{name}.enabled"] = False
            settings.update(runtimeWorkspaceRoots=[], config=overrides)
        return settings

    def decide(
        self,
        observation: str,
        output_schema: dict[str, Any],
        images: Mapping[str, CapturedImage] | None = None,
        content: tuple[ContentPart, ...] | None = None,
    ) -> dict[str, Any]:
        """Ask Codex for one structured tool selection."""
        self.last_request_timing = {}
        self._request_started_at = None
        with model_call("codex", self.model) as call:
            self._usage_call = call
            self._pending_usage = {}
            try:
                return self._decide(observation, output_schema, images, content)
            finally:
                if self._request_started_at is not None:
                    # Client round trip, including transport, provider wait and
                    # response parsing; never a server-only inference measure.
                    self.last_request_timing["request_elapsed_s"] = (
                        time.perf_counter() - self._request_started_at
                    )
                self._usage_call = None
                self._pending_usage = {}

    def _decide(self, observation, output_schema, images=None, content=None):
        prepare_started = time.perf_counter()
        if self.thread_id is None:
            raise RuntimeError("Codex thread has not been started")
        validate_input_size(tuple(content or ()), observation, images or ())
        self._usage_call.update(request_started=True, thread_id=self.thread_id)
        inputs = self._input(
            observation, images, content,
            self.convert_camera_images_to_jpeg, self.camera_jpeg_quality,
        )
        self.last_request_timing["input_prepare_s"] = time.perf_counter() - prepare_started
        self._request_started_at = time.perf_counter()
        result = self._request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "input": inputs,
                "outputSchema": output_schema,
                "effort": self.effort,
            },
        )
        turn_id = str(result["turn"]["id"])
        self._usage_call["turn_id"] = turn_id
        self._usage_call["raw_usage"] = self._pending_usage.get(turn_id)
        answer: str | None = None
        while True:
            event = self._receive()
            self._record_usage(event)
            params = event.get("params", {})
            if params.get("threadId") != self.thread_id:
                continue
            if params.get("turnId", turn_id) != turn_id:
                continue
            if event.get("method") == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage" and item.get("phase") in {
                    None,
                    "final_answer",
                }:
                    answer = str(item.get("text", ""))
            if event.get("method") == "turn/completed":
                turn = params.get("turn", {})
                if str(turn.get("id")) != turn_id:
                    continue
                self._usage_call["usage_final"] = True
                if turn.get("status") != "completed":
                    error = turn.get("error")
                    message = f"Codex turn failed: {error}"
                    if turn.get("status") == "failed":
                        raise codex_failure(error)
                    raise RuntimeError(message)
                if answer is None:
                    raise RuntimeError("Codex 没有返回工具选择")
                selection = json.loads(answer)
                if not isinstance(selection, dict):
                    raise RuntimeError("Codex 工具选择不是 JSON 对象")
                arguments_value = selection.get("arguments")
                # New protocol uses a native object. Accept the old encoded
                # string during migration so an interrupted/older run can be
                # resumed without changing the robot layer.
                if isinstance(arguments_value, str):
                    arguments = json.loads(arguments_value)
                else:
                    arguments = arguments_value
                if not isinstance(arguments, dict):
                    raise RuntimeError("Codex arguments 必须是对象")
                return {
                    "name": selection.get("name"),
                    "arguments": arguments,
                    "_wire": selection,
                }

    @staticmethod
    def _input(
        observation: str,
        images: Mapping[str, CapturedImage] | None,
        content: tuple[ContentPart, ...] | None = None,
        convert_camera_images_to_jpeg: bool = False,
        camera_jpeg_quality: int = 85,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if content:
            items.extend(to_app_server_items(content))
        items.append({"type": "text", "text": observation})
        for name, image in (images or {}).items():
            items.append({"type": "text", "text": f"Camera image: {name}"})
            items.append(
                {
                    "type": "image",
                    "url": _camera_data_url(
                        image, convert_camera_images_to_jpeg, camera_jpeg_quality
                    ),
                }
            )
        return items

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                process_group = process.pid
                signal_name = signal.SIGTERM
                os.killpg(process_group, signal_name)
            except (OSError, ProcessLookupError):
                process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            process.wait(timeout=2)

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"id": request_id, "method": method, "params": params})
        while True:
            message = self._receive()
            self._record_usage(message)
            if message.get("id") != request_id:
                self._reject_unexpected_request(message)
                continue
            if "error" in message:
                raise codex_failure(message["error"], method)
            result = message.get("result", {})
            if not isinstance(result, dict):
                raise RuntimeError(f"Codex {method} returned an invalid result")  # noqa: TRY004 - wire protocol failure
            return result

    def _record_usage(self, event):
        call = getattr(self, "_usage_call", None)
        if call is None or event.get("method") != "thread/tokenUsage/updated":
            return
        params = event.get("params", {})
        if params.get("threadId") != self.thread_id or not params.get("turnId"):
            return
        turn_id = str(params["turnId"])
        if call.get("turn_id") not in (None, turn_id):
            return
        last = params.get("tokenUsage", {}).get("last")
        if isinstance(last, dict):
            # Notifications can precede the turn/start response. Keep LAST for
            # this turn, never add cumulative thread totals or duplicate updates.
            self._pending_usage[turn_id] = last
            if call.get("turn_id") == turn_id:
                call["raw_usage"] = last

    def _send(self, message: dict[str, Any]) -> None:
        if self.process is None:
            raise RuntimeError("Codex 已关闭")
        try:
            self._stdin.write(json.dumps(message) + "\n")
            self._stdin.flush()
        except OSError as error:
            raise AgentRecoverableError(
                f"Codex app-server write failed: {error}",
                provider="codex", code="appServerDisconnected",
            ) from error

    def _receive(self) -> dict[str, Any]:
        message = receive_event(self._events)
        if isinstance(message, (EOFError, OSError)):
            raise AgentRecoverableError(
                f"Codex app-server disconnected: {message}",
                provider="codex", code="appServerDisconnected",
            ) from message
        if isinstance(message, Exception):
            raise RuntimeError(f"Codex app-server 已停止: {message}") from message  # noqa: TRY004 - transport failure
        if getattr(self, "decision_only", False) and message.get("method") in {
            "item/started", "item/completed",
        }:
            kind = message.get("params", {}).get("item", {}).get("type")
            if kind not in {"userMessage", "agentMessage", "reasoning", "contextCompaction"}:
                raise RuntimeError(f"Decision-only session rejected unexpected item type: {kind}")
        return message

    def _reject_unexpected_request(self, message: dict[str, Any]) -> None:
        if "id" in message and "method" in message:
            self._send(
                {
                    "id": message["id"],
                    "error": {"code": -32601, "message": "Only robot decisions are supported"},
                }
            )
            raise RuntimeError(f"Codex 请求了未启用的交互: {message['method']}")

    def _read_events(self, stream: Any) -> None:
        try:
            for line in stream:
                if line.strip():
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("JSON-RPC message must be an object")
                    self._events.put(message)
        except Exception as exc:  # noqa: BLE001 - forward reader failures to the caller
            self._events.put(exc)
        finally:
            # A clean child exit used to leave _receive blocked on an empty queue.
            self._events.put(EOFError("Codex app-server stdout closed"))


def _camera_data_url(image: CapturedImage, convert_to_jpeg: bool, quality: int) -> str:
    """Encode only live camera inputs as JPEG while preserving their pixel dimensions."""
    if not convert_to_jpeg or image.mime_type == "image/jpeg":
        return image.data_url()

    if image.rgb_data is not None:
        rgb = Image.frombytes("RGB", (image.width, image.height), image.rgb_data)
    else:
        with Image.open(BytesIO(image.data)) as source:
            rgb = source.convert("RGB")

    with BytesIO() as stream:
        rgb.save(stream, format="JPEG", quality=quality)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
