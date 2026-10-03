#!/usr/bin/env python3
"""Check a configured Claude/Kimi profile with two image turns, without hardware."""

import argparse
import io
import json
import secrets
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from gpt_policy.harness.config import named_agent_config
from gpt_policy.harness.factory import create_agent, preflight_agent
from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.protocol import output_schema, tool_schemas


ROOT = Path(__file__).resolve().parents[1]


def images(colors):
    result = {}
    for name, color in zip(("left", "right", "top"), colors):
        stream = io.BytesIO()
        Image.new("RGB", (96, 96), color).save(stream, format="PNG")
        result[name] = SimpleNamespace(
            data=stream.getvalue(), mime_type="image/png", width=96, height=96, rgb_data=None,
        )
    return result


def check(config):
    """Use the production bimanual schema; inspect decisions without executing them."""
    context = AgentContext(
        instructions=(
            "This is a software protocol test with synthetic images and no robot. "
            "Return only a done selection using StructuredOutput. In arguments.summary, "
            "put a JSON string with current_colors, first_colors and session_word. "
            "Color arrays must follow image order: left, right, top. Choose basic names "
            "from red, green, blue, cyan, magenta, yellow. Remember the first turn's "
            "colors and session_word across turns. arguments.hindsight should describe "
            "the software check. Do not request any motion or other tools."
        ),
        tools=tool_schemas(6, ("left", "right")),
        output_schema=output_schema(6, ("left", "right")),
    )
    word = secrets.token_hex(5)
    turns = [
        AgentTurn(f"First turn. session_word={word}", images(("red", "lime", "blue"))),
        AgentTurn("Second turn. Identify the new colors and recall the original colors and session_word.",
                  images(("cyan", "magenta", "yellow"))),
    ]
    expected_colors = [["red", "green", "blue"], ["cyan", "magenta", "yellow"]]
    session = create_agent(config, config.model, convert_images=True)
    reports = []
    try:
        session.start(context)
        for turn, expected in zip(turns, expected_colors):
            decision = session.decide(turn)
            if decision["name"] != "done":
                raise ValueError(f"Expected done, received {decision['name']}")
            report = json.loads(decision["arguments"]["summary"])
            if (report.get("current_colors") != expected
                    or report.get("first_colors") != expected_colors[0]
                    or report.get("session_word") != word):
                raise ValueError(f"Image/context check failed: {report}")
            reports.append(report)
    finally:
        session.close()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("claude", "kimi"), required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    config = None
    result = {"profile": args.agent, "hardware_opened": False, "ok": False}
    try:
        config = named_agent_config(args.agent, args.config_dir)
        result.update(model=config.model, effort=config.effort, transport=config.type)
        preflight_agent(config)
        result["turns"] = check(config)
        result["ok"] = True
    except Exception as exc:
        message = str(exc)
        for key, value in (config.environment if config else {}).items():
            if "TOKEN" in key or "KEY" in key:
                message = message.replace(value, "<redacted>")
        result["error"] = f"{type(exc).__name__}: {message}"
    result["elapsed_s"] = round(time.monotonic() - started, 2)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
