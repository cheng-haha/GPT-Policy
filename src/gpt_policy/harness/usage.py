"""Record model calls separately from decisions; never feed accounting to the model.

The run-scoped context includes naming before the recording directory exists and
short-lived demonstration clients. Transport readers report counters; this module
alone normalizes and prices them. No network pricing lookup runs on the robot.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

from ..paths import generated_var_path


PRICES = json.loads(Path(__file__).with_name("prices.json").read_text())
_ledger = ContextVar("model_usage_ledger", default=None)
_phase = ContextVar("model_usage_phase", default="decision")
TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
              "cache_write_1h_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens")


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalize_usage(raw, provider):
    if not isinstance(raw, dict) or not raw:
        return None
    if provider == "codex":
        names = dict(zip(TOKEN_KEYS, ("inputTokens", "cachedInputTokens", "cacheWriteInputTokens",
                                     "cacheWrite1hInputTokens", "outputTokens", "reasoningOutputTokens", "totalTokens")))
        result = {key: _count(raw.get(name)) for key, name in names.items()}
    else:
        # Anthropic's input_tokens EXCLUDES cached reads and writes. Normalize
        # to inclusive input_tokens so provider totals have the same meaning.
        reads = _count(raw.get("cache_read_input_tokens"))
        writes = _count(raw.get("cache_creation_input_tokens"))
        creation = raw.get("cache_creation")
        creation = creation if isinstance(creation, dict) else {}
        result = {
            "input_tokens": _count(raw.get("input_tokens")),
            "cached_input_tokens": reads,
            "cache_write_input_tokens": writes,
            "cache_write_1h_input_tokens": _count(creation.get("ephemeral_1h_input_tokens")),
            "output_tokens": _count(raw.get("output_tokens")),
            "reasoning_output_tokens": _count(raw.get("reasoning_output_tokens")),
            "total_tokens": None,
        }
        if result["input_tokens"] is not None:
            result["input_tokens"] += (reads or 0) + (writes or 0)
    if result["input_tokens"] is not None and result["output_tokens"] is not None:
        # Reasoning is a subset of output, cached reads/writes a subset of input.
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def estimate_cost(model, usage):
    model = model.removesuffix("[1m]")
    rate = PRICES["models"].get(model)
    result = {"estimated_cost_usd": None, "pricing": rate, "assumptions": []}
    if not rate or "unavailable_reason" in rate:
        result["cost_unavailable_reason"] = (rate or {}).get("unavailable_reason", "model_price_unknown")
        return result
    if not usage or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        result["cost_unavailable_reason"] = "token_usage_unavailable"
        return result
    inp, out = usage["input_tokens"], usage["output_tokens"]
    read = usage.get("cached_input_tokens") or 0
    write = usage.get("cache_write_input_tokens") or 0
    hour = usage.get("cache_write_1h_input_tokens") or 0
    if read + write > inp or hour > write:
        result["cost_unavailable_reason"] = "inconsistent_token_usage"
        return result
    if write and "cache_write_input" not in rate:
        result["cost_unavailable_reason"] = "cache_write_price_unknown"
        return result
    for key in ("cached_input_tokens", "cache_write_input_tokens"):
        if usage.get(key) is None:
            result["assumptions"].append(f"{key}_not_reported_assumed_zero")
    if write and "cache_write_1h_input" in rate and usage.get("cache_write_1h_input_tokens") is None:
        result["assumptions"].append("cache_write_ttl_not_reported_assumed_5m")
    long = inp > rate.get("long_context_threshold", math.inf)
    input_scale = rate.get("long_input_multiplier", 1) if long else 1
    output_scale = rate.get("long_output_multiplier", 1) if long else 1
    cost = ((inp - read - write) * rate["input"] + read * rate["cached_input"]
            + (write - hour) * rate.get("cache_write_input", 0)
            + hour * rate.get("cache_write_1h_input", rate.get("cache_write_input", 0))) * input_scale
    cost += out * rate["output"] * output_scale
    result.update(estimated_cost_usd=round(cost / 1_000_000, 10), long_context_pricing=long)
    return result


def summarize(calls):
    known = [c for c in calls if c["estimated_cost_usd"] is not None]
    priced = round(sum(c["estimated_cost_usd"] for c in known), 10)
    missing = sum(not c.get("usage") or c["usage"].get("total_tokens") is None for c in calls)
    incomplete = sum(not c.get("usage_final") or not c.get("usage")
                     or c["usage"].get("total_tokens") is None for c in calls)
    return {
        "calls": len(calls), "failed_calls": sum(c["status"] != "completed" for c in calls),
        "elapsed_s": round(sum(c["elapsed_s"] for c in calls), 6),
        "tokens": {k: sum((c.get("usage") or {}).get(k) or 0 for c in calls) for k in TOKEN_KEYS},
        "usage_missing_calls": missing, "usage_incomplete_calls": incomplete,
        "token_totals_complete": not incomplete,
        "estimated_cost_usd": priced if len(known) == len(calls) and not incomplete else None,
        "priced_cost_usd": priced, "unpriced_calls": len(calls) - len(known),
        "cost_status": "complete" if len(known) == len(calls) and not incomplete else "partial" if known else "unknown",
    }


class UsageLedger:
    def __init__(self):
        self.calls = []
        self.root = None
        self.run_root = generated_var_path("runs", "gpt")
        self._written = 0
        self.write_error = None

    def attach(self, root):
        self.root = Path(root)
        self.flush()

    def record(self, call):
        call = {**call, "call": len(self.calls) + 1, "phase": _phase.get()}
        call["usage"] = normalize_usage(call.get("raw_usage"), call["provider"])
        call.update(estimate_cost(call["model"], call["usage"]))
        self.calls.append(call)
        self.flush()

    def summary(self):
        return {
            "schema_version": 1, "currency": PRICES["currency"], "cost_basis": PRICES["basis"],
            "pricing_checked_on": PRICES["checked_on"],
            "scope": "model turns in this invocation, including naming and demonstration preparation",
            "price_assumption": "Standard public API rates; excludes subscription, gateway, service-tier and account-specific charges",
            **summarize(self.calls),
            "by_model": {m: summarize([c for c in self.calls if c["model"] == m])
                         for m in sorted({c["model"] for c in self.calls})},
            "by_phase": {p: summarize([c for c in self.calls if c["phase"] == p])
                         for p in sorted({c["phase"] for c in self.calls})},
            "write_error": self.write_error,
        }

    def flush(self):
        if self.root is None:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "usage.jsonl").open("a", encoding="utf-8") as stream:
                for call in self.calls[self._written:]:
                    stream.write(json.dumps(call, ensure_ascii=False, allow_nan=False) + "\n")
                    self._written += 1
            temporary = self.root / "usage.json.tmp"
            temporary.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.root / "usage.json")
        except OSError as exc:
            # Accounting must not interrupt an in-progress hardware action or
            # replace the original provider error during finalization.
            if self.write_error is None:
                print(f"Usage recording failed: {exc}", file=sys.stderr)
            self.write_error = str(exc)


@contextmanager
def collect_usage():
    ledger = UsageLedger()
    token = _ledger.set(ledger)
    try:
        yield ledger
    finally:
        _ledger.reset(token)
        if ledger.calls and (ledger.root is None or not ledger.root.exists()):
            # Naming can fail before a semantic task directory can be created.
            root = ledger.run_root / f"{datetime.now():%Y%m%d-%H%M%S-%f}-initialization_failed"
            ledger.attach(root)
            print(f"Usage saved: {root}", file=sys.stderr)


@contextmanager
def usage_phase(name):
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


@contextmanager
def model_call(provider, model):
    started = time.monotonic()
    call = {"at_s": time.time(), "provider": provider, "model": model,
            "status": "completed", "raw_usage": None, "usage_final": False}
    try:
        yield call
    except BaseException as exc:
        call["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        call["error_type"] = type(exc).__name__
        raise
    finally:
        call["elapsed_s"] = round(time.monotonic() - started, 6)
        if (ledger := _ledger.get()) and call.get("request_started"):
            ledger.record(call)


def attach_usage(root):
    if ledger := _ledger.get():
        ledger.attach(root)


def set_usage_run_root(root):
    """Keep early naming failures with the selected task agent's recordings."""
    if ledger := _ledger.get():
        ledger.run_root = Path(root)


def usage_summary():
    ledger = _ledger.get()
    return ledger.summary() if ledger is not None else None
