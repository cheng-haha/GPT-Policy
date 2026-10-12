"""Opt-in, account-bound use of one free Codex reset; no purchases or robot commands."""

import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ...harness.codex import CodexAppServer
from ...harness.waiting import monitor_health, wait_with_health
from .io import append_json, write_json


def sanitized(snapshot):
    """Keep quota evidence, never authentication or account profile contents."""
    return {
        "checked_at_s": time.time(),
        "ordinaryUsageAllowed": snapshot.get("ordinaryUsageAllowed"),
        "rateLimits": snapshot.get("rateLimits"),
        "rateLimitsByLimitId": snapshot.get("rateLimitsByLimitId"),
        "rateLimitResetCredits": snapshot.get("rateLimitResetCredits"),
    }


@contextmanager
def quota_client(config, health_check=None):
    with monitor_health(health_check, deadline=time.monotonic() + 45):
        client = CodexAppServer(config["model"], config["codex_bin"], decision_only=True)
        try:
            yield client
        finally:
            client.close()


def read_quota(config):
    with quota_client(config) as client:
        return sanitized(client._request("account/rateLimits/read", {}))


def recover_quota(config, *, health_check=None):
    """Called only before obtaining a model action. Share one idempotent reset across GPUs."""
    policy = config.get("quota_recovery", {})
    if policy.get("allow_one_free_reset") is not True:
        return False
    expected_account = policy.get("account_id")
    if not expected_account or not policy.get("state_directory"):
        raise ValueError("Quota recovery requires an explicit account and shared state directory")
    root = Path(policy["state_directory"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if health_check:
                    health_check()
                wait_with_health(.2)
        # A lost response can retry the SAME idempotency key, never consume another credit.
        for attempt in range(3):
            try:
                with quota_client(config, health_check) as client:
                    return _recover_locked(client, root, expected_account)
            except Exception as error:  # noqa: BLE001 - bounded account RPC recovery; no model/action retries
                append_json(root / "events.jsonl", {
                    "at_s": time.time(), "event": "quota_probe_error",
                    "attempt": attempt + 1, "error": f"{type(error).__name__}: {error}",
                })
                if health_check:
                    health_check()
                if attempt == 2:
                    return False
                wait_with_health((5, 15)[attempt])
    return False


def _recover_locked(client, root, expected_account):
    snapshot = client._request("account/rateLimits/read", {})
    if snapshot.get("accountId") != expected_account:
        write_json(root / "status.json", {"status": "account_mismatch", **sanitized(snapshot)})
        return False

    def record(status, value):
        write_json(root / "status.json", {"status": status, **sanitized(value)})

    # Only backend permission establishes exhaustion/recovery; percentages are display data.
    allowed = snapshot.get("ordinaryUsageAllowed")
    if allowed is True:
        record("available", snapshot)
        return True
    if allowed is not False:
        record("permission_unknown", snapshot)
        return False
    buckets = snapshot.get("rateLimitsByLimitId") or {"codex": snapshot.get("rateLimits")}
    if any(
        b.get("spendControlReached") is True
        or b.get("rateLimitReachedType") not in (None, "rate_limit_reached")
        for b in buckets.values() if isinstance(b, dict)
    ):
        record("spending_or_workspace_limit", snapshot)
        return False
    ledger = root / "reset.json"
    state = json.loads(ledger.read_text()) if ledger.exists() else {}
    if state.get("outcome") is not None:
        record("free_reset_already_attempted", snapshot)
        return False
    if not state:
        credits = (snapshot.get("rateLimitResetCredits") or {}).get("credits") or []
        available = [
            credit for credit in credits
            if credit.get("status") == "available"
            and credit.get("resetType") == "codexRateLimits"
            and (credit.get("expiresAt") is None or credit["expiresAt"] > time.time())
        ]
        if not available:
            record("no_available_free_reset", snapshot)
            return False
        state = {
            "requested_at_s": time.time(), "account_id": expected_account,
            "request": {"creditId": available[0]["id"], "idempotencyKey": str(uuid.uuid4())},
            "before": sanitized(snapshot),
        }
        # Persist intent BEFORE the call. A crash or disconnect reuses this exact request.
        write_json(ledger, state)
    if state["account_id"] != expected_account:
        record("reset_ledger_account_mismatch", snapshot)
        return False
    record("resetting", snapshot)
    response = client._request("account/rateLimitResetCredit/consume", state["request"])
    state.update(outcome=response.get("outcome", "unrecognized"), responded_at_s=time.time())
    write_json(ledger, state)
    append_json(root / "events.jsonl", {
        "at_s": time.time(), "event": "free_reset_response", "outcome": state["outcome"],
    })
    after = client._request("account/rateLimits/read", {})
    recovered = after.get("accountId") == expected_account and after.get("ordinaryUsageAllowed") is True
    record("available" if recovered else "reset_not_confirmed", after)
    return recovered
