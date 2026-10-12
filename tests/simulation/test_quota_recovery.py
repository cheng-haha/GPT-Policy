"""Quota recovery is opt-in, serialized, idempotent and separate from physical actions."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpt_policy.simulation.robodojo import quota


def limits(allowed=False, *, used=100, credits=True):
    return {
        "accountId": "authorized-account", "ordinaryUsageAllowed": allowed,
        "rateLimits": {"primary": {"usedPercent": used}, "spendControlReached": False},
        "rateLimitResetCredits": {"availableCount": int(credits), "credits": [
            {"id": "one-free-reset", "status": "available", "resetType": "codexRateLimits"}
        ] if credits else []},
    }


@pytest.fixture
def backend(monkeypatch, tmp_path):
    state = {"snapshot": limits(), "requests": [], "lose_response": False}
    class Client:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def _request(self, method, params):
            if method == "account/rateLimits/read":
                return copy.deepcopy(state["snapshot"])
            assert method == "account/rateLimitResetCredit/consume"
            state["requests"].append(dict(params))
            if state["lose_response"] and len(state["requests"]) == 1:
                raise ConnectionError("ambiguous response")
            state["snapshot"] = limits(True, used=0, credits=False)
            return {"outcome": "reset" if len(state["requests"]) == 1 else "alreadyRedeemed"}
    monkeypatch.setattr(quota, "CodexAppServer", Client)
    monkeypatch.setattr(quota, "wait_with_health", lambda _: None)
    config = {
        "model": "test", "codex_bin": "test",
        "quota_recovery": {
            "allow_one_free_reset": True, "account_id": "authorized-account",
            "state_directory": str(tmp_path / "quota"),
        },
    }
    return config, state


def test_eight_workers_share_exactly_one_free_reset(backend):
    config, state = backend
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(lambda _: quota.recover_quota(config), range(8)))
    assert len(state["requests"]) == 1
    assert state["requests"][0]["creditId"] == "one-free-reset"
    # A later exhaustion cannot consume another reset, even if one appears available.
    state["snapshot"] = limits()
    assert quota.recover_quota(config) is False
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("scenario", [
    "not_authorized", "available", "permission_unknown", "no_credit",
    "wrong_account", "spend_control", "workspace_limit", "expired_credit",
])
def test_no_unjustified_reset(backend, scenario):
    config, state = backend
    if scenario == "not_authorized":
        config["quota_recovery"]["allow_one_free_reset"] = False
    elif scenario == "available":
        state["snapshot"]["ordinaryUsageAllowed"] = True
    elif scenario == "permission_unknown":
        state["snapshot"]["ordinaryUsageAllowed"] = None
    elif scenario == "no_credit":
        state["snapshot"]["rateLimitResetCredits"] = {"availableCount": 0}
    elif scenario == "wrong_account":
        state["snapshot"]["accountId"] = "different-account"
    elif scenario == "spend_control":
        state["snapshot"]["rateLimits"]["spendControlReached"] = True
    elif scenario == "workspace_limit":
        state["snapshot"]["rateLimits"]["rateLimitReachedType"] = "workspace_owner_usage_limit_reached"
    elif scenario == "expired_credit":
        state["snapshot"]["rateLimitResetCredits"]["credits"][0]["expiresAt"] = 1
    assert quota.recover_quota(config) is (scenario == "available")
    assert not state["requests"]


def test_ambiguous_reset_response_reuses_durable_idempotency_key(backend):
    config, state = backend
    state["lose_response"] = True
    assert quota.recover_quota(config) is True
    assert len(state["requests"]) == 2 and state["requests"][0] == state["requests"][1]
    ledger = json.loads((quota.Path(config["quota_recovery"]["state_directory"]) / "reset.json").read_text())
    assert ledger["outcome"] == "alreadyRedeemed"


def test_reset_response_alone_does_not_establish_recovery(backend, monkeypatch):
    config, state = backend
    class Unconfirmed:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def _request(self, method, params):
            if method == "account/rateLimits/read":
                return limits(None) if state["requests"] else limits()
            state["requests"].append(params)
            return {"outcome": "reset"}
    monkeypatch.setattr(quota, "CodexAppServer", Unconfirmed)
    assert quota.recover_quota(config) is False
    assert len(state["requests"]) == 1
