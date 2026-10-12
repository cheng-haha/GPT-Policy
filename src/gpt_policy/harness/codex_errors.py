"""Codex protocol error classification, verified against installed 0.160.0 schema."""

import json

from .errors import AgentOverloadedError, AgentRecoverableError, AgentUsageLimitError


def codex_failure(error, label="turn"):
    message = f"Codex {label} failed: {error}"
    if not isinstance(error, dict):
        return RuntimeError(message)
    info = error.get("codexErrorInfo")
    code = (
        info
        if isinstance(info, str)
        else next(iter(info), None)
        if isinstance(info, dict)
        else None
    )
    status = (
        info.get(code, {}).get("httpStatusCode")
        if isinstance(info, dict) and isinstance(info.get(code), dict)
        else None
    )
    # Never reconnect around explicit credentials, quotas, policy or local input errors.
    if code in {"usageLimitExceeded", "sessionBudgetExceeded"}:
        return AgentUsageLimitError(message, provider="codex", code=code)
    if (
        error.get("misalignment") is not None
        or code
        in {
            "unauthorized",
            "cyberPolicy",
            "misalignmentPolicyViolation",
            "tooManyDenials",
            "sandboxError",
            "contextWindowExceeded",
        }
        or status in {401, 402, 403}
        or error.get("code") in {-32600, -32601, -32602}
    ):
        return RuntimeError(message)
    if code == "serverOverloaded":
        return AgentOverloadedError(message, provider="codex", code=code)
    transient = code in {"internalServerError", "rateLimitExceeded"}
    transport = code in {
        "httpConnectionFailed",
        "responseStreamConnectionFailed",
        "responseStreamDisconnected",
        "responseTooManyFailedAttempts",
    }
    transient |= transport and (
        status is None
        or (isinstance(status, int) and 200 <= status < 300)
        or status in {408, 429, 500, 502, 503, 504}
    )
    detail = error.get("message", "")
    # Native compaction wraps the same opaque backend error. Strip only this
    # observed wrapper; explicit input/auth/policy errors remain non-retryable.
    if isinstance(detail, str):
        detail = detail.removeprefix("Error running remote compact task: ")
    try:
        body = json.loads(detail)
    except (TypeError, ValueError):
        body = None
    if isinstance(body, dict) and set(body) == {"detail"}:
        detail = body["detail"]
    # The archived failure supplied ONLY {"detail":"Bad Request"}, classified
    # as "other". Retry that opaque case, not specific invalid schema/model errors.
    opaque_bad_request = (
        code in {None, "other", "badRequest"}
        and isinstance(detail, str)
        and detail.strip().lower() == "bad request"
        and not error.get("additionalDetails")
    )
    if transient or opaque_bad_request:
        return AgentRecoverableError(
            message,
            provider="codex",
            code="opaqueBadRequest" if opaque_bad_request else code,
        )
    return RuntimeError(message)
