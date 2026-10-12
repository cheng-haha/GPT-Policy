"""Errors at the agent boundary; no robot action is retried here."""


class AgentError(RuntimeError):
    pass


class AgentProtocolError(AgentError):
    pass


class AgentTimeoutError(AgentError):
    pass


class AgentOverloadedError(AgentError):
    """An explicitly failed provider turn; no decision may be executed."""

    def __init__(self, message: str, *, provider: str, code: str):
        super().__init__(message)
        self.provider = provider
        self.code = code


class AgentUsageLimitError(AgentError):
    """Provider quota is exhausted; retrying the same model will not help."""

    def __init__(self, message: str, *, provider: str, code: str):
        super().__init__(message)
        self.provider = provider
        self.code = code


class AgentRecoverableError(AgentError):
    """Model transport/provider failure before any host tool selection is returned."""

    def __init__(self, message: str, *, provider: str, code: str):
        super().__init__(message)
        self.provider = provider
        self.code = code
