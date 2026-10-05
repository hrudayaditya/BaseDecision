"""Provider errors with stable codes and sanitized messages."""
class ProviderError(RuntimeError):
    def __init__(self, code, *, retryable=False, status_code=None):
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(f'Provider request failed: {code}')

class ProviderResponseError(ProviderError):
    """Refusal, incomplete generation, or invalid structured output."""
