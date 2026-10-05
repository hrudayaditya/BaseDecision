"""Provider errors with stable codes and sanitized messages."""

from typing import Any


class ProviderError(RuntimeError):
    """A provider request failed. ``detail`` is an API-defined identifier (never free text)."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Create the error; the message holds only the code and an optional safe detail."""
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Provider request failed: {code}" + (f" ({detail})" if detail else ""))

    def __reduce__(self) -> tuple[Any, ...]:
        # Default pickling would re-run __init__ with the formatted message and wrap it twice.
        return (
            _restore_error,
            (type(self), self.code, self.retryable, self.status_code, self.detail),
        )


def _restore_error(
    cls: type[ProviderError],
    code: str,
    retryable: bool,
    status_code: int | None,
    detail: str | None,
) -> ProviderError:
    return cls(code, retryable=retryable, status_code=status_code, detail=detail)


class ProviderResponseError(ProviderError):
    """Refusal, incomplete generation, or invalid structured output."""
