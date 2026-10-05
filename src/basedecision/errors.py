"""Provider errors with stable codes and sanitized messages, and pickling support."""

import copyreg
from typing import Any

from .types import ContextLengthError


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
        """Rebuild the error from its fields when pickled or copied."""
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


# ContextLengthError(required, maximum) is defined in types.py, a module pinned byte-for-byte by
# the calibration contract, and passes only a message to Exception.__init__. Default pickling
# therefore re-runs __init__ with that one message and fails ("missing 'maximum'"), which breaks
# concurrent.futures.ProcessPoolExecutor, multiprocessing, Celery, Ray and Dask: the parent sees
# BrokenProcessPool instead of the real error. The copyreg registry teaches pickle and copy how to
# rebuild it without touching types.py. It applies to this exact class (there are no subclasses).
def _restore_context_length_error(required: int, maximum: int) -> ContextLengthError:
    # types.py is a pinned legacy module without annotations, hence the ignore.
    return ContextLengthError(required, maximum)  # type: ignore[no-untyped-call]


def _reduce_context_length_error(error: ContextLengthError) -> tuple[Any, ...]:
    return (_restore_context_length_error, (error.required_tokens, error.maximum_tokens))


copyreg.pickle(ContextLengthError, _reduce_context_length_error)
