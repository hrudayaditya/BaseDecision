"""Provider response normalization: parse the wire, not the SDK objects.

The OpenAI Responses API returns a *list of heterogeneous output items* (messages, reasoning,
and some two dozen tool-call types) and Anthropic returns a list of content blocks. Probing
attributes on those SDK objects is fragile: an attribute can exist but be ``None`` (a reasoning
item's ``content``), be missing, or differ between SDK versions. This module instead

1. converts a response of any kind (SDK model, namespace, plain dict) to plain JSON-like data
   with :func:`to_plain`, and
2. applies one closed, explicit policy to that data (:func:`openai_text`,
   :func:`anthropic_text`).

Every item/block type has exactly one role:

* **decision text**: an OpenAI ``message`` with ``output_text`` parts, an Anthropic ``text`` block.
* **refusal**: an OpenAI ``refusal`` content part, or Anthropic ``stop_reason == "refusal"``.
* **benign**: OpenAI ``reasoning`` items and Anthropic ``thinking``/``redacted_thinking`` blocks.
  They are the model's private deliberation, are ignored, and their cost is reported in usage.
* **forbidden**: anything else. BaseDecision sends no tools, so a tool call or an unknown item
  type means the response is not a plain decision and is rejected, never guessed at.

The functions here are *total*: for any input they either return text or raise
:class:`~basedecision.errors.ProviderResponseError` with a stable code. They never raise
``TypeError``/``KeyError``/``AttributeError`` and never put provider-controlled free text into an
error (only identifiers that match a strict pattern).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

from .errors import ProviderResponseError

__all__ = ["anthropic_text", "openai_text", "to_plain", "usage_counts"]

_MAX_DEPTH: Final = 32
_LABEL: Final = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")
_BENIGN_OPENAI_ITEMS: Final = frozenset({"reasoning"})
_BENIGN_ANTHROPIC_BLOCKS: Final = frozenset({"thinking", "redacted_thinking"})


def to_plain(value: object, _depth: int = 0) -> object:
    """Convert an SDK response (pydantic model, namespace, mapping, ...) to JSON-like data.

    Containers are copied recursively; pydantic-style objects are dumped with ``model_dump``;
    other objects contribute their public ``__dict__``. Anything unconvertible, and anything
    nested deeper than 32 levels (including reference cycles), becomes ``None``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _depth >= _MAX_DEPTH:
        return None
    if isinstance(value, Mapping):
        return {str(key): to_plain(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item, _depth + 1) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump(mode="json", warnings=False)
        except Exception:  # a broken dump falls back to the attribute view below
            dumped = None
        else:
            return to_plain(dumped, _depth + 1)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {
            str(key): to_plain(item, _depth + 1)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return None


def _kind(node: object) -> str | None:
    """Return the ``type`` of an item, block or part, but only if it is a string."""
    value = node.get("type") if isinstance(node, Mapping) else None
    return value if isinstance(value, str) else None


def _label(value: object) -> str | None:
    """Return ``value`` only if it looks like an API-defined identifier, never free text."""
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else None


def _items(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _unexpected(kind: str | None) -> ProviderResponseError:
    return ProviderResponseError("unexpected_output_block", detail=_label(kind))


def openai_text(payload: object) -> str:
    """Return the decision text of an OpenAI Responses API response.

    Raises:
        ProviderResponseError: ``refusal``; ``output_budget_exhausted``/``content_filtered``/
            ``incomplete_output`` for an unfinished response; ``unexpected_output_block`` for
            tool calls or unknown item types; ``invalid_output`` for malformed content.
    """
    if not isinstance(payload, Mapping):
        raise ProviderResponseError("invalid_output")
    texts: list[str] = []
    refused = False
    unexpected: ProviderResponseError | None = None
    for item in _items(payload.get("output")):
        kind = _kind(item)
        if kind in _BENIGN_OPENAI_ITEMS:
            continue
        if kind != "message":
            unexpected = unexpected or _unexpected(kind)
            continue
        for part in _items(item.get("content")):
            part_kind = _kind(part)
            if part_kind == "refusal":
                refused = True
            elif part_kind == "output_text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ProviderResponseError("invalid_output")
                texts.append(text)
            else:
                unexpected = unexpected or _unexpected(part_kind)
    if refused:
        raise ProviderResponseError("refusal")
    if payload.get("status") != "completed":
        raise _incomplete(payload.get("incomplete_details"))
    if unexpected is not None:
        raise unexpected
    if not texts and isinstance(payload.get("output_text"), str):
        # Clients and test doubles may expose only the SDK's convenience ``output_text``.
        texts.append(payload["output_text"])
    return "".join(texts)


def _incomplete(details: object) -> ProviderResponseError:
    reason = details.get("reason") if isinstance(details, Mapping) else None
    if reason == "max_output_tokens":
        return ProviderResponseError("output_budget_exhausted")
    if reason == "content_filter":
        return ProviderResponseError("content_filtered")
    return ProviderResponseError("incomplete_output")


def anthropic_text(payload: object) -> str:
    """Return the decision text of an Anthropic Messages API response.

    Raises:
        ProviderResponseError: ``refusal``; ``output_budget_exhausted`` (``max_tokens``);
            ``incomplete_output`` for any other unfinished stop reason;
            ``unexpected_output_block`` for tool use or unknown block types.
    """
    if not isinstance(payload, Mapping):
        raise ProviderResponseError("invalid_output")
    reason = payload.get("stop_reason")
    if reason == "refusal":
        raise ProviderResponseError("refusal")
    if reason == "max_tokens":
        raise ProviderResponseError("output_budget_exhausted")
    if reason != "end_turn":
        raise ProviderResponseError("incomplete_output")
    texts: list[str] = []
    for block in _items(payload.get("content")):
        kind = _kind(block)
        if kind in _BENIGN_ANTHROPIC_BLOCKS:
            continue
        if kind != "text":
            raise _unexpected(kind)
        text = block.get("text")
        if not isinstance(text, str):
            raise ProviderResponseError("invalid_output")
        texts.append(text)
    return "".join(texts)


def usage_counts(payload: object) -> dict[str, int]:
    """Token counts reported by the provider (non-negative integers only).

    Always ``input_tokens`` and ``output_tokens`` when present; ``reasoning_tokens`` as well when
    the provider reports hidden reasoning tokens (they are already included in ``output_tokens``
    and are billed).
    """
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    if not isinstance(usage, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if type(value) is int and value >= 0:
            counts[key] = value
    details = usage.get("output_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    if type(reasoning) is int and reasoning >= 0:
        counts["reasoning_tokens"] = reasoning
    return counts
