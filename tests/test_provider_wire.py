"""Tests for provider response handling (``basedecision._wire`` and the provider backends).

Layers, from cheap to strict:

* ``PolicyTests``: the explicit role of every item/block type (text, refusal, benign, forbidden).
* ``TotalityFuzzTests``: for *any* JSON input the parsers either return text or raise
  ``ProviderResponseError``: never ``TypeError``/``KeyError``/``AttributeError``. This is the
  invariant whose violation was the original reasoning-item crash.
* ``SdkUnionTests``: enumerate the installed OpenAI/Anthropic SDK's own item-type unions and
  require a defined outcome for every member, so an SDK upgrade that adds a type is covered
  automatically.
* ``TransportTests``: the official SDKs over an in-memory HTTP transport, with real response
  shapes for reasoning models.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import pickle
import random
import types
import typing
import unittest
from typing import Any

from basedecision import BaseDecision, InputError, Option, Request
from basedecision._wire import anthropic_text, openai_text, to_plain, usage_counts
from basedecision.errors import ProviderError, ProviderResponseError
from basedecision.providers import parse_selection

DECISION = '{"option_index":1}'
RESPONSE_CODES = {
    "refusal",
    "output_budget_exhausted",
    "content_filtered",
    "incomplete_output",
    "unexpected_output_block",
    "invalid_output",
}


def message(text: str = DECISION) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def reasoning(**fields: Any) -> dict[str, Any]:
    return {"id": "rs_1", "type": "reasoning", "summary": [], **fields}


def openai_response(output: list[Any], status: str = "completed", **extra: Any) -> dict[str, Any]:
    return {"id": "resp_1", "object": "response", "status": status, "output": output, **extra}


def anthropic_response(content: list[Any], stop_reason: str | None = "end_turn") -> dict[str, Any]:
    return {"id": "msg_1", "type": "message", "stop_reason": stop_reason, "content": content}


def code_of(function: Any, payload: Any) -> str:
    """Return the ProviderResponseError code raised by ``function(payload)``."""
    with unittest.TestCase().assertRaises(ProviderResponseError) as caught:
        function(payload)
    return str(caught.exception.code)


class OpenAIPolicyTests(unittest.TestCase):
    def test_reasoning_item_is_ignored(self) -> None:
        for item in (
            reasoning(),
            reasoning(content=None),
            reasoning(content=[]),
            reasoning(content=[{"type": "reasoning_text", "text": "thinking"}]),
            reasoning(summary=[{"type": "summary_text", "text": "short"}], encrypted_content="x"),
            {"type": "reasoning"},
        ):
            with self.subTest(item=item):
                self.assertEqual(openai_text(openai_response([item, message()])), DECISION)

    def test_several_reasoning_items_around_the_message(self) -> None:
        output = [reasoning(), reasoning(id="rs_2"), message(), reasoning(id="rs_3")]
        self.assertEqual(openai_text(openai_response(output)), DECISION)

    def test_message_text_parts_are_joined(self) -> None:
        item = message()
        item["content"] = [
            {"type": "output_text", "text": '{"option_'},
            {"type": "output_text", "text": 'index":1}'},
        ]
        self.assertEqual(openai_text(openai_response([item])), DECISION)

    def test_refusal_wins_over_text_and_over_completed_status(self) -> None:
        item = message()
        item["content"].append({"type": "refusal", "refusal": "no"})
        self.assertEqual(code_of(openai_text, openai_response([reasoning(), item])), "refusal")
        self.assertEqual(code_of(openai_text, openai_response([item], "incomplete")), "refusal")

    def test_tool_and_unknown_items_are_forbidden(self) -> None:
        for kind in (
            "function_call",
            "web_search_call",
            "mcp_call",
            "shell_call",
            "brand_new_item",
        ):
            with self.subTest(kind=kind):
                payload = openai_response([{"type": kind}, message()])
                with self.assertRaises(ProviderResponseError) as caught:
                    openai_text(payload)
                self.assertEqual(caught.exception.code, "unexpected_output_block")
                self.assertEqual(caught.exception.detail, kind)

    def test_malformed_items_are_forbidden_not_crashes(self) -> None:
        for item in (None, 5, "message", [], {"no": "type"}, {"type": None}, {"type": ["message"]}):
            with self.subTest(item=item):
                self.assertEqual(
                    code_of(openai_text, openai_response([item, message()])),
                    "unexpected_output_block",
                )

    def test_unknown_or_malformed_content_parts(self) -> None:
        for part in ({"type": "image"}, None, 7, {}, {"type": "audio", "data": "x"}):
            item = message()
            item["content"].append(part)
            with self.subTest(part=part):
                self.assertEqual(
                    code_of(openai_text, openai_response([item])), "unexpected_output_block"
                )
        for text in (None, 5, ["x"], {"a": 1}):
            item = message()
            item["content"] = [{"type": "output_text", "text": text}]
            with self.subTest(text=text):
                self.assertEqual(code_of(openai_text, openai_response([item])), "invalid_output")

    def test_unfinished_responses_have_specific_codes(self) -> None:
        cases = [
            ({"reason": "max_output_tokens"}, "output_budget_exhausted"),
            ({"reason": "content_filter"}, "content_filtered"),
            ({"reason": "something_new"}, "incomplete_output"),
            ({}, "incomplete_output"),
            (None, "incomplete_output"),
            ("max_output_tokens", "incomplete_output"),
        ]
        for details, code in cases:
            with self.subTest(details=details):
                payload = openai_response([reasoning()], "incomplete", incomplete_details=details)
                self.assertEqual(code_of(openai_text, payload), code)

    def test_status_must_be_completed(self) -> None:
        for status in ("failed", "in_progress", "queued", "cancelled", None, 5):
            with self.subTest(status=status):
                payload = {"output": [message()], "status": status}
                self.assertEqual(code_of(openai_text, payload), "incomplete_output")
        self.assertEqual(code_of(openai_text, {"output": [message()]}), "incomplete_output")

    def test_missing_or_malformed_output_yields_empty_text(self) -> None:
        for output in (None, [], "x", {"a": 1}, 5):
            with self.subTest(output=output):
                self.assertEqual(openai_text({"status": "completed", "output": output}), "")
        with self.assertRaises(ProviderResponseError):  # the selection parser then rejects it
            parse_selection("", 2)

    def test_convenience_output_text_is_a_fallback(self) -> None:
        self.assertEqual(openai_text({"status": "completed", "output_text": DECISION}), DECISION)
        both = openai_response([message('{"option_index":0}')], output_text=DECISION)
        self.assertEqual(openai_text(both), '{"option_index":0}')  # real items take precedence

    def test_non_mapping_payloads_are_invalid_output(self) -> None:
        for payload in (None, 5, "x", [], [message()], object()):
            with self.subTest(payload=payload):
                self.assertEqual(code_of(openai_text, payload), "invalid_output")


class AnthropicPolicyTests(unittest.TestCase):
    def text(self, text: str = DECISION) -> dict[str, Any]:
        return {"type": "text", "text": text}

    def test_thinking_blocks_are_ignored(self) -> None:
        thinking = {"type": "thinking", "thinking": "hmm", "signature": "s"}
        redacted = {"type": "redacted_thinking", "data": "x"}
        for content in ([thinking, self.text()], [redacted, thinking, self.text(), thinking]):
            self.assertEqual(anthropic_text(anthropic_response(content)), DECISION)

    def test_text_blocks_are_joined(self) -> None:
        content = [self.text('{"option_'), self.text('index":1}')]
        self.assertEqual(anthropic_text(anthropic_response(content)), DECISION)

    def test_tool_and_unknown_blocks_are_forbidden(self) -> None:
        for kind in ("tool_use", "server_tool_use", "web_search_tool_result", "brand_new"):
            with self.subTest(kind=kind), self.assertRaises(ProviderResponseError) as caught:
                anthropic_text(anthropic_response([self.text(), {"type": kind}]))
            self.assertEqual(
                (caught.exception.code, caught.exception.detail), ("unexpected_output_block", kind)
            )

    def test_malformed_blocks(self) -> None:
        for block in (None, 5, "text", [], {}, {"type": None}):
            with self.subTest(block=block):
                self.assertEqual(
                    code_of(anthropic_text, anthropic_response([block])), "unexpected_output_block"
                )
        for text in (None, 5, ["x"]):
            with self.subTest(text=text):
                self.assertEqual(
                    code_of(anthropic_text, anthropic_response([{"type": "text", "text": text}])),
                    "invalid_output",
                )

    def test_stop_reasons(self) -> None:
        cases = {
            "refusal": "refusal",
            "max_tokens": "output_budget_exhausted",
            "tool_use": "incomplete_output",
            "stop_sequence": "incomplete_output",
            "pause_turn": "incomplete_output",
            None: "incomplete_output",
            "": "incomplete_output",
        }
        for reason, code in cases.items():
            with self.subTest(reason=reason):
                payload = anthropic_response([self.text()], reason)
                self.assertEqual(code_of(anthropic_text, payload), code)

    def test_missing_or_malformed_content(self) -> None:
        for content in (None, "x", {"a": 1}, 5):
            with self.subTest(content=content):
                self.assertEqual(
                    anthropic_text({"stop_reason": "end_turn", "content": content}), ""
                )

    def test_non_mapping_payloads_are_invalid_output(self) -> None:
        for payload in (None, 5, "x", [], object()):
            with self.subTest(payload=payload):
                self.assertEqual(code_of(anthropic_text, payload), "invalid_output")


class UsageTests(unittest.TestCase):
    def test_reports_reasoning_tokens_when_present(self) -> None:
        usage = {
            "input_tokens": 10,
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 256},
        }
        self.assertEqual(
            usage_counts({"usage": usage}),
            {"input_tokens": 10, "output_tokens": 300, "reasoning_tokens": 256},
        )

    def test_plain_counts(self) -> None:
        self.assertEqual(
            usage_counts({"usage": {"input_tokens": 1, "output_tokens": 2}}),
            {"input_tokens": 1, "output_tokens": 2},
        )

    def test_only_non_negative_integers_are_accepted(self) -> None:
        bad = {
            "input_tokens": True,
            "output_tokens": -1,
            "output_tokens_details": {"reasoning_tokens": 1.5},
        }
        self.assertEqual(usage_counts({"usage": bad}), {})
        for usage in (None, [], "x", 5, {"input_tokens": "3"}):
            self.assertEqual(usage_counts({"usage": usage}), {})
        for payload in (None, 5, [], "x"):
            self.assertEqual(usage_counts(payload), {})


class ToPlainTests(unittest.TestCase):
    @staticmethod
    def as_namespaces(value: Any) -> Any:
        if isinstance(value, dict):
            return types.SimpleNamespace(
                **{k: ToPlainTests.as_namespaces(v) for k, v in value.items()}
            )
        if isinstance(value, list):
            return [ToPlainTests.as_namespaces(v) for v in value]
        return value

    def test_json_survives_a_namespace_round_trip(self) -> None:
        for payload in (
            openai_response([reasoning(), message()]),
            anthropic_response([{"type": "text", "text": "x"}]),
        ):
            self.assertEqual(to_plain(self.as_namespaces(payload)), payload)

    def test_round_trip_property_on_random_json(self) -> None:
        rng = random.Random(7)
        for _ in range(500):
            payload = TotalityFuzzTests.random_json(rng)
            if isinstance(payload, dict):
                self.assertEqual(to_plain(self.as_namespaces(payload)), payload)
            self.assertEqual(to_plain(payload), payload)

    def test_pydantic_style_objects_are_dumped(self) -> None:
        class Model:
            def model_dump(self, **kwargs: Any) -> dict[str, Any]:
                self.kwargs = kwargs
                return {"a": [1, {"b": None}]}

        model = Model()
        self.assertEqual(to_plain(model), {"a": [1, {"b": None}]})
        self.assertEqual(model.kwargs, {"mode": "json", "warnings": False})

    def test_broken_dump_falls_back_to_attributes(self) -> None:
        class Broken:
            def __init__(self) -> None:
                self.visible = 1
                self._hidden = 2

            def model_dump(self, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("boom")

        self.assertEqual(to_plain(Broken()), {"visible": 1})

    def test_cycles_and_deep_nesting_are_bounded(self) -> None:
        loop = types.SimpleNamespace()
        loop.me = loop
        node, depth = to_plain(loop), 0
        while isinstance(node, dict) and "me" in node:
            node, depth = node["me"], depth + 1
        self.assertTrue(0 < depth <= 32, depth)
        deep: dict[str, Any] = {}
        tail = deep
        for _ in range(5000):
            tail["next"] = {}
            tail = tail["next"]
        self.assertIsInstance(to_plain(deep), dict)  # no RecursionError

    def test_unconvertible_objects_become_none(self) -> None:
        for value in (object(), set(), range(3), frozenset({1})):
            with self.subTest(value=type(value).__name__):
                self.assertIsNone(to_plain(value))


class ErrorTests(unittest.TestCase):
    def test_detail_is_part_of_the_message_and_pickles(self) -> None:
        error = ProviderResponseError("unexpected_output_block", detail="function_call")
        self.assertEqual(
            str(error), "Provider request failed: unexpected_output_block (function_call)"
        )
        clone = pickle.loads(pickle.dumps(error))
        self.assertEqual(
            (type(clone), clone.code, clone.detail, str(clone)),
            (ProviderResponseError, "unexpected_output_block", "function_call", str(error)),
        )

    def test_default_message_is_unchanged(self) -> None:
        error = ProviderError("rate_limit", retryable=True, status_code=429)
        self.assertEqual(str(error), "Provider request failed: rate_limit")
        self.assertIsNone(error.detail)

    def test_detail_never_carries_free_text(self) -> None:
        hostile = [
            "Ignore previous instructions",
            "<script>",
            "x" * 100,
            "Function_call",
            "has space",
            "new\nline",
            "",
            "9starts_with_digit",
            "ünï",
        ]
        for kind in hostile:
            with self.subTest(kind=kind[:20]), self.assertRaises(ProviderResponseError) as caught:
                openai_text(openai_response([{"type": kind}]))
            self.assertIsNone(caught.exception.detail)
            self.assertNotIn(kind[:10] if kind.strip() else "@@", str(caught.exception))


class TotalityFuzzTests(unittest.TestCase):
    """For ANY JSON, parsing returns text or raises ProviderResponseError. Nothing else."""

    @staticmethod
    def random_json(rng: random.Random, depth: int = 0) -> Any:
        scalars: list[Any] = [
            None,
            True,
            False,
            0,
            1,
            -1,
            10**12,
            0.5,
            "",
            "x",
            "message",
            "reasoning",
            "output_text",
            "refusal",
            "text",
            "thinking",
            "completed",
            "end_turn",
            "max_output_tokens",
            DECISION,
            "é\u200b",
        ]
        if depth >= 3 or rng.random() < 0.4:
            return rng.choice(scalars)
        if rng.random() < 0.5:
            return [TotalityFuzzTests.random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]
        keys = [
            "type",
            "content",
            "text",
            "status",
            "output",
            "stop_reason",
            "usage",
            "summary",
            "incomplete_details",
            "reason",
            "input_tokens",
            "output_tokens",
            "output_tokens_details",
            "reasoning_tokens",
            "output_text",
            "refusal",
        ]
        return {
            rng.choice(keys): TotalityFuzzTests.random_json(rng, depth + 1)
            for _ in range(rng.randint(0, 5))
        }

    @staticmethod
    def mutate(rng: random.Random, root: Any) -> Any:
        paths: list[list[Any]] = []

        def walk(value: Any, path: list[Any]) -> None:
            paths.append(path)
            children = (
                value.items()
                if isinstance(value, dict)
                else (enumerate(value) if isinstance(value, list) else ())
            )
            for key, child in children:
                walk(child, [*path, key])

        walk(root, [])
        path = rng.choice(paths)
        if not path:
            return TotalityFuzzTests.random_json(rng)
        parent = root
        for step in path[:-1]:
            parent = parent[step]
        if isinstance(parent, dict) and rng.random() < 0.25:
            del parent[path[-1]]
        else:
            parent[path[-1]] = TotalityFuzzTests.random_json(rng)
        return root

    def check(self, function: Any, payload: Any) -> str:
        try:
            result = function(payload)
        except ProviderResponseError as error:
            self.assertIn(error.code, RESPONSE_CODES)
            return "rejected"
        self.assertIsInstance(result, str)
        return "text"

    def test_mutated_valid_responses(self) -> None:
        rng = random.Random(20261005)
        bases = [
            (openai_text, openai_response([reasoning(), message()], usage={"input_tokens": 1})),
            (
                anthropic_text,
                anthropic_response(
                    [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": DECISION}]
                ),
            ),
        ]
        outcomes = {"text": 0, "rejected": 0}
        for _ in range(6000):
            function, base = rng.choice(bases)
            payload = copy.deepcopy(base)
            for _ in range(rng.randint(1, 3)):
                payload = self.mutate(rng, payload)
            outcomes[self.check(function, payload)] += 1
            counts = usage_counts(payload)
            self.assertTrue(all(type(v) is int and v >= 0 for v in counts.values()))
        self.assertGreater(outcomes["text"], 200, outcomes)
        self.assertGreater(outcomes["rejected"], 1000, outcomes)

    def test_arbitrary_json_and_objects(self) -> None:
        rng = random.Random(99)
        for _ in range(3000):
            payload = self.random_json(rng)
            for function in (openai_text, anthropic_text):
                self.check(function, payload)
                self.check(
                    function,
                    to_plain(ToPlainTests.as_namespaces(payload))
                    if isinstance(payload, dict)
                    else payload,
                )


def union_members(alias: Any) -> list[type]:
    """All concrete classes of a (possibly Annotated) Union alias."""
    args = typing.get_args(alias)
    union_types = (typing.Union, getattr(types, "UnionType", typing.Union))
    while args and typing.get_origin(args[0]) in union_types:
        args = typing.get_args(args[0])
    return [arg for arg in args if isinstance(arg, type)]


def literal_type_name(cls: type) -> str:
    annotation = cls.model_fields["type"].annotation  # type: ignore[attr-defined]
    return str(typing.get_args(annotation)[0])


def field_variants(cls: Any, name: str) -> list[Any]:
    """The SDK class instantiated bare, and with each declared field set to awkward values."""
    variants = [cls.model_construct(type=name)]
    for field in cls.model_fields:
        if field != "type":
            variants += [
                cls.model_construct(type=name, **{field: v}) for v in (None, [], "x", {}, 0)
            ]
    return variants


@unittest.skipUnless(importlib.util.find_spec("openai"), "openai SDK not installed")
class OpenAISdkUnionTests(unittest.TestCase):
    def test_every_output_item_type_has_a_defined_outcome(self) -> None:
        from openai.types.responses import ResponseOutputItem

        members = union_members(ResponseOutputItem)
        names = {literal_type_name(cls) for cls in members}
        self.assertLessEqual({"message", "reasoning"}, names, "union introspection failed")
        self.assertGreater(len(names), 10, "union introspection failed")
        for cls in members:
            name = literal_type_name(cls)
            for variant in field_variants(cls, name):
                payload = openai_response([to_plain(variant), message()])
                with self.subTest(item=name):
                    if name in ("message", "reasoning"):
                        self.assertEqual(openai_text(payload), DECISION)
                    else:
                        with self.assertRaises(ProviderResponseError) as caught:
                            openai_text(payload)
                        self.assertEqual(caught.exception.code, "unexpected_output_block")
                        self.assertEqual(caught.exception.detail, name)


@unittest.skipUnless(importlib.util.find_spec("anthropic"), "anthropic SDK not installed")
class AnthropicSdkUnionTests(unittest.TestCase):
    def test_every_content_block_type_has_a_defined_outcome(self) -> None:
        from anthropic.types import ContentBlock

        members = union_members(ContentBlock)
        names = {literal_type_name(cls) for cls in members}
        self.assertLessEqual({"text", "thinking", "redacted_thinking", "tool_use"}, names)
        for cls in members:
            name = literal_type_name(cls)
            for variant in field_variants(cls, name):
                payload = anthropic_response(
                    [to_plain(variant), {"type": "text", "text": DECISION}]
                )
                with self.subTest(block=name):
                    if name in ("thinking", "redacted_thinking"):
                        self.assertEqual(anthropic_text(payload), DECISION)
                    elif name == "text":
                        try:
                            self.assertTrue(anthropic_text(payload).endswith(DECISION))
                        except ProviderResponseError as error:
                            self.assertEqual(error.code, "invalid_output")
                    else:
                        with self.assertRaises(ProviderResponseError) as caught:
                            anthropic_text(payload)
                        self.assertEqual(caught.exception.code, "unexpected_output_block")
                        self.assertEqual(caught.exception.detail, name)


def _http_flavours() -> list[Any]:
    """httpx modules to try: newer SDKs require ``httpx2``; older ones accept ``httpx``."""
    modules = []
    for name in ("httpx2", "httpx"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            continue
    return modules


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("openai", "anthropic")) and _http_flavours(),
    "official SDKs and an httpx flavour are required",
)
class TransportTests(unittest.TestCase):
    """The official SDKs over an in-memory transport, with real reasoning-model response shapes."""

    def build(self, provider: str, respond: Any, **options: Any) -> tuple[BaseDecision, list[Any]]:
        from anthropic import Anthropic
        from openai import OpenAI

        seen: list[Any] = []
        decision = BaseDecision.from_provider(
            provider, "test-model", api_key="test-key", max_retries=0, **options
        )
        base_url = str(decision._client.base_url)
        decision._client.close()
        factory = OpenAI if provider == "openai" else Anthropic
        last_error: Exception | None = None
        for module in _http_flavours():

            def handler(request: Any, module: Any = module) -> Any:
                seen.append(json.loads(request.content))
                return respond(module, request)

            try:
                decision._client = factory(
                    api_key="test-key",
                    base_url=base_url,
                    max_retries=0,
                    http_client=module.Client(transport=module.MockTransport(handler)),
                )
                return decision, seen
            except TypeError as error:  # this SDK generation wants the other httpx flavour
                last_error = error
        raise AssertionError(f"no usable httpx flavour: {last_error}")

    request = Request("some customer text", "Which?", (Option("a", "Alpha"), Option("b", "Beta")))

    def openai(self, output: list[Any], status: str = "completed", **extra: Any) -> Any:
        body = {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "status": status,
            "model": "m",
            "output": output,
            **extra,
        }
        return lambda module, request: module.Response(200, json=body)

    def anthropic(self, content: list[Any], stop_reason: str = "end_turn") -> Any:
        body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
        return lambda module, request: module.Response(200, json=body)

    def test_openai_reasoning_response_yields_a_decision_and_reports_reasoning_tokens(self) -> None:
        usage = {
            "input_tokens": 12,
            "output_tokens": 300,
            "total_tokens": 312,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 256},
        }
        decision, _ = self.build("openai", self.openai([reasoning(), message()], usage=usage))
        result = decision.predict(self.request)
        self.assertEqual(result.answer, "b")
        self.assertEqual(
            result.usage, {"input_tokens": 12, "output_tokens": 300, "reasoning_tokens": 256}
        )

    def test_openai_budget_exhausted_by_reasoning_has_its_own_code(self) -> None:
        response = self.openai(
            [reasoning()], "incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        decision, _ = self.build("openai", response)
        with self.assertRaises(ProviderResponseError) as caught:
            decision.predict(self.request)
        self.assertEqual(
            (caught.exception.code, caught.exception.retryable), ("output_budget_exhausted", False)
        )

    def test_openai_content_filter_refusal_and_tool_call(self) -> None:
        cases = [
            (
                self.openai([], "incomplete", incomplete_details={"reason": "content_filter"}),
                "content_filtered",
            ),
            (
                self.openai(
                    [
                        {
                            "id": "m",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "refusal", "refusal": "no"}],
                        }
                    ]
                ),
                "refusal",
            ),
            (
                self.openai(
                    [
                        {
                            "id": "f",
                            "type": "function_call",
                            "call_id": "c",
                            "name": "n",
                            "arguments": "{}",
                        },
                        message(),
                    ]
                ),
                "unexpected_output_block",
            ),
        ]
        for response, code in cases:
            decision, _ = self.build("openai", response)
            with self.subTest(code), self.assertRaises(ProviderResponseError) as caught:
                decision.predict(self.request)
            self.assertEqual(caught.exception.code, code)

    def test_anthropic_thinking_blocks_are_ignored(self) -> None:
        content = [
            {"type": "thinking", "thinking": "hmm", "signature": "s"},
            {"type": "text", "text": DECISION},
        ]
        decision, _ = self.build("anthropic", self.anthropic(content))
        self.assertEqual(decision.predict(self.request).answer, "b")

    def test_anthropic_budget_and_tool_use(self) -> None:
        decision, _ = self.build("anthropic", self.anthropic([], "max_tokens"))
        with self.assertRaises(ProviderResponseError) as caught:
            decision.predict(self.request)
        self.assertEqual(caught.exception.code, "output_budget_exhausted")
        tool = [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]
        decision, _ = self.build("anthropic", self.anthropic(tool))
        with self.assertRaises(ProviderResponseError) as caught:
            decision.predict(self.request)
        self.assertEqual(
            (caught.exception.code, caught.exception.detail),
            ("unexpected_output_block", "tool_use"),
        )

    def test_reasoning_effort_is_sent_only_when_requested(self) -> None:
        decision, seen = self.build("openai", self.openai([message()]))
        decision.predict(self.request)
        self.assertNotIn("reasoning", seen[0])
        decision, seen = self.build("openai", self.openai([message()]), reasoning_effort="low")
        decision.predict(self.request)
        self.assertEqual(seen[0]["reasoning"], {"effort": "low"})

    def test_reasoning_effort_validation(self) -> None:
        for bad in ("", "Low", "a b", 5, "x" * 40, ["low"]):
            with self.subTest(bad=bad), self.assertRaises(InputError):
                BaseDecision.from_provider("openai", "m", api_key="k", reasoning_effort=bad)
        with self.assertRaises(InputError):
            BaseDecision.from_provider("anthropic", "m", api_key="k", reasoning_effort="low")


if __name__ == "__main__":
    unittest.main()
