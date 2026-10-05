"""Tests for the Jev/SystemOne adapter (``basedecision.systemone``).

Most tests use a scripted backend so the wire format is checked exactly and quickly, with
no model and no Torch. ``RealModelTests`` runs the same API against a real checkpoint when
``BASEDECISION_TEST_MODEL`` points at one (``BASEDECISION_TEST_DEVICE``/``..._PRECISION``
default to ``cpu``/``fp32``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, get_args
from unittest.mock import patch

import basedecision
from basedecision import (
    BaseDecision,
    CalibratedDecision,
    ContextLengthError,
    InputError,
    Option,
    Request,
    Result,
    SystemOne,
    SystemOneError,
)
from basedecision.calibration import _DATA, _sha
from basedecision.systemone import ErrorCode, parse_request_body, systemone

REFERENCE_REQUEST: dict[str, Any] = {
    "model": "clef",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}


def _text_hash_logit(text: str) -> float:
    """A stable pseudo-logit in [-3, 3) derived from option text."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 2**32 * 6 - 3


class ScriptedBackend:
    """Deterministic stand-in for a local model; mirrors ``BaseDecision`` result shapes."""

    max_context_chars = 10**9

    def __init__(self, logits: Any = None) -> None:
        self.calls: list[list[Request]] = []
        self.model_id = "scripted"
        self._logits = logits or self._default_logits

    @staticmethod
    def _default_logits(request: Request) -> list[float]:
        return [_text_hash_logit(option.text) for option in request.options]

    def packed_tokens(self, request: Request) -> int:
        return len(request.context) + 7

    def count_tokens(self, request: Request) -> int:
        if len(request.context) > self.max_context_chars:
            raise ContextLengthError(len(request.context), self.max_context_chars)
        return self.packed_tokens(request)

    def predict_batch(self, requests: Any) -> list[Result]:
        requests = list(requests)
        self.calls.append(requests)
        for request in requests:
            self.count_tokens(request)  # like the real model: validate everything first
        return [self._result(request) for request in requests]

    def _result(self, request: Request) -> Result:
        logits = self._logits(request)
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        probs = [value / math.fsum(exps) for value in exps]
        best = logits.index(peak)
        chosen = request.options[best]
        return Result(
            answer=(chosen.id == "true") if request.kind == "noul" else chosen.id,
            option_id=chosen.id,
            label=chosen.text,
            probabilities={o.id: p for o, p in zip(request.options, probs, strict=True)},
            raw_logits={o.id: v for o, v in zip(request.options, logits, strict=True)},
            packed_tokens=self.packed_tokens(request),
            model_id=self.model_id,
            kind=request.kind,
        )


def golden_logits(request: Request) -> list[float]:
    """Logits chosen so the expected probabilities are easy to compute by hand."""
    if request.question == "Which team should handle the message?":
        return [0.0, 2.0]  # sorted options: billing, technical
    if request.kind == "score":
        return [0.0, 0.0, math.log(8.0)]  # p = 1/10, 1/10, 8/10
    return [0.0, math.log(3.0)]  # noul: false, true -> p(true) = 0.75


class WireFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ScriptedBackend(golden_logits)
        self.service = SystemOne(self.backend)

    def test_reference_example_matches_documented_format(self) -> None:
        tokens = len(REFERENCE_REQUEST["state"]) + 7
        expected = {
            "model": "clef",
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": "technical",
                    "confidence": 0.8808,
                    "probabilities": {"billing": 0.1192, "technical": 0.8808},
                },
                "urgency": {
                    "type": "score",
                    "score": 1.7,  # 0 * 0.1 + 1 * 0.1 + 2 * 0.8
                    "confidence": 0.8,
                    "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                },
                "outage": {"type": "noul", "noul": 0.75},
            },
            "usage": {"input_tokens": 3 * tokens, "output_tokens": 0},
        }
        self.assertEqual(self.service(REFERENCE_REQUEST), expected)

    def test_response_is_plain_json(self) -> None:
        response = self.service(REFERENCE_REQUEST)
        self.assertEqual(json.loads(json.dumps(response)), response)

    def test_answers_keep_question_order(self) -> None:
        names = ["zeta", "alpha", "mid"]
        questions = {n: {"type": "noul"} for n in names}
        answers = self.service({"model": "m", "state": "s", "questions": questions})["answers"]
        self.assertEqual(list(answers), names)

    def test_choice_probabilities_follow_request_order_not_sorted_order(self) -> None:
        request = copy.deepcopy(REFERENCE_REQUEST)
        request["questions"]["department"]["criteria"] = {
            "technical": "Bugs or outages",
            "billing": "Payments or invoices",
        }
        answer = self.service(request)["answers"]["department"]
        self.assertEqual(list(answer["probabilities"]), ["technical", "billing"])
        self.assertEqual(answer["choice"], "technical")

    def test_model_name_is_echoed(self) -> None:
        for name in ("clef", "anything-else", "ünïcode"):
            request = {**REFERENCE_REQUEST, "model": name}
            self.assertEqual(self.service(request)["model"], name)

    def test_one_batched_backend_call_for_all_questions(self) -> None:
        self.service(REFERENCE_REQUEST)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(len(self.backend.calls[0]), 3)

    def test_score_legend_keeps_raw_descriptions(self) -> None:
        request = {
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "score", "criteria": [1, {"a": [2]}, "three"]}},
        }
        legend = self.service(request)["answers"]["q"]["legend"]
        self.assertEqual(legend, {"0": 1, "1": {"a": [2]}, "2": "three"})

    def test_values_are_rounded_to_four_places(self) -> None:
        answer = self.service(REFERENCE_REQUEST)["answers"]["urgency"]
        for value in (answer["score"], answer["confidence"], *answer["probabilities"].values()):
            self.assertEqual(value, round(value, 4))

    def test_function_form_matches_class_form(self) -> None:
        self.assertEqual(
            systemone(self.backend, REFERENCE_REQUEST), self.service(REFERENCE_REQUEST)
        )

    def test_request_is_not_mutated(self) -> None:
        snapshot = copy.deepcopy(REFERENCE_REQUEST)
        self.service(REFERENCE_REQUEST)
        self.assertEqual(REFERENCE_REQUEST, snapshot)


class StateAndInstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ScriptedBackend()
        self.service = SystemOne(self.backend)

    def context_for(self, state: Any) -> str:
        body = {"model": "m", "state": state, "questions": {"q": {"type": "noul"}}}
        self.service(body)
        return self.backend.calls[-1][0].context

    def question_for(self, question: dict[str, Any]) -> str:
        body = {"model": "m", "state": "s", "questions": {"q": {"type": "noul", **question}}}
        self.service(body)
        return self.backend.calls[-1][0].question

    def test_string_state_is_used_verbatim(self) -> None:
        self.assertEqual(self.context_for("  keep  this\n"), "  keep  this\n")

    def test_json_state_is_rendered_compact_with_sorted_keys(self) -> None:
        self.assertEqual(
            self.context_for({"b": [True, None], "a": 1.5}), '{"a":1.5,"b":[true,null]}'
        )

    def test_unicode_state_is_not_escaped(self) -> None:
        self.assertEqual(self.context_for({"k": "é✓"}), '{"k":"é✓"}')

    def test_scalar_states_are_allowed(self) -> None:
        self.assertEqual(self.context_for(None), "null")
        self.assertEqual(self.context_for(7), "7")
        self.assertEqual(self.context_for(False), "false")
        self.assertEqual(self.context_for([]), "[]")
        self.assertEqual(self.context_for(""), "")

    def test_state_key_order_does_not_change_the_context(self) -> None:
        a = self.context_for({"x": 1, "y": {"p": 1, "q": 2}})
        b = self.context_for({"y": {"q": 2, "p": 1}, "x": 1})
        self.assertEqual(a, b)

    def test_unserializable_states_are_rejected_with_a_field(self) -> None:
        for bad in (float("nan"), float("inf"), {1, 2}, object(), {1: "a", "b": 2}):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(SystemOneError) as ctx:
                self.context_for(bad)
            self.assertEqual(
                (ctx.exception.code, ctx.exception.field), ("invalid_request", "state")
            )

    def test_recursion_errors_while_rendering_are_mapped(self) -> None:
        with (
            patch("basedecision.systemone.json.dumps", side_effect=RecursionError),
            self.assertRaises(SystemOneError) as ctx,
        ):
            self.context_for({"deeply": "nested"})
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("invalid_request", "state"))

    def test_instructions_default_to_the_question_id(self) -> None:
        for question in (
            {},
            {"instructions": None},
            {"instructions": ""},
            {"instructions": "  \n"},
        ):
            with self.subTest(question=question):
                self.assertEqual(self.question_for(question), "q")

    def test_instructions_are_used_and_non_strings_are_rendered(self) -> None:
        self.assertEqual(self.question_for({"instructions": "Is it?"}), "Is it?")
        self.assertEqual(self.question_for({"instructions": {"b": 1, "a": 2}}), '{"a":2,"b":1}')


class OptionMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ScriptedBackend()
        self.service = SystemOne(self.backend)

    def answer(self, question: dict[str, Any]) -> Any:
        body = {"model": "m", "state": "s", "questions": {"q": question}}
        return self.service(body)["answers"]["q"]

    def request_for(self, question: dict[str, Any]) -> Request:
        self.answer(question)
        return self.backend.calls[-1][0]

    def test_choice_options_are_packed_in_sorted_id_order(self) -> None:
        criteria = {"zz": "last", "aa": "first", "mm": "middle"}
        request = self.request_for({"type": "choice", "criteria": criteria})
        self.assertEqual([o.id for o in request.options], ["aa", "mm", "zz"])
        self.assertEqual([o.text for o in request.options], ["first", "middle", "last"])

    def test_choice_decision_is_independent_of_json_key_order(self) -> None:
        criteria = {f"id{i}": f"description number {i}" for i in range(8)}
        reference = self.answer({"type": "choice", "criteria": criteria})
        rng = random.Random(3)
        for _ in range(20):
            items = list(criteria.items())
            rng.shuffle(items)
            shuffled = self.answer({"type": "choice", "criteria": dict(items)})
            self.assertEqual(shuffled["choice"], reference["choice"])
            self.assertEqual(
                sorted(shuffled["probabilities"].items()),
                sorted(reference["probabilities"].items()),
            )

    def test_blank_missing_and_non_string_descriptions(self) -> None:
        criteria = {"a": "", "b": None, "c": {"k": 1}, "d": "  ", "e": 5}
        request = self.request_for({"type": "choice", "criteria": criteria})
        texts = {o.id: o.text for o in request.options}
        self.assertEqual(texts, {"a": "a", "b": "b", "c": '{"k":1}', "d": "d", "e": "5"})

    def test_option_ids_are_not_part_of_the_packed_text(self) -> None:
        request = self.request_for({"type": "choice", "criteria": {"x1": "alpha", "x2": "beta"}})
        self.assertTrue(all(o.id not in o.text for o in request.options))

    def test_score_levels_are_indexed_from_zero(self) -> None:
        request = self.request_for({"type": "score", "criteria": ["low", "mid", "high"]})
        self.assertEqual(
            [(o.id, o.text) for o in request.options], [("0", "low"), ("1", "mid"), ("2", "high")]
        )
        self.assertEqual(request.kind, "score")

    def test_score_blank_descriptions_fall_back_to_the_index(self) -> None:
        request = self.request_for({"type": "score", "criteria": ["", None, "high"]})
        self.assertEqual([o.text for o in request.options], ["0", "1", "high"])

    def test_noul_defaults_match_the_native_check_api(self) -> None:
        request = self.request_for({"type": "noul"})
        self.assertEqual(
            [(o.id, o.text) for o in request.options], [("false", "false"), ("true", "true")]
        )

    def test_noul_accepts_custom_true_false_descriptions(self) -> None:
        for criteria in (
            None,
            {},
            {"true": "yes"},
            {"false": "no"},
            {"true": "Yes", "false": "No"},
        ):
            with self.subTest(criteria=criteria):
                self.request_for({"type": "noul", "criteria": criteria})
        request = self.request_for({"type": "noul", "criteria": {"true": "Yes", "false": "No"}})
        self.assertEqual(
            [(o.id, o.text) for o in request.options], [("false", "No"), ("true", "Yes")]
        )

    def test_noul_answer_is_the_probability_of_true(self) -> None:
        backend = ScriptedBackend(lambda request: [0.0, math.log(3.0)])
        body = {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}}
        self.assertEqual(SystemOne(backend)(body)["answers"]["q"], {"type": "noul", "noul": 0.75})

    def test_invalid_criteria_are_rejected(self) -> None:
        cases: list[tuple[str, dict[str, Any], str]] = [
            ("choice without criteria", {"type": "choice"}, "invalid_criteria"),
            ("choice empty", {"type": "choice", "criteria": {}}, "invalid_criteria"),
            ("choice list", {"type": "choice", "criteria": ["a", "b"]}, "invalid_criteria"),
            ("choice one option", {"type": "choice", "criteria": {"a": "x"}}, "invalid_criteria"),
            (
                "choice duplicate text",
                {"type": "choice", "criteria": {"a": "x", "b": "x"}},
                "invalid_criteria",
            ),
            (
                "choice blank id",
                {"type": "choice", "criteria": {" ": "x", "b": "y"}},
                "invalid_criteria",
            ),
            ("score dict", {"type": "score", "criteria": {"0": "a", "1": "b"}}, "invalid_criteria"),
            ("score empty", {"type": "score", "criteria": []}, "invalid_criteria"),
            ("score one level", {"type": "score", "criteria": ["a"]}, "invalid_criteria"),
            ("score duplicate", {"type": "score", "criteria": ["a", "a"]}, "invalid_criteria"),
            ("noul unknown key", {"type": "noul", "criteria": {"maybe": "x"}}, "invalid_criteria"),
            ("noul list", {"type": "noul", "criteria": ["true"]}, "invalid_criteria"),
            (
                "noul same text",
                {"type": "noul", "criteria": {"true": "x", "false": "x"}},
                "invalid_criteria",
            ),
        ]
        for name, question, code in cases:
            with self.subTest(name), self.assertRaises(SystemOneError) as ctx:
                self.answer(question)
            self.assertEqual(ctx.exception.code, code)
            self.assertTrue(ctx.exception.field and ctx.exception.field.startswith("questions.q"))

    def test_option_count_limits(self) -> None:
        ok = {f"o{i}": f"description {i}" for i in range(255)}
        self.assertEqual(len(self.answer({"type": "choice", "criteria": ok})["probabilities"]), 255)
        ok_score = [f"level {i}" for i in range(255)]
        self.assertEqual(len(self.answer({"type": "score", "criteria": ok_score})["legend"]), 255)
        too_many = {f"o{i}": f"description {i}" for i in range(256)}
        with self.assertRaises(SystemOneError) as ctx:
            self.answer({"type": "choice", "criteria": too_many})
        self.assertEqual(ctx.exception.code, "limit_exceeded")


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ScriptedBackend()
        self.service = SystemOne(self.backend)

    def base(self) -> dict[str, Any]:
        return copy.deepcopy(REFERENCE_REQUEST)

    def assertRejected(self, body: Any, code: str, field: str | None = None) -> SystemOneError:
        with self.assertRaises(SystemOneError) as ctx:
            self.service(body)
        self.assertEqual(ctx.exception.code, code, str(ctx.exception))
        if field is not None:
            self.assertEqual(ctx.exception.field, field)
        self.assertEqual(self.backend.calls, [], "no inference may run for an invalid request")
        return ctx.exception

    def test_body_must_be_an_object(self) -> None:
        for body in ([], "text", None, 5, [REFERENCE_REQUEST]):
            with self.subTest(body=body):
                self.assertRejected(body, "invalid_request")

    def test_required_top_level_fields(self) -> None:
        for field in ("model", "state", "questions"):
            body = self.base()
            del body[field]
            with self.subTest(missing=field):
                self.assertRejected(body, "invalid_request", field)

    def test_bad_model_values(self) -> None:
        for value in ("", "   ", 5, None, ["clef"]):
            with self.subTest(model=value):
                self.assertRejected({**self.base(), "model": value}, "invalid_request", "model")

    def test_bad_questions_containers(self) -> None:
        for value in ({}, [], "q", None, 3):
            with self.subTest(questions=value):
                self.assertRejected(
                    {**self.base(), "questions": value}, "invalid_request", "questions"
                )

    def test_unknown_fields_are_rejected_when_strict(self) -> None:
        self.assertRejected({**self.base(), "temperature": 0}, "unknown_field", "temperature")
        body = self.base()
        body["questions"]["outage"]["instruction"] = "typo for instructions"
        self.assertRejected(body, "unknown_field", "questions.outage.instruction")

    def test_unknown_fields_are_ignored_when_not_strict(self) -> None:
        lenient = SystemOne(self.backend, strict=False)
        body = {**self.base(), "temperature": 0, "metadata": {"a": 1}}
        body["questions"]["outage"]["extra"] = True
        self.assertIn("answers", lenient(body))

    def test_request_id_is_accepted_and_ignored(self) -> None:
        self.assertIn("answers", self.service({**self.base(), "id": "req-1"}))

    def test_bad_question_shapes(self) -> None:
        cases: list[tuple[str, Any, str]] = [
            ("not an object", "choice", "invalid_question"),
            ("none", None, "invalid_question"),
            ("missing type", {"criteria": ["a", "b"]}, "invalid_question"),
            ("unknown type", {"type": "multi", "criteria": ["a", "b"]}, "invalid_question"),
            ("type none", {"type": None}, "invalid_question"),
            ("type unhashable", {"type": ["noul"]}, "invalid_question"),
            ("type wrong case", {"type": "Noul"}, "invalid_question"),
        ]
        for name, question, code in cases:
            body = {"model": "m", "state": "s", "questions": {"q": question}}
            with self.subTest(name):
                self.assertRejected(body, code)

    def test_bad_question_ids(self) -> None:
        for qid in ("", "  ", 5, None):
            body = {"model": "m", "state": "s", "questions": {qid: {"type": "noul"}}}
            with self.subTest(qid=qid):
                self.assertRejected(body, "invalid_question")

    def test_media_is_rejected_not_ignored(self) -> None:
        for key in ("images", "videos"):
            for value in ([object()], ["a.png"], [[1, 2]]):
                with self.subTest(key=key):
                    self.assertRejected({**self.base(), key: value}, "unsupported_modality", key)
        self.assertRejected(
            {**self.base(), "media_kwargs": {"a": 1}}, "unsupported_modality", "media_kwargs"
        )

    def test_empty_media_fields_are_accepted(self) -> None:
        body = {**self.base(), "images": [], "videos": None, "media_kwargs": {}}
        self.assertIn("answers", self.service(body))

    def test_media_with_wrong_type_is_an_invalid_request(self) -> None:
        for value in ("img.png", 5, {"a": 1}):
            with self.subTest(images=value):
                self.assertRejected({**self.base(), "images": value}, "invalid_request", "images")

    def test_limits(self) -> None:
        small = SystemOne(self.backend, max_questions=2, max_state_chars=10)
        with self.assertRaises(SystemOneError) as ctx:
            small(self.base())  # 3 questions, state longer than 10 characters
        self.assertEqual(ctx.exception.code, "limit_exceeded")
        body = {"model": "m", "state": "x" * 11, "questions": {"q": {"type": "noul"}}}
        with self.assertRaises(SystemOneError) as ctx:
            small(body)
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("limit_exceeded", "state"))
        body["state"] = "x" * 10
        self.assertIn("answers", small(body))
        unlimited = SystemOne(self.backend, max_questions=None, max_state_chars=None)
        many = {f"q{i}": {"type": "noul"} for i in range(200)}
        self.assertEqual(
            len(unlimited({"model": "m", "state": "s", "questions": many})["answers"]), 200
        )

    def test_default_question_limit(self) -> None:
        many = {f"q{i}": {"type": "noul"} for i in range(65)}
        self.assertRejected(
            {"model": "m", "state": "s", "questions": many}, "limit_exceeded", "questions"
        )

    def test_invalid_limits_are_rejected_at_construction(self) -> None:
        for name in ("max_questions", "max_state_chars"):
            for value in (0, -1, 1.5, True, "5"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    SystemOne(self.backend, **{name: value})

    def test_accepted_models(self) -> None:
        gated = SystemOne(self.backend, accepted_models={"basedecision", "clef"})
        self.assertIn("answers", gated(self.base()))
        with self.assertRaises(SystemOneError) as ctx:
            gated({**self.base(), "model": "gpt"})
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("model_not_found", "model"))
        with self.assertRaises(TypeError):
            SystemOne(self.backend, accepted_models="clef")

    def test_a_late_invalid_question_prevents_all_inference(self) -> None:
        body = self.base()
        body["questions"]["zzz_last"] = {"type": "choice", "criteria": {"only": "one option"}}
        self.assertRejected(body, "invalid_criteria", "questions.zzz_last.criteria")


class BackendTests(unittest.TestCase):
    def body(self, state: str = "state") -> dict[str, Any]:
        return {
            "model": "m",
            "state": state,
            "questions": {"a": {"type": "noul"}, "b": {"type": "noul", "instructions": "Other?"}},
        }

    def test_object_without_predict_batch_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            SystemOne(object())  # type: ignore[arg-type]

    def test_cloud_providers_are_rejected_at_construction(self) -> None:
        class FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                self.calls: list[Any] = []

            def close(self) -> None:
                pass

        fake = {
            "openai": types.SimpleNamespace(OpenAI=FakeClient),
            "anthropic": types.SimpleNamespace(Anthropic=FakeClient),
        }
        with patch.dict(sys.modules, fake):
            for provider in ("openai", "anthropic"):
                backend = BaseDecision.from_provider(provider, "model", api_key="k")
                with self.subTest(provider), self.assertRaises(SystemOneError) as ctx:
                    SystemOne(backend)
                self.assertEqual(ctx.exception.code, "unsupported_backend")
                self.assertEqual(backend._client.calls, [], "no request may be sent")

    def test_backend_without_probabilities_is_rejected(self) -> None:
        class NoProbabilities(ScriptedBackend):
            def _result(self, request: Request) -> Result:
                result = super()._result(request)
                return Result(**{**result.__dict__, "probabilities": None})

        with self.assertRaises(SystemOneError) as ctx:
            SystemOne(NoProbabilities())(self.body())
        self.assertEqual(ctx.exception.code, "unsupported_backend")

    def test_wrong_result_count_is_an_internal_error(self) -> None:
        class Short(ScriptedBackend):
            def predict_batch(self, requests: Any) -> list[Result]:
                return super().predict_batch(requests)[:1]

        with self.assertRaises(RuntimeError):
            SystemOne(Short())(self.body())

    def test_context_length_error_is_attributed_with_numbers(self) -> None:
        backend = ScriptedBackend()
        backend.max_context_chars = 10
        with self.assertRaises(SystemOneError) as ctx:
            SystemOne(backend)(self.body("x" * 50))
        error = ctx.exception
        self.assertEqual(error.code, "context_length_exceeded")
        self.assertEqual(error.field, "questions.a")
        self.assertEqual(error.details, {"required_tokens": 50, "maximum_tokens": 10})
        self.assertIn("Nothing was truncated", str(error))

    def test_context_length_error_without_count_tokens(self) -> None:
        class Bare:
            def predict_batch(self, requests: Any) -> list[Result]:
                raise ContextLengthError(9000, 8192)

        with self.assertRaises(SystemOneError) as ctx:
            SystemOne(Bare())(self.body())  # type: ignore[arg-type]
        self.assertEqual(ctx.exception.code, "context_length_exceeded")
        self.assertEqual(ctx.exception.details, {"required_tokens": 9000, "maximum_tokens": 8192})

    def test_other_input_errors_are_attributed_to_the_question(self) -> None:
        class Colliding(ScriptedBackend):
            def count_tokens(self, request: Request) -> int:
                if request.question == "Other?":
                    raise InputError("Options tokenize identically")
                return super().count_tokens(request)

            def predict_batch(self, requests: Any) -> list[Result]:
                raise InputError("Options tokenize identically")

        with self.assertRaises(SystemOneError) as ctx:
            SystemOne(Colliding())(self.body())
        self.assertEqual(
            (ctx.exception.code, ctx.exception.field), ("invalid_criteria", "questions.b")
        )

    def test_unattributable_input_error_is_an_invalid_request(self) -> None:
        class Opaque(ScriptedBackend):
            def predict_batch(self, requests: Any) -> list[Result]:
                raise InputError("scope error")

        with self.assertRaises(SystemOneError) as ctx:
            SystemOne(Opaque())(self.body())
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_unexpected_backend_errors_propagate_unchanged(self) -> None:
        class Broken(ScriptedBackend):
            def predict_batch(self, requests: Any) -> list[Result]:
                raise FloatingPointError("Nonfinite model outputs")

        with self.assertRaises(FloatingPointError):
            SystemOne(Broken())(self.body())

    def test_shared_service_is_thread_safe(self) -> None:
        service = SystemOne(ScriptedBackend())
        bodies = [self.body(f"state number {i}") for i in range(64)]
        expected = [service(b) for b in bodies]
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(service, bodies))
        self.assertEqual(got, expected)


class CalibratedBackendTests(unittest.TestCase):
    """``CalibratedDecision`` (choice-only) works as a backend and returns calibrated values."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        artifact = json.loads(_DATA.read_text())
        artifact["checkpoint"] = {}
        for name in (
            "model.safetensors",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            artifact["checkpoint"][name] = _sha(path)
        artifact["scopes"]["sgd_identifier"].update(
            min_tokens=1, max_tokens=1000, min_options=2, max_options=3
        )
        artifact_path = root / "calibration.json"
        artifact_path.write_text(json.dumps(artifact))

        class CudaScripted(ScriptedBackend):
            precision = "bf16"
            device = types.SimpleNamespace(type="cuda")

        self.raw = CudaScripted(lambda request: [0.0, 4.0][: len(request.options)])
        self.raw.model_id = str(root)
        calibrated = CalibratedDecision(self.raw, profile="sgd_identifier", artifact=artifact_path)
        self.service = SystemOne(calibrated)

    def test_choice_answers_are_calibrated_and_keep_the_decision(self) -> None:
        body = {
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "choice", "criteria": {"a": "first", "b": "second"}}},
        }
        answer = self.service(body)["answers"]["q"]
        raw = self.raw.predict_batch(self.raw.calls[-1])[0]
        self.assertEqual(answer["choice"], raw.option_id)
        self.assertLess(answer["confidence"], round(raw.probabilities[raw.option_id], 4))
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, delta=2e-4)

    def test_boolean_and_score_questions_are_refused_with_a_clear_message(self) -> None:
        for question in ({"type": "noul"}, {"type": "score", "criteria": ["lo", "hi"]}):
            with self.subTest(question["type"]), self.assertRaises(SystemOneError) as ctx:
                self.service({"model": "m", "state": "s", "questions": {"q": question}})
            self.assertEqual(ctx.exception.code, "invalid_request")
            self.assertIn("choice only", str(ctx.exception))


class ErrorObjectTests(unittest.TestCase):
    def test_is_an_input_error_and_a_value_error(self) -> None:
        self.assertTrue(issubclass(SystemOneError, InputError))
        self.assertTrue(issubclass(SystemOneError, ValueError))

    def test_pickle_round_trip_keeps_all_attributes(self) -> None:
        error = SystemOneError(
            "context_length_exceeded",
            "too long",
            field="questions.q",
            required_tokens=9000,
            maximum_tokens=8192,
        )
        clone = pickle.loads(pickle.dumps(error))
        self.assertEqual(
            (type(clone), clone.code, str(clone), clone.field, clone.details),
            (
                SystemOneError,
                "context_length_exceeded",
                "too long",
                "questions.q",
                {"required_tokens": 9000, "maximum_tokens": 8192},
            ),
        )

    def test_to_dict_is_json_serializable(self) -> None:
        error = SystemOneError("invalid_request", "bad", field="state")
        self.assertEqual(
            error.to_dict(),
            {"error": {"code": "invalid_request", "message": "bad", "field": "state"}},
        )
        bare = SystemOneError("unsupported_backend", "no")
        self.assertEqual(
            bare.to_dict(), {"error": {"code": "unsupported_backend", "message": "no"}}
        )
        json.dumps(error.to_dict())

    def test_messages_never_echo_the_state(self) -> None:
        secret = "SECRET-CUSTOMER-TEXT-7731"
        backend = ScriptedBackend()
        backend.max_context_chars = 5
        service = SystemOne(backend, max_state_chars=100)
        attempts: list[Any] = [
            {"model": "m", "state": secret, "questions": {"q": {"type": "noul"}}},  # too long
            {"model": "m", "state": secret, "questions": {"q": {"type": "bogus"}}},
            {
                "model": "m",
                "state": secret,
                "images": [secret],
                "questions": {"q": {"type": "noul"}},
            },
            {"model": "m", "state": {secret: {1, 2}}, "questions": {"q": {"type": "noul"}}},
            {
                "model": "m",
                "state": secret,
                "questions": {"q": {"type": "choice", "criteria": {"a": secret, "b": secret}}},
            },
        ]
        for body in attempts:
            with self.assertRaises(SystemOneError) as ctx:
                service(body)
            self.assertNotIn(secret, str(ctx.exception))
            self.assertNotIn(secret, json.dumps(ctx.exception.to_dict()))


class ParseRequestBodyTests(unittest.TestCase):
    def test_parses_text_and_bytes(self) -> None:
        self.assertEqual(parse_request_body('{"a": [1, 2.5, "é"]}'), {"a": [1, 2.5, "é"]})
        self.assertEqual(parse_request_body('{"a": "é"}'.encode()), {"a": "é"})

    def test_rejects_what_a_lenient_parser_would_accept(self) -> None:
        bad = [
            b"",
            "not json",
            "[1, 2]",
            "5",
            '"text"',
            "null",
            '{"a": NaN}',
            '{"a": Infinity}',
            '{"a": -Infinity}',
            '{"a": 1, "a": 2}',
            '{"a": 1} trailing',
            b"\xff\xfe\x00{",
            "[" * 100000,
        ]
        for body in bad:
            with self.subTest(body=body if len(body) < 30 else "deeply nested"):
                with self.assertRaises(SystemOneError) as ctx:
                    parse_request_body(body)
                self.assertEqual(ctx.exception.code, "invalid_request")

    def test_duplicate_question_ids_are_rejected_not_merged(self) -> None:
        text = '{"model":"m","state":"s","questions":{"q":{"type":"noul"},"q":{"type":"noul"}}}'
        with self.assertRaises(SystemOneError):
            parse_request_body(text)

    def test_error_messages_do_not_echo_the_body(self) -> None:
        with self.assertRaises(SystemOneError) as ctx:
            parse_request_body('{"secret-value-123": oops}')
        self.assertNotIn("secret-value-123", str(ctx.exception))


class PackageSurfaceTests(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertIs(basedecision.SystemOne, SystemOne)
        self.assertIs(basedecision.SystemOneError, SystemOneError)
        self.assertIn("SystemOne", basedecision.__all__)

    def test_submodule_is_not_shadowed_by_a_function(self) -> None:
        import basedecision.systemone as module

        self.assertTrue(isinstance(module, types.ModuleType))
        self.assertIs(module.SystemOne, SystemOne)

    def test_importing_the_adapter_does_not_load_torch(self) -> None:
        import subprocess

        code = "import basedecision.systemone, sys; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)


class FuzzTests(unittest.TestCase):
    """Random corruption of a valid request: only a valid response or SystemOneError may result."""

    ERROR_CODES = set(get_args(ErrorCode))

    @staticmethod
    def random_json(rng: random.Random, depth: int = 0) -> Any:
        kinds = ["null", "bool", "int", "float", "str", "empty", "unicode"]
        if depth < 3:
            kinds += ["list", "dict"]
        kind = rng.choice(kinds)
        if kind == "null":
            return None
        if kind == "bool":
            return rng.random() < 0.5
        if kind == "int":
            return rng.choice([0, 1, -1, 2, 255, 256, 10**12, -(10**12)])
        if kind == "float":
            return rng.choice([0.0, 0.5, -1.5, 1e300, 1e-300])
        if kind == "str":
            return rng.choice(["a", "choice", "score", "noul", "true", "false", "x" * 40, " "])
        if kind == "empty":
            return rng.choice(["", [], {}])
        if kind == "unicode":
            return rng.choice(["é", "✓✓", "你好", "\u200b", "\x00", "á"])
        if kind == "list":
            return [FuzzTests.random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]
        return {
            rng.choice(
                ["a", "b", "type", "criteria", "true", "false", "0", "1", "x y"]
            ): FuzzTests.random_json(rng, depth + 1)
            for _ in range(rng.randint(0, 4))
        }

    @staticmethod
    def mutate(rng: random.Random, node: Any) -> Any:
        """Replace/delete one random descendant of ``node`` (in place); return the root."""
        paths: list[list[Any]] = []

        def walk(value: Any, path: list[Any]) -> None:
            paths.append(path)
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, [*path, key])
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, [*path, index])

        walk(node, [])
        path = rng.choice(paths)
        if not path:
            return FuzzTests.random_json(rng)
        parent = node
        for step in path[:-1]:
            parent = parent[step]
        if rng.random() < 0.25 and isinstance(parent, dict):
            del parent[path[-1]]
        else:
            parent[path[-1]] = FuzzTests.random_json(rng)
        return node

    def test_mutated_requests_never_raise_unexpected_exceptions(self) -> None:
        service = SystemOne(ScriptedBackend())
        rng = random.Random(20261005)
        outcomes = {"ok": 0, "rejected": 0}
        for _ in range(4000):
            body = copy.deepcopy(REFERENCE_REQUEST)
            for _ in range(rng.randint(1, 3)):
                body = self.mutate(rng, body)
            try:
                response = service(body)
            except SystemOneError as error:
                outcomes["rejected"] += 1
                self.assertIn(error.code, self.ERROR_CODES)
                json.dumps(error.to_dict())
                self.assertIsInstance(str(error), str)
            else:
                outcomes["ok"] += 1
                json.dumps(response)  # must be plain JSON
                self.assertEqual(set(response), {"model", "answers", "usage"})
                for answer in response["answers"].values():
                    self.assertIn(answer["type"], ("choice", "score", "noul"))
        self.assertGreater(outcomes["ok"], 100, outcomes)
        self.assertGreater(outcomes["rejected"], 1000, outcomes)


@unittest.skipUnless(os.environ.get("BASEDECISION_TEST_MODEL"), "set BASEDECISION_TEST_MODEL")
class RealModelTests(unittest.TestCase):
    """The adapter against a real checkpoint, cross-checked with the native API."""

    model: Any
    service: SystemOne

    @classmethod
    def setUpClass(cls) -> None:
        from basedecision import load

        cls.model = load(
            os.environ["BASEDECISION_TEST_MODEL"],
            device=os.environ.get("BASEDECISION_TEST_DEVICE", "cpu"),
            precision=os.environ.get("BASEDECISION_TEST_PRECISION", "fp32"),
        )
        cls.service = SystemOne(cls.model)

    def test_reference_example_is_well_formed_and_sensible(self) -> None:
        response = self.service(REFERENCE_REQUEST)
        answers = response["answers"]
        department = answers["department"]
        self.assertEqual(
            department["choice"],
            max(department["probabilities"], key=department["probabilities"].get),
        )
        self.assertAlmostEqual(sum(department["probabilities"].values()), 1.0, delta=2e-4)
        urgency = answers["urgency"]
        self.assertTrue(0.0 <= urgency["score"] <= 2.0)
        self.assertAlmostEqual(sum(urgency["probabilities"].values()), 1.0, delta=3e-4)
        self.assertTrue(0.0 <= answers["outage"]["noul"] <= 1.0)
        self.assertGreater(response["usage"]["input_tokens"], 0)
        self.assertEqual(json.loads(json.dumps(response)), response)

    def test_answers_equal_the_native_api(self) -> None:
        state = REFERENCE_REQUEST["state"]
        response = self.service(REFERENCE_REQUEST)["answers"]
        native = self.model.predict(
            Request(
                state,
                "Which team should handle the message?",
                (Option("billing", "Payments or invoices"), Option("technical", "Bugs or outages")),
            )
        )
        for option_id, probability in native.probabilities.items():
            self.assertEqual(
                response["department"]["probabilities"][option_id], round(probability, 4)
            )
        self.assertEqual(response["department"]["choice"], native.option_id)
        check = self.model.check(context=state, question="Is a service down?")
        self.assertEqual(response["outage"]["noul"], round(check.probabilities["true"], 4))
        levels = ["Can wait", "This week", "Today"]
        score = self.model.score(context=state, question="urgency", levels=levels, values=[0, 1, 2])
        self.assertEqual(response["urgency"]["score"], round(score.expected_value, 4))

    def test_json_state_and_key_order_invariance(self) -> None:
        body = {
            "model": "m",
            "state": {"invoice": {"total": 1250.0, "status": "overdue"}},
            "questions": {
                "status": {
                    "type": "choice",
                    "instructions": "What is the invoice status?",
                    "criteria": {
                        "paid": "Invoice is paid.",
                        "overdue": "Invoice is past due.",
                        "draft": "Not sent.",
                    },
                }
            },
        }
        first = self.service(body)["answers"]["status"]
        body["questions"]["status"]["criteria"] = dict(
            reversed(list(body["questions"]["status"]["criteria"].items()))
        )
        second = self.service(body)["answers"]["status"]
        self.assertEqual(first["choice"], second["choice"])
        self.assertEqual(
            sorted(first["probabilities"].items()), sorted(second["probabilities"].items())
        )

    def test_documentation_snippets_run(self) -> None:
        """The SystemOne code in the README and docs executes and returns an answers dict."""
        root = Path(__file__).resolve().parents[1]
        sources = {
            "docs/SYSTEMONE.md": (root / "docs" / "SYSTEMONE.md").read_text(),
            "README.md": (root / "README.md").read_text().split("## Jev / SystemOne API", 1)[-1],
        }
        for name, text in sources.items():
            block = re.search(r"```python\n(.*?)```", text, re.S)
            self.assertIsNotNone(block, name)
            code = re.sub(
                r"load\(['\"]/path/to/[^'\"]*['\"](?:,[^)]*)?\)",
                "load(MODEL, device=DEVICE, precision=PRECISION)",
                block.group(1),  # type: ignore[union-attr]
            )
            namespace: dict[str, Any] = {
                "MODEL": os.environ["BASEDECISION_TEST_MODEL"],
                "DEVICE": os.environ.get("BASEDECISION_TEST_DEVICE", "cpu"),
                "PRECISION": os.environ.get("BASEDECISION_TEST_PRECISION", "fp32"),
            }
            with self.subTest(name):
                exec(compile(code, name, "exec"), namespace)  # noqa: S102
                self.assertIn("answers", namespace["response"])

    def test_oversized_state_is_rejected_not_truncated(self) -> None:
        body = {"model": "m", "state": "word " * 20000, "questions": {"q": {"type": "noul"}}}
        with self.assertRaises(SystemOneError) as ctx:
            self.service(body)
        self.assertEqual(ctx.exception.code, "context_length_exceeded")
        self.assertGreater(
            ctx.exception.details["required_tokens"], ctx.exception.details["maximum_tokens"]
        )


if __name__ == "__main__":
    unittest.main()
