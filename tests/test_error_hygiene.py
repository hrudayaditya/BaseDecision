"""Exception hygiene: no leaks through exception chains, and every exception survives pickling.

Two properties, tested systematically rather than case by case:

* **No retained input.** An error that has been sanitized must not keep the original exception
  alive. Python attaches the exception being handled as ``__context__`` to anything raised inside
  its ``except`` block (``from None`` only hides it from printing), and such originals hold the
  request: an SDK exception holds the HTTP request including the API key header, a
  ``json.JSONDecodeError`` holds the whole document in ``.doc``. Tests walk the whole chain.
* **Every exception pickles.** Errors cross process boundaries (``ProcessPoolExecutor``,
  ``multiprocessing``, Celery, Ray, Dask). An exception whose ``__init__`` takes different arguments
  than ``args`` cannot be unpickled and breaks the whole pool. A registry covers *every* exception
  class the package defines, and a guard fails when a new one is added without being listed.
"""

from __future__ import annotations

import concurrent.futures
import copy
import importlib
import importlib.util
import multiprocessing
import os
import pickle
import sys
import traceback
import unittest
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

from basedecision import (
    BaseDecision,
    CalibrationError,
    ContextLengthError,
    InputError,
    Option,
    Request,
    SystemOne,
    SystemOneError,
)
from basedecision.errors import ProviderError, ProviderResponseError
from basedecision.providers import parse_selection
from basedecision.systemone import parse_request_body

SECRET = "SECRET-7f3a-customer-text"
API_KEY = "sk-test-KEY-9d41"
SDK_MODULES = ("openai", "anthropic", "httpx", "httpx2", "httpcore", "anyio", "h11")
# Attributes that mean "this exception carries the data that caused it".
DATA_ATTRIBUTES = ("doc", "request", "response", "body", "object")


def chain(error: BaseException) -> Iterator[BaseException]:
    """Yield ``error`` and every exception reachable through ``__cause__``/``__context__``."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending += [e for e in (current.__cause__, current.__context__) if e is not None]


def traceback_modules(error: BaseException) -> set[str]:
    """Names of every module that has a frame in the traceback of ``error`` or its chain."""
    modules: set[str] = set()
    for item in chain(error):
        tb = item.__traceback__
        while tb is not None:
            modules.add(str(tb.tb_frame.f_globals.get("__name__", "")))
            tb = tb.tb_next
    return modules


class HygieneCase(unittest.TestCase):
    def assertClean(self, error: BaseException) -> None:
        """The error holds no original exception, no SDK frames, and no secret in its text."""
        self.assertEqual(list(chain(error)), [error], "an exception is attached to the error")
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        for item in chain(error):
            for name in DATA_ATTRIBUTES:
                self.assertFalse(hasattr(item, name), f"{type(item).__name__}.{name}")
        for module in traceback_modules(error):
            self.assertFalse(module.split(".")[0] in SDK_MODULES, f"frame from {module}")
        rendered = "".join(traceback.format_exception(error)) + repr(error) + str(error)
        for secret in (SECRET, API_KEY):
            self.assertNotIn(secret, rendered)


class ParserChainTests(HygieneCase):
    def capture(self, function: Any, *args: Any) -> BaseException:
        with self.assertRaises((ProviderResponseError, SystemOneError)) as caught:
            function(*args)
        return caught.exception

    def test_parse_selection_does_not_keep_the_model_reply(self) -> None:
        for reply in (
            '{"option_index": "' + SECRET + '" ',  # truncated JSON
            "not json " + SECRET,
            "",
            '{"option_index": 1, "option_index": 2, "x": "' + SECRET + '"}',  # duplicate key
            "[" * 5000,  # recursion
        ):
            with self.subTest(reply=reply[:30]):
                self.assertClean(self.capture(parse_selection, reply, 2))

    def test_parse_request_body_does_not_keep_the_request(self) -> None:
        bodies: list[str | bytes] = [
            '{"state": "' + SECRET + '" oops}',
            '{"state": "' + SECRET + '"} trailing',
            ('{"state": "' + SECRET + '"} ').encode() + b"\xff\xfe",  # invalid UTF-8
            '{"state": NaN, "x": "' + SECRET + '"}',
            '{"' + SECRET + '": 1, "' + SECRET + '": 2}',  # duplicate key: name must not be echoed
            "[" * 5000 + SECRET,
            "[1, 2] " + SECRET,
        ]
        for body in bodies:
            with self.subTest(body=body[:30]):
                self.assertClean(self.capture(parse_request_body, body))

    def test_unserializable_state_does_not_keep_the_value(self) -> None:
        class Backend:
            def predict_batch(self, requests: Any) -> list[Any]:
                raise AssertionError("must not run")

        service = SystemOne(Backend())  # type: ignore[arg-type]
        for state in ({SECRET, 1}, {1: SECRET, "b": 2}, float("nan"), object()):
            with self.subTest(state=type(state).__name__):
                request = {"model": "m", "state": state, "questions": {"q": {"type": "noul"}}}
                self.assertClean(self.capture(service, request))

    def test_systemone_error_battery_never_chains_to_data_holding_exceptions(self) -> None:
        class Backend:
            def predict_batch(self, requests: Any) -> list[Any]:
                raise ContextLengthError(10, 5)

            def count_tokens(self, request: Any) -> int:
                raise ContextLengthError(10, 5)

        service = SystemOne(Backend())  # type: ignore[arg-type]
        question = {"q": {"type": "choice", "criteria": {"a": SECRET, "b": SECRET + "2"}}}
        bad_requests: list[Any] = [
            {"model": "m", "state": SECRET, "questions": {"q": {"type": "bogus"}}},
            {"model": "m", "state": SECRET, "questions": question},
            {"model": "m", "state": SECRET, "images": [SECRET], "questions": question},
            {
                "model": "m",
                "state": SECRET,
                "questions": {"q": {"type": "choice", "criteria": {"a": SECRET}}},
            },
            {"model": "m", "state": SECRET * 3, "questions": {"q": {"type": "noul"}}},
        ]
        for body in bad_requests:
            with self.subTest(body=str(body)[:40]), self.assertRaises(SystemOneError) as caught:
                service(body)
            for item in chain(caught.exception):
                for name in DATA_ATTRIBUTES:
                    self.assertFalse(hasattr(item, name), f"{type(item).__name__}.{name}")
            self.assertNotIn(SECRET, str(caught.exception))


def _http_flavours() -> list[Any]:
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
class ProviderChainTests(HygieneCase):
    """Real SDKs over an in-memory transport: every failure mode yields a clean error."""

    request = Request(SECRET, "Which?", (Option("a", "Alpha"), Option("b", "Beta")))

    def provider(self, provider: str, respond: Any, retries: int = 0) -> BaseDecision:
        from anthropic import Anthropic
        from openai import OpenAI

        decision = BaseDecision.from_provider(
            provider, "test-model", api_key=API_KEY, max_retries=retries
        )
        base_url = str(decision._client.base_url)
        decision._client.close()
        factory = OpenAI if provider == "openai" else Anthropic
        for module in _http_flavours():
            try:
                decision._client = factory(
                    api_key=API_KEY,
                    base_url=base_url,
                    max_retries=retries,
                    http_client=module.Client(
                        transport=module.MockTransport(lambda r, m=module: respond(m, r))
                    ),
                )
                return decision
            except TypeError:
                continue
        raise AssertionError("no usable httpx flavour")

    def failure(self, provider: str, respond: Any, retries: int = 0) -> ProviderError:
        decision = self.provider(provider, respond, retries)
        with patch("time.sleep"), self.assertRaises(ProviderError) as caught:
            decision.predict(self.request)
        return caught.exception

    def test_http_errors_leave_nothing_attached(self) -> None:
        body = {"error": {"message": f"leak {API_KEY} {SECRET}", "type": "x"}}
        for provider in ("openai", "anthropic"):
            for status in (400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 503, 529):
                with self.subTest(provider=provider, status=status):
                    error = self.failure(
                        provider, lambda m, r, s=status: m.Response(s, json=body), retries=1
                    )
                    self.assertEqual(error.status_code, status)
                    self.assertClean(error)

    def test_transport_failures_leave_nothing_attached(self) -> None:
        for provider in ("openai", "anthropic"):
            for exception_name in ("ReadTimeout", "ConnectError", "ConnectTimeout", "ReadError"):

                def respond(module: Any, request: Any, name: str = exception_name) -> Any:
                    raise getattr(module, name)(f"{SECRET} {API_KEY}", request=request)

                with self.subTest(provider=provider, failure=exception_name):
                    self.assertClean(self.failure(provider, respond, retries=1))

    def test_unexpected_exceptions_from_the_sdk_leave_nothing_attached(self) -> None:
        decision = self.provider("openai", lambda m, r: m.Response(200, json={}))

        class Exploding:
            def create(self, **kwargs: Any) -> Any:
                raise ValueError(f"{SECRET} {API_KEY} {kwargs['input'][:20]}")

        decision._client.responses = Exploding()  # type: ignore[assignment]
        with self.assertRaises(ProviderError) as caught:
            decision.predict(self.request)
        self.assertEqual(caught.exception.code, "provider_failure")
        self.assertClean(caught.exception)

    def test_malformed_model_replies_leave_nothing_attached(self) -> None:
        def reply(text: str) -> Any:
            body = {
                "id": "r",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "m",
                "output": [
                    {
                        "id": "m",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            }
            return lambda module, request: module.Response(200, json=body)

        for text in ('{"option_index": "' + SECRET + '" ', "nonsense " + SECRET, ""):
            with self.subTest(text=text[:25]):
                decision = self.provider("openai", reply(text))
                with self.assertRaises(ProviderResponseError) as caught:
                    decision.predict(self.request)
                self.assertClean(caught.exception)


# --------------------------------------------------------------------------------------------
# Pickling
# --------------------------------------------------------------------------------------------

SAMPLES: dict[str, Any] = {
    "InputError": lambda: InputError("bad input"),
    "ContextLengthError": lambda: ContextLengthError(9000, 8192, at_least=True),
    "CalibrationError": lambda: CalibrationError("out of scope"),
    "ProviderError": lambda: ProviderError("rate_limit", retryable=True, status_code=429),
    "ProviderResponseError": lambda: ProviderResponseError(
        "unexpected_output_block", detail="function_call"
    ),
    "SystemOneError": lambda: SystemOneError(
        "context_length_exceeded",
        "too long",
        field="questions.q",
        required_tokens=9000,
        maximum_tokens=8192,
    ),
}


def package_exception_classes() -> dict[str, type[BaseException]]:
    """Every exception class defined by the package (all light modules are imported with it)."""
    found: dict[str, type[BaseException]] = {}
    pending: list[type[BaseException]] = [Exception]
    while pending:
        for cls in pending.pop().__subclasses__():
            pending.append(cls)
            if cls.__module__.split(".")[0] == "basedecision":
                found[cls.__name__] = cls
    return found


def raise_in_worker(name: str) -> None:
    """Module-level so that a spawned worker can import it by name."""
    raise SAMPLES[name]()


def assert_equivalent(
    case: unittest.TestCase, original: BaseException, clone: BaseException
) -> None:
    case.assertIs(type(clone), type(original))
    case.assertEqual(str(clone), str(original))
    case.assertEqual(clone.args, original.args)
    case.assertEqual(vars(clone), vars(original))


class ExceptionPicklingTests(unittest.TestCase):
    def test_the_registry_covers_every_exception_class_in_the_package(self) -> None:
        # Fails when someone adds an exception class: add a sample for it to SAMPLES.
        self.assertEqual(set(package_exception_classes()), set(SAMPLES))

    def test_every_exception_round_trips_through_every_pickle_protocol(self) -> None:
        for name, factory in SAMPLES.items():
            original = factory()
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(exception=name, protocol=protocol):
                    clone = pickle.loads(pickle.dumps(original, protocol))
                    assert_equivalent(self, original, clone)

    def test_every_exception_survives_copy_and_deepcopy(self) -> None:
        for name, factory in SAMPLES.items():
            original = factory()
            for copier in (copy.copy, copy.deepcopy):
                with self.subTest(exception=name, copier=copier.__name__):
                    assert_equivalent(self, original, copier(original))

    def test_context_length_error_keeps_its_numbers(self) -> None:
        clone = pickle.loads(pickle.dumps(ContextLengthError(9000, 8192)))
        self.assertEqual((clone.required_tokens, clone.maximum_tokens), (9000, 8192))
        self.assertIn("9000", str(clone))
        self.assertIsInstance(clone, InputError)

    def test_a_raised_error_with_a_traceback_still_pickles(self) -> None:
        try:
            raise ContextLengthError(9000, 8192)
        except ContextLengthError as error:
            clone = pickle.loads(pickle.dumps(error))
        self.assertEqual(clone.required_tokens, 9000)

    def test_provider_error_message_is_not_doubled(self) -> None:
        clone = pickle.loads(pickle.dumps(ProviderError("rate_limit", status_code=429)))
        self.assertEqual(str(clone), "Provider request failed: rate_limit")

    def test_importing_the_package_registers_the_reducer(self) -> None:
        import subprocess

        code = (
            "import pickle, basedecision; "
            "e = pickle.loads(pickle.dumps(basedecision.ContextLengthError(9000, 8192))); "
            "assert (e.required_tokens, e.maximum_tokens) == (9000, 8192)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


def available_start_methods() -> list[str]:
    return [
        m for m in ("spawn", "fork", "forkserver") if m in multiprocessing.get_all_start_methods()
    ]


class MultiprocessingTests(unittest.TestCase):
    """The real scenario: an error raised in a worker process reaches the parent intact."""

    def test_errors_raised_in_workers_arrive_intact(self) -> None:
        for method in available_start_methods():
            context = multiprocessing.get_context(method)
            with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
                for name, factory in SAMPLES.items():
                    with self.subTest(start_method=method, exception=name):
                        future = pool.submit(raise_in_worker, name)
                        with self.assertRaises(type(factory())) as caught:
                            future.result(timeout=120)
                        assert_equivalent(self, factory(), caught.exception)

    def test_the_pool_survives_and_keeps_working_after_a_context_length_error(self) -> None:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
            with self.assertRaises(ContextLengthError) as caught:
                pool.submit(raise_in_worker, "ContextLengthError").result(timeout=120)
            self.assertEqual(caught.exception.required_tokens, 9000)
            # Before the fix this was a BrokenProcessPool: the pool was dead after one error.
            self.assertEqual(pool.submit(abs, -5).result(timeout=120), 5)


def predict_too_long_in_worker(model_path: str) -> None:
    """Load the real model in a worker process and give it an input that cannot fit."""
    from basedecision import load

    model = load(model_path)
    model.check(context="word " * 20000, question="Is this too long?")


@unittest.skipUnless(os.environ.get("BASEDECISION_TEST_MODEL"), "set BASEDECISION_TEST_MODEL")
class RealModelWorkerTests(unittest.TestCase):
    def test_context_length_error_from_a_real_worker_reaches_the_parent(self) -> None:
        """The reported scenario: ProcessPoolExecutor + the real model + an over-long input."""
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
            future = pool.submit(predict_too_long_in_worker, os.environ["BASEDECISION_TEST_MODEL"])
            with self.assertRaises(ContextLengthError) as caught:
                future.result(timeout=600)
        self.assertGreater(caught.exception.required_tokens, caught.exception.maximum_tokens)
        self.assertEqual(caught.exception.maximum_tokens, 8192)


if __name__ == "__main__":
    unittest.main()
