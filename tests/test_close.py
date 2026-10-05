"""``close()`` on the local model, the cloud backend and the context-manager protocol."""

from __future__ import annotations

import gc
import os
import threading
import unittest
import weakref
from types import SimpleNamespace
from typing import Any

from basedecision import BaseDecision, InputError, Option, Request
from basedecision.cpu import CPUFastDecision

MODEL = os.environ.get("BASEDECISION_TEST_MODEL", "")


class Tokenizer:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> Any:
        return {"input_ids": [ord(c) + 10 for c in text]}


class Weights:
    """Stands in for the network: something that can be garbage collected."""


def request(context: str = "abc") -> Request:
    return Request(context, "Which?", (Option("a", "Alpha"), Option("b", "Beta")))


def make_engine(device_type: str = "cpu", forward: Any = None) -> tuple[BaseDecision, list[str]]:
    """A local model without weights: ``_forward`` is scripted, ``torch.cuda`` is recorded."""
    cache_calls: list[str] = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: cache_calls.append("c")))
    engine = BaseDecision.__new__(BaseDecision)
    engine._lock = threading.RLock()
    engine._tokenizer = Tokenizer()
    engine._model = Weights()
    engine._torch = fake_torch
    engine.device = SimpleNamespace(type=device_type)
    engine.precision, engine.model_id = "fp32", "/checkpoint"
    engine.maximum_tokens, engine.max_batch_size, engine.max_batch_tokens = 8192, 1, 8192
    engine._forward = forward or (lambda items: [None] * len(items))
    engine._result = lambda r, p, z: r.context
    return engine, cache_calls


class LocalCloseTests(unittest.TestCase):
    def test_close_blocks_every_request_method_with_a_clear_error(self) -> None:
        engine, _ = make_engine()
        self.assertFalse(engine.closed)
        self.assertEqual(engine.predict(request("x")), "x")
        engine.close()
        self.assertTrue(engine.closed)
        calls = {
            "predict": lambda: engine.predict(request()),
            "predict_batch": lambda: engine.predict_batch([request()]),
            "predict_iter": lambda: list(engine.predict_iter([request()])),
            "choose": lambda: engine.choose(context="c", question="q", options=["a", "b"]),
            "check": lambda: engine.check(context="c", question="q"),
            "score": lambda: engine.score(context="c", question="q", levels=["a", "b"]),
            "count_tokens": lambda: engine.count_tokens(request()),
        }
        for name, call in calls.items():
            with self.subTest(method=name), self.assertRaisesRegex(InputError, "closed"):
                call()

    def test_close_releases_the_weights_and_the_tokenizer(self) -> None:
        engine, _ = make_engine()
        weights, tokenizer = weakref.ref(engine._model), weakref.ref(engine._tokenizer)
        engine.close()
        gc.collect()
        self.assertIsNone(weights(), "the network is still referenced after close()")
        self.assertIsNone(tokenizer(), "the tokenizer is still referenced after close()")

    def test_close_is_idempotent_and_leaves_descriptive_state_readable(self) -> None:
        engine, _ = make_engine()
        engine.close()
        engine.close()
        self.assertTrue(engine.closed)
        self.assertEqual((engine.model_id, engine.precision), ("/checkpoint", "fp32"))
        self.assertEqual(engine.device.type, "cpu")

    def test_the_gpu_cache_is_emptied_for_cuda_only(self) -> None:
        for device_type, expected in (("cuda", ["c"]), ("cpu", [])):
            with self.subTest(device=device_type):
                engine, cache_calls = make_engine(device_type)
                engine.close()
                engine.close()
                self.assertEqual(cache_calls, expected)

    def test_context_manager_closes_on_exit_and_on_error(self) -> None:
        engine, _ = make_engine()
        with engine as entered:
            self.assertIs(entered, engine)
            self.assertFalse(engine.closed)
        self.assertTrue(engine.closed)

        engine, _ = make_engine()
        with self.assertRaisesRegex(RuntimeError, "boom"), engine:
            raise RuntimeError("boom")  # not swallowed, and the model is still released
        self.assertTrue(engine.closed)

    def test_a_cpu_fast_model_inherits_close(self) -> None:
        self.assertIs(CPUFastDecision.close, BaseDecision.close)
        self.assertIs(CPUFastDecision.__exit__, BaseDecision.__exit__)

    def test_close_waits_for_a_running_request_which_still_finishes(self) -> None:
        started, release = threading.Event(), threading.Event()

        def slow_forward(items: Any) -> list[None]:
            started.set()
            release.wait(10)
            return [None] * len(items)

        engine, _ = make_engine(forward=slow_forward)
        outcome: list[Any] = []
        runner = threading.Thread(target=lambda: outcome.append(engine.predict(request("done"))))
        runner.start()
        self.assertTrue(started.wait(10))
        closer = threading.Thread(target=engine.close)
        closer.start()
        closer.join(0.3)
        self.assertTrue(closer.is_alive(), "close() did not wait for the running request")
        self.assertFalse(engine.closed)
        release.set()
        runner.join(10)
        closer.join(10)
        self.assertEqual(outcome, ["done"])  # the request completed normally
        self.assertTrue(engine.closed)
        with self.assertRaises(InputError):
            engine.predict(request())

    def test_concurrent_requests_either_finish_or_get_the_closed_error(self) -> None:
        enough = threading.Event()
        counter = {"n": 0}

        def forward(items: Any) -> list[None]:
            counter["n"] += 1
            if counter["n"] >= 20:
                enough.set()
            return [None] * len(items)

        engine, _ = make_engine(forward=forward)
        finished: list[str] = []  # how each worker's loop ended
        completed = {"n": 0}
        guard = threading.Lock()

        def worker() -> None:
            for _ in range(100_000):  # keeps going until it meets the closed model
                try:
                    engine.predict(request())
                except InputError:
                    verdict = "closed"
                    break
                except BaseException as error:  # anything else is a bug (e.g. AttributeError)
                    verdict = f"BUG {type(error).__name__}: {error}"
                    break
                with guard:
                    completed["n"] += 1
            else:
                verdict = "never closed"
            with guard:
                finished.append(verdict)

        workers = [threading.Thread(target=worker) for _ in range(8)]
        for thread in workers:
            thread.start()
        self.assertTrue(enough.wait(10))
        engine.close()
        for thread in workers:
            thread.join(30)
        self.assertEqual(finished, ["closed"] * 8)
        self.assertGreaterEqual(completed["n"], 20)  # requests ran before and while closing


class CloudCloseTests(unittest.TestCase):
    def test_the_cloud_backend_reports_closed_like_the_local_model(self) -> None:
        from basedecision.providers import ProviderDecision

        backend = ProviderDecision.__new__(ProviderDecision)
        closed_clients: list[str] = []
        backend._client = SimpleNamespace(close=lambda: closed_clients.append("x"))
        backend._closed = False
        self.assertFalse(backend.closed)
        with backend as entered:
            self.assertIs(entered, backend)
        self.assertTrue(backend.closed)
        backend.close()
        self.assertEqual(closed_clients, ["x"])  # the HTTP client is closed exactly once


@unittest.skipUnless(MODEL, "set BASEDECISION_TEST_MODEL to a checkpoint directory")
class RealModelCloseTests(unittest.TestCase):
    def test_with_block_frees_the_real_weights_and_the_error_is_clear(self) -> None:
        from basedecision import load

        with load(MODEL) as model:
            answer = model.check(context="The account is active.", question="Is it active?").answer
            network = weakref.ref(model._model)
            self.assertIs(answer, True)
        gc.collect()
        self.assertTrue(model.closed)
        self.assertIsNone(network(), "the real network is still in memory after the with block")
        with self.assertRaisesRegex(InputError, "closed"):
            model.check(context="The account is active.", question="Is it active?")

    def test_a_model_can_be_loaded_again_after_closing_another(self) -> None:
        from basedecision import load

        first = load(MODEL)
        first.close()
        with load(MODEL) as second:
            result = second.choose(
                context="Please refund my purchase.",
                question="What does the customer request?",
                options=["Refund", "Delivery status", "Other"],
            )
        self.assertEqual(result.answer, "Refund")

    def test_cpu_fast_models_close_too(self) -> None:
        from basedecision import CPUFastUnavailable, load

        try:
            fast = load(MODEL, backend="cpu_fast")
        except CPUFastUnavailable:
            self.skipTest("cpu_fast is unavailable on this installation")
        with fast:
            self.assertEqual(fast.check(context="It is on.", question="Is it on?").answer, True)
        self.assertTrue(fast.closed)


if __name__ == "__main__":
    unittest.main()
