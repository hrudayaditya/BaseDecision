"""Checks for the experimental ``cpu_fast`` backend: version gate, self-test and the adapter itself.

``cpu_fast`` does not demand exact torch/transformers releases. It accepts a range (torch >= 2.6,
transformers 4.48-4.57) and proves, with a known-answer self-test on the installed versions, that
its patched attention reproduces the original. These tests cover the gate, the self-test (including
deliberately broken adapters, which it must catch), and the adapter. They run on every supported
install and skip elsewhere; with ``BASEDECISION_TEST_MODEL`` the real checkpoint is compared against
the default FP32 backend, and on an unsupported install ``cpu_fast`` must refuse clearly.
"""

from __future__ import annotations

import os
import threading
import unittest
import warnings
from typing import Any
from unittest.mock import patch

from basedecision import CPUFastDecision, CPUFastUnavailable, InputError
from basedecision import _cpu_selftest as selftest

try:
    import torch
    import transformers

    PROBLEM = selftest.version_problem(torch.__version__, transformers.__version__)
except ImportError:  # torch/transformers are optional
    PROBLEM = "torch and transformers are not installed"

SUPPORTED = PROBLEM is None
MODEL = os.environ.get("BASEDECISION_TEST_MODEL", "")
REASON = f"cpu_fast is not supported on this install ({PROBLEM})"
# The real checkpoint's attention geometry: 16 heads of size 64, a 128-token window, a global layer
# every third layer (hidden size and layer count do not matter for the geometry).
REAL_GEOMETRY = {
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "local_attention": 128,
    "global_attn_every_n_layers": 3,
}


def real_config() -> Any:
    from transformers import ModernBertConfig

    return ModernBertConfig(**REAL_GEOMETRY)


class VersionGateTests(unittest.TestCase):
    def test_supported_ranges(self) -> None:
        ok = [
            ("2.6.0", "4.48.0"),
            ("2.9.1", "4.57.6"),
            ("2.14.1+cu126", "4.57.6"),
            ("2.7.0a0+git123", "4.50.3"),
            ("2.99.0", "4.99.0"),
        ]
        for torch_version, transformers_version in ok:
            with self.subTest(torch=torch_version, transformers=transformers_version):
                self.assertIsNone(selftest.version_problem(torch_version, transformers_version))

    def test_unsupported_versions_say_why_and_what_to_do(self) -> None:
        bad = [
            ("2.5.1", "4.57.6", "torch 2.6"),
            ("1.13.0", "4.57.6", "torch 2.6"),
            ("3.0.0", "4.57.6", "torch 2.6"),
            ("2.9.1", "4.47.1", "4.48 to 4.57"),
            ("2.9.1", "5.0.0", "transformers 5"),
            ("2.14.1", "5.18.0", "transformers 5"),
            ("garbage", "4.57.6", "torch 2.6"),
            ("2.9.1", "", "4.48 to 4.57"),
        ]
        for torch_version, transformers_version, mention in bad:
            with self.subTest(torch=torch_version, transformers=transformers_version):
                problem = selftest.version_problem(torch_version, transformers_version)
                self.assertIsNotNone(problem)
                self.assertIn(mention, problem)
                self.assertIn("load(path)", problem)  # the default backend that does work

    def test_the_exception_is_an_input_error_and_can_be_caught_to_fall_back(self) -> None:
        self.assertTrue(issubclass(CPUFastUnavailable, InputError))
        self.assertTrue(issubclass(CPUFastUnavailable, ValueError))

    def test_an_unsupported_install_is_refused_before_any_weights_are_loaded(self) -> None:
        with patch("basedecision.cpu.version_problem", return_value="not supported here"):
            for path in ("/does/not/exist", "/another/missing/dir"):
                with self.subTest(path=path), self.assertRaises(CPUFastUnavailable) as caught:
                    CPUFastDecision(path)  # a FileNotFoundError would mean it tried to load
                self.assertEqual(str(caught.exception), "not supported here")

    @unittest.skipUnless(SUPPORTED, REASON)
    def test_wrong_device_or_precision_is_still_an_input_error_first(self) -> None:
        for kwargs in ({"device": "cuda"}, {"precision": "fp32"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(InputError) as caught:
                CPUFastDecision("/missing", **kwargs)
            self.assertNotIsInstance(caught.exception, CPUFastUnavailable)


@unittest.skipUnless(SUPPORTED, REASON)
class SelfTestTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(selftest._CACHE, {}, clear=True)  # every test starts uncached
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_passes_on_a_supported_install_and_reports_what_it_saw(self) -> None:
        result = selftest.verify(real_config())
        self.assertEqual(result.torch, torch.__version__)
        self.assertEqual(result.transformers, transformers.__version__)
        self.assertLess(
            result.relative_difference, selftest.FP32_TOLERANCE / 5, "should agree closely"
        )
        self.assertLess(result.seconds, 30)

    def test_the_result_is_cached_per_geometry(self) -> None:
        first = selftest.verify(real_config())
        self.assertIs(selftest.verify(real_config()), first)
        other = real_config()
        other.local_attention = 256
        self.assertIsNot(selftest.verify(other), first, "a different geometry must be re-tested")

    def test_concurrent_callers_share_one_run(self) -> None:
        runs = []
        original = selftest._run_self_test

        def counting(*args: Any) -> Any:
            runs.append(1)
            return original(*args)

        with patch.object(selftest, "_run_self_test", counting):
            threads = [
                threading.Thread(target=selftest.verify, args=(real_config(),)) for _ in range(6)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(runs), 1)

    def test_it_changes_no_global_state(self) -> None:
        from transformers.models.modernbert import modeling_modernbert as mb

        rng = torch.get_rng_state().clone()
        registry = dict(mb.MODERNBERT_ATTENTION_FUNCTION)
        flag = torch.backends.mha.get_fastpath_enabled()
        threads = torch.get_num_threads()
        attention_forward = mb.ModernBertAttention.forward
        selftest.verify(real_config())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()), "the caller's RNG state changed")
        self.assertEqual(mb.MODERNBERT_ATTENTION_FUNCTION, registry)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), flag)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertIs(mb.ModernBertAttention.forward, attention_forward)

    # -- the self-test must catch an adapter that does not work -------------------------------
    def assert_refused(self, expected: str) -> str:
        with self.assertRaises(CPUFastUnavailable) as caught:
            selftest.verify(real_config())
        message = str(caught.exception)
        self.assertIn(expected, message)
        self.assertIn("load(path)", message, "the message must say what to do instead")
        self.assertEqual(selftest._CACHE, {}, "a failure must never be cached as a success")
        return message

    def test_catches_an_adapter_that_crashes(self) -> None:
        def broken(model: Any) -> Any:
            raise AttributeError("module has no attribute 'MODERNBERT_ATTENTION_FUNCTION'")

        with patch("basedecision._cpu_attention.install", broken):
            message = self.assert_refused("AttributeError")
        self.assertIn(torch.__version__, message)
        self.assertIn(transformers.__version__, message)

    def test_catches_an_adapter_that_computes_something_else(self) -> None:
        from basedecision import _cpu_attention

        real_install = _cpu_attention.install

        def corrupting(model: Any) -> Any:
            counters = real_install(model)
            for module in model.modules():
                if hasattr(module, "Wo"):
                    module.Wo.weight.mul_(3.0)  # silently changes the numerics
            return counters

        with patch("basedecision._cpu_attention.install", corrupting):
            self.assert_refused("differ from the reference")

    def test_catches_even_a_one_percent_error(self) -> None:
        from basedecision import _cpu_attention

        real_install = _cpu_attention.install

        def slightly_off(model: Any) -> Any:
            counters = real_install(model)
            for module in model.modules():
                if hasattr(module, "Wo"):
                    module.Wo.weight.mul_(1.01)  # a subtle numeric error in one matrix per layer
            return counters

        with patch("basedecision._cpu_attention.install", slightly_off):
            self.assert_refused("differ from the reference")

    def test_catches_non_finite_outputs(self) -> None:
        from basedecision import _cpu_attention

        real_install = _cpu_attention.install

        def poisoning(model: Any) -> Any:
            counters = real_install(model)
            for module in model.modules():
                if hasattr(module, "Wo"):
                    module.Wo.weight.fill_(float("nan"))
                    break
            return counters

        with patch("basedecision._cpu_attention.install", poisoning):
            self.assert_refused("non-finite")

    def test_catches_an_adapter_that_is_never_used(self) -> None:
        with patch(
            "basedecision._cpu_attention.install",
            lambda model: {"tiled_calls": 0, "reference_calls": 7},
        ):
            self.assert_refused("not exercised")

    def test_a_failed_check_does_not_poison_a_later_good_one(self) -> None:
        with (
            patch("basedecision._cpu_attention.install", side_effect=RuntimeError("boom")),
            self.assertRaises(CPUFastUnavailable),
        ):
            selftest.verify(real_config())
        self.assertIsInstance(selftest.verify(real_config()), selftest.Verification)


@unittest.skipUnless(SUPPORTED, REASON)
class KernelEquivalenceTests(unittest.TestCase):
    def make_toy(self) -> Any:
        """A tiny random DecisionModel with the real attention geometry; identical every time."""
        from transformers import ModernBertConfig, ModernBertModel

        from basedecision._model import DecisionModel

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(92)
            config = ModernBertConfig(
                vocab_size=100,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=6,
                num_attention_heads=2,
                max_position_embeddings=2048,
                local_attention=REAL_GEOMETRY["local_attention"],
                global_attn_every_n_layers=REAL_GEOMETRY["global_attn_every_n_layers"],
                reference_compile=False,
            )
            config._attn_implementation = "sdpa"
            toy = DecisionModel(ModernBertModel(config), head_layers=2).eval()
            toy.requires_grad_(False)
        return toy

    def setUp(self) -> None:
        from transformers.models.modernbert import modeling_modernbert as mb

        self.mb = mb
        self.toy = self.make_toy()
        self.batches = []
        generator = torch.Generator().manual_seed(5)
        for size, length in [(1, 257), (1, 513), (1, 1100), (2, 513)]:
            mask = torch.ones(size, length, dtype=torch.long)
            if size == 2:
                mask[1, -50:] = 0  # padding: the fast path must fall back to the reference
            self.batches.append(
                {
                    "input_ids": torch.randint(0, 100, (size, length), generator=generator),
                    "attention_mask": mask,
                    "marker_pos": torch.tensor([[1, 3]] * size),
                    "marker_mask": torch.ones(size, 2, dtype=torch.bool),
                    "qtype": torch.zeros(size, dtype=torch.long),
                }
            )

    def run_batches(self, model: Any = None) -> list[Any]:
        model = model or self.toy
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            return [model(**batch)[0].clone() for batch in self.batches]

    def test_tiled_attention_and_head_match_the_reference(self) -> None:
        from basedecision._cpu_attention import install, install_head

        before = self.run_batches()
        counters = install(self.toy)
        install_head(self.toy)
        after = self.run_batches()
        for expected, actual in zip(before, after, strict=True):
            torch.testing.assert_close(expected, actual, atol=0.05, rtol=0.01)
        self.assertGreater(counters["tiled_calls"], 0, "the tiled path was never used")
        self.assertGreater(counters["reference_calls"], 0, "padded batches must use the reference")

    def test_installation_is_instance_local(self) -> None:
        from basedecision._cpu_attention import install, install_head

        attention_class, layer_class = self.mb.ModernBertAttention, torch.nn.TransformerEncoderLayer
        original_attention, original_layer = attention_class.forward, layer_class.forward
        registry = dict(self.mb.MODERNBERT_ATTENTION_FUNCTION)
        flag = torch.backends.mha.get_fastpath_enabled()
        untouched = self.make_toy()  # identical weights, never patched
        before = self.run_batches(untouched)
        install(self.toy)
        install_head(self.toy)
        self.run_batches()  # exercise the patched instance
        # Structural checks: the tiled and reference kernels may give bit-identical outputs, so
        # outputs alone cannot prove isolation. Nothing global or class-level may have changed...
        self.assertIs(attention_class.forward, original_attention)
        self.assertIs(layer_class.forward, original_layer)
        self.assertEqual(self.mb.MODERNBERT_ATTENTION_FUNCTION, registry)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), flag)
        # ...only the patched instance carries overrides, and the untouched one carries none.
        patched = [m for m in self.toy.modules() if "forward" in vars(m)]
        self.assertTrue(patched, "install() should have patched this instance")
        self.assertEqual([m for m in untouched.modules() if "forward" in vars(m)], [])
        for expected, actual in zip(before, self.run_batches(untouched), strict=True):
            self.assertTrue(torch.equal(expected, actual), "another instance was affected")


@unittest.skipUnless(SUPPORTED and MODEL, REASON + " and BASEDECISION_TEST_MODEL")
class RealCheckpointTests(unittest.TestCase):
    def test_fast_backend_agrees_with_fp32_and_leaves_other_instances_alone(self) -> None:
        from transformers.models.modernbert import modeling_modernbert as mb

        from basedecision import ContextLengthError, Option, Request, load

        registry = dict(mb.MODERNBERT_ATTENTION_FUNCTION)
        standard = load(MODEL, device="cpu", precision="fp32")
        options = (Option("a", "Refund"), Option("b", "Delivery"))
        requests = [
            Request("Please refund this purchase.", "What is requested?", options),
            Request(
                "No refund is requested.",
                "Refund requested?",
                (Option("false", "False"), Option("true", "True")),
                "noul",
            ),
            Request(
                "Service was okay.",
                "Rate service.",
                (Option("0", "Bad"), Option("1", "Okay"), Option("2", "Good")),
                "score",
            ),
            Request("Observation. " * 300 + "Please refund.", "What is requested?", options),
        ]
        reference = [standard.predict(r) for r in requests]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # the documented experimental warning
            fast = load(MODEL, backend="cpu_fast")
        for request, expected in zip(requests, reference, strict=True):
            self.assertEqual(fast.predict(request).option_id, expected.option_id)
        info = fast.backend_info()
        self.assertGreater(info["attention_calls"]["tiled_calls"], 0)
        self.assertEqual(info["self_test"]["torch"], torch.__version__)
        self.assertEqual(info["self_test"]["transformers"], transformers.__version__)
        self.assertLess(info["self_test"]["relative_difference"], selftest.FP32_TOLERANCE)
        self.assertEqual(mb.MODERNBERT_ATTENTION_FUNCTION, registry)
        self.assertEqual(
            [standard.predict(r).raw_logits for r in requests],
            [r.raw_logits for r in reference],
            "the default instance was affected",
        )
        with self.assertRaises(ContextLengthError):
            fast.count_tokens(Request("word " * 20000, "Question?", options))


@unittest.skipUnless(
    MODEL and not SUPPORTED, "needs BASEDECISION_TEST_MODEL on an unsupported install"
)
class UnsupportedInstallTests(unittest.TestCase):
    def test_cpu_fast_refuses_clearly_and_the_default_backend_still_works(self) -> None:
        import time

        from basedecision import load

        started = time.perf_counter()
        with self.assertRaises(CPUFastUnavailable) as caught:
            load(MODEL, backend="cpu_fast")
        self.assertLess(time.perf_counter() - started, 2.0, "must refuse before loading weights")
        self.assertIn("load(path)", str(caught.exception))
        model = load(MODEL)  # exactly what the message recommends
        self.assertEqual(
            model.check(context="The account is active.", question="Active?").kind, "noul"
        )


if __name__ == "__main__":
    unittest.main()
