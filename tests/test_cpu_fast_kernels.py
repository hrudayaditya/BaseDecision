"""Checks for the experimental ``cpu_fast`` backend (``basedecision._cpu_attention``).

The adapter is derived from Transformers 4.57.6 and only supported with torch 2.9.1 and
transformers 4.57.6 (``CPUFastDecision`` refuses other versions), so these tests are skipped
elsewhere. ``KernelEquivalenceTests`` needs no checkpoint: it builds a tiny random ModernBERT and
compares the original attention/head against the instance-local tiled versions. With
``BASEDECISION_TEST_MODEL`` the real checkpoint is compared against the default FP32 backend.
"""

from __future__ import annotations

import os
import unittest
import warnings
from typing import Any

try:
    import torch
    import transformers

    SUPPORTED = transformers.__version__ == "4.57.6" and torch.__version__.split("+")[0] == "2.9.1"
except ImportError:  # torch/transformers are optional
    SUPPORTED = False

MODEL = os.environ.get("BASEDECISION_TEST_MODEL", "")
REASON = "cpu_fast requires torch 2.9.1 and transformers 4.57.6"


@unittest.skipUnless(SUPPORTED, REASON)
class KernelEquivalenceTests(unittest.TestCase):
    def make_toy(self) -> Any:
        """A tiny random DecisionModel; identical every time (fixed seed)."""
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
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=1024,
                local_attention=256,
                global_attn_every_n_layers=2,
                reference_compile=False,
            )
            config._attn_implementation = "sdpa"
            toy = DecisionModel(ModernBertModel(config), head_layers=1).eval()
            toy.requires_grad_(False)
        return toy

    def setUp(self) -> None:
        from transformers.models.modernbert import modeling_modernbert as mb

        self.mb = mb
        self.toy = self.make_toy()
        self.batches = []
        generator = torch.Generator().manual_seed(5)
        for size, length in [(1, 257), (1, 513), (2, 513)]:
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

        flag = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)  # the reference head path
            before = self.run_batches()
        finally:
            torch.backends.mha.set_fastpath_enabled(flag)
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
        self.assertGreater(fast.backend_info()["attention_calls"]["tiled_calls"], 0)
        self.assertEqual(mb.MODERNBERT_ATTENTION_FUNCTION, registry)
        self.assertEqual(
            [standard.predict(r).raw_logits for r in requests],
            [r.raw_logits for r in reference],
            "the default instance was affected",
        )
        with self.assertRaises(ContextLengthError):
            fast.count_tokens(Request("word " * 20000, "Question?", options))


if __name__ == "__main__":
    unittest.main()
