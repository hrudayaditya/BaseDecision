"""Tests for hardware selection: ``resolve_device``, ``load`` and ``load_from_hub``.

A fake ``torch`` module stands in for each kind of machine, so every hardware scenario is
covered on any machine without a GPU or even without Torch installed.
"""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from basedecision import (
    BaseDecision,
    CalibrationError,
    CPUFastDecision,
    InputError,
    load,
    load_from_hub,
    resolve_device,
)


def fake_torch(
    cuda: bool = False, bf16: Any = True, accepts_emulation_flag: bool = True
) -> types.SimpleNamespace:
    """A minimal ``torch`` for one kind of machine; ``bf16`` may be an exception to raise."""

    def is_bf16_supported(*args: Any, **kwargs: Any) -> bool:
        if kwargs and not accepts_emulation_flag:
            raise TypeError("is_bf16_supported() got an unexpected keyword argument")
        if isinstance(bf16, Exception):
            raise bf16
        # Without emulation the answer is "native BF16 only"; with it, always True.
        return bool(bf16) if kwargs.get("including_emulation") is False else True

    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda, is_bf16_supported=MagicMock(side_effect=is_bf16_supported)
    )
    return types.SimpleNamespace(cuda=cuda_ns)


CPU_ONLY = fake_torch(cuda=False, bf16=False)  # a CUDA-less Torch reports no BF16
MODERN_GPU = fake_torch(cuda=True, bf16=True)  # e.g. A100/H100/RTX 30xx+
PRE_AMPERE_GPU = fake_torch(cuda=True, bf16=False)  # e.g. T4/V100: BF16 only by emulation


class ResolveDeviceTests(unittest.TestCase):
    def resolve(self, torch: Any, *args: Any) -> tuple[str, str]:
        with patch.dict(sys.modules, {"torch": torch}):
            return resolve_device(*args)

    def test_no_gpu_means_cpu_fp32(self) -> None:
        self.assertEqual(self.resolve(CPU_ONLY), ("cpu", "fp32"))
        self.assertEqual(self.resolve(CPU_ONLY, "auto"), ("cpu", "fp32"))
        self.assertEqual(self.resolve(CPU_ONLY, None, None), ("cpu", "fp32"))

    def test_modern_gpu_means_cuda_bf16(self) -> None:
        self.assertEqual(self.resolve(MODERN_GPU), ("cuda", "bf16"))
        self.assertEqual(self.resolve(MODERN_GPU, "auto"), ("cuda", "bf16"))

    def test_gpu_without_native_bf16_falls_back_to_fp32(self) -> None:
        self.assertEqual(self.resolve(PRE_AMPERE_GPU), ("cuda", "fp32"))

    def test_native_bf16_is_asked_for_without_emulation(self) -> None:
        self.resolve(MODERN_GPU)
        MODERN_GPU.cuda.is_bf16_supported.assert_called_with(including_emulation=False)

    def test_older_torch_without_the_keyword(self) -> None:
        torch = fake_torch(cuda=True, bf16=True, accepts_emulation_flag=False)
        self.assertEqual(self.resolve(torch), ("cuda", "bf16"))

    def test_unusual_cuda_setup_is_handled_conservatively(self) -> None:
        torch = fake_torch(cuda=True, bf16=RuntimeError("driver problem"))
        self.assertEqual(self.resolve(torch), ("cuda", "fp32"))

    def test_explicit_device_with_default_precision(self) -> None:
        self.assertEqual(self.resolve(MODERN_GPU, "cpu"), ("cpu", "fp32"))
        self.assertEqual(self.resolve(MODERN_GPU, "cuda"), ("cuda", "bf16"))
        self.assertEqual(self.resolve(MODERN_GPU, "cuda:1"), ("cuda:1", "bf16"))
        self.assertEqual(self.resolve(PRE_AMPERE_GPU, "cuda"), ("cuda", "fp32"))
        # An explicit CUDA request is not second-guessed even when no GPU exists.
        self.assertEqual(self.resolve(CPU_ONLY, "cuda"), ("cuda", "fp32"))

    def test_torch_device_like_objects(self) -> None:
        cuda_like = types.SimpleNamespace(type="cuda", index=0)
        cpu_like = types.SimpleNamespace(type="cpu", index=None)
        self.assertEqual(self.resolve(MODERN_GPU, cuda_like), (cuda_like, "bf16"))
        self.assertEqual(self.resolve(MODERN_GPU, cpu_like), (cpu_like, "fp32"))

    def test_explicit_precision_is_kept_and_never_validated_here(self) -> None:
        self.assertEqual(self.resolve(CPU_ONLY, None, "fp32"), ("cpu", "fp32"))
        self.assertEqual(self.resolve(MODERN_GPU, None, "fp32"), ("cuda", "fp32"))
        self.assertEqual(self.resolve(MODERN_GPU, "cuda", "fp32"), ("cuda", "fp32"))
        # Invalid combinations are returned as given so the model constructor can reject them
        # with its own clear message.
        self.assertEqual(self.resolve(CPU_ONLY, "cpu", "bf16"), ("cpu", "bf16"))
        self.assertEqual(self.resolve(CPU_ONLY, "mps", "fp16"), ("mps", "fp16"))

    def test_explicit_choices_do_not_need_torch(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):  # makes "import torch" fail
            self.assertEqual(resolve_device("cpu"), ("cpu", "fp32"))
            self.assertEqual(resolve_device("cpu", "fp32"), ("cpu", "fp32"))
            self.assertEqual(resolve_device("cuda", "bf16"), ("cuda", "bf16"))

    def test_auto_selection_needs_torch_and_says_how_to_get_it(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            for args in ((), ("auto",), (None, None), ("cuda",)):
                with self.subTest(args=args), self.assertRaises(ImportError) as caught:
                    resolve_device(*args)
                self.assertIn("basedecision[runtime]", str(caught.exception))

    def test_importing_the_package_still_does_not_load_torch(self) -> None:
        import subprocess

        code = "import basedecision, sys; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)


class LoadTests(unittest.TestCase):
    def test_load_without_arguments_uses_the_machine_default(self) -> None:
        for torch, expected in (
            (CPU_ONLY, ("cpu", "fp32")),
            (MODERN_GPU, ("cuda", "bf16")),
            (PRE_AMPERE_GPU, ("cuda", "fp32")),
        ):
            with (
                patch.dict(sys.modules, {"torch": torch}),
                patch.object(BaseDecision, "from_pretrained", return_value="model") as factory,
            ):
                self.assertEqual(load("/model"), "model")
            factory.assert_called_once_with("/model", device=expected[0], precision=expected[1])

    def test_explicit_arguments_and_other_options_pass_through(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": MODERN_GPU}),
            patch.object(BaseDecision, "from_pretrained", return_value="m") as factory,
        ):
            load("/model", device="cpu", precision="fp32", max_batch_size=4)
        factory.assert_called_once_with("/model", device="cpu", precision="fp32", max_batch_size=4)

    def test_partial_arguments_are_completed(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": CPU_ONLY}),
            patch.object(BaseDecision, "from_pretrained", return_value="m") as factory,
        ):
            load("/model", precision="fp32")
            factory.assert_called_with("/model", device="cpu", precision="fp32")
            load("/model", device="cpu")
            factory.assert_called_with("/model", device="cpu", precision="fp32")

    def test_cpu_fast_keeps_its_own_defaults(self) -> None:
        with patch.object(CPUFastDecision, "from_pretrained", return_value="fast") as factory:
            self.assertEqual(load("/model", backend="cpu_fast"), "fast")
        factory.assert_called_once_with("/model", device="cpu", precision="bf16")

    def test_validation_happens_before_any_loading(self) -> None:
        with patch.object(BaseDecision, "from_pretrained") as factory:
            with self.assertRaises(InputError):
                load("/model", backend="magic")
            for profile in ("clinc", "does-not-exist"):
                with self.assertRaises(CalibrationError):
                    load("/model", calibration_profile=profile)
            with self.assertRaises(CalibrationError):
                load("/model", backend="cpu_fast", calibration_profile="sgd_schema")
        factory.assert_not_called()

    def test_constructor_errors_are_not_hidden(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": CPU_ONLY}),
            patch.object(
                BaseDecision,
                "from_pretrained",
                side_effect=InputError("CPU inference requires fp32"),
            ),
            self.assertRaises(InputError),
        ):
            load("/model", device="cpu", precision="bf16")


class FakeTokenizer:
    def __init__(self) -> None:
        self.deprecation_warnings: dict[str, bool] = {}


class FakeLoaded:
    def __init__(self) -> None:
        self._tokenizer = FakeTokenizer()


NOTICE = "sequence-length-is-longer-than-the-specified-maximum"


class LengthNoticeTests(unittest.TestCase):
    """Transformers' "will result in indexing errors" notice is wrong here: we raise instead."""

    def loaded_by(self, call: Any) -> FakeLoaded:
        loaded = FakeLoaded()
        hub = types.SimpleNamespace(snapshot_download=lambda **_: "/cache/snapshot")
        with (
            patch.dict(sys.modules, {"torch": CPU_ONLY, "huggingface_hub": hub}),
            patch.object(BaseDecision, "from_pretrained", return_value=loaded),
            patch.object(BaseDecision, "from_hub", return_value=loaded),
            patch.object(CPUFastDecision, "from_pretrained", return_value=loaded),
            patch.object(CPUFastDecision, "from_hub", return_value=loaded),
        ):
            self.assertIs(call(), loaded)
        return loaded

    def test_every_loader_marks_the_notice_as_already_shown(self) -> None:
        calls = {
            "load": lambda: load("/model"),
            "load cpu_fast": lambda: load("/model", backend="cpu_fast"),
            "load_from_hub": lambda: load_from_hub("owner/model"),
            "load_from_hub cpu_fast": lambda: load_from_hub("owner/model", backend="cpu_fast"),
        }
        for name, call in calls.items():
            with self.subTest(loader=name):
                loaded = self.loaded_by(call)
                self.assertIs(loaded._tokenizer.deprecation_warnings[NOTICE], True)

    def test_other_notices_are_untouched_and_odd_objects_are_left_alone(self) -> None:
        loaded = FakeLoaded()
        loaded._tokenizer.deprecation_warnings["something-else"] = False
        with (
            patch.dict(sys.modules, {"torch": CPU_ONLY}),
            patch.object(BaseDecision, "from_pretrained", return_value=loaded),
        ):
            load("/model")
        self.assertEqual(loaded._tokenizer.deprecation_warnings["something-else"], False)
        for odd in (object(), types.SimpleNamespace(_tokenizer=None)):
            with (
                patch.dict(sys.modules, {"torch": CPU_ONLY}),
                patch.object(BaseDecision, "from_pretrained", return_value=odd),
            ):
                self.assertIs(load("/model"), odd)


class LoadFromHubTests(unittest.TestCase):
    def test_defaults_are_machine_appropriate_and_the_download_is_explicit(self) -> None:
        downloads: list[dict[str, Any]] = []

        def snapshot_download(**kwargs: Any) -> str:
            downloads.append(kwargs)
            return "/cache/snapshot"

        hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with (
            patch.dict(sys.modules, {"torch": CPU_ONLY, "huggingface_hub": hub}),
            patch.object(BaseDecision, "from_pretrained", return_value="model") as factory,
        ):
            result = load_from_hub("owner/model", revision="abc123", local_files_only=True)
        self.assertEqual(result, "model")
        factory.assert_called_once_with("/cache/snapshot", device="cpu", precision="fp32")
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0]["repo_id"], "owner/model")
        self.assertEqual(downloads[0]["revision"], "abc123")
        self.assertTrue(downloads[0]["local_files_only"])
        self.assertIn("*.py", downloads[0]["ignore_patterns"])  # no remote code, ever

    def test_gpu_machine_gets_cuda_bf16(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": MODERN_GPU}),
            patch.object(BaseDecision, "from_hub", return_value="m") as factory,
        ):
            load_from_hub("owner/model")
        factory.assert_called_once_with(
            "owner/model", revision=None, device="cuda", precision="bf16"
        )

    def test_explicit_options_pass_through(self) -> None:
        with patch.object(BaseDecision, "from_hub", return_value="m") as factory:
            load_from_hub("o/m", revision="r", device="cpu", precision="fp32", cache_dir="/c")
        factory.assert_called_once_with(
            "o/m", revision="r", device="cpu", precision="fp32", cache_dir="/c"
        )

    def test_cpu_fast_backend(self) -> None:
        with patch.object(CPUFastDecision, "from_hub", return_value="fast") as factory:
            self.assertEqual(load_from_hub("o/m", revision="r", backend="cpu_fast"), "fast")
        factory.assert_called_once_with("o/m", revision="r", device="cpu", precision="bf16")

    def test_validation_happens_before_the_download(self) -> None:
        with patch.object(BaseDecision, "from_hub") as factory:
            with self.assertRaises(InputError):
                load_from_hub("o/m", backend="magic")
            with self.assertRaises(CalibrationError):
                load_from_hub("o/m", calibration_profile="clinc")
        factory.assert_not_called()

    def test_empty_repo_id_is_rejected(self) -> None:
        for repo in ("", "  ", None):
            with self.subTest(repo=repo), self.assertRaises(InputError):
                load_from_hub(repo)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
