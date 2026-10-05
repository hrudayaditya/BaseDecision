# CPU support

No flag is needed to run without a GPU: `load(path)` (and `load_from_hub(...)`) automatically use the portable CPU/FP32 path when no suitable CUDA GPU is present. This page is about the separate, experimental `cpu_fast` backend.

`load(path, backend='cpu_fast')` selects experimental BF16 autocast with FP32 weights and tiled local backbone attention. `CPUFastDecision.from_pretrained` and `.from_hub` are also available. OpenAI/Anthropic adapters and default local inference are unchanged.

The implementation patches only this model instance's attention methods. It never changes the Transformers attention registry, PyTorch MHA switch, thread counts, affinity, environment variables, or weights. An instance serializes its own requests with the existing SDK lock. Different instances share hardware resources but not optimization settings.

Global attention and padded or batch>1 backbone attention use the original implementation. The head still uses functional SDPA and BF16. Fallback is not a switch to FP32. Results retain raw probabilities and add `backend` and `inference_precision`; `backend_info()` reports cumulative path counters. The inherited `precision` attribute describes FP32 weight/base-loader precision; actual computation is identified explicitly by `inference_precision`.

Only Torch 2.9.1 / Transformers 4.57.6 are accepted for fast mode. Unsupported versions fail with a clear error before model loading. Standard CPU FP32 remains available; fast mode never silently chooses a different backend. BF16 may be slow on hardware without efficient support. This release does not promise a latency target on arbitrary CPUs. CPU calibration profiles remain disabled.

Prior *experimental script* measurements on Xeon Platinum 8480+ achieved about 1.02s at 4k and 2.83s near 8k. Those numbers are not yet measurements of the RC5 instance-local implementation. A 471-case comparison of that earlier path changed three labels (one gain, two losses); probabilities can shift. On one 10-core Apple-silicon laptop `cpu_fast` was 10-15x *slower* than the default CPU/FP32 path, so treat it as an opt-in for the hardware it was developed on, not a general speed-up.

The adapter is covered by `tests/test_cpu_fast_kernels.py`: a tiny randomly initialized model compares the tiled attention and head against the reference (including padded and batched inputs, which fall back to the reference) and proves that installing it changes only that instance; with `BASEDECISION_TEST_MODEL` the real checkpoint is compared against the default FP32 backend. These tests need torch 2.9.1 and transformers 4.57.6 and are skipped otherwise. They do not establish model accuracy or calibrated confidence. Choose thread settings externally before startup, for example OMP_NUM_THREADS=16 and MKL_NUM_THREADS=16. Do not run concurrent performance jobs when measuring latency.
