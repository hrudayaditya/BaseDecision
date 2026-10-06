# CPU support

No flag is needed to run without a GPU: `load(path)` (and `load_from_hub(...)`) automatically use the portable CPU/FP32 path when no suitable CUDA GPU is present. This page is about the separate, experimental `cpu_fast` backend.

`load(path, backend='cpu_fast')` selects experimental BF16 autocast with FP32 weights and tiled local backbone attention. `CPUFastDecision.from_pretrained` and `.from_hub` are also available. OpenAI/Anthropic adapters and default local inference are unchanged.

The implementation patches only this model instance's attention methods. It never changes the Transformers attention registry, PyTorch MHA switch, thread counts, affinity, environment variables, or weights. An instance serializes its own requests with the existing SDK lock. Different instances share hardware resources but not optimization settings.

Global attention and padded or batch>1 backbone attention use the original implementation. The head still uses functional SDPA and BF16. Fallback is not a switch to FP32. Results retain raw probabilities and add `backend` and `inference_precision`; `backend_info()` reports cumulative path counters. The inherited `precision` attribute describes FP32 weight/base-loader precision; actual computation is identified explicitly by `inference_precision`.

## Versions and the self-test

`cpu_fast` does not require particular torch or transformers releases, and this package never installs or pins them: your environment stays yours. It accepts **torch 2.6 or newer with transformers 4.48 to 4.57** and, when it loads, *proves on your installation* that its patched attention reproduces the original. The proof is a known-answer test of about 0.3 s on a tiny random model that has this checkpoint's attention geometry (head size, window, global-layer pattern). In FP32 a correct adapter agrees with the reference to about 1e-7, so the check catches even a 1% numeric error; a looser BF16 pass confirms that the dtype the backend runs in works on your install. The result is cached for the process, and `model.backend_info()['self_test']` records the versions it was verified on. The self-test changes no global state.

Outside that range (transformers 5 removed the internals the adapter patches), or if the self-test fails, `load(path, backend='cpu_fast')` raises `CPUFastUnavailable` with a message that says what to do, instantly and before any weights are loaded in the version case. Fast mode never silently chooses a different backend; to fall back deliberately:

```python
from basedecision import CPUFastUnavailable, load

try:
    model = load(path, backend='cpu_fast')
except CPUFastUnavailable:
    model = load(path)  # the default CPU/FP32 backend, which works with your installation
```

Verified with the real checkpoint (same labels as the default backend, tiled path used) on torch 2.6.0 / transformers 4.48.0, torch 2.9.1 / 4.57.6 and torch 2.14.1 / 4.57.6; refused cleanly on transformers 5.18. BF16 may be slow on hardware without efficient support. This release does not promise a latency target on arbitrary CPUs. CPU calibration profiles remain disabled.

Prior *experimental script* measurements on Xeon Platinum 8480+ achieved about 1.02s at 4k and 2.83s near 8k. Those numbers are not measurements of the current instance-local implementation. A 471-case comparison of that earlier path changed three labels (one gain, two losses); probabilities can shift. On one 10-core Apple-silicon laptop `cpu_fast` was 10-15x *slower* than the default CPU/FP32 path, so treat it as an opt-in for the hardware it was developed on, not a general speed-up.

The adapter is covered by `tests/test_cpu_fast_kernels.py`: a tiny randomly initialized model compares the tiled attention and head against the reference (including padded and batched inputs, which fall back to the reference) and proves that installing it changes only that instance; with `BASEDECISION_TEST_MODEL` the real checkpoint is compared against the default FP32 backend. The tests also cover the version gate, the self-test (including deliberately broken and subtly wrong adapters, which it must catch) and the refusal on an unsupported install; they run on every supported install. They do not establish model accuracy or calibrated confidence. Choose thread settings externally before startup, for example OMP_NUM_THREADS=16 and MKL_NUM_THREADS=16. Do not run concurrent performance jobs when measuring latency.
