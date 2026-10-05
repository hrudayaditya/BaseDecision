# Measured batching throughput

A100-SXM4-80GB; Torch 2.9.1+cu126; BF16 autocast, FP32 weights.
Synthetic similar-length requests, eight per workload and three warmed timing repetitions.
All tested labels matched single-request inference. Throughput is not per-request latency.

| Approximate length | Batch 1 req/s | Fastest tested batch | Req/s | Speedup | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| 256 | 43.84 | 8 | 273.31 | 6.23x | 1.65 |
| 512 | 41.88 | 8 | 157.26 | 3.75x | 1.73 |
| 2048 | 13.24 | 8 | 30.67 | 2.32x | 2.37 |
| 4096 | 4.98 | 8 | 11.44 | 2.30x | 3.71 |
| 8176 | 1.49 | 4 | 2.96 | 1.98x | 5.19 |

The earlier mixed-length fixture was slower batched (0.820 vs 0.687 seconds).
Default batch size remains 1. Choose explicit batches for similar-length workloads.
These measurements do not justify unmeasured quantization, compilation, automatic
batch sizing or claims about unrelated GPUs and deployments.
