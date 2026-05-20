# Performance tuning

Practical "few-line wins" for fit/predict throughput and memory.
Performance regression is gated nightly by `tests/perf/` against
checked-in baselines per A13 (`cpu-x86`, `t4`); this page is the
user-facing guidance.

## GPU vs CPU

seq-sklearn auto-detects hardware via `seq_sklearn.hardware.detect()`
and selects the best available device. Single-GPU training is the
intended target; distributed/multi-GPU is out of scope for v1.

```python
from seq_sklearn import detect, HardwareTier
print(detect())   # HardwareTier.CPU / VOLTA_TURING / AMPERE_ADA / ...
```

You don't manually set the device; the trainer reads `detect()` at
fit time. Forcing CPU on a GPU box (rare, but useful for reproducing
issues) is a `precision`-and-environment concern; see
[the determinism reference](../reference/determinism).

## Precision

- **`precision="32-true"`** (default) — float32, the deterministic
  path. Use for reproducibility-critical runs, the N1 quickstart
  contract, and the criterion-9 release validation.
- **`precision="16-mixed"`** — bf16 (Ampere+) or fp16 with autocast.
  ~1.5-2x speedup on modern GPUs. Slight loss of bit-exact
  reproducibility but usually accuracy-neutral.

On a Hopper or Blackwell GPU, mixed precision is essentially free;
on Volta/Turing (T4), it can be slower than `32-true` due to the
overhead of casts on older tensor cores.

## Batch size

Larger batches → more parallel work per gradient step → faster
epochs (until memory caps out). Typical knee points:

- CPU-only: `batch_size=64` is a reasonable default; `128`/`256`
  saturate few cores.
- T4 / older GPUs: `64`-`128`.
- A10 / A100: `128`-`512`, memory permitting.

If you hit out-of-memory mid-epoch, halve `batch_size` and try
again. The library does not transparently chunk; you set the size.

## Lookback

`lookback` is the time-axis dimension of every window. Doubling
`lookback` roughly doubles the per-step compute and the memory
footprint of the LSTM. Empirically, `12`–`24` is the sweet spot for
monthly cadence; very long lookbacks (>50) start to hurt convergence
without proportionate accuracy gains.

## Inference latency

Per-sample inference latency on a batch of 1024 windows is the N7
reference; v1's budget is `< 100 ms` on CPU, `< 10 ms` on GPU. To
optimize inference specifically:

- Use the largest `batch_size` your serving environment allows;
  per-sample latency drops as batches get larger.
- Export to ONNX (see [the ONNX export how-to](export_onnx.md)). The exported
  graph drops Python/Lightning overhead and the gather-preserving
  LSTM path runs in fewer ops than the eager forward.
- Pin CPU-frequency governor to `performance` on bare-metal serving.

## When perf matters less than you think

- **Single fit** of a small panel (under ~500 entities): the trainer
  is dominated by Lightning init overhead, not the math. Optimizing
  the architecture knobs barely moves wall-clock.
- **Hyperparameter search**: lean on Optuna's pruning (see
  [Tune with Optuna](tune_with_optuna)) rather than micro-optimizing
  each trial.

## What the nightly perf gate actually checks

`tests/perf/` measures the median + p95 train-step time, peak
memory, and inference latency on a fixed proxy panel per cell
(`cpu-x86`, `t4`). The gate fires on a >15% regression to median
step time or >10% to peak memory vs the checked-in baseline. You
will get a nightly alert before a slow-down ships, not a PR-blocker;
A13 caps the PR-CI budget so per-PR perf is observational, not
mandatory.

```{testcode}
from seq_sklearn import HardwareTier, detect

tier = detect()
assert isinstance(tier, HardwareTier)
```
