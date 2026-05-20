# Determinism and reproducibility

What seq-sklearn guarantees, what it doesn't, and the side effects
of the strict-mode path. This is the standalone reference; the
underlying design rationale is in
[explanation/determinism_and_precision](../explanation/determinism_and_precision).

## The contract

For a fixed:

- `seed` on the estimator,
- `precision="32-true"`,
- single device (same CPU or same GPU model),
- same `seq-sklearn`, `torch`, `numpy`, `lightning` versions,

`fit` produces bit-identical weights and `predict` produces
bit-identical outputs across re-runs. The N1 quickstart-in-CI test
asserts the binary-classifier accuracy threshold at `seed=42`
exactly.

## What is NOT guaranteed

- **Bit-exactness across PyTorch / CUDA / cuDNN versions.** PyTorch
  point releases regularly change reduction orders, kernel
  selection, and rounding in non-determinism-critical paths.
  Reproducibility requires version-pinning.
- **Bit-exactness across hardware tiers.** A V100 and a T4 are both
  `VOLTA_TURING`-tier but use different kernels; outputs match to
  high tolerance, not to ULP.
- **Bit-exactness with `precision="16-mixed"`.** Mixed precision
  introduces non-determinism by design (the autocast policy depends
  on tensor history); use `"32-true"` for strict reproducibility.

## The strict-mode side effects

When `seed` is set and `precision="32-true"`, the library calls
`seq_sklearn.training._determinism.enable_strict_mode()`, which:

- Sets `torch.use_deterministic_algorithms(True, warn_only=False)`.
- Sets `torch.backends.cudnn.deterministic = True`.
- Sets `torch.backends.cudnn.benchmark = False`.
- Exports `CUBLAS_WORKSPACE_CONFIG=":4096:8"` if it is unset
  (a caller-supplied non-default value is left untouched).

Idempotent: re-calling makes the same assertions, never logs or
warns on the second call.

## Performance cost

Strict determinism costs throughput. The `cudnn.benchmark=False`
flag disables kernel autotuning; `torch.use_deterministic_algorithms`
forbids the faster non-deterministic kernels for some ops (most
notably some pooling and scatter-add). Expect 1.1x–1.5x slower
training than the non-deterministic path on a modern GPU. For the
N7 absolute-budget validation, this cost is accepted; for casual
exploration, prefer not setting `seed`.

## Multi-process and DataLoader

PyTorch's DataLoader with `num_workers > 0` requires
`worker_init_fn` + a `Generator` per worker to be deterministic.
seq-sklearn's internal DataLoader does this automatically when
`seed` is set. If you pass your own DataLoader (rare, undocumented
hook), you are responsible for the worker-seeding.

## Re-checking determinism

The N7-absolute test (`tests/perf/test_n7_absolute.py`,
`gpu and slow`-marked) is the canonical determinism + budget
re-check at release time. Run it manually on an A100/T4/4090; record
the four numeric budgets in the CHANGELOG v1.0.0 entry per
[the release checklist](../about/release_checklist).
