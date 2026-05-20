# Determinism and precision: the model

For the operational contract and how to reach the strict-mode path,
see the [determinism reference](../reference/determinism). This page
explains the design rationale.

## The trilemma

There are three things every ML library can offer:

- **Speed.** Faster training and inference.
- **Bit-exact reproducibility.** Re-running the same code on the
  same data and same hardware gives the same bytes out.
- **Cross-hardware portability.** Same code on a different GPU
  produces the same answer.

You can have any two. PyTorch's design point is that the library
caller chooses the trade. seq-sklearn surfaces that choice through
two knobs: `seed` and `precision`.

## What `seed` does

Setting `seed` activates the strict determinism path:

- Python's `random`, NumPy's RNG, PyTorch's CPU RNG, and PyTorch's
  CUDA RNG are all seeded from the same value.
- `torch.use_deterministic_algorithms(True, warn_only=False)`
  forbids the non-deterministic fast paths in PyTorch.
- `torch.backends.cudnn.deterministic = True` and
  `cudnn.benchmark = False` disable cuDNN kernel autotuning (which
  selects different kernels run-to-run).
- `CUBLAS_WORKSPACE_CONFIG=":4096:8"` is exported (if unset) so
  cuBLAS picks deterministic GEMM kernels.

These are the four flags PyTorch's own determinism docs name. The
library applies them once; the call is idempotent.

## What `precision` does

`precision="32-true"` keeps every tensor in float32. Math is
deterministic at the float32 level (within the per-kernel
guarantees PyTorch makes).

`precision="16-mixed"` uses autocast to run most ops in bf16
(Ampere+) or fp16 (older GPUs) while keeping a master copy in
float32. The autocast policy depends on tensor history; identical
inputs through different code paths can converge to different bf16
intermediates, so 16-mixed sacrifices bit-exactness for speed.

## Why not just always be deterministic

The cost. Deterministic cuDNN is 1.1x-1.5x slower than the
benchmarked path. The deterministic CPU implementations of some
ops are also slower than their non-deterministic counterparts.
For casual exploration, the cost outweighs the benefit; for the
N1 quickstart-in-CI reproducibility test, the cost is required.

The library makes the choice explicit: set `seed`, get
determinism + the strict-mode side effects + the cost. Don't set
`seed`, get the default fast path.

## What's NOT in the determinism guarantee

- **Bit-exactness across `seq-sklearn` / `torch` / `cudnn`
  versions.** PyTorch point releases routinely change reduction
  orders, kernel selection, and rounding. Pinning versions is the
  only way to get bit-exactness over time.
- **Bit-exactness across hardware tiers.** A V100 and a T4 are
  both `HardwareTier.VOLTA_TURING` but use different cuBLAS
  kernels. Outputs match to a high tolerance, not to ULP.
- **Bit-exactness with `precision="16-mixed"`.** By design.

## The release-checklist tie-in

Acceptance criterion 9 (`docs/requirements.md`) is "all N7 budgets
met". `tests/perf/test_n7_absolute.py` is the canonical re-check on
an A100/T4/4090, run manually as a release-checklist step. The test
uses `seed` + `precision="32-true"` so the budgets are measured on
the deterministic path. The four numbers go into the CHANGELOG
v1.0.0 entry.
