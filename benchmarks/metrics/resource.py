"""Resource metrics (B5.3).

Wall-clock fit seconds, peak process RSS, peak CUDA memory (when a
CUDA device is visible), and per-sample inference latency (median +
p95). Recorded on every run; the harness routes them into the
manifest alongside the predict-quality metrics.

The latency helper takes a single-batch predict callable; the
harness measures over a fixed number of repetitions and reports the
median + p95 in milliseconds per batch.
"""

import resource as _stdlib_resource
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FitResource:
    """Resource snapshot from a single `adapter.fit` call.

    ``wall_seconds`` is the elapsed wall-clock. ``peak_rss_bytes``
    is the process resident-set-size high-water mark observed AFTER
    the fit (via `resource.getrusage` on POSIX; bytes on Linux,
    kilobytes on macOS — the unit difference is documented in the
    field name). ``peak_cuda_bytes`` is `torch.cuda.max_memory_
    allocated()` when CUDA is visible, otherwise None.
    """

    wall_seconds: float
    peak_rss_bytes: int
    peak_cuda_bytes: int | None


def measure_fit(
    fit_callable: Callable[[], None],
    *,
    cuda_available: bool = False,
    cuda_reset_callable: Callable[[], None] | None = None,
    cuda_peak_callable: Callable[[], int] | None = None,
) -> FitResource:
    """Run ``fit_callable`` once, returning a `FitResource`.

    `cuda_reset_callable` is `torch.cuda.reset_peak_memory_stats`
    when the caller wants a fresh CUDA high-water mark; injected so
    the metrics module does not import torch directly.
    `cuda_peak_callable` is `torch.cuda.max_memory_allocated` for
    the same reason.
    """
    if cuda_available and cuda_reset_callable is not None:
        cuda_reset_callable()
    start = time.perf_counter()
    fit_callable()
    wall = time.perf_counter() - start
    usage = _stdlib_resource.getrusage(_stdlib_resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)  # Linux: bytes; macOS: kilobytes
    peak_cuda: int | None = None
    if cuda_available and cuda_peak_callable is not None:
        peak_cuda = int(cuda_peak_callable())
    return FitResource(wall_seconds=wall, peak_rss_bytes=peak_rss, peak_cuda_bytes=peak_cuda)


@dataclass(frozen=True, slots=True)
class InferenceLatency:
    """Median and p95 wall-clock latency (milliseconds) of a single
    predict call on a 1024-row batch (or whatever batch size the
    caller times). N7 reads these per-batch, NOT per-sample."""

    median_ms: float
    p95_ms: float
    n_reps: int


def measure_inference_latency(
    predict_callable: Callable[[], None],
    *,
    n_reps: int = 20,
    n_warmup: int = 3,
) -> InferenceLatency:
    """Time `predict_callable` `n_reps` times after `n_warmup`
    untimed reps, returning median + p95 in ms. The caller is
    responsible for sizing the batch (B5.3 / N7 read per 1024-window
    batch)."""
    for _ in range(n_warmup):
        predict_callable()
    samples: list[float] = []
    for _ in range(n_reps):
        start = time.perf_counter()
        predict_callable()
        samples.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(samples)
    return InferenceLatency(
        median_ms=float(np.median(arr)),
        p95_ms=float(np.quantile(arr, 0.95)),
        n_reps=n_reps,
    )
