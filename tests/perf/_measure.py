"""Phase 11 measurement functions (PB.1 / PB.2 / PB.3).

The three `measure_*` functions are the SINGLE code path shared by
the capture CLI (PE.1) and the gate benchmark tests, so a captured
baseline and a gated run can never diverge. PG.6 monkeypatches these
three to known constants to test the capture CLI offline. Imports
torch via `_workload`, so this module is perf-path only (PC.1a).
"""

import logging
import multiprocessing as mp
import os
import queue as _queue
import resource
import time
import tracemalloc
from typing import Any

from tests.perf._constants import (
    BENCH_MIN_ROUNDS,
    INFERENCE_BATCH,
    INFERENCE_REPEATS,
    INFERENCE_WARMUP,
    PEAK_MEM_CHILD_TIMEOUT_S,
)
from tests.perf._gate import percentile_linear

# `_workload` (torch + TFTClassifier) is imported lazily inside each
# function body, NOT at module scope, so `import tests.perf._measure`
# stays torch-free. `tests.perf.capture` imports this module at its
# module scope; PG.6 monkeypatches the `measure_*` functions, so the
# fast-suite capture-CLI test never triggers the lazy torch import
# (post-Gemini code-review C1/C2: keep the fast PR suite torch-free).

__all__ = [
    "device_name",
    "measure_inference_latency",
    "measure_peak_memory",
    "measure_train_step",
    "one_train_step",
]

logger = logging.getLogger("seq_sklearn.perf")


def device_name() -> str:
    """`"cpu"` or the CUDA device string (PC.1 auditability)."""
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def one_train_step() -> None:
    """One fit of a freshly built proxy. The Gemini-NIT is documented
    in PB.1: on CPU this is dominated by Lightning init overhead, which
    is acceptable for a RELATIVE gate because it is identical
    run-to-run and the baseline captures it too."""
    from tests.perf._workload import build_proxy_estimator_and_panel

    est, panel, y = build_proxy_estimator_and_panel()
    est.fit(panel, y)


def measure_train_step() -> tuple[float, float]:
    """Median and P95 step seconds over `BENCH_MIN_ROUNDS` timed fits.
    The same timed unit (`one_train_step`) the PB.1 pytest-benchmark
    test wraps, so capture and gate measure the identical execution."""
    samples: list[float] = []
    for _ in range(BENCH_MIN_ROUNDS):
        start = time.perf_counter()
        one_train_step()
        samples.append(time.perf_counter() - start)
    return percentile_linear(samples, 50), percentile_linear(samples, 95)


def _peak_memory_child(queue: "mp.Queue[dict[str, Any]]") -> None:
    """Runs in a fresh spawned process so `ru_maxrss` (a non-resettable
    process-lifetime high-water mark) is attributable to the proxy
    alone, not contaminated by prior pytest activity (Gemini-C3)."""
    try:
        import torch

        from tests.perf._workload import build_proxy_estimator_and_panel

        tracemalloc.start()
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        est, panel, y = build_proxy_estimator_and_panel()
        est.fit(panel, y)
        est.predict(panel)
        _, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if cuda:
            value: float = float(torch.cuda.max_memory_allocated())
            metric = "cuda_max_alloc_bytes"
            dev = torch.cuda.get_device_name(0)
        else:
            value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            metric = "ru_maxrss_kb"
            dev = "cpu"
        queue.put(
            {
                "value": value,
                "peak_memory_metric": metric,
                "device_name": dev,
                "tracemalloc_peak_bytes": int(tm_peak),
                "pid": os.getpid(),
                "start_method": mp.get_start_method(),
            }
        )
    except Exception as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def measure_peak_memory() -> dict[str, Any]:
    """Spawn an isolated child, run fit+predict, return its record.
    A crashed/OOM-killed/timed-out child is a HARD failure, never a
    silent skip (qa-I1)."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue[dict[str, Any]] = ctx.Queue()
    proc = ctx.Process(target=_peak_memory_child, args=(queue,))
    proc.start()
    try:
        record = queue.get(timeout=PEAK_MEM_CHILD_TIMEOUT_S)
    except _queue.Empty as exc:
        proc.terminate()
        proc.join(10)
        raise RuntimeError(
            f"peak-memory child did not return within {PEAK_MEM_CHILD_TIMEOUT_S}s"
        ) from exc
    proc.join(10)
    if proc.exitcode is None:
        # Returned a record but the process is wedged in teardown:
        # surface it, never leave a zombie or trust a half-exit (code-I2).
        proc.kill()
        proc.join(10)
        raise RuntimeError("peak-memory child wedged after returning its record")
    if proc.exitcode != 0:
        raise RuntimeError(f"peak-memory child exited with code {proc.exitcode}")
    if "error" in record:
        raise RuntimeError(f"peak-memory child failed: {record['error']}")
    tm_peak = record.pop("tracemalloc_peak_bytes", None)
    logger.info("perf peak-memory: tracemalloc_peak_bytes=%s (logged-only, Q2)", tm_peak)
    return record


def measure_inference_latency() -> tuple[float, float]:
    """Per-sample inference latency over `INFERENCE_REPEATS` timed
    predicts on `INFERENCE_BATCH` windows, warm-ups excluded."""
    from tests.perf._workload import build_proxy_estimator_and_panel

    est, panel, y = build_proxy_estimator_and_panel()
    est.fit(panel, y)
    batch = panel.head(min(INFERENCE_BATCH, len(panel)))
    n = max(1, len(batch))
    for _ in range(INFERENCE_WARMUP):
        est.predict(batch)
    per_sample: list[float] = []
    for _ in range(INFERENCE_REPEATS):
        start = time.perf_counter()
        est.predict(batch)
        per_sample.append((time.perf_counter() - start) / n)
    return percentile_linear(per_sample, 50), percentile_linear(per_sample, 95)
