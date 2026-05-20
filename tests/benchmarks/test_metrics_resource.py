"""Phase B4 resource-metric tests (B5.3, B4-R1 / qa-C1).

`measure_fit` and `measure_inference_latency` were originally
shipped without unit tests; the R1 swarm flagged this as the only
new module with zero coverage. These tests exercise both functions
on CPU only, using lambdas for the torch CUDA hooks so the metrics
module stays torch-free at test time.
"""

import time

import pytest
from benchmarks.metrics.resource import (
    FitResource,
    InferenceLatency,
    measure_fit,
    measure_inference_latency,
)


def test_measure_fit_cpu_only_returns_fitresource_with_correct_types() -> None:
    res = measure_fit(lambda: None, cuda_available=False)
    assert isinstance(res, FitResource)
    assert res.wall_seconds >= 0.0
    assert res.peak_rss_bytes > 0
    assert res.peak_cuda_bytes is None


def test_measure_fit_cuda_callables_are_invoked_in_order() -> None:
    # Tracks invocation order so the test catches a refactor that
    # reads `max_memory_allocated` BEFORE the fit (which would read
    # the previous-fit high-water mark).
    calls: list[str] = []

    def reset() -> None:
        calls.append("reset")

    def fit_callable() -> None:
        calls.append("fit")

    def peak() -> int:
        calls.append("peak")
        return 42

    res = measure_fit(
        fit_callable,
        cuda_available=True,
        cuda_reset_callable=reset,
        cuda_peak_callable=peak,
    )
    assert isinstance(res, FitResource)
    assert res.peak_cuda_bytes == 42
    assert calls == ["reset", "fit", "peak"]


def test_measure_fit_cuda_available_but_no_callables_returns_none_peak() -> None:
    # Defensive: cuda_available=True but the harness forgot to wire
    # the torch hooks. The function does NOT crash; peak_cuda is None.
    res = measure_fit(lambda: None, cuda_available=True)
    assert isinstance(res, FitResource)
    assert res.peak_cuda_bytes is None


def test_measure_fit_wall_seconds_is_nontrivial_for_slow_callable() -> None:
    def slow() -> None:
        time.sleep(0.05)

    res = measure_fit(slow, cuda_available=False)
    assert res.wall_seconds >= 0.05


def test_measure_inference_latency_calls_warmup_then_reps() -> None:
    counter = {"n": 0}

    def predict() -> None:
        counter["n"] += 1

    res = measure_inference_latency(predict, n_reps=7, n_warmup=2)
    assert isinstance(res, InferenceLatency)
    # The contract: n_warmup untimed reps followed by n_reps timed reps.
    assert counter["n"] == 7 + 2
    assert res.n_reps == 7


def test_measure_inference_latency_p95_is_at_least_median() -> None:
    res = measure_inference_latency(lambda: None, n_reps=20, n_warmup=3)
    assert res.median_ms >= 0.0
    assert res.p95_ms >= res.median_ms


def test_fit_resource_is_frozen_and_forbids_extra() -> None:
    # Pydantic v2 ConfigDict(extra="forbid", frozen=True) is the
    # contract: a downstream maintainer cannot silently mutate the
    # resource snapshot or extend it via the constructor.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FitResource(
            wall_seconds=1.0,
            peak_rss_bytes=1024,
            peak_cuda_bytes=None,
            extra_field=1,  # pyright: ignore[reportCallIssue]
        )
    rec = FitResource(wall_seconds=1.0, peak_rss_bytes=1024, peak_cuda_bytes=None)
    with pytest.raises(ValidationError):
        rec.wall_seconds = 2.0  # pyright: ignore[reportAttributeAccessIssue]
