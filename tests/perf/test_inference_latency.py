"""PB.3 inference-latency benchmark. `perf`-marked, nightly-only.

Per-sample latency over INFERENCE_REPEATS timed predicts on
INFERENCE_BATCH windows (warm-ups excluded). Latency is recorded but
NOT hard-gated in v1 (PD.1: A13 gates only step time and peak
memory); running it through `assert_within_baseline` still exercises
the load/provisional/missing precedence for the latency path.
"""

import pytest

pytestmark = pytest.mark.perf


def test_inference_latency_within_baseline(perf_determinism: None) -> None:
    from tests.perf._gate import assert_within_baseline, placeholder_measured, resolve_cell
    from tests.perf._measure import measure_inference_latency

    cell = resolve_cell()
    median, p95 = measure_inference_latency()
    measured = placeholder_measured(
        cell,
        inference_latency_median_s=median,
        inference_latency_p95_s=p95,
    )
    assert_within_baseline(cell, measured)
