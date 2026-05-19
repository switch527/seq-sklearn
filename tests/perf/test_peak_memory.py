"""PB.2 peak-memory benchmark. `perf`-marked, nightly-only.

The fit+predict payload runs in a fresh spawned child process so
`ru_maxrss` (a non-resettable process-lifetime high-water mark) is
attributable to the proxy alone, not contaminated by prior pytest
activity (Gemini-C3). A crashed/timed-out child is a hard failure.
"""

import pytest

pytestmark = pytest.mark.perf


def test_peak_memory_within_baseline(perf_determinism: None) -> None:
    from tests.perf._gate import assert_within_baseline, placeholder_measured, resolve_cell
    from tests.perf._measure import measure_peak_memory

    cell = resolve_cell()
    record = measure_peak_memory()
    measured = placeholder_measured(
        cell,
        peak_memory_value=float(record["value"]),
        peak_memory_metric=record["peak_memory_metric"],
    )
    assert_within_baseline(cell, measured)
