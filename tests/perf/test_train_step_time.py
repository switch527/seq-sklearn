"""PB.1 train-step-time benchmark. `perf`-marked, nightly-only.

Times `one_train_step` via the pytest-benchmark `benchmark` fixture
(BENCH_MIN_ROUNDS rounds), gates the median against the resolved
cell's baseline. Selected by the `-m perf` MARKER, never
`--benchmark-only` (Gemini-C1: that flag would silently drop PB.2/PB.3).
"""

import pytest

pytestmark = pytest.mark.perf


def test_train_step_time_within_baseline(benchmark: object, perf_determinism: None) -> None:
    from tests.perf._gate import (
        assert_within_baseline,
        percentile_linear,
        placeholder_measured,
        resolve_cell,
    )
    from tests.perf._constants import BENCH_MIN_ROUNDS
    from tests.perf._measure import one_train_step

    cell = resolve_cell()
    benchmark.pedantic(  # type: ignore[attr-defined]
        one_train_step, rounds=BENCH_MIN_ROUNDS, iterations=1, warmup_rounds=1
    )
    stats = benchmark.stats.stats  # type: ignore[attr-defined]
    median = float(stats.median)
    p95 = percentile_linear([float(x) for x in stats.data], 95)

    measured = placeholder_measured(cell, train_step_median_s=median, train_step_p95_s=p95)
    assert_within_baseline(cell, measured)
