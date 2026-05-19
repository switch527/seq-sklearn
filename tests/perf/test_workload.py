"""PG.4 (proxy config + CPU budget + peak-memory isolation) and PG.8
`test_bench_min_rounds_applied`. The config/isolation tests are
`perf`-marked (they construct the estimator / spawn a child); the
budget test is subprocess-isolated.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from tests.perf._workload import (
    BENCH_MIN_ROUNDS,
    PROXY_ATTENTION_HEADS,
    PROXY_BUILD_TIMEOUT_S,
    PROXY_HIDDEN_SIZE,
    PROXY_N,
    PROXY_P,
)


def _repo_root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[2])


@pytest.mark.perf
def test_proxy_estimator_config_matches_spec(perf_determinism: None) -> None:
    """qa-C4: all three benchmarks must measure the spec'd workload."""
    from tests.perf._workload import build_proxy_estimator_and_panel

    est, panel, y = build_proxy_estimator_and_panel()
    assert est.hidden_size == PROXY_HIDDEN_SIZE
    assert est.attention_heads == PROXY_ATTENTION_HEADS
    assert len(panel) == PROXY_N * PROXY_P
    assert len(y) == PROXY_N * PROXY_P


def test_proxy_builds_within_cpu_budget() -> None:
    """qa-I1 / R1: an oversized proxy must fail loudly, not silently
    time the nightly job out. Subprocess with a hard timeout."""
    script = textwrap.dedent(
        """
        from seq_sklearn.training._determinism import enable_strict_mode
        from tests.perf._workload import build_proxy_estimator_and_panel
        enable_strict_mode()
        est, panel, y = build_proxy_estimator_and_panel()
        est.fit(panel, y)
        print("OK")
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            timeout=PROXY_BUILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"proxy build+fit exceeded {PROXY_BUILD_TIMEOUT_S}s budget; "
            f"shrink (N, P, L) in _workload.py (PERF_BASELINE_REVIEWED:)"
        )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout


@pytest.mark.perf
def test_peak_memory_payload_runs_in_child_process(perf_determinism: None) -> None:
    """Gemini-C3 / qa-I1: the ru_maxrss payload must run in a SPAWNED
    child (distinct PID, spawn not fork) or the measurement is
    process-lifetime-contaminated and order-dependent."""
    from tests.perf._measure import measure_peak_memory

    record = measure_peak_memory()
    assert record, "peak-memory child returned no record"
    assert record["pid"] != os.getpid(), "payload ran in the pytest process"
    assert record["start_method"] == "spawn", (
        f"child start_method is {record['start_method']!r}, not 'spawn'; "
        f"fork would reinherit the parent heap into ru_maxrss"
    )


def test_bench_min_rounds_applied() -> None:
    """PG.8 / arch R3-N: the train-step benchmark is invoked with the
    named `BENCH_MIN_ROUNDS` constant, not a prose-only number."""
    from pathlib import Path

    import tests.perf.test_train_step_time as mod

    src = Path(mod.__file__).read_text() if mod.__file__ else ""
    assert "rounds=BENCH_MIN_ROUNDS" in src
    assert isinstance(BENCH_MIN_ROUNDS, int)
    assert BENCH_MIN_ROUNDS >= 1
