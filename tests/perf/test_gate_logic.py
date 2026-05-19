"""PG.1 (gate-branch + P4 done-when), PG.9 (latency non-gate), PG.8
`test_p95_matches_numpy`. NON-`perf`-marked: runs in the fast default
suite, imports only `_gate` (no torch), discharges P4 offline and
deterministically.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from tests.perf._gate import (
    PerfBaseline,
    PerfRegressionError,
    assert_within_baseline,
    percentile_linear,
    placeholder_measured,
)

CELL = "cpu-x86"


def _write_baseline(directory: Path, **overrides: object) -> None:
    fields: dict[str, object] = {
        "cell": CELL,
        "captured_git_sha": "base",
        "torch_version": "base",
        "python_version": "base",
        "device_name": "cpu",
        "provisional": False,
        "train_step_median_s": 1.0,
        "train_step_p95_s": 1.0,
        "peak_memory_value": 1000.0,
        "peak_memory_metric": "ru_maxrss_kb",
        "inference_latency_median_s": 0.001,
        "inference_latency_p95_s": 0.001,
    }
    fields.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{CELL}.json").write_text(PerfBaseline.model_validate(fields).model_dump_json())


def test_enforce_step_breach_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    _write_baseline(tmp_path)
    measured = placeholder_measured(CELL, train_step_median_s=1.20)  # +20% > 15%
    with pytest.raises(PerfRegressionError):
        assert_within_baseline(CELL, measured, directory=tmp_path)


def test_enforce_step_within_band_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    _write_baseline(tmp_path)
    measured = placeholder_measured(CELL, train_step_median_s=1.05)  # +5% < 15%
    assert_within_baseline(CELL, measured, directory=tmp_path)


def test_enforce_memory_breach_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    _write_baseline(tmp_path)
    measured = placeholder_measured(CELL, peak_memory_value=1200.0)  # +20% > 10%
    with pytest.raises(PerfRegressionError):
        assert_within_baseline(CELL, measured, directory=tmp_path)


def test_enforce_memory_within_band_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    _write_baseline(tmp_path)
    measured = placeholder_measured(CELL, peak_memory_value=1050.0)  # +5% < 10%
    assert_within_baseline(CELL, measured, directory=tmp_path)


def test_warn_mode_breach_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "warn")
    _write_baseline(tmp_path)
    measured = placeholder_measured(CELL, train_step_median_s=1.20)
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.perf"):
        assert_within_baseline(CELL, measured, directory=tmp_path)
    assert any("perf regression" in r.message for r in caplog.records)


def test_unset_env_defaults_to_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("SEQ_SKLEARN_PERF_GATE", raising=False)
    _write_baseline(tmp_path)  # provisional=False, ru_maxrss_kb metric
    measured = placeholder_measured(CELL, train_step_median_s=1.20)
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.perf"):
        assert_within_baseline(CELL, measured, directory=tmp_path)
    assert any("perf regression" in r.message for r in caplog.records)


def test_metric_name_mismatch_raises_even_in_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "warn")
    _write_baseline(tmp_path, peak_memory_metric="ru_maxrss_kb")
    measured = placeholder_measured(CELL)
    measured = measured.model_copy(update={"peak_memory_metric": "cuda_max_alloc_bytes"})
    with pytest.raises(PerfRegressionError, match="mismatch"):
        assert_within_baseline(CELL, measured, directory=tmp_path)


def test_provisional_baseline_never_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    _write_baseline(tmp_path, provisional=True)
    measured = placeholder_measured(CELL, train_step_median_s=1.20)
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.perf"):
        assert_within_baseline(CELL, measured, directory=tmp_path)
    assert any("provisional" in r.message for r in caplog.records)


def test_provisional_precedes_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    _write_baseline(tmp_path, provisional=True, peak_memory_metric="ru_maxrss_kb")
    measured = placeholder_measured(CELL).model_copy(
        update={"peak_memory_metric": "cuda_max_alloc_bytes"}
    )
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.perf"):
        assert_within_baseline(CELL, measured, directory=tmp_path)  # no raise
    assert any("provisional" in r.message for r in caplog.records)


def test_latency_breach_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PG.9: latency is observational, not gated (PD.1)."""
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    _write_baseline(tmp_path)
    measured = placeholder_measured(CELL, inference_latency_median_s=0.003)  # 3x
    assert_within_baseline(CELL, measured, directory=tmp_path)  # no raise


@pytest.mark.parametrize("q", [50.0, 95.0, 0.0, 100.0])
def test_p95_matches_numpy(q: float) -> None:
    """PG.8: the stdlib percentile helper equals numpy's linear method."""
    data = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    assert percentile_linear(data, q) == pytest.approx(
        float(np.percentile(data, q, method="linear"))
    )
