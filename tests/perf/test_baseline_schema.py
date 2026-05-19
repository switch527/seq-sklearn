"""PG.2: the PC.1 schema is enforced, not assumed, and PD.1c
missing/corrupt-baseline behavior. NON-`perf`, fast suite, no torch.
"""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.perf._gate import (
    PerfBaseline,
    PerfRegressionError,
    assert_within_baseline,
    baselines_dir,
    placeholder_measured,
)

_VALID: dict[str, object] = {
    "cell": "cpu-x86",
    "captured_git_sha": "abc",
    "torch_version": "2.0",
    "python_version": "3.12",
    "device_name": "cpu",
    "provisional": False,
    "train_step_median_s": 1.0,
    "train_step_p95_s": 1.1,
    "peak_memory_value": 1000.0,
    "peak_memory_metric": "ru_maxrss_kb",
    "inference_latency_median_s": 0.001,
    "inference_latency_p95_s": 0.002,
}


@pytest.mark.parametrize("cell", ["cpu-x86", "t4"])
def test_perf_baselines_present_and_valid(cell: str) -> None:
    path = baselines_dir() / f"{cell}.json"
    assert path.exists(), f"missing checked-in baseline {path}"
    PerfBaseline.model_validate_json(path.read_text())


def test_missing_metric_key_raises() -> None:
    bad = dict(_VALID)
    del bad["train_step_p95_s"]
    with pytest.raises(ValidationError):
        PerfBaseline.model_validate(bad)


def test_extra_key_raises() -> None:
    with pytest.raises(ValidationError):
        PerfBaseline.model_validate({**_VALID, "unknown_field": 1})


def test_bad_peak_memory_metric_literal_raises() -> None:
    with pytest.raises(ValidationError):
        PerfBaseline.model_validate({**_VALID, "peak_memory_metric": "cuda_bytes"})


def test_tracemalloc_key_rejected() -> None:
    """Q2 / Gemini-C2: tracemalloc is logged, never persisted."""
    with pytest.raises(ValidationError):
        PerfBaseline.model_validate({**_VALID, "tracemalloc_peak_kb": 123.0})


def test_missing_baseline_enforce_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    with pytest.raises(PerfRegressionError, match="missing or invalid"):
        assert_within_baseline("cpu-x86", placeholder_measured("cpu-x86"), directory=tmp_path)


def test_corrupt_baseline_enforce_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "enforce")
    (tmp_path / "cpu-x86.json").write_text('{"cell": "cpu-x86", trunc')
    with pytest.raises(PerfRegressionError, match="missing or invalid"):
        assert_within_baseline("cpu-x86", placeholder_measured("cpu-x86"), directory=tmp_path)


def test_missing_baseline_warn_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "warn")
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.perf"):
        assert_within_baseline("cpu-x86", placeholder_measured("cpu-x86"), directory=tmp_path)
    assert any("missing or invalid" in r.message for r in caplog.records)


def test_corrupt_baseline_warn_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_PERF_GATE", "warn")
    (tmp_path / "cpu-x86.json").write_text("not json at all")
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.perf"):
        assert_within_baseline("cpu-x86", placeholder_measured("cpu-x86"), directory=tmp_path)
    assert any("missing or invalid" in r.message for r in caplog.records)
