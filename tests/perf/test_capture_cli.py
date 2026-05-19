"""PG.6: the capture CLI round-trips through `PerfBaseline` and writes
a correct `device_name` (qa-I2 / R3-C1 / R4-C1). Monkeypatches the
three `_measure` functions so it runs offline without a real perf
run; pins capture/gate schema parity before any nightly run.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.perf import capture
from tests.perf._gate import PerfBaseline


def _patch_measures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.perf import _measure

    monkeypatch.setattr(_measure, "measure_train_step", lambda: (1.25, 1.40))
    monkeypatch.setattr(
        _measure,
        "measure_peak_memory",
        lambda: {
            "value": 999.0,
            "peak_memory_metric": "ru_maxrss_kb",
            "device_name": "cpu",
            "pid": 1,
            "start_method": "spawn",
        },
    )
    monkeypatch.setattr(_measure, "measure_inference_latency", lambda: (0.0011, 0.0019))
    monkeypatch.setattr(capture, "baselines_dir", lambda: tmp_path)


def test_capture_cli_roundtrips_through_pydantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_measures(monkeypatch, tmp_path)
    baseline = capture.capture("cpu-x86")

    written = PerfBaseline.model_validate_json((tmp_path / "cpu-x86.json").read_text())
    assert written == baseline
    assert written.train_step_median_s == pytest.approx(1.25)
    assert written.train_step_p95_s == pytest.approx(1.40)
    assert written.peak_memory_value == pytest.approx(999.0)
    assert written.inference_latency_median_s == pytest.approx(0.0011)

    if shutil.which("git"):
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.SubprocessError:
            pytest.skip("git present but HEAD unresolvable")
        assert written.captured_git_sha == sha


def test_capture_writes_device_name_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """qa R3-C1 / R4-C1: `device_name` is a free `str`; only running
    the capture path catches a bug that copies the CUDA branch on CPU
    or writes an empty string."""
    _patch_measures(monkeypatch, tmp_path)
    baseline = capture.capture("cpu-x86")
    assert baseline.device_name == "cpu"
