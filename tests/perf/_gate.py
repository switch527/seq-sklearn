"""Phase 11 perf-baseline gate (PC.1 / PC.2 / PD.1).

Module-import boundary (PC.1a, tightened by Gemini-C2): this module
imports ONLY pydantic + stdlib at load. It must NOT module-import
`seq_sklearn.hardware` (that module does an unconditional
`import torch` at `src/seq_sklearn/hardware.py:14`, which would
transitively pull torch at `_gate` import and void the fast-PR
boundary). :func:`resolve_cell` therefore imports ``detect`` and
``torch`` lazily, inside the function body, and is only ever called
from ``perf``-marked benchmark tests at run time.
"""

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

__all__ = [
    "PEAK_MEMORY_GATE",
    "STEP_TIME_GATE",
    "PerfBaseline",
    "PerfRegressionError",
    "assert_within_baseline",
    "baselines_dir",
    "expected_peak_memory_metric",
    "percentile_linear",
    "placeholder_measured",
    "resolve_cell",
]

logger = logging.getLogger("seq_sklearn.perf")

# A13 regression thresholds (P3): median step time +15%, peak memory +10%.
STEP_TIME_GATE = 1.15
PEAK_MEMORY_GATE = 1.10

Cell = Literal["cpu-x86", "t4"]
PeakMemoryMetric = Literal["ru_maxrss_kb", "cuda_max_alloc_bytes"]


class PerfRegressionError(Exception):
    """A gated perf metric breached its baseline (or the baseline file
    is missing/invalid under enforce, or a metric-name mismatch)."""


class PerfBaseline(BaseModel):
    """Checked-in per-cell baseline (PC.1). ``extra="forbid"`` so an
    unknown key is a hard ``ValidationError``, never silently dropped;
    every metric field is a required ``float`` so a missing one raises
    at load. `tracemalloc` is deliberately absent: it is logged at
    measurement time, never persisted (Q2 / Gemini-C2)."""

    model_config = ConfigDict(extra="forbid")

    cell: Cell
    captured_git_sha: str
    torch_version: str
    python_version: str
    device_name: str
    provisional: bool = False

    train_step_median_s: float
    train_step_p95_s: float
    peak_memory_value: float
    peak_memory_metric: PeakMemoryMetric
    inference_latency_median_s: float
    inference_latency_p95_s: float


def percentile_linear(data: list[float], q: float) -> float:
    """The ``q``-th percentile (0..100) with numpy's ``method="linear"``
    interpolation, implemented in pure stdlib so this module stays
    pydantic+stdlib-only (PC.1a). `PG.8 test_p95_matches_numpy` pins
    this against ``numpy.percentile(..., method="linear")``."""
    if not data:
        raise ValueError("percentile_linear: empty data")
    ordered = sorted(data)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def baselines_dir() -> Path:
    """Directory holding the checked-in per-cell baseline JSON files."""
    return Path(__file__).parent / "_baselines"


def expected_peak_memory_metric(cell: str) -> PeakMemoryMetric:
    """The peak-memory metric a cell measures: RSS on the CPU cell,
    CUDA bytes otherwise. A per-benchmark test that does not itself
    measure memory uses this to fill a non-breaching, metric-matching
    placeholder so it gates only its own metric without tripping the
    PD.1a mode-independent mismatch raise."""
    return "ru_maxrss_kb" if cell == "cpu-x86" else "cuda_max_alloc_bytes"


def resolve_cell() -> Cell:
    """Resolve the runtime cell (PC.2). Two-step, NOT tier-only:
    ``HardwareTier.VOLTA_TURING`` spans V100 (CC 7.0) and T4 (CC 7.5),
    so a CUDA device is the ``t4`` cell only when it is a name-confirmed
    T4. Any other CUDA device ``pytest.skip``s with a reason naming
    both the tier and the device (arch-I1). torch + ``detect`` are
    imported HERE, lazily, never at module scope (PC.1a / Gemini-C2)."""
    import pytest
    import torch

    from seq_sklearn.hardware import HardwareTier, detect

    tier = detect()
    if tier == HardwareTier.CPU:
        return "cpu-x86"
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    if tier == HardwareTier.VOLTA_TURING and "T4" in device:
        return "t4"
    pytest.skip(
        f"no perf baseline for tier {tier.name} device {device!r}: the v1 "
        f"public cells are cpu-x86 and t4 only (A13/P2; other devices are "
        f"optional contributor cells, D2)"
    )
    raise AssertionError("unreachable: pytest.skip always raises")


def placeholder_measured(cell: str, **metrics: object) -> PerfBaseline:
    """A `measured` `PerfBaseline` with non-breaching placeholders for
    every field, updated by ``metrics``. A per-benchmark test fills
    only its own metric(s); the unset gated metrics stay 0.0 (which
    can never exceed ``baseline * gate``) and ``peak_memory_metric``
    defaults to the cell's expected metric so the PD.1a mismatch raise
    is not spuriously tripped."""
    base: dict[str, object] = {
        "cell": cell,
        "captured_git_sha": "measured",
        "torch_version": "measured",
        "python_version": "measured",
        "device_name": "measured",
        "provisional": False,
        "train_step_median_s": 0.0,
        "train_step_p95_s": 0.0,
        "peak_memory_value": 0.0,
        "peak_memory_metric": expected_peak_memory_metric(cell),
        "inference_latency_median_s": 0.0,
        "inference_latency_p95_s": 0.0,
    }
    base.update(metrics)
    return PerfBaseline.model_validate(base)


def _gate_mode() -> str:
    """``enforce`` raises on a breach; anything else (incl. unset) warns
    (PD.2: ``warn`` is the safe default)."""
    return os.environ.get("SEQ_SKLEARN_PERF_GATE", "warn")


def _load_baseline(cell: str, directory: Path) -> PerfBaseline:
    return PerfBaseline.model_validate_json((directory / f"{cell}.json").read_text())


def assert_within_baseline(
    cell: str,
    measured: PerfBaseline,
    *,
    directory: Path | None = None,
) -> None:
    """Compare ``measured`` against the checked-in baseline for ``cell``.

    Precedence (PD.1b), strictly ordered:
      0. load+validate the baseline file; on any failure apply the
         PD.1c missing/corrupt branch and stop.
      1. ``provisional`` baseline => warn + return (PD.1b / PE.3).
      2. ``peak_memory_metric`` mismatch => raise, mode-independent
         (PD.1a): a unit mismatch is a config bug, not a regression.
      3. mode dispatch: ``enforce`` raises on a step (>15%) or memory
         (>10%) breach; ``warn``/unset logs a warning and returns.
    Latency and P95 are recorded but never gated in v1 (PD.1).
    """
    directory = directory if directory is not None else baselines_dir()
    mode = _gate_mode()

    # (0) load+validate; missing/corrupt -> PD.1c (never a silent pass).
    try:
        baseline = _load_baseline(cell, directory)
    except (OSError, ValueError, ValidationError) as exc:
        msg = f"baseline for cell {cell} missing or invalid: {exc}"
        if mode == "enforce":
            raise PerfRegressionError(msg) from exc
        logger.warning("perf gate (warn): %s", msg)
        return

    # (1) provisional short-circuit (precedes the metric-name guard:
    # a provisional metric name is itself a hand-seeded placeholder).
    if baseline.provisional:
        logger.warning("perf gate: gating skipped: provisional baseline for cell %s", cell)
        return

    # (2) metric-name guard, mode-independent (PD.1a).
    if measured.peak_memory_metric != baseline.peak_memory_metric:
        raise PerfRegressionError(
            f"peak_memory_metric mismatch for cell {cell}: measured "
            f"{measured.peak_memory_metric!r} vs baseline "
            f"{baseline.peak_memory_metric!r} (RSS vs CUDA bytes are "
            f"incommensurable; this is a configuration bug)"
        )

    # (3) mode dispatch on the two gated metrics (A13: step 15%, mem 10%).
    breaches: list[str] = []
    if measured.train_step_median_s > baseline.train_step_median_s * STEP_TIME_GATE:
        breaches.append(
            f"train_step_median_s {measured.train_step_median_s:.6g} > "
            f"{baseline.train_step_median_s:.6g} * {STEP_TIME_GATE} "
            f"(+{STEP_TIME_GATE - 1:.0%} gate)"
        )
    if measured.peak_memory_value > baseline.peak_memory_value * PEAK_MEMORY_GATE:
        breaches.append(
            f"peak_memory_value {measured.peak_memory_value:.6g} > "
            f"{baseline.peak_memory_value:.6g} * {PEAK_MEMORY_GATE} "
            f"(+{PEAK_MEMORY_GATE - 1:.0%} gate)"
        )

    if not breaches:
        logger.info("perf gate: cell %s within baseline", cell)
        return

    detail = f"perf regression on cell {cell}: " + "; ".join(breaches)
    if mode == "enforce":
        raise PerfRegressionError(detail)
    logger.warning("perf gate (warn, non-blocking): %s", detail)
