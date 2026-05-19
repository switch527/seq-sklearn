"""Phase 11 baseline capture CLI (PE.1).

``python -m tests.perf.capture --cell <cell>`` runs the three
measurement functions and writes ``_baselines/<cell>.json`` with the
current git SHA. The SAME `measure_*` code path the gate uses
produces the baseline, so capture and check can never diverge. PG.6
monkeypatches the three `tests.perf._measure` functions to drive this
offline.
"""

import argparse
import importlib.metadata
import platform
import subprocess
import sys

from tests.perf import _measure
from tests.perf._gate import PerfBaseline, baselines_dir

__all__ = ["capture", "main"]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def capture(cell: str, *, provisional: bool = False) -> PerfBaseline:
    """Measure and return the `PerfBaseline` for ``cell`` (also written
    to ``_baselines/<cell>.json``). `torch_version` is read from
    installed-distribution metadata (stdlib `importlib.metadata`), NOT
    by importing torch, so the monkeypatched PG.6 path stays torch-free
    (post-Gemini code-review C1)."""
    step_median, step_p95 = _measure.measure_train_step()
    mem = _measure.measure_peak_memory()
    lat_median, lat_p95 = _measure.measure_inference_latency()

    baseline = PerfBaseline(
        cell=cell,  # type: ignore[arg-type]
        captured_git_sha=_git_sha(),
        torch_version=importlib.metadata.version("torch"),
        python_version=platform.python_version(),
        device_name=mem["device_name"],
        provisional=provisional,
        train_step_median_s=step_median,
        train_step_p95_s=step_p95,
        peak_memory_value=mem["value"],
        peak_memory_metric=mem["peak_memory_metric"],
        inference_latency_median_s=lat_median,
        inference_latency_p95_s=lat_p95,
    )
    out = baselines_dir() / f"{cell}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(baseline.model_dump_json(indent=2) + "\n")
    return baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tests.perf.capture")
    parser.add_argument("--cell", required=True, choices=["cpu-x86", "t4"])
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args(argv)
    baseline = capture(args.cell, provisional=args.provisional)
    sys.stdout.write(baseline.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
