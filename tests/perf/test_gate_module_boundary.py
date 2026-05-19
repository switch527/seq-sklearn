"""PG.3 (import boundary + resolver logic), PG.5 autouse pin, PG.10
(Gemini-C1 collection guard). NON-`perf`, fast default suite.
"""

import subprocess
import sys
import textwrap

import pytest

_BOUNDARY_SNIPPET = textwrap.dedent(
    """
    import sys
    {imp}
    bad = [m for m in ("torch", "seq_sklearn") if m in sys.modules]
    assert not bad, f"heavy import leaked at module load: {{bad}}"
    print("OK")
    """
)


@pytest.mark.parametrize(
    "imp",
    ["import tests.perf._gate", "import tests.perf.conftest"],
)
def test_gate_module_has_no_heavy_imports(imp: str) -> None:
    """PG.3 (a)/(b): importing `_gate` or `conftest` must not pull
    torch/seq_sklearn (PC.1a / Gemini-C2). Subprocess uses
    `sys.executable` so it is the same venv as pytest (qa NEW-I2).
    The resolver is deliberately NOT called here."""
    proc = subprocess.run(
        [sys.executable, "-c", _BOUNDARY_SNIPPET.format(imp=imp)],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    assert proc.returncode == 0, f"{imp!r}: {proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout


# Helper modules the fast suite imports plus every non-perf test
# module. `test_boundary_guard_list_is_exhaustive` pins that this
# list stays in sync with what pytest actually collects, so a new
# fast test file or a dropped `@pytest.mark.perf` cannot silently
# re-open the R1 torch-bleed class (arch R2-I1).
_FAST_SUITE_MODULES = [
    "tests.perf._constants",
    "tests.perf._measure",
    "tests.perf.capture",
    "tests.perf.test_gate_logic",
    "tests.perf.test_baseline_schema",
    "tests.perf.test_gate_module_boundary",
    "tests.perf.test_workload",
    "tests.perf.test_capture_cli",
    "tests.perf.test_check_perf_baselines_script",
]


def test_boundary_guard_list_is_exhaustive() -> None:
    """arch R2-I1: the hand-maintained `_FAST_SUITE_MODULES` must cover
    every test module pytest actually collects in the fast
    (not perf/slow/gpu) suite. Adding a new fast test file without
    adding it here fails THIS test, so the torch-bleed guard cannot
    drift out of date."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "not perf and not slow and not gpu",
            "tests/perf/",
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    collected = {
        "tests.perf." + line.split("/")[2].split("::")[0].removesuffix(".py")
        for line in proc.stdout.splitlines()
        if line.startswith("tests/perf/") and "::" in line
    }
    missing = collected - set(_FAST_SUITE_MODULES)
    assert not missing, (
        f"fast-suite test modules not in the no-heavy-import guard list: "
        f"{sorted(missing)}; add them to _FAST_SUITE_MODULES"
    )


@pytest.mark.parametrize("imp", [f"import {m}" for m in _FAST_SUITE_MODULES])
def test_fast_suite_modules_have_no_heavy_imports(imp: str) -> None:
    """Post-Gemini code-review C1/C2 + I1: EVERY module the fast
    (non-`perf`) suite collects must be torch/seq_sklearn-free at
    import. PG.3 (a)/(b) only guarded `_gate`/`conftest`; a
    module-scope `_workload`/`capture` import in a test file silently
    reintroduced the boundary violation. This whole-suite guard makes
    that class of regression impossible to land unnoticed."""
    proc = subprocess.run(
        [sys.executable, "-c", _BOUNDARY_SNIPPET.format(imp=imp)],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    assert proc.returncode == 0, f"{imp!r}: {proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout


def _repo_root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[2])


def test_cell_resolver_branch_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    """PG.3 resolver-logic test (in-process, NOT asserting sys.modules;
    Gemini-C2: calling the resolver legitimately imports torch via
    `seq_sklearn.hardware`, so the no-torch boundary is enforced only
    by the two import checks above, never here)."""
    import torch

    import seq_sklearn.hardware as hw
    from tests.perf._gate import resolve_cell

    monkeypatch.setattr(hw, "detect", lambda: hw.HardwareTier.CPU)
    assert resolve_cell() == "cpu-x86"

    monkeypatch.setattr(hw, "detect", lambda: hw.HardwareTier.VOLTA_TURING)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _i=0: "Tesla T4")
    assert resolve_cell() == "t4"

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _i=0: "Tesla V100-SXM2")
    with pytest.raises(pytest.skip.Exception) as ei:
        resolve_cell()
    assert "VOLTA_TURING" in str(ei.value)
    assert "V100" in str(ei.value)

    for tier in ("PASCAL", "AMPERE_ADA", "HOPPER", "BLACKWELL"):
        monkeypatch.setattr(hw, "detect", lambda t=tier: getattr(hw.HardwareTier, t))
        # Device string deliberately does NOT contain the tier name, so
        # asserting both proves PC.2's "tier AND device" skip reason is
        # specific, not coincidental (arch-NIT / qa-NIT-1).
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _i=0: "GPU-XYZ-9000")
        with pytest.raises(pytest.skip.Exception) as ei:
            resolve_cell()
        assert tier in str(ei.value)
        assert "GPU-XYZ-9000" in str(ei.value)


def test_perf_fixture_not_autouse() -> None:
    """PG.5: the determinism fixture must be `autouse=False` so it does
    not pull torch at the start of the fast PR suite. Importing the
    conftest to introspect it must itself stay torch-free (lazy
    imports inside the fixture body, Gemini-C2)."""
    import tests.perf.conftest as cf

    marker = cf.perf_determinism._fixture_function_marker  # type: ignore[attr-defined]
    assert marker.autouse is False
    assert marker.scope == "session"


def test_all_three_perf_tests_collected() -> None:
    """PG.10 (Gemini-C1): `-m perf` must collect all three benchmark
    files. The earlier "collect with --benchmark-only, expect fewer"
    second assertion was retracted (post-Gemini qa-I2): that flag
    filters at the RUN phase, not collection, so it is mechanistically
    unsound under --collect-only. WHY --benchmark-only is forbidden in
    PF.1/PF.2 is documented in PB.1, not asserted here."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "perf",
            "tests/perf/",
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    out = proc.stdout
    for name in (
        "test_train_step_time.py",
        "test_peak_memory.py",
        "test_inference_latency.py",
    ):
        assert name in out, f"{name} not collected under -m perf:\n{out}\n{proc.stderr}"
