"""Sphinx build is the docs trust gate (Phase 12 PD.4).

Runs in a temporary build directory so the test is isolated. Skips
loudly via `importorskip` when the `[docs]` extra is absent (the
default unit-job invocation includes `[docs]` per Phase 12 PD.3, so
on CI this never silently no-ops; on a `[dev]`-only dev box the skip
is visible in the test output).
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Visible skip if Sphinx is not installed (e.g. `uv sync --extra dev`
# without `--extra docs`).
pytest.importorskip("sphinx")

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
DOCS_DOCTEST_BUDGET_S = 240


def test_sphinx_build_warns_as_errors(tmp_path: Path) -> None:
    """`sphinx-build -W --keep-going -b html` exits 0 from a clean tree."""
    out = tmp_path / "html"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx.cmd.build",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(_DOCS_DIR),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"sphinx-build -W exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-4000:]}\n"
        f"stderr:\n{proc.stderr[-4000:]}"
    )
    assert (out / "index.html").exists()


def test_autodoc_pydantic_field_tables_rendered(tmp_path: Path) -> None:
    """The pydantic config tables actually carry field names, not an
    empty Fields section. `sphinx-build -W` alone passes on an
    empty-but-clean render; this asserts content."""
    out = tmp_path / "html"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx.cmd.build",
            "-W",
            "-b",
            "html",
            str(_DOCS_DIR),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    config_html = (out / "reference" / "config.html").read_text()
    # Expected pydantic field names from TFTConfig / TabularToSequenceConfig.
    for token in ("hidden_size", "attention_heads", "lookback"):
        assert token in config_html, (
            f"autodoc-pydantic rendered an empty / non-content config "
            f"page: token {token!r} missing from reference/config.html"
        )


def test_doc_snippet_suite_fast(tmp_path: Path) -> None:
    """The `-b doctest` build wall-clock stays inside the PR budget
    (`DOCS_DOCTEST_BUDGET_S = 240` seconds; the typical run takes
    ~20-40 s, so the budget is generous, not tight). A single hanging
    TFT-fit doctest block cannot blow the budget."""
    import time

    out = tmp_path / "doctest"
    start = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx.cmd.build",
            "-W",
            "-b",
            "doctest",
            str(_DOCS_DIR),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert elapsed < DOCS_DOCTEST_BUDGET_S, (
        f"docs `-b doctest` build took {elapsed:.1f}s "
        f"(budget {DOCS_DOCTEST_BUDGET_S}s)"
    )
