"""The remaining small docs gates (Phase 12 PD.4 / PE.1).

- `test_every_doc_has_a_toctree_home`: explicit orphan check that
  complements `sphinx-build -W` nav validation.
- `test_nonexecutable_block_ratio_bounded`: caps the share of
  non-executable python blocks in the docs (no silent opt-out).
- `test_n7_absolute_test_present_and_unskipped`: pins criterion 9's
  release-checklist artifact.
- `test_changelog_has_1_0_0_entry`: PE.1 regex guard for the
  Keep-a-Changelog v1.0.0 header.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_MAX_NONEXEC_RATIO = 0.6  # ample headroom; the gate exists to stop a flip-to-zero, not to micromanage.


def _all_docs_md() -> list[Path]:
    skip = {"research", "references", "_build", "_static"}
    return sorted(
        p
        for p in _DOCS.rglob("*.md")
        if not any(part in skip for part in p.relative_to(_DOCS).parts)
    )


def _all_toctree_refs() -> set[str]:
    """Collect the docnames referenced by any `toctree` directive
    anywhere under `docs/`. MyST uses ``` ```{toctree} ``` ``` fences;
    reST uses `.. toctree::`."""
    refs: set[str] = set()
    fence_re = re.compile(r"```\{toctree\}(.*?)```", re.DOTALL)
    rst_re = re.compile(r"\.\.\s*toctree::.*?\n((?:^[ \t]+.*\n|^\n)+)", re.MULTILINE)
    for md in _DOCS.rglob("*.md"):
        text = md.read_text()
        for match in fence_re.finditer(text):
            for line in match.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                refs.add(_resolve(md, line))
    for rst in _DOCS.rglob("*.rst"):
        text = rst.read_text()
        for match in rst_re.finditer(text):
            for line in match.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                refs.add(_resolve(rst, line))
    return refs


def _resolve(source: Path, ref: str) -> str:
    """Resolve a relative toctree entry to a docname-from-docs-root."""
    base = source.parent
    candidate = (base / ref).resolve()
    try:
        rel = candidate.relative_to(_DOCS)
    except ValueError:
        return ref
    name = str(rel).removesuffix(".md").removesuffix(".rst")
    return name.replace("\\", "/")


def test_every_doc_has_a_toctree_home() -> None:
    """Every published `.md` is reachable from a toctree. The root
    `index.md` is the entrypoint and exempt from being toctree'd
    from elsewhere."""
    refs = _all_toctree_refs()
    refs.add("index")  # root entrypoint
    docs_without_home: list[str] = []
    for md in _all_docs_md():
        name = str(md.relative_to(_DOCS)).removesuffix(".md").replace("\\", "/")
        if name not in refs:
            docs_without_home.append(name)
    assert not docs_without_home, (
        f"docs without a toctree home (orphan pages, would fail "
        f"`sphinx-build -W`): {docs_without_home}"
    )


def test_nonexecutable_block_ratio_bounded() -> None:
    """The ratio of plain `python` fenced blocks (non-doctest) vs
    total python fenced blocks across the prose pages stays below
    `_MAX_NONEXEC_RATIO`. Prevents an author silently opting the
    whole suite out of execution (R3 gate teeth)."""
    plain = 0
    executable = 0
    # A doctest block in MyST is either the RST directive
    # `.. testcode::` / `.. doctest::` (inside `{eval-rst}`) or the
    # MyST-native fence ``` ```{testcode} ``` ``` / ``` ```{doctest} ```.
    # The plain `python` fence is the non-executable variant.
    plain_re = re.compile(r"```python\n")
    testcode_re = re.compile(r"(\.\.\s*(?:testcode|doctest)::|```\{(?:testcode|doctest)\})")
    # Skip files whose role is to host code that is INTENTIONALLY not
    # doctested:
    #   - migration_template.md, changelog.md: out-of-tree snippet hosts
    #   - governance docs under design/ (architecture, requirements,
    #     implementation_plan, phase plans, research/strategy briefs)
    #     contain illustrative design code, not user snippets. They
    #     still live at `docs/` root pending the design/ physical move,
    #     so we enumerate them here.
    skip = {
        "migration_template.md",
        "changelog.md",
        "architecture.md",
        "requirements.md",
        "implementation_plan.md",
        "phase10_onnx_deploy.md",
        "phase11_perf_baselines.md",
        "phase12_docs_release.md",
        "phase_1_refactor_plan.md",
        "benchmark_suite_design.md",
        "benchmark_suite_implementation_plan.md",
        "refactor_prediction_step.md",
        "hyperparameter_strategy.md",
        "docs_strategy_research.md",
        "readme_and_docs_plan.md",
    }
    for md in _all_docs_md():
        if md.name in skip:
            continue
        text = md.read_text()
        plain += len(plain_re.findall(text))
        executable += len(testcode_re.findall(text))
    total = plain + executable
    if total == 0:
        return  # vacuously fine; the gate exists to bound an active codebase
    ratio = plain / total
    assert ratio < _MAX_NONEXEC_RATIO, (
        f"non-executable python fenced blocks are {ratio:.0%} of all "
        f"python blocks (cap {_MAX_NONEXEC_RATIO:.0%}); add `.. testcode::` "
        f"to more prose snippets or move illustrative-only ones to the "
        f"migration template."
    )


def test_n7_absolute_test_present_and_unskipped() -> None:
    """Criterion 9 (N7 absolute budgets) has a real release-checklist
    artifact (`tests/perf/test_n7_absolute.py`), not just an
    enumerated line.

    The skip-detection rule: a top-of-line `@pytest.mark.skip` marker
    (bare or parameterised) on a test function is an unconditional
    skip and fails the gate. A `pytest.skip(...)` call inside a test
    body is conditional and is allowed (the N7 artifact skips
    conditionally on CUDA availability, which is fine).
    """
    path = _ROOT / "tests" / "perf" / "test_n7_absolute.py"
    assert path.exists(), f"missing {path.relative_to(_ROOT)}"
    text = path.read_text()
    assert "def test_" in text, "no test function in test_n7_absolute.py"
    unconditional = re.search(r"^\s*@pytest\.mark\.skip\b", text, re.MULTILINE)
    assert unconditional is None, (
        "test_n7_absolute.py carries an unconditional `@pytest.mark.skip` "
        "decorator; the release-checklist artifact is degenerate"
    )


def test_changelog_has_1_0_0_entry() -> None:
    """PE.1: `[Unreleased]` -> `[1.0.0]` shape, Keep-a-Changelog."""
    text = (_ROOT / "CHANGELOG.md").read_text()
    assert re.search(r"^## \[1\.0\.0\]", text, re.MULTILINE), (
        "CHANGELOG.md is missing the `## [1.0.0]` entry; the release "
        "engineer will not have a header to date-stamp."
    )
    assert re.search(r"^## \[Unreleased\]", text, re.MULTILINE), (
        "CHANGELOG.md is missing the fresh `[Unreleased]` section at "
        "the top; new changes have nowhere to land."
    )
