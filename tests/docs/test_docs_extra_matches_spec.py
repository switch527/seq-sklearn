"""The architecture A12/A18 `[docs]` pin list either matches the live
`pyproject.toml [docs]` extra OR matches it plus exactly
`{sphinx-gallery, autodoc-pydantic}` (the PA.2 adds). Catches the
target-vs-live divergence regressing into real drift.
"""

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _extract_docs_extra(text: str) -> set[str]:
    """Pull package names out of a `docs = [...]` TOML or doc block."""
    match = re.search(r"docs\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert match is not None, "no `docs = [...]` block found"
    body = match.group(1)
    # Capture the package name before any version specifier.
    names = re.findall(r'"\s*([a-zA-Z0-9_.-]+?)\s*(?:[<>=!~\[]|")', body)
    return set(names)


if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = None  # type: ignore[assignment]


def _pyproject_docs() -> set[str]:
    if tomllib is None:
        pytest.skip("tomllib unavailable on Python < 3.11")
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]["docs"]
    return {re.match(r"[a-zA-Z0-9_.-]+", p).group(0) for p in extras}


def _arch_docs() -> set[str]:
    text = (_ROOT / "docs" / "architecture.md").read_text()
    # The A12 prose `[docs]` extra block is the FIRST `docs = [...]`;
    # the duplicate A18 example block follows the same shape.
    return _extract_docs_extra(text)


def test_docs_extra_is_target_or_live() -> None:
    live = _pyproject_docs()
    target = _arch_docs()
    # Two PA.2 additions take the live to the target.
    expected_target = live | {"sphinx-gallery", "autodoc-pydantic"}
    assert target == live or target == expected_target, (
        f"architecture A12 [docs] target diverged from live pyproject and\n"
        f"is not the documented target-equals-live-plus-{{sphinx-gallery, "
        f"autodoc-pydantic}} state.\n"
        f"  arch (target): {sorted(target)}\n"
        f"  pyproject (live): {sorted(live)}\n"
        f"  expected target: {sorted(expected_target)}"
    )
