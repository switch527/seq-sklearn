"""P-0 guard: the Phase 12 Sphinx reconciliation cannot silently
regress. Every `mkdocs`/`mkdocstrings`/`griffe-pydantic` mention in
the four spec docs must be on the named allowlist of superseded /
analog / policy contexts.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_DOCS = (
    _ROOT / "docs" / "architecture.md",
    _ROOT / "docs" / "requirements.md",
    _ROOT / "docs" / "implementation_plan.md",
    _ROOT / "docs" / "readme_and_docs_plan.md",
)
_PATTERN = re.compile(r"mkdocs|mkdocstrings|griffe-pydantic", re.IGNORECASE)

# Allowlist anchors (case-insensitive) for legitimate residual
# mentions. A hit is allowed when the hit line OR an adjacent line
# (±2-line window) contains any of these substrings.
_ANCHORS = (
    "superseded",
    "overturn",
    "reconcil",
    "ratified",
    "interim",
    "earlier mkdocs",
    "prior mkdocs",
    "not mkdocs",
    "sphinx analog",
    "resolved to sphinx",
    "mkdocs<2",
    "the 2.0",
    "hostile",
)


def _line_or_neighbors_contain_anchor(lines: list[str], idx: int, window: int = 2) -> bool:
    lo, hi = max(0, idx - window), min(len(lines), idx + window + 1)
    region = "\n".join(lines[lo:hi]).lower()
    return any(anchor in region for anchor in _ANCHORS)


@pytest.mark.parametrize("doc", _SPEC_DOCS, ids=lambda p: p.name)
def test_no_unreconciled_mkdocs_residue(doc: Path) -> None:
    lines = doc.read_text().splitlines()
    leaks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if _PATTERN.search(line) and not _line_or_neighbors_contain_anchor(lines, i):
            leaks.append((i + 1, line.rstrip()))
    assert not leaks, (
        f"unreconciled mkdocs/mkdocstrings/griffe-pydantic residue in "
        f"{doc.name} (Phase 12 P-0 / R1 arch-C1):\n"
        + "\n".join(f"  {doc.name}:{ln}: {text}" for ln, text in leaks)
    )
