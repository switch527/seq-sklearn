"""The architecture A12 `[docs]` pin block must match the live
`pyproject.toml [docs]` extra exactly. (Pre-PA.2 this test admitted
two states; Phase 12 R1 collapsed it back to a single equality check
once PA.2 landed both spec and live to the same nine pins.)
"""

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _extract_docs_extra(text: str) -> set[str]:
    """Pull package names out of a `docs = [...]` TOML or doc block."""
    match = re.search(r"docs\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert match is not None, "no `docs = [...]` block found"
    body = match.group(1)
    names = re.findall(r'"\s*([a-zA-Z0-9_.-]+?)\s*(?:[<>=!~\[]|")', body)
    return set(names)


def _pyproject_docs() -> set[str]:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]["docs"]
    out: set[str] = set()
    for p in extras:
        m = re.match(r"[a-zA-Z0-9_.-]+", p)
        assert m is not None, f"unparseable [docs] requirement: {p!r}"
        out.add(m.group(0))
    return out


def _arch_docs() -> set[str]:
    text = (_ROOT / "docs" / "architecture.md").read_text()
    # The A12 prose `[docs]` extra block is the FIRST `docs = [...]`;
    # the duplicate A18 example block follows the same shape.
    return _extract_docs_extra(text)


def test_docs_extra_matches_live() -> None:
    live = _pyproject_docs()
    target = _arch_docs()
    assert target == live, (
        f"architecture A12 [docs] block diverged from live pyproject.\n"
        f"  arch:       {sorted(target)}\n"
        f"  pyproject:  {sorted(live)}\n"
        f"  arch-only:  {sorted(target - live)}\n"
        f"  live-only:  {sorted(live - target)}"
    )
