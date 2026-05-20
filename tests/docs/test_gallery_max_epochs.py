"""Static pre-execution scan: every gallery script bounds
`max_epochs` to a small pinned ceiling, so a heavy example cannot
silently blow the docs CI budget at build time (caught BEFORE
sphinx-gallery executes it)."""

import ast
from pathlib import Path

import pytest

_DOC_MAX_EPOCHS = 2  # the same pinned bound used by the README block test
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "examples"


def _gallery_scripts() -> list[Path]:
    return sorted(p for p in _EXAMPLES_DIR.glob("plot_*.py"))


def _max_epochs_kwargs(tree: ast.Module) -> list[int]:
    values: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "max_epochs":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                values.append(node.value.value)
            else:
                values.append(-1)  # non-literal => fail loudly
    return values


@pytest.mark.parametrize("script", _gallery_scripts(), ids=lambda p: p.name)
def test_every_gallery_script_has_max_epochs_bounded(script: Path) -> None:
    tree = ast.parse(script.read_text())
    found = _max_epochs_kwargs(tree)
    assert found, (
        f"gallery script {script.name} has no `max_epochs=<int>` kwarg; "
        f"plain code-paths could run any number of epochs and blow the "
        f"build budget. Add `max_epochs=1` to every estimator construction."
    )
    bad = [v for v in found if not (1 <= v <= _DOC_MAX_EPOCHS)]
    assert not bad, (
        f"gallery script {script.name} has max_epochs outside [1, "
        f"{_DOC_MAX_EPOCHS}]: {bad} (non-literal values show as -1)"
    )
