"""README `## Quickstart` block is parseable AND executable.

Closes the qa R3-C1 gap: the README is at the repo root, outside
`docs/`, so neither `sphinx.ext.doctest` nor `pytest --doctest-modules
src/` reaches it. This is R3 mechanism (d). The README block is
extracted, parsed, and `exec`d with the same CPU monkeypatches the
e2e quickstart test uses.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

_README = Path(__file__).resolve().parents[2] / "README.md"


def _quickstart_block(text: str) -> str:
    """Extract the FIRST fenced ```python block AFTER `## Quickstart`."""
    quickstart_idx = text.find("## Quickstart")
    if quickstart_idx == -1:
        pytest.fail(
            "README.md is missing a `## Quickstart` heading (Phase 12 PE.0): "
            "the Planned-API stubs were not rewritten into a real Quickstart."
        )
    tail = text[quickstart_idx:]
    match = re.search(r"```python\n(.*?)\n```", tail, re.DOTALL)
    if match is None:
        pytest.fail(
            "README.md has `## Quickstart` but no fenced ```python block "
            "beneath it"
        )
    return match.group(1)


def test_quickstart_block_parses() -> None:
    """The README quickstart block is valid Python (cheap structural pin)."""
    block = _quickstart_block(_README.read_text())
    ast.parse(block)  # raises SyntaxError if not valid Python


def test_quickstart_block_imports_real_public_symbols() -> None:
    """Every `from <mod> import <name>` in the README quickstart resolves
    to a live attribute on the imported module.

    Catches a rename inside any imported module (façade, adapter,
    synthetic generator) at parse time, before the slower exec test
    has to detect it via a runtime AttributeError. Validates EVERY
    `ImportFrom` node, not just the `seq_sklearn` façade.
    """
    block = _quickstart_block(_README.read_text())
    tree = ast.parse(block)
    facade_imports: set[str] = set()
    unresolved: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        mod = importlib.import_module(node.module)
        for alias in node.names:
            if not hasattr(mod, alias.name):
                unresolved.append(f"{node.module}.{alias.name}")
            if node.module == "seq_sklearn":
                facade_imports.add(alias.name)
    assert not unresolved, (
        f"README quickstart imports symbols that don't exist on the "
        f"target module (rename or removal): {unresolved}"
    )
    # The README quickstart should use the public façade for the
    # estimator (PE.0); at minimum TFTClassifier from the façade.
    assert "TFTClassifier" in facade_imports, (
        "README quickstart should import TFTClassifier from seq_sklearn "
        f"(the public façade); got façade imports: {facade_imports}"
    )


def test_quickstart_block_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The README quickstart EXECUTES end-to-end on CPU.

    Mirrors `tests/e2e/test_quickstart.py::_force_cpu`: patches
    `seq_sklearn.training.trainer.detect`, `torch.cuda.is_available`,
    and `torch.cuda.device_count` so the block runs on the CPU path
    regardless of host hardware (a Blackwell box still exec's the
    block in the CPU codepath, matching CI).
    """
    pytest.importorskip("seq_sklearn")
    import torch

    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr(
        "seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    block = _quickstart_block(_README.read_text())
    # Trim the block down so a CPU exec is fast: replace any
    # `max_epochs=<N>` with `max_epochs=1`; replace `num_entities=<N>`
    # and `periods_per_entity=<N>` with small ints. This keeps the
    # source legible while making the test cheap; the e2e quickstart
    # remains the heavy-fit gate.
    # `\s*=\s*` tolerates a future README reformat that introduces
    # spaces around `=`; the substitution must catch the heavy default
    # values regardless of whitespace, else the unit job will silently
    # fit on a 160-entity panel and blow the budget.
    block = re.sub(r"max_epochs\s*=\s*\d+", "max_epochs=1", block)
    block = re.sub(r"num_entities\s*=\s*\d+", "num_entities=32", block)
    block = re.sub(r"periods_per_entity\s*=\s*\d+", "periods_per_entity=6", block)
    namespace: dict[str, object] = {}
    exec(compile(block, "<README ## Quickstart>", "exec"), namespace)
