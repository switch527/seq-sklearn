"""README `## Quickstart` block is parseable AND executable.

Closes the qa R3-C1 gap: the README is at the repo root, outside
`docs/`, so neither `sphinx.ext.doctest` nor `pytest --doctest-modules
src/` reaches it. This is R3 mechanism (d) — the README block is
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
    """The block imports the same public symbols as `examples/quickstart.py`,
    via the `seq_sklearn` façade. A non-vacuous structural pin layered on
    the execution check below."""
    block = _quickstart_block(_README.read_text())
    tree = ast.parse(block)
    names_imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "seq_sklearn":
            for alias in node.names:
                names_imported.add(alias.name)
    # The README quickstart should use the public façade for the
    # estimator (PE.0); at minimum TFTClassifier from the façade.
    assert "TFTClassifier" in names_imported, (
        "README quickstart should import TFTClassifier from seq_sklearn "
        f"(the public façade); got imports from seq_sklearn: {names_imported}"
    )
    # TFTClassifier resolves to a real class on `seq_sklearn`.
    seq_sklearn = importlib.import_module("seq_sklearn")
    assert hasattr(seq_sklearn, "TFTClassifier")


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
    block = re.sub(r"max_epochs=\d+", "max_epochs=1", block)
    block = re.sub(r"num_entities=\d+", "num_entities=32", block)
    block = re.sub(r"periods_per_entity=\d+", "periods_per_entity=6", block)
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - executing trusted README content under monkeypatch
        compile(block, "<README ## Quickstart>", "exec"), namespace
    )
