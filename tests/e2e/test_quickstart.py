"""Quickstart-in-CI (N1 / A14).

Imports the same ``examples/quickstart.py`` the README mirrors and
asserts the N1 binary-classifier threshold (accuracy >= 0.75) at
seed=42, dgp_version=1 per A14 ("Not a smoke test"). README and CI
cannot drift because both point at this one example file.

This is the ONLY e2e test that runs on every PR: it is marked ``e2e``
but deliberately NOT ``slow`` (per the implementation_plan: marking it
slow would silently drop the N1 quickstart-in-CI requirement).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.e2e

_QUICKSTART = Path(__file__).parents[2] / "examples" / "quickstart.py"


def _load_quickstart() -> object:
    spec = importlib.util.spec_from_file_location("seq_sklearn_quickstart_example", _QUICKSTART)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def test_quickstart_meets_n1_binary_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    module = _load_quickstart()
    accuracy = module.run_quickstart(seed=42)  # type: ignore[attr-defined]
    assert accuracy >= 0.75, (
        f"quickstart binary accuracy {accuracy:.3f} < 0.75 (N1 / A14); "
        f"the README example and the CI threshold are wired to the same "
        f"examples/quickstart.py so this is a real regression, not drift"
    )
