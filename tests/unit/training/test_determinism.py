"""Side-effect assertions for ``enable_strict_mode`` (per requirements N1).

Two scenarios per the N1 spec: Scenario A starts with
``CUBLAS_WORKSPACE_CONFIG`` unset and asserts the four side effects fire
plus idempotency on a second call; Scenario B pre-sets the env var to a
non-default value and asserts the three torch flags fire while the env
var is left untouched. The autouse ``strict_mode_globals`` fixture
restores the process globals between tests.
"""

import os

import torch

from seq_sklearn.training._determinism import enable_strict_mode


def test_scenario_a_env_unset_sets_all_four_and_is_idempotent() -> None:
    """N1 Scenario A: env var unset; all four side effects fire; idempotent."""
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    enable_strict_mode()

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    enable_strict_mode()

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_scenario_b_env_preset_nondefault_is_left_untouched() -> None:
    """N1 Scenario B: pre-set non-default env var stays untouched."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    enable_strict_mode()

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
