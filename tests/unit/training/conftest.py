"""Training-unit fixtures.

Declares ``strict_mode_globals`` per architecture A14: an autouse
fixture that snapshots and restores the four process-global N4 flags
around EVERY training-unit test. ``enable_strict_mode`` mutates true
process globals; under ``pytest-randomly`` test order a determinism
test can run before a sibling callback or factory test, so the
snapshot/restore must wrap all tests in this directory (not just
``test_determinism.py``) or strict-mode state leaks and the suite goes
nondeterministically red. The snapshot is four attribute reads; the
isolation guarantee is worth that cost.
"""

import os
from collections.abc import Generator

import pytest
import torch


@pytest.fixture(autouse=True)
def strict_mode_globals() -> Generator[None]:
    """Snapshot / restore the four N4 process globals around each test.

    Captures ``torch.are_deterministic_algorithms_enabled()``,
    ``torch.backends.cudnn.deterministic``,
    ``torch.backends.cudnn.benchmark``, and
    ``os.environ.get("CUBLAS_WORKSPACE_CONFIG")`` at setup; restores all
    four at teardown so no training-unit test can leak strict-mode state
    into another under ``pytest-randomly`` ordering.
    """
    det_algorithms = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(det_algorithms, warn_only=False)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas
