"""Training-unit fixtures.

Declares ``strict_mode_globals`` per architecture A14: an autouse
fixture that snapshots and restores the four process-global N4 flags,
scoped to ``test_determinism.py`` only (a marker check inside the
fixture body) so the snapshot / restore overhead does not apply to the
other training-unit tests in the same directory. ``pytest-randomly``
permutes test order; without restoration, Scenario B's preconditions
become non-deterministic.
"""

import os
from collections.abc import Generator

import pytest
import torch


@pytest.fixture(autouse=True)
def strict_mode_globals(request: pytest.FixtureRequest) -> Generator[None]:
    """Snapshot / restore the four N4 process globals around determinism tests.

    Active only for ``test_determinism.py`` (scoped via an fspath check
    so the other training-unit tests skip the snapshot work). Captures
    ``torch.are_deterministic_algorithms_enabled()``,
    ``torch.backends.cudnn.deterministic``,
    ``torch.backends.cudnn.benchmark``, and
    ``os.environ.get("CUBLAS_WORKSPACE_CONFIG")`` at setup; restores all
    four at teardown.
    """
    if "test_determinism" not in request.node.fspath.basename:
        yield
        return
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
