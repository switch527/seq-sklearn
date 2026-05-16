"""Integration-suite fixtures (Phase 6a owns this conftest).

``integration_strict_mode_globals`` snapshots the four N4 determinism
globals at setup and restores them at teardown, but only for tests
carrying ``@pytest.mark.determinism``. ``pytest-randomly`` can order a
``tests/unit/`` test that disables determinism (e.g. a hypothesis test)
before the integration determinism test; without the restore the
bit-identical assertion would pass or fail by accident. ``tmp_cwd``
keeps Lightning's default ``./checkpoints`` out of the repo by running
every integration test from a fresh temp dir.
"""

import os
from collections.abc import Generator

import pytest
import torch


@pytest.fixture(autouse=True)
def tmp_cwd(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each integration test from a throwaway working directory."""
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def integration_strict_mode_globals(
    request: pytest.FixtureRequest,
) -> Generator[None]:
    """Snapshot / restore the four N4 globals around determinism tests.

    No-op unless the test is marked ``determinism`` (only those tests
    assert bit-identical output and therefore need contamination-proof
    global state); the snapshot is four attribute reads.
    """
    if request.node.get_closest_marker("determinism") is None:
        yield
        return
    det = torch.are_deterministic_algorithms_enabled()
    cudnn_det = torch.backends.cudnn.deterministic
    cudnn_bench = torch.backends.cudnn.benchmark
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(det, warn_only=False)
        torch.backends.cudnn.deterministic = cudnn_det
        torch.backends.cudnn.benchmark = cudnn_bench
        if cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas
