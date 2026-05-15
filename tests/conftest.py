"""Shared pytest fixtures and hypothesis profile registration.

Phase 1 lands the ``propagate_seq_sklearn_logger`` autouse fixture per
architecture A14 so F11-event-emission tests can capture records via
pytest's ``caplog``. Additional fixtures land alongside the modules they
test across Phases 2-9 per docs/implementation_plan.md.
"""

import logging
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def propagate_seq_sklearn_logger() -> Generator[None]:
    """Ensure the ``seq_sklearn`` logger propagates so ``caplog`` captures.

    Library best practice is for application code (not the library) to
    attach handlers; pytest's ``caplog`` plugs into the root logger and
    sees emitted records only if ``propagate=True`` on every ancestor
    logger. The library never disables propagation; this fixture asserts
    that invariant so a future change that flips it surfaces as a test
    failure.

    Uses ``yield`` so any test that flips ``propagate`` mid-run gets the
    original value restored at teardown, preventing cross-test leakage
    under ``pytest-randomly`` ordering.
    """
    seq_logger = logging.getLogger("seq_sklearn")
    original = seq_logger.propagate
    assert original is True, "seq_sklearn logger.propagate must stay True so caplog captures"
    try:
        yield
    finally:
        seq_logger.propagate = original
