"""Shared pytest fixtures, hypothesis profiles, and the N1 check sets.

Phase 1 landed the ``propagate_seq_sklearn_logger`` autouse fixture per
architecture A14 so F11-event-emission tests can capture records via
pytest's ``caplog``. Phase 9 adds:

* the F1.1 ``EXPECTED_FAILED_CHECKS`` / ``EXPECTED_PASSING_CHECKS`` sets
  (kept as two separate constants per the requirements-doc Gemini-pass
  fix: only the former is passed to
  ``parametrize_with_checks(..., expected_failed_checks=...)``; the
  latter drives a meta-test that the named checks are still collected),
* the ``inner_loop`` / ``nightly`` hypothesis profiles (N1: a profile is
  always loaded so a hypothesis test never runs profile-less),
* the ``--snapshot-update`` option and the ``snapshot`` fixture backing
  ``tests/snapshot/`` against pinned artifacts in ``tests/_snapshots/``.
"""

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

# --- F1.1 / N1 sklearn check_estimator contract ----------------------

# Checks that legitimately FAIL for a panel-DataFrame attention
# estimator. Passed to ``parametrize_with_checks(...,
# expected_failed_checks=...)`` with strict xfail so a silent
# compatibility win breaks CI and forces a doc update. requirements.md
# F1.1 marks these four mandatory; the set is the floor, not a ceiling
# (a sklearn release may legitimately add more, calibrated when the
# suite runs). Maps check name -> reason string (the sklearn 1.6
# ``expected_failed_checks`` mapping shape).
EXPECTED_FAILED_CHECKS: dict[str, str] = {
    "check_methods_sample_order_invariance": (
        "attention is intentionally order-sensitive; reordering input rows changes the prediction"
    ),
    "check_fit_idempotent": (
        "stochastic optimizer plus non-deterministic CUDA paths under "
        "mixed precision; idempotence holds only on the CPU FP32 "
        "deterministic path (N4)"
    ),
    "check_estimators_dtypes": (
        "DataFrame panel input contract; numpy-array equivalents are not supported"
    ),
    "check_dtype_object": (
        "DataFrame panel input contract; numpy-array equivalents are not supported"
    ),
}

# Checks intentionally expected to PASS, documented so a sklearn upgrade
# that renames or drops one is recognizable. NOT passed to
# ``parametrize_with_checks``; drives the meta-test in
# ``test_check_estimator.py``.
EXPECTED_PASSING_CHECKS: tuple[str, ...] = (
    "check_pandas_column_name_consistency",
    "check_n_features_in",
    "check_n_features_in_after_fitting",
    "check_methods_subset_invariance",
)

# --- hypothesis profiles (N1) ----------------------------------------

settings.register_profile(
    "inner_loop",
    settings(
        deadline=2000,
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    ),
)
settings.register_profile("nightly", settings(deadline=None, max_examples=500))

# A profile is ALWAYS loaded (default inner_loop) so no hypothesis test
# ever runs profile-less; CI sets HYPOTHESIS_PROFILE=nightly nightly.
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "inner_loop"))


# --- logger propagation invariant (Phase 1) --------------------------


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


# --- snapshot regression infra (N1 / A14) ----------------------------

_SNAPSHOT_DIR = Path(__file__).parent / "_snapshots"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--snapshot-update`` (A14 snapshot refresh procedure)."""
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Regenerate tests/_snapshots/ artifacts instead of asserting.",
    )


class _Snapshot:
    """Pinned-artifact snapshot helper.

    Each snapshot is a ``<name>.npy`` output tensor plus a
    ``<name>.manifest.json`` (dgp_version, seed, config hash) under
    ``tests/_snapshots/``. ``check`` asserts byte-identical equality on
    the deterministic CPU FP32 path, or regenerates under
    ``--snapshot-update``.
    """

    def __init__(self, *, update: bool) -> None:
        self._update = update

    def check(self, name: str, array: np.ndarray, manifest: dict[str, Any]) -> None:
        arr_path = _SNAPSHOT_DIR / f"{name}.npy"
        man_path = _SNAPSHOT_DIR / f"{name}.manifest.json"
        array = np.asarray(array)
        if self._update:
            _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            np.save(arr_path, array)
            man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            pytest.skip(f"snapshot {name!r} regenerated (--snapshot-update)")
        if not arr_path.exists() or not man_path.exists():
            pytest.fail(
                f"snapshot {name!r} missing; run "
                f"`pytest tests/snapshot/ --snapshot-update` to create it"
            )
        expected_manifest = json.loads(man_path.read_text())
        assert manifest == expected_manifest, (
            f"snapshot {name!r} manifest drift: the DGP version / seed / "
            f"config changed. Expected {expected_manifest}, got {manifest}. "
            f"Refresh with --snapshot-update only if the change is intended."
        )
        expected = np.load(arr_path)
        np.testing.assert_array_equal(
            array,
            expected,
            err_msg=f"snapshot {name!r} output drift on the CPU FP32 path",
        )


@pytest.fixture
def snapshot(request: pytest.FixtureRequest) -> _Snapshot:
    """Snapshot helper bound to the ``--snapshot-update`` flag."""
    return _Snapshot(update=bool(request.config.getoption("--snapshot-update")))
