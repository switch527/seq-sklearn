"""Phase B14 tests for `benchmarks/report/_bootstrap_aggregate.py`.

Covers the constants module's defensive paths that the broader
aggregator tests don't reach.
"""

from importlib.metadata import PackageNotFoundError

import pytest
from benchmarks.report import _bootstrap_aggregate
from benchmarks.report._bootstrap_aggregate import numpy_version


def test_numpy_version_returns_unknown_when_package_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-1 qa-I2: the `PackageNotFoundError` fallback returns
    `"unknown"` rather than raising. The branch is defensive
    (it should never fire in a properly installed env) but the
    contract is that `numpy_version()` never raises."""

    def _raise(_name: str) -> str:
        raise PackageNotFoundError("simulated missing distribution")

    monkeypatch.setattr(_bootstrap_aggregate, "_pkg_version", _raise)
    assert numpy_version() == "unknown"


def test_numpy_version_returns_pep_440_string_in_normal_env() -> None:
    """Happy path: `numpy_version()` returns the installed numpy
    version (a non-empty string). Cross-pins with the aggregator's
    end-to-end `bootstrap_numpy_version` field assertion."""
    version = numpy_version()
    assert version
    assert version != "unknown"
