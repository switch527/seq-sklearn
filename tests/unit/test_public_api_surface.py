"""Owning test for the architecture-A3 public-API façade (Phase 12 R13/PE.4).

For 11 phases nothing imported via `seq_sklearn` itself: every test used
the deep module path, `check_estimator` validated classes handed to it
directly, and coverage measured lines-executed, not surface-correctness.
This is the test that owns the package façade going forward, so the
same gap class cannot reopen silently.
"""

import importlib
import re
import subprocess
import sys
import types
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import pytest

import seq_sklearn

_ARCH_DOC = Path(__file__).resolve().parents[2] / "docs" / "architecture.md"

# Architecture A3 names that are INTERNAL-tier and must never leak into
# the public surface (per the `## A3:` section's INTERNAL-tier prose
# in docs/architecture.md, which lists `RecurrentSequenceEstimator{,Config}`
# as the named INTERNAL-tier symbols in v1).
_INTERNAL_TIER_FORBIDDEN: frozenset[str] = frozenset(
    {"RecurrentSequenceEstimator", "RecurrentSequenceEstimatorConfig"}
)


def _spec_public_api() -> set[str]:
    """Extract the literal A3 `__all__` from docs/architecture.md.

    Anchored to the `## A3:` heading so a future doc edit that adds an
    illustrative `__all__` block earlier in the file (e.g. a migration
    template, a stability-tier discussion) cannot silently re-anchor
    the test against the wrong block.
    """
    text = _ARCH_DOC.read_text()
    a3_heading = re.search(r"^##\s+A3:", text, re.MULTILINE)
    if a3_heading is None:
        pytest.fail(f"could not locate `## A3:` heading in {_ARCH_DOC}")
    next_heading = re.search(r"^##\s+A\d", text[a3_heading.end() :], re.MULTILINE)
    # The `next_heading is None` arm runs when A3 is the LAST `## A`
    # section in the file (today there's always an A4 trailing A3, so
    # this branch is reachable but unexercised in CI; the slice
    # extension to `len(text)` keeps the regex from silently returning
    # an empty result if the doc structure changes).
    end = a3_heading.end() + next_heading.start() if next_heading else len(text)
    a3_slice = text[a3_heading.start() : end]
    match = re.search(r"__all__\s*=\s*\[(.*?)\]", a3_slice, re.DOTALL)
    if match is None:
        pytest.fail(
            f"could not locate the `__all__ = [...]` block under `## A3:` "
            f"in {_ARCH_DOC}; the A3 public-API source has moved or been removed"
        )
    body = match.group(1)
    names = re.findall(r'"([^"]+)"', body)
    if not names:
        pytest.fail(f"parsed empty name set from A3 `__all__` in {_ARCH_DOC}")
    return set(names)


def test_public_api_all_matches_spec() -> None:
    """(a) `seq_sklearn.__all__` is exactly the A3 set."""
    assert set(seq_sklearn.__all__) == _spec_public_api()


def test_public_api_names_resolve_to_deep_path_objects() -> None:
    """(b) every name in `__all__` is the SAME object as the deep import.

    Identity-compare, so a re-bound stub (different `id`) fails.
    """
    deep_paths = {
        "TFTClassifier": ("seq_sklearn.models.transformer.tft.classifier", "TFTClassifier"),
        "TFTRegressor": ("seq_sklearn.models.transformer.tft.regressor", "TFTRegressor"),
        "TabularToSequence": ("seq_sklearn.data.tabular_to_sequence", "TabularToSequence"),
        "TabularToSequenceConfig": ("seq_sklearn.config.tabular", "TabularToSequenceConfig"),
        "TFTConfig": ("seq_sklearn.config.tft", "TFTConfig"),
        "TabularConfigParams": ("seq_sklearn.config.adapters", "TabularConfigParams"),
        "OptimizerParams": ("seq_sklearn.config.adapters", "OptimizerParams"),
        "SchedulerParams": ("seq_sklearn.config.adapters", "SchedulerParams"),
        "LossParams": ("seq_sklearn.config.adapters", "LossParams"),
        "SamplerParams": ("seq_sklearn.config.adapters", "SamplerParams"),
        "TFTAdvancedParams": ("seq_sklearn.config.adapters", "TFTAdvancedParams"),
        "EntityTimeSeriesSplit": ("seq_sklearn.model_selection.split", "EntityTimeSeriesSplit"),
        "HardwareTier": ("seq_sklearn.hardware", "HardwareTier"),
        "detect": ("seq_sklearn.hardware", "detect"),
        "SeqSklearnError": ("seq_sklearn.errors", "SeqSklearnError"),
        "ConfigError": ("seq_sklearn.errors", "ConfigError"),
        "DataContractError": ("seq_sklearn.errors", "DataContractError"),
        "TrainingError": ("seq_sklearn.errors", "TrainingError"),
        "PredictionError": ("seq_sklearn.errors", "PredictionError"),
        "NotFittedError": ("seq_sklearn.errors", "NotFittedError"),
        "AttentionOutput": ("seq_sklearn.inference.attention", "AttentionOutput"),
        "RegressionAttentionOutput": (
            "seq_sklearn.inference.attention",
            "RegressionAttentionOutput",
        ),
        "suggest_params": ("seq_sklearn.tuning.suggest_params", "suggest_params"),
        "optuna_trial_guard": ("seq_sklearn.tuning.pruning", "optuna_trial_guard"),
    }
    assert set(deep_paths) == set(seq_sklearn.__all__)
    for name, (mod_path, attr) in deep_paths.items():
        mod = importlib.import_module(mod_path)
        assert hasattr(seq_sklearn, name), f"seq_sklearn missing public name {name!r}"
        assert getattr(seq_sklearn, name) is getattr(mod, attr), (
            f"seq_sklearn.{name} is not the same object as {mod_path}.{attr} "
            f"(re-bound stub or shadowed)"
        )


def test_public_api_version_single_source() -> None:
    """(c) `__version__` reads from installed-distribution metadata.

    Rejects a hardcoded literal that can drift from `pyproject.toml`.
    """
    try:
        dist_version = _pkg_version("seq-sklearn")
    except PackageNotFoundError:
        pytest.fail(
            "seq-sklearn not installed; run `uv sync` or `pip install -e .` "
            "(the façade test requires the dist metadata to verify version single-source)"
        )
    assert seq_sklearn.__version__ == dist_version


def test_public_api_no_internal_tier_leak() -> None:
    """(d) INTERNAL-tier names never appear in the public surface."""
    sentinel = object()
    for name in _INTERNAL_TIER_FORBIDDEN:
        assert name not in seq_sklearn.__all__, (
            f"INTERNAL-tier name {name!r} leaked into seq_sklearn.__all__"
        )
        assert getattr(seq_sklearn, name, sentinel) is sentinel, (
            f"INTERNAL-tier name {name!r} reachable on seq_sklearn (namespace leak)"
        )


def test_public_api_no_namespace_leakage() -> None:
    """(d') Every non-underscore, non-dunder attribute set on the module
    must be in `__all__`. Uses `vars`, not `dir`, to scope strictly to
    module-level names (not inherited module-type attrs)."""
    declared = set(seq_sklearn.__all__)
    leaked = sorted(
        name
        for name in vars(seq_sklearn)
        if not name.startswith("_") and name not in declared
        # Re-exported submodules are reachable via attribute access but
        # not part of the documented surface; the contract is that the
        # documented public API is in `__all__`, and any other reachable
        # name is a leak.
    )
    # Filter to non-module objects (submodules can show up after a child
    # import and are NOT part of the public surface per the A3 contract;
    # they remain importable as fully-qualified paths).
    leaked = [n for n in leaked if not isinstance(getattr(seq_sklearn, n), types.ModuleType)]
    assert not leaked, (
        f"namespace leakage on `seq_sklearn`: non-underscore attrs not in __all__: {leaked}"
    )


def test_public_api_subprocess_imports() -> None:
    """(e) `from seq_sklearn import <name>` works in a fresh subprocess.

    Proves the façade is genuinely installed-and-importable, not just
    monkey-imported in the test session. One import per name, so a
    failure identifies which name is broken.
    """
    script = (
        "\n".join(f"from seq_sklearn import {name}" for name in seq_sklearn.__all__)
        + "\nprint('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"subprocess façade import failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout
