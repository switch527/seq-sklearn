"""Shared fixtures for `tests/benchmarks/`.

Phase B0+B1 ships:
- ``isolated_registry`` (autouse): snapshots the three registry
  dicts (datasets `_REGISTRY`, datasets `_LOADERS`, models
  `_REGISTRY`) on entry and restores them on exit, so the
  scaffold tests can register new specs without leaking state
  into later tests. Phase B1 imports loader modules that register
  real dataset names at import time; without this fixture, a
  `pytest-randomly` reordering would surface intermittent failures
  once those modules are imported by the test collection.
- ``minimal_config_toml`` (per-test): writes a minimal TOML config to
  ``tmp_path`` and returns its path, used by the CLI subprocess test.

Later phases will add fixtures for tmp-path caches, fake adapters,
and per-experiment fakes.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from benchmarks.registry import datasets as _datasets_reg
from benchmarks.registry import models as _models_reg


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Snapshot+restore the dataset + model registries around every
    test.

    Three module-level dicts must move in lockstep: dataset specs,
    dataset loaders, and model specs. The Phase B1 `_LOADERS`
    companion is new; without restoring it, side-effect loader
    registrations from one test (e.g. registering a stub loader for
    a unique name) leave `_LOADERS` populated and a same-name
    registration in a later test trips the duplicate-loader guard
    spuriously. The autouse scope is per-function (the cheapest
    scope) since each scaffold test mutates the registry at most
    once.
    """
    ds_reg = _datasets_reg._REGISTRY  # pyright: ignore[reportPrivateUsage]
    ds_loaders = _datasets_reg._LOADERS  # pyright: ignore[reportPrivateUsage]
    m_reg = _models_reg._REGISTRY  # pyright: ignore[reportPrivateUsage]
    ds_snapshot = dict(ds_reg)
    ds_loaders_snapshot = dict(ds_loaders)
    m_snapshot = dict(m_reg)
    try:
        yield
    finally:
        ds_reg.clear()
        ds_reg.update(ds_snapshot)
        ds_loaders.clear()
        ds_loaders.update(ds_loaders_snapshot)
        m_reg.clear()
        m_reg.update(m_snapshot)


@pytest.fixture
def minimal_config_toml(tmp_path: Path) -> Path:
    """Write a minimal `BenchmarkConfig` TOML to `tmp_path` and
    return its path. The config references no real datasets or
    models (the names are dummy strings); the harness validates the
    schema only, the registry cross-check happens at run time and
    is intentionally absent from the dry-run path."""
    path = tmp_path / "benchmark.toml"
    path.write_text(
        'datasets = ["dummy_dataset"]\n'
        'models = ["dummy_model"]\n'
        f'output_dir = "{tmp_path / "out"}"\n'
        "\n"
        "[[experiments]]\n"
        'kind = "raw_loss"\n'
        "seeds = [0, 1, 2]\n",
        encoding="utf-8",
    )
    return path
