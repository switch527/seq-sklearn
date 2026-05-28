"""Phase B12 TSC adapter tests (aeon-missing path).

These tests run regardless of whether `aeon` is installed; they
patch `sys.modules` so the lazy-import path inside
`_check_aeon_available` sees `aeon` as missing. Mandates
`monkeypatch.setitem(sys.modules, "aeon", None)` rather than a
bare assignment so pytest auto-restores under `pytest-randomly`
reordering.

Aeon-installed smoke tests live in `test_tsc_adapter_smoke.py`
behind `pytest.importorskip("aeon")`.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import benchmarks.adapters  # type: ignore[reportUnusedImport]  # noqa: F401  --  side-effect import: registers seq + gbm + tsc
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters._base import (
    OptionalDependencyMissingError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.adapters.tsc import (
    _check_aeon_available,
    _TSCAdapter,
)
from benchmarks.config import BenchmarkConfig, DatasetSpec, ExperimentSpec, TaskType
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.registry.models import register_adapter

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

# --- _check_aeon_available --------------------------------------------------


def test_check_aeon_available_raises_typed_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy-import path raises a typed
    `OptionalDependencyMissingError` carrying the package name.
    Uses `monkeypatch.setitem` so pytest auto-restores
    `sys.modules` on teardown (B12.5 mandate)."""
    monkeypatch.setitem(sys.modules, "aeon", None)
    with pytest.raises(OptionalDependencyMissingError) as exc_info:
        _check_aeon_available()
    assert exc_info.value.package_name == "aeon"


def test_optional_dependency_missing_error_does_not_subclass_importerror() -> None:
    """B12 design Gemini-I2 / R-B12 arch-I2: the MRO must subclass
    `Exception` directly, NOT `ImportError`. Existing
    `except ImportError` clauses at `raw_loss.py:201` (torch probe)
    and `run_manifest.py:185, 196` would otherwise silently swallow
    the typed skip error if a future code move puts an adapter
    import on their path."""
    err = OptionalDependencyMissingError("aeon")
    assert isinstance(err, Exception)
    assert not isinstance(err, ImportError)


# --- _TSCAdapter base-instantiation block ----------------------------------


def test_tsc_adapter_post_init_blocks_direct_base_instantiation() -> None:
    """`_TSCAdapter` is abstract; instantiating it directly raises
    `TypeError` so the runtime `SeqSklearnAdapter` Protocol gate
    cannot accept the half-configured base. Matches the GBM
    pattern at `tests/benchmarks/test_gbm_adapter.py:294-303`."""
    spec = _make_binary_spec()
    with pytest.raises(TypeError, match="abstract base"):
        _TSCAdapter(spec=spec, hyperparameters={})


# --- predict_quantiles ------------------------------------------------------


def test_tsc_adapter_predict_quantiles_raises_quantiles_unsupported() -> None:
    """TSC adapters are classification-only at v1; the
    `predict_quantiles` hook always raises (D-B12.3)."""
    from benchmarks.adapters.tsc import _RocketClassifierAdapter

    spec = _make_binary_spec()
    adapter = _RocketClassifierAdapter(spec=spec, hyperparameters={})
    panel = _make_binary_panel()
    with pytest.raises(QuantilesUnsupportedError, match="classification-only"):
        adapter.predict_quantiles(panel)


# --- B5 driver integration: aeon-missing produces typed skip ----------------


def test_run_raw_loss_records_optional_dep_skip_when_aeon_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: run `run_raw_loss` with a TSC fake adapter +
    `sys.modules["aeon"] = None`; assert
    `cells_skipped_optional_dep_missing == n_tsc_cells_in_run`,
    `cells_skipped_adapter_error == 0`, the run returns cleanly,
    and every TSC ResultRow carries the typed
    `skipped_reason="optional_dep_missing: aeon"`. Pins the B5
    wire-up touch points (b)-(f)."""
    monkeypatch.setitem(sys.modules, "aeon", None)
    register_all_fakes_and_get_panels()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("rocket_classifier",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    result = run_raw_loss(config, output_root=output_root, env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_optional_dep_missing > 0
    assert result.cells_skipped_adapter_error == 0


# --- fake binary panel helpers ---------------------------------------------


def _make_binary_spec() -> DatasetSpec:
    return DatasetSpec(
        name="ds_tsc_test",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/x.csv",
        integrity_sha256="0" * 64,
        archive_basename="x.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x1",),
        feature_categorical_cols=(),
        lookback=3,
        positive_label=1,
        citation="tsc adapter test",
    )


def _make_binary_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for entity in range(3):
        for t in range(5):
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "x1": float(rng.normal()),
                    "y": int(entity % 2),
                }
            )
    return pd.DataFrame(rows)


# --- fake TSC adapter for the aeon-missing e2e test ------------------------


@dataclass
class _FakeTSCAdapter:
    """A `family="tsc"` test adapter whose `fit` raises
    `OptionalDependencyMissingError("aeon")` so the B5 driver
    routes it to the typed `optional_dep_missing` skip path
    without requiring aeon to be installed in CI."""

    name: ClassVar[str] = "fake_tsc_adapter"
    family: ClassVar[str] = "tsc"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        raise OptionalDependencyMissingError("aeon")

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(panel))

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full((len(panel), 2), 0.5)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


def _register_fake_tsc_adapter() -> None:
    from benchmarks.config import ModelSpec

    spec = ModelSpec(
        name="fake_tsc_adapter",
        family="tsc",
        task_types=("binary",),
        supports_proba=True,
        reason="aeon-missing skip-wire-up test fixture",
    )

    def _factory(
        *,
        spec: DatasetSpec,
        hyperparameters: dict[str, Any] | None = None,
    ) -> SeqSklearnAdapter:
        return cast(
            SeqSklearnAdapter,
            _FakeTSCAdapter(spec=spec, hyperparameters=hyperparameters or {}),
        )

    register_adapter(spec, _factory)


def test_run_raw_loss_with_fake_tsc_adapter_routes_to_optional_dep_skip(
    tmp_path: Path,
) -> None:
    """Same intent as the aeon-monkeypatch test but routes through
    a fake adapter that raises the typed error directly. Pins the
    skip-reason format `"optional_dep_missing: aeon"` on every
    emitted row."""
    register_all_fakes_and_get_panels()
    _register_fake_tsc_adapter()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_tsc_adapter",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    result = run_raw_loss(config, output_root=output_root, env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_adapter_error == 0
    assert result.cells_skipped_optional_dep_missing > 0
    # Validate one of the written rows carries the typed format.
    from benchmarks.manifest import load_run

    manifest = load_run(output_root)
    tsc_rows = manifest.loc[manifest["model_name"] == "fake_tsc_adapter"]
    assert len(tsc_rows) > 0
    for _, row in tsc_rows.iterrows():
        assert str(row["skipped_reason"]).startswith("optional_dep_missing: aeon")


# --- aeon-version capture in the manifest -----------------------------------


def test_aeon_appears_in_pinned_packages_list() -> None:
    """B12 touch point (k): `_PINNED_PACKAGES` includes `"aeon"`
    so the manifest fingerprint records the version on every run
    (None when aeon is absent, the version string when installed)."""
    from benchmarks.run_manifest import _PINNED_PACKAGES

    assert "aeon" in _PINNED_PACKAGES


# --- R1 unit-level pins on adapter branches --------------------------------


def test_tsc_adapter_fit_rejects_unsupported_task_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1 qa-C3: `_TSCAdapter.fit` rejects datasets whose
    `task_type` is not in the adapter's `task_types` tuple. The
    check happens BEFORE `_check_aeon_available` so an aeon-
    missing CI still surfaces the typed `ValueError` rather than
    `OptionalDependencyMissingError`."""
    from benchmarks.adapters.tsc import _RocketClassifierAdapter

    monkeypatch.setitem(sys.modules, "aeon", None)
    # ROCKET only supports binary + multiclass; build a regression spec.
    spec = DatasetSpec(
        name="ds_tsc_regr",
        task_type="regression_point",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/x.csv",
        integrity_sha256="0" * 64,
        archive_basename="x.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x1",),
        feature_categorical_cols=(),
        lookback=3,
        citation="task-type mismatch test",
    )
    adapter = _RocketClassifierAdapter(spec=spec, hyperparameters={})
    panel = _make_binary_panel()
    with pytest.raises(ValueError, match="does not support task_type"):
        adapter.fit(panel, np.zeros(len(panel)))


def test_tsc_adapter_predict_raises_not_fitted_before_fit() -> None:
    """R1 qa-I2: `_TSCAdapter.predict` raises `NotFittedError`
    when called before `fit`. The B5 driver's narrow catch tuple
    includes `NotFittedError` so an unfitted-adapter bug surfaces
    as a typed skip rather than an `AttributeError`."""
    from benchmarks.adapters.tsc import _RocketClassifierAdapter

    from seq_sklearn import NotFittedError

    spec = _make_binary_spec()
    adapter = _RocketClassifierAdapter(spec=spec, hyperparameters={})
    panel = _make_binary_panel()
    with pytest.raises(NotFittedError, match="before fit"):
        adapter.predict(panel)


def test_tsc_adapter_predict_proba_raises_not_fitted_before_fit() -> None:
    """R1 qa-I2: same intent as the `predict` test, for the
    proba surface."""
    from benchmarks.adapters.tsc import _RocketClassifierAdapter

    from seq_sklearn import NotFittedError

    spec = _make_binary_spec()
    adapter = _RocketClassifierAdapter(spec=spec, hyperparameters={})
    panel = _make_binary_panel()
    with pytest.raises(NotFittedError, match="before fit"):
        adapter.predict_proba(panel)


def test_tsc_adapter_predict_proba_raises_when_supports_proba_false() -> None:
    """R1 qa-I3: the `supports_proba=False` guard in
    `predict_proba`. All three concrete adapters set
    `supports_proba=True` so the branch is otherwise unreachable
    through the registered roster. Pin via a synthetic subclass
    with `supports_proba=False`."""
    from dataclasses import dataclass

    from benchmarks.adapters._base import ProbaUnsupportedError
    from benchmarks.adapters.tsc import _RocketClassifierAdapter

    @dataclass
    class _NoProbaAdapter(_RocketClassifierAdapter):
        supports_proba: ClassVar[bool] = False

    spec = _make_binary_spec()
    adapter = _NoProbaAdapter(spec=spec, hyperparameters={})
    # Manually mark "fitted" so the supports_proba check runs
    # before the NotFittedError check.
    adapter._est = object()  # type: ignore[assignment]
    panel = _make_binary_panel()
    with pytest.raises(ProbaUnsupportedError, match="check `supports_proba`"):
        adapter.predict_proba(panel)


def test_tsc_adapter_reshape_caches_by_panel_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1 qa-I4: `_reshape` caches the (tensor, mapping) pair by
    `id(panel)`. A second call on the same panel object reuses
    the cached arrays rather than re-reshaping. Pin via a
    monkey-patched `panel_to_tensor` that counts calls."""
    from benchmarks.adapters.tsc import _RocketClassifierAdapter
    from benchmarks.protocol import raw_mts as _raw_mts_module

    spec = _make_binary_spec()
    adapter = _RocketClassifierAdapter(spec=spec, hyperparameters={})
    panel = _make_binary_panel()

    call_count = 0
    real_panel_to_tensor = _raw_mts_module.panel_to_tensor

    def _counting_panel_to_tensor(p: pd.DataFrame, s: DatasetSpec) -> Any:
        nonlocal call_count
        call_count += 1
        return real_panel_to_tensor(p, s)

    monkeypatch.setattr("benchmarks.adapters.tsc.panel_to_tensor", _counting_panel_to_tensor)

    # First call populates the cache.
    adapter._reshape(panel)
    assert call_count == 1
    # Second call on the SAME panel object reuses the cache.
    adapter._reshape(panel)
    assert call_count == 1, "cache hit expected; panel_to_tensor was called twice"


# --- broadcast preserves identity, not NaN, for above-floor early rows -----


def test_broadcast_above_floor_early_rows_carry_instance_value_not_nan() -> None:
    """R1 code-I2: an entity with more than L_resolved rows
    has its trailing window IN the tensor; the early (pre-window)
    rows are NOT below-floor and the broadcast MUST emit the
    instance value (not NaN) for them. A bug that NaN-fills
    early-window rows would silently strip them via
    `_strip_below_floor_rows` and corrupt the metric."""
    import benchmarks.protocol.raw_mts as _raw_mts_module

    spec = DatasetSpec(
        name="ds_above_floor",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/x.csv",
        integrity_sha256="0" * 64,
        archive_basename="x.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x1",),
        feature_categorical_cols=(),
        lookback=3,
        positive_label=1,
        citation="above-floor broadcast pin",
    )
    # One entity with 5 rows; L=3. The trailing window is rows
    # [2, 3, 4]; rows [0, 1] are EARLY rows of a KEPT entity.
    panel = pd.DataFrame(
        [{"entity_id": "e_0", "period": t, "x1": float(t), "y": 1} for t in range(5)]
    )
    _, mapping = _raw_mts_module.panel_to_tensor(panel, spec)
    # Every panel row (early + kept) maps to instance 0; none -1.
    assert (mapping >= 0).all()
    # Simulate a per-instance proba [[0.3, 0.7]] and broadcast.
    per_instance = np.array([[0.3, 0.7]], dtype=np.float64)
    broadcast_out = _raw_mts_module.broadcast_per_instance_to_per_row(per_instance, mapping)
    # Every row, including early rows 0 and 1, carries (0.3, 0.7).
    for row_idx in range(len(panel)):
        np.testing.assert_array_equal(broadcast_out[row_idx], np.array([0.3, 0.7]))
    # No NaN anywhere.
    assert not np.isnan(broadcast_out).any()
