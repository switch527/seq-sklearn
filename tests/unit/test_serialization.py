"""Tests for the safetensors + JSON save / load primitives (A17)."""

import json
import warnings
from pathlib import Path

import pytest
import torch

from seq_sklearn import serialization
from seq_sklearn.errors import PredictionError
from seq_sklearn.serialization import (
    CURRENT_SCHEMA_VERSION,
    _migrate,
    load_weights_and_state,
    save_weights_and_state,
)


def test_migrate_v1_to_v1_is_identity() -> None:
    weights = {"w": torch.zeros(2)}
    state = {"schema_version": 1, "extra": "kept"}
    out_w, out_s = _migrate(weights, state)
    assert out_w is weights
    assert out_s is state
    assert out_s["schema_version"] == 1


def test_migrate_schema_too_new_raises() -> None:
    with pytest.raises(PredictionError, match="newer than library"):
        _migrate({}, {"schema_version": CURRENT_SCHEMA_VERSION + 1})


def test_migrate_schema_too_old_raises() -> None:
    with pytest.raises(PredictionError, match="older than oldest supported"):
        _migrate({}, {"schema_version": 0})


def test_migrate_absent_schema_version_treated_as_too_old() -> None:
    with pytest.raises(PredictionError, match="older than oldest supported"):
        _migrate({}, {})


def test_migrate_non_int_schema_version_raises() -> None:
    with pytest.raises(PredictionError, match="must be an int"):
        _migrate({}, {"schema_version": "1"})
    with pytest.raises(PredictionError, match="must be an int"):
        _migrate({}, {"schema_version": True})


def test_save_load_preserves_bfloat16_dtype(tmp_path: Path) -> None:
    weights = {"w": torch.ones(3, dtype=torch.bfloat16)}
    save_weights_and_state(
        tmp_path / "m",
        weights,
        {"schema_version": 1, "seq_sklearn_version": serialization.__version__},
    )
    loaded_w, _ = load_weights_and_state(tmp_path / "m")
    assert loaded_w["w"].dtype == torch.bfloat16
    assert torch.equal(loaded_w["w"], weights["w"])


def _no_op_step(
    weights: serialization.WeightDict, state: serialization.StateDict
) -> tuple[serialization.WeightDict, serialization.StateDict]:
    return weights, state  # forgets to bump schema_version


def _advancing_step(
    weights: serialization.WeightDict, state: serialization.StateDict
) -> tuple[serialization.WeightDict, serialization.StateDict]:
    state = {**state, "schema_version": 2, "migrated": True}
    return weights, state


def test_migrate_detects_no_op_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    bad: dict[tuple[int, int], serialization.Migration] = {(1, 2): _no_op_step}
    monkeypatch.setattr(serialization, "MIGRATIONS", bad)
    monkeypatch.setattr(serialization, "CURRENT_SCHEMA_VERSION", 2)
    with pytest.raises(PredictionError, match="schema 1 to 2"):
        serialization._migrate({}, {"schema_version": 1})


def test_migrate_missing_step_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serialization, "MIGRATIONS", {})
    monkeypatch.setattr(serialization, "CURRENT_SCHEMA_VERSION", 2)
    with pytest.raises(PredictionError, match="no migration registered from schema 1 to 2"):
        serialization._migrate({}, {"schema_version": 1})


def test_migrate_applies_advancing_step(monkeypatch: pytest.MonkeyPatch) -> None:
    good: dict[tuple[int, int], serialization.Migration] = {(1, 2): _advancing_step}
    monkeypatch.setattr(serialization, "MIGRATIONS", good)
    monkeypatch.setattr(serialization, "CURRENT_SCHEMA_VERSION", 2)
    weights = {"w": torch.ones(2)}
    out_w, out_s = serialization._migrate(weights, {"schema_version": 1})
    assert out_s["schema_version"] == 2
    assert out_s["migrated"] is True
    assert torch.equal(out_w["w"], weights["w"])


def test_save_load_round_trips_byte_identical(tmp_path: Path) -> None:
    weights = {"layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    state = {
        "schema_version": 1,
        "seq_sklearn_version": serialization.__version__,
        "feature_names_in": ["a", "b"],
        "scaler_mean": [0.5, 1.5],
    }
    save_weights_and_state(tmp_path / "model", weights, state)

    loaded_w, loaded_s = load_weights_and_state(tmp_path / "model")
    assert torch.equal(loaded_w["layer.weight"], weights["layer.weight"])
    assert loaded_w["layer.weight"].dtype == torch.float32
    assert loaded_s == state
    # state.json is human-readable and reparses to the same object.
    raw = json.loads((tmp_path / "model" / "state.json").read_text())
    assert raw == state


def test_save_creates_missing_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "model"
    save_weights_and_state(target, {"w": torch.ones(1)}, {"schema_version": 1})
    assert (target / "weights.safetensors").is_file()
    assert (target / "state.json").is_file()


def test_load_emits_version_mismatch_warning(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    save_weights_and_state(
        model_dir,
        {"w": torch.zeros(1)},
        {"schema_version": 1, "seq_sklearn_version": "0.0.0-ancient"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_weights_and_state(model_dir)

    mismatch = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(mismatch) == 1
    message = str(mismatch[0].message)
    assert "version mismatch" in message
    assert "0.0.0-ancient" in message
    assert serialization.__version__ in message


def test_load_no_warning_when_versions_match(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    save_weights_and_state(
        model_dir,
        {"w": torch.zeros(1)},
        {"schema_version": 1, "seq_sklearn_version": serialization.__version__},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_weights_and_state(model_dir)
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
