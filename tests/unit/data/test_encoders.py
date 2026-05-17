"""Tests for the categorical encoder and real-feature scalers (F3)."""

import json
import logging

import numpy as np
import pytest

from seq_sklearn.data.encoders import (
    UNK_INDEX,
    UNK_TOKEN,
    CategoricalEncoder,
    make_scaler,
)
from seq_sklearn.errors import NotFittedError
from seq_sklearn.logging import Event


def test_fit_pins_unk_at_index_zero() -> None:
    enc = CategoricalEncoder()
    enc.fit({"col": np.array(["b", "a", "a", "c"])})
    assert enc.vocabularies_["col"][UNK_INDEX] == UNK_TOKEN
    assert enc.vocabularies_["col"] == [UNK_TOKEN, "a", "b", "c"]


def test_known_levels_round_trip() -> None:
    enc = CategoricalEncoder()
    enc.fit({"col": np.array(["x", "y", "z"])})
    encoded = enc.transform({"col": np.array(["z", "x", "y"])})
    decoded = enc.inverse_transform(encoded)
    assert decoded["col"].tolist() == ["z", "x", "y"]


def test_unseen_maps_to_unk_without_raising(caplog: pytest.LogCaptureFixture) -> None:
    enc = CategoricalEncoder()
    enc.fit({"col": np.array(["a", "b"])})
    with caplog.at_level(logging.WARNING, logger="seq_sklearn"):
        encoded = enc.transform({"col": np.array(["a", "ZZZ", "b", "QQQ"])})
    assert encoded["col"].tolist() == [1, UNK_INDEX, 2, UNK_INDEX]

    events = [
        r for r in caplog.records if getattr(r, "event", None) == Event.DATA_UNSEEN_CATEGORIES
    ]
    assert len(events) == 1
    assert events[0].payload["count"] == 2


def test_no_log_when_all_seen(caplog: pytest.LogCaptureFixture) -> None:
    enc = CategoricalEncoder()
    enc.fit({"col": np.array(["a", "b"])})
    with caplog.at_level(logging.WARNING, logger="seq_sklearn"):
        enc.transform({"col": np.array(["a", "b"])})
    events = [
        r for r in caplog.records if getattr(r, "event", None) == Event.DATA_UNSEEN_CATEGORIES
    ]
    assert events == []


def test_unk_index_inverse_decodes_to_literal() -> None:
    enc = CategoricalEncoder()
    enc.fit({"col": np.array(["a"])})
    decoded = enc.inverse_transform({"col": np.array([UNK_INDEX])})
    assert decoded["col"].tolist() == [UNK_TOKEN]


def test_encoder_raises_before_fit() -> None:
    enc = CategoricalEncoder()
    with pytest.raises(NotFittedError):
        enc.transform({"col": np.array(["a"])})
    with pytest.raises(NotFittedError):
        enc.inverse_transform({"col": np.array([0])})


def test_encoder_sklearn_is_fitted() -> None:
    enc = CategoricalEncoder()
    assert enc.__sklearn_is_fitted__() is False
    enc.fit({"col": np.array(["a"])})
    assert enc.__sklearn_is_fitted__() is True


@pytest.mark.parametrize("strategy", ["standard", "robust", "quantile_uniform", "none"])
def test_scaler_round_trips(strategy: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(200, 4))
    scaler = make_scaler(strategy)  # type: ignore[arg-type]
    scaler.fit(x)
    transformed = scaler.transform(x)
    recovered = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(recovered, x, atol=1e-6)


@pytest.mark.parametrize("strategy", ["standard", "robust", "quantile_uniform", "none"])
def test_scaler_raises_before_fit(strategy: str) -> None:
    scaler = make_scaler(strategy)  # type: ignore[arg-type]
    x = np.ones((3, 2))
    with pytest.raises(NotFittedError):
        scaler.transform(x)
    with pytest.raises(NotFittedError):
        scaler.inverse_transform(x)


@pytest.mark.parametrize("strategy", ["standard", "robust", "quantile_uniform", "none"])
def test_scaler_sklearn_is_fitted(strategy: str) -> None:
    scaler = make_scaler(strategy)  # type: ignore[arg-type]
    assert scaler.__sklearn_is_fitted__() is False  # type: ignore[attr-defined]
    scaler.fit(np.ones((4, 2)))
    assert scaler.__sklearn_is_fitted__() is True  # type: ignore[attr-defined]


def test_identity_scaler_is_noop() -> None:
    scaler = make_scaler("none")
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler.fit(x)
    np.testing.assert_array_equal(scaler.transform(x), x)


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown scaler strategy"):
        make_scaler("nope")  # type: ignore[arg-type]


# --- A17 fitted-state serialization (Phase 6a save/load hook) ---------


@pytest.mark.parametrize("strategy", ["standard", "robust", "quantile_uniform", "none"])
def test_scaler_fitted_state_json_roundtrip(strategy: str) -> None:
    # Fitted state must survive json.dumps/loads and reconstruct a
    # scaler whose transform is byte-equal (N1 FP32 CPU exact equality),
    # with no re-fit and no pickled sklearn object.
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 5.0, size=(150, 3))
    src = make_scaler(strategy)  # type: ignore[arg-type]
    src.fit(x)
    blob = json.loads(json.dumps(src.get_fitted_state()))
    restored = make_scaler(strategy).set_fitted_state(blob)  # type: ignore[arg-type]
    assert np.array_equal(restored.transform(x), src.transform(x))


@pytest.mark.parametrize("strategy", ["standard", "robust", "quantile_uniform", "none"])
def test_scaler_get_fitted_state_before_fit_raises(strategy: str) -> None:
    with pytest.raises(NotFittedError):
        make_scaler(strategy).get_fitted_state()  # type: ignore[arg-type]


def test_categorical_encoder_fitted_state_json_roundtrip() -> None:
    enc = CategoricalEncoder()
    enc.fit({"c1": np.array(["a", "b", "a"]), "c2": np.array([1, 2, 3])})
    cols = {"c1": np.array(["a", "z", "b"]), "c2": np.array([2, 9, 1])}
    blob = json.loads(json.dumps(enc.get_fitted_state()))
    restored = CategoricalEncoder().set_fitted_state(blob)
    for name in ("c1", "c2"):
        np.testing.assert_array_equal(restored.transform(cols)[name], enc.transform(cols)[name])
    assert restored.__sklearn_is_fitted__() is True


def test_categorical_encoder_get_fitted_state_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        CategoricalEncoder().get_fitted_state()


def test_categorical_encoder_set_fitted_state_bad_payload_raises() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        CategoricalEncoder().set_fitted_state({"vocabularies": ["not", "a", "dict"]})
