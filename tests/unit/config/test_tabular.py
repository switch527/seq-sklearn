"""Tests for the TabularToSequenceConfig pydantic model at architecture A4."""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.tabular import TabularToSequenceConfig


def _minimal(**overrides: object) -> TabularToSequenceConfig:
    return TabularToSequenceConfig(id_col="id", time_col="time", **overrides)


def test_minimal_construction_uses_documented_defaults() -> None:
    cfg = _minimal()
    assert cfg.lookback == 12
    assert cfg.prediction_step == 1
    assert cfg.min_periods == 1
    assert cfg.scaling_real == "standard"


def test_frozen_post_construction() -> None:
    cfg = _minimal()
    with pytest.raises(ValidationError):
        cfg.lookback = 6  # type: ignore[misc]


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(not_a_field=True)


def test_lookback_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        _minimal(lookback=0)


def test_prediction_step_zero_allowed() -> None:
    cfg = _minimal(prediction_step=0)
    assert cfg.prediction_step == 0


def test_prediction_step_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(prediction_step=-1)


def test_config_is_hashable() -> None:
    """Frozen pydantic v2 models with hashable fields are hashable.

    Mapping-typed ``categorical_embed_dims`` must NOT make the model
    unhashable; A4 calls out this idiom explicitly.
    """
    cfg = _minimal()
    h = hash(cfg)
    assert isinstance(h, int)


def test_two_equal_configs_hash_equal() -> None:
    a = _minimal()
    b = _minimal()
    assert a == b
    assert hash(a) == hash(b)


def test_categorical_embed_dims_default_is_empty_mapping() -> None:
    cfg = _minimal()
    assert dict(cfg.categorical_embed_dims) == {}


def test_categorical_embed_dims_accepts_dict() -> None:
    cfg = _minimal(categorical_embed_dims={"industry": 8})
    assert dict(cfg.categorical_embed_dims) == {"industry": 8}


def test_categorical_embed_dims_accepts_tuple_of_tuples() -> None:
    cfg = _minimal(categorical_embed_dims=(("country", 4), ("industry", 8)))
    assert dict(cfg.categorical_embed_dims) == {"country": 4, "industry": 8}


def test_categorical_embed_dims_accepts_iterable_of_pairs() -> None:
    cfg = _minimal(categorical_embed_dims=[("country", 4), ("industry", 8)])
    assert dict(cfg.categorical_embed_dims) == {"country": 4, "industry": 8}


def test_categorical_embed_dims_sorted_canonically() -> None:
    """Stored form is sorted so hash is order-independent."""
    a = _minimal(categorical_embed_dims={"z": 8, "a": 4})
    b = _minimal(categorical_embed_dims={"a": 4, "z": 8})
    assert a.categorical_embed_dims == (("a", 4), ("z", 8))
    assert a == b
    assert hash(a) == hash(b)


def test_static_categorical_cols_stored_as_tuple() -> None:
    cfg = _minimal(static_categorical_cols=("industry", "country"))
    assert cfg.static_categorical_cols == ("industry", "country")


def test_scaling_static_real_inherit_default() -> None:
    cfg = _minimal()
    assert cfg.scaling_static_real == "inherit"


def test_unknown_scaling_real_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(scaling_real="unsupported")  # type: ignore[arg-type]
