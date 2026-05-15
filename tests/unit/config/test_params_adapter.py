"""Tests for the BaseEstimator adapter at architecture A4 step 3."""

import pytest
import sklearn
import sklearn.base
from pydantic import ValidationError

from seq_sklearn.config._params_adapter import TabularConfigParams
from seq_sklearn.config.tabular import TabularToSequenceConfig


def test_default_construction_mirrors_pydantic_defaults() -> None:
    adapter = TabularConfigParams()
    assert adapter.id_col == "id"
    assert adapter.time_col == "time"
    assert adapter.lookback == 12
    assert adapter.scaling_real == "standard"


def test_get_params_returns_flat_dict() -> None:
    adapter = TabularConfigParams(lookback=6)
    params = adapter.get_params(deep=False)
    assert params["lookback"] == 6
    assert params["id_col"] == "id"


def test_set_params_mutates_in_place() -> None:
    adapter = TabularConfigParams()
    adapter.set_params(lookback=24, prediction_step=2)
    assert adapter.lookback == 24
    assert adapter.prediction_step == 2


def test_sklearn_clone_produces_independent_instance() -> None:
    """sklearn.base.clone copies via get_params; the clone must be
    independent so subsequent mutation of the original does not leak.
    """
    original = TabularConfigParams(lookback=6)
    cloned = sklearn.base.clone(original)
    assert isinstance(cloned, TabularConfigParams)
    assert cloned is not original
    cloned.set_params(lookback=24)
    assert original.lookback == 6
    assert cloned.lookback == 24


def test_to_pydantic_produces_frozen_config() -> None:
    adapter = TabularConfigParams(lookback=6, prediction_step=2)
    cfg = adapter.to_pydantic()
    assert isinstance(cfg, TabularToSequenceConfig)
    assert cfg.lookback == 6
    assert cfg.prediction_step == 2


def test_to_pydantic_propagates_validation_errors() -> None:
    """A4 step 4: outer estimators wrap this into ConfigError at _build_config."""
    adapter = TabularConfigParams(lookback=0)  # below the ge=1 floor
    with pytest.raises(ValidationError):
        adapter.to_pydantic()


def test_categorical_embed_dims_none_default_is_empty_mapping_in_pydantic() -> None:
    adapter = TabularConfigParams()
    cfg = adapter.to_pydantic()
    assert dict(cfg.categorical_embed_dims) == {}


def test_categorical_embed_dims_dict_round_trips() -> None:
    adapter = TabularConfigParams(categorical_embed_dims={"industry": 16})
    cfg = adapter.to_pydantic()
    assert dict(cfg.categorical_embed_dims) == {"industry": 16}


def test_set_params_chains_via_clone() -> None:
    """clone() should not freeze the field set; set_params on the clone
    must still work.
    """
    original = TabularConfigParams()
    cloned = sklearn.base.clone(original)
    cloned.set_params(lookback=18)
    assert cloned.lookback == 18
