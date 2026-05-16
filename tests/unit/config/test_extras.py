"""Tests for the Tier 4 ``extra`` escape hatch and deprecation-alias helper.

The pure-validator and alias-helper tests use a local frozen pydantic
stub so they stay isolated from any one family config's other fields and
validators. The round-trip tests that must exercise the full real
pydantic field path use ``OptimizerConfig``.
"""

import json
import warnings

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined

from seq_sklearn.config._extras import (
    # Module-private by design; imported here as a test seam so the
    # alias-helper tests can monkeypatch a fake promotion. Not public API.
    _PROMOTED_KEYS_BY_FAMILY,
    ExtraDict,
    extract_deprecated_extras,
)
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.optimizer import OptimizerConfig
from seq_sklearn.config.sampler import SamplerConfig
from seq_sklearn.config.scheduler import SchedulerConfig
from seq_sklearn.errors import ConfigError


class _ExtraHolder(BaseModel):
    """Frozen stub carrying only the escape hatch, mirroring a family config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extra: ExtraDict = ()


class _PromotionStub(BaseModel):
    """Stub with a promotable typed field, for the alias-helper tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fake_field: bool = False
    extra: ExtraDict = ()


def test_extra_dict_rejects_non_primitive_value() -> None:
    with pytest.raises(TypeError):
        _ExtraHolder(extra={"weights": np.array([1, 2, 3])})


def test_extra_dict_round_trips_each_primitive_type() -> None:
    original = OptimizerConfig(
        extra={
            "a_str": "x",
            "b_int": 3,
            "c_float": 1.5,
            "d_bool": True,
            "e_none": None,
        }
    )
    payload = json.loads(json.dumps(original.model_dump(mode="json")))
    restored = OptimizerConfig.model_validate(payload)
    got = dict(restored.extra)
    assert got["a_str"] == "x"
    assert type(got["a_str"]) is str
    assert got["b_int"] == 3
    assert type(got["b_int"]) is int
    assert got["c_float"] == 1.5
    assert type(got["c_float"]) is float
    assert got["d_bool"] is True
    assert type(got["d_bool"]) is bool
    assert got["e_none"] is None


def test_extra_dict_stored_as_sorted_tuple() -> None:
    forward = _ExtraHolder(extra=(("alpha", 1), ("beta", 2)))
    reversed_ = _ExtraHolder(extra=(("beta", 2), ("alpha", 1)))
    assert forward.extra == reversed_.extra
    assert forward.extra == (("alpha", 1), ("beta", 2))
    assert hash(forward) == hash(reversed_)


def test_extra_dict_survives_json_roundtrip() -> None:
    cfg = OptimizerConfig(extra=(("flag", True), ("count", 3)))
    payload = json.loads(json.dumps(cfg.model_dump(mode="json")))
    reconstructed = OptimizerConfig.model_validate(payload)
    assert cfg == reconstructed


def test_extract_deprecated_extras_meta_promoted_keys_exist() -> None:
    """Every registered promotion names a real typed field with a default.

    The v1 registry is empty, so the loop is vacuous; it becomes
    load-bearing automatically once a promotion is registered and
    catches a maintainer who registers one without adding the typed
    field, or who points it at a no-default field (whose
    ``FieldInfo.default`` is ``PydanticUndefined``, which makes the
    helper's "both paths set" detection ambiguous).
    """
    family_cls = {
        "optimizer": OptimizerConfig,
        "scheduler": SchedulerConfig,
        "loss": LossConfig,
        "sampler": SamplerConfig,
    }
    assert set(_PROMOTED_KEYS_BY_FAMILY) == set(family_cls)
    for family, promoted in _PROMOTED_KEYS_BY_FAMILY.items():
        for _extra_key, typed_name in promoted.items():
            fields = family_cls[family].model_fields
            assert typed_name in fields
            assert fields[typed_name].default is not PydanticUndefined


def test_extract_deprecated_extras_both_typed_and_extra_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_PROMOTED_KEYS_BY_FAMILY["optimizer"], "fake", "fake_field")
    cfg = _PromotionStub(fake_field=True, extra=(("fake", True),))
    with (
        pytest.warns(DeprecationWarning, match=r"deprecated"),
        pytest.raises(ConfigError, match="both extra and the typed"),
    ):
        extract_deprecated_extras(cfg, "optimizer")


def test_extract_deprecated_extras_happy_path_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unpromoted keys pass through unchanged with no DeprecationWarning.

    Covers two passthrough shapes: (a) no promotion registered for the
    family, and (b) a promotion IS registered but this config did not
    use the deprecated key, so the alias route is skipped.
    """
    cfg = _ExtraHolder(extra=(("amsgrad", True),))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        returned_cfg, result = extract_deprecated_extras(cfg, "optimizer")
    assert returned_cfg is cfg  # no promoted key: cfg returned unchanged
    assert result == {"amsgrad": True}

    monkeypatch.setitem(_PROMOTED_KEYS_BY_FAMILY["optimizer"], "fake", "fake_field")
    unrelated = _PromotionStub(extra=(("amsgrad", True),))  # "fake" not present
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        skip_cfg, result_skip = extract_deprecated_extras(unrelated, "optimizer")
    assert skip_cfg is unrelated
    assert result_skip == {"amsgrad": True}


def test_extract_deprecated_extras_mock_promotion_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_PROMOTED_KEYS_BY_FAMILY["optimizer"], "fake", "fake_field")
    cfg = _PromotionStub(extra=(("fake", True),))  # fake_field left at default
    with pytest.warns(DeprecationWarning, match=r"deprecated"):
        returned_cfg, result = extract_deprecated_extras(cfg, "optimizer")
    # The promoted key is consumed from the returned dict and its value
    # is routed onto the typed field of the returned (copied) cfg, so
    # the alias is behavior-preserving. The original frozen cfg is
    # untouched.
    assert "fake" not in result
    assert returned_cfg.fake_field is True
    assert cfg.fake_field is False


def test_extra_dict_rejects_non_string_key() -> None:
    with pytest.raises(TypeError):
        _ExtraHolder(extra=((1, "v"),))


def test_normalize_extras_accepts_none_produces_empty_tuple() -> None:
    assert _ExtraHolder(extra=None).extra == ()
