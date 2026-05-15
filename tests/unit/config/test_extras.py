"""Tests for the Tier 4 ``extra`` escape hatch and deprecation-alias helper.

Commit 1 ships ``_extras.py`` only; the family sub-configs
(``OptimizerConfig`` et al.) land in commit 2. To exercise the full
pydantic field path (not a bare ``_normalize_extras`` call) without a
forward dependency, these tests use a local frozen pydantic stub with an
``extra: ExtraDict`` field. The coverage intent (BeforeValidator +
frozen-model interaction) is identical to using a real family config.
"""

import json
import warnings

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from seq_sklearn.config._extras import (
    _PROMOTED_KEYS_BY_FAMILY,
    ExtraDict,
    extract_deprecated_extras,
)
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
    original = _ExtraHolder(
        extra={
            "a_str": "x",
            "b_int": 3,
            "c_float": 1.5,
            "d_bool": True,
            "e_none": None,
        }
    )
    payload = json.loads(json.dumps(original.model_dump(mode="json")))
    restored = _ExtraHolder.model_validate(payload)
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
    cfg = _ExtraHolder(extra=(("flag", True), ("count", 3)))
    payload = json.loads(json.dumps(cfg.model_dump(mode="json")))
    reconstructed = _ExtraHolder.model_validate(payload)
    assert cfg == reconstructed


def test_extract_deprecated_extras_meta_promoted_keys_exist() -> None:
    """Every registered promotion names a real typed field on its family config.

    The v1 registry is empty, so the field-existence loop is vacuous
    today; it becomes substantive automatically once a promotion is
    registered (post commit 2, when the family configs exist). The
    family-config import is lazy so this stays green at commit 1.
    """
    assert set(_PROMOTED_KEYS_BY_FAMILY) == {
        "optimizer",
        "scheduler",
        "loss",
        "sampler",
    }
    for family, promoted in _PROMOTED_KEYS_BY_FAMILY.items():
        for _extra_key, typed_name in promoted.items():
            from seq_sklearn import config as _cfg

            family_cls = {
                "optimizer": getattr(_cfg, "OptimizerConfig", None),
                "scheduler": getattr(_cfg, "SchedulerConfig", None),
                "loss": getattr(_cfg, "LossConfig", None),
                "sampler": getattr(_cfg, "SamplerConfig", None),
            }[family]
            assert family_cls is not None
            assert typed_name in family_cls.model_fields


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
        result = extract_deprecated_extras(cfg, "optimizer")
    assert result == {"amsgrad": True}

    monkeypatch.setitem(_PROMOTED_KEYS_BY_FAMILY["optimizer"], "fake", "fake_field")
    unrelated = _PromotionStub(extra=(("amsgrad", True),))  # "fake" not present
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result_skip = extract_deprecated_extras(unrelated, "optimizer")
    assert result_skip == {"amsgrad": True}


def test_extract_deprecated_extras_mock_promotion_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_PROMOTED_KEYS_BY_FAMILY["optimizer"], "fake", "fake_field")
    cfg = _PromotionStub(extra=(("fake", True),))  # fake_field left at default
    with pytest.warns(DeprecationWarning, match=r"deprecated"):
        result = extract_deprecated_extras(cfg, "optimizer")
    # The promoted key is consumed from the returned dict; the typed
    # field is the canonical home going forward.
    assert "fake" not in result


def test_extra_dict_rejects_non_string_key() -> None:
    with pytest.raises(TypeError):
        _ExtraHolder(extra=((1, "v"),))


def test_normalize_extras_accepts_none_produces_empty_tuple() -> None:
    assert _ExtraHolder(extra=None).extra == ()
