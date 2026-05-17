"""A15.1 attention-output dataclass contract (Phase 6b).

The field set of each dataclass is the v1 snapshot: a MINOR-release
field addition breaks the exact-tuple assertion and forces a deliberate
bump. Tuple-unpacking is unsupported (BETA field set), so iterating an
instance raises ``TypeError``. The regression variant has no ``logits``
field by design (qa r2-I3).
"""

import dataclasses

import numpy as np
import pytest

from seq_sklearn.inference.attention import AttentionOutput, RegressionAttentionOutput

_CLASSIFIER_FIELDS = (
    "predictions",
    "probabilities",
    "logits",
    "var_selection_weights",
    "static_var_selection_weights",
    "attention_weights",
    "padding_mask",
    "entity_id",
)

_REGRESSION_FIELDS = (
    "predictions",
    "quantiles_used",
    "var_selection_weights",
    "static_var_selection_weights",
    "attention_weights",
    "padding_mask",
    "entity_id",
)


def test_attention_output_field_snapshot() -> None:
    assert tuple(f.name for f in dataclasses.fields(AttentionOutput)) == _CLASSIFIER_FIELDS


def test_regression_attention_output_field_snapshot() -> None:
    assert (
        tuple(f.name for f in dataclasses.fields(RegressionAttentionOutput)) == _REGRESSION_FIELDS
    )


def test_regression_variant_has_no_logits_field() -> None:
    # A15.1: regression intentionally omits logits (predictions already
    # carries the raw scalars a classifier exposes via logits).
    assert "logits" not in {f.name for f in dataclasses.fields(RegressionAttentionOutput)}


def _classifier_instance() -> AttentionOutput:
    z = np.zeros(1)
    return AttentionOutput(
        predictions=z,
        probabilities=z,
        logits=z,
        var_selection_weights=z,
        static_var_selection_weights=z,
        attention_weights=z,
        padding_mask=z,
        entity_id=z,
    )


def _regression_instance() -> RegressionAttentionOutput:
    z = np.zeros(1)
    return RegressionAttentionOutput(
        predictions=z,
        quantiles_used=None,
        var_selection_weights=z,
        static_var_selection_weights=z,
        attention_weights=z,
        padding_mask=z,
        entity_id=z,
    )


@pytest.mark.parametrize("out", [_classifier_instance(), _regression_instance()])
def test_tuple_unpacking_raises_typeerror(out: object) -> None:
    with pytest.raises(TypeError):
        _a, _b = out  # type: ignore[misc]


@pytest.mark.parametrize("out", [_classifier_instance(), _regression_instance()])
def test_frozen_instances_reject_mutation(out: object) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.predictions = np.ones(1)  # type: ignore[misc]


def test_quantiles_used_accepts_vector_and_none() -> None:
    z = np.zeros(1)
    pt = _regression_instance()
    assert pt.quantiles_used is None
    q = RegressionAttentionOutput(
        predictions=z,
        quantiles_used=(0.1, 0.5, 0.9),
        var_selection_weights=z,
        static_var_selection_weights=z,
        attention_weights=z,
        padding_mask=z,
        entity_id=z,
    )
    assert q.quantiles_used == (0.1, 0.5, 0.9)
