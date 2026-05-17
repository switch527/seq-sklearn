"""RecurrentSequenceEstimatorConfig skeleton (Phase 6b, A6.1).

The config is a v1 schema only (INTERNAL). These pin the default
field values, that it inherits the frozen / ``extra="forbid"`` /
required-``task_type``-``loss`` contract from
:class:`BaseModelConfig`, and one validation-failure path per the
testing convention.
"""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.recurrent import RecurrentSequenceEstimatorConfig


def _cfg(**overrides: object) -> RecurrentSequenceEstimatorConfig:
    return RecurrentSequenceEstimatorConfig(
        task_type="binary",
        loss=LossConfig(strategy="cross_entropy"),
        **overrides,
    )


def test_default_field_values_match_a6_1() -> None:
    cfg = _cfg()
    assert cfg.bidirectional is False
    assert cfg.recurrent_dropout == 0.1
    assert cfg.recurrent_dropout_kind == "weight_drop"
    assert cfg.hidden_init_strategy == "zero"
    assert cfg.readout == "last_valid"
    assert cfg.bptt_window is None


def test_overrides_round_trip() -> None:
    cfg = _cfg(
        bidirectional=True,
        recurrent_dropout=0.3,
        recurrent_dropout_kind="variational",
        hidden_init_strategy="per_entity",
        readout="attention",
        bptt_window=32,
    )
    assert cfg.bidirectional is True
    assert cfg.recurrent_dropout == 0.3
    assert cfg.recurrent_dropout_kind == "variational"
    assert cfg.hidden_init_strategy == "per_entity"
    assert cfg.readout == "attention"
    assert cfg.bptt_window == 32


def test_inherits_frozen_contract() -> None:
    cfg = _cfg()
    with pytest.raises(ValidationError):
        cfg.bidirectional = True


def test_inherits_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        _cfg(not_a_recurrent_field=1)


def test_requires_task_type_and_loss() -> None:
    with pytest.raises(ValidationError):
        RecurrentSequenceEstimatorConfig()  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_recurrent_dropout_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        _cfg(recurrent_dropout=bad)


def test_invalid_recurrent_dropout_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        _cfg(recurrent_dropout_kind="dropconnect")


def test_bptt_window_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _cfg(bptt_window=0)


def test_inherits_validity_matrix_validator() -> None:
    # regression task + cross_entropy loss is an F5-illegal cell; the
    # inherited BaseModelConfig validator must still reject it.
    with pytest.raises(ValidationError):
        RecurrentSequenceEstimatorConfig(
            task_type="regression_point",
            loss=LossConfig(strategy="cross_entropy"),
        )
