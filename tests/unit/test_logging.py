"""Tests for the structured-logging surface at architecture A15."""

import logging

import pytest

from seq_sklearn.logging import Event, emit


def test_event_enum_str_values() -> None:
    """Every Event value is a dotted string."""
    for member in Event:
        assert "." in member.value
        # The class derives from str, so each member is also a str instance.
        assert isinstance(member, str)


def test_emit_attaches_event_attribute_to_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("seq_sklearn.test_logging")
    with caplog.at_level(logging.INFO, logger="seq_sklearn"):
        emit(logger, Event.TRAIN_EPOCH, epoch=3, train_loss=0.5)
    matching = [r for r in caplog.records if r.message == "train.epoch"]
    assert len(matching) == 1
    record = matching[0]
    # Per A15, the extra keys land as record attributes (not record.extra).
    assert record.event == "train.epoch"
    assert record.payload == {"epoch": 3, "train_loss": 0.5}


def test_emit_default_level_is_info(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("seq_sklearn.test_logging")
    with caplog.at_level(logging.DEBUG, logger="seq_sklearn"):
        emit(logger, Event.CALIBRATION_FIT, strategy="temperature", cal_size=500)
    matching = [r for r in caplog.records if r.message == "calibration.fit"]
    assert len(matching) == 1
    assert matching[0].levelno == logging.INFO


def test_emit_respects_explicit_level(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("seq_sklearn.test_logging")
    with caplog.at_level(logging.WARNING, logger="seq_sklearn"):
        emit(
            logger,
            Event.TRAIN_NAN_STEP_SKIPPED,
            level=logging.WARNING,
            step=42,
            consecutive_nan_count=1,
        )
    matching = [r for r in caplog.records if r.message == "train.nan_step_skipped"]
    assert len(matching) == 1
    assert matching[0].levelno == logging.WARNING
    assert matching[0].payload == {"step": 42, "consecutive_nan_count": 1}


def test_emit_copies_payload_so_caller_mutation_is_isolated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The emitted record must not alias the caller's payload dict."""
    logger = logging.getLogger("seq_sklearn.test_logging")
    with caplog.at_level(logging.INFO, logger="seq_sklearn"):
        emit(logger, Event.HARDWARE_DETECT, tier="CPU", selected_precision="32-true")
    matching = [r for r in caplog.records if r.message == "hardware.detect"]
    assert len(matching) == 1
    # Mutating the captured payload should not crash; emit() copies.
    matching[0].payload["tier"] = "GPU"
    # A second emission with new content should land on its own record.
    with caplog.at_level(logging.INFO, logger="seq_sklearn"):
        emit(logger, Event.HARDWARE_DETECT, tier="HOPPER", selected_precision="bf16-mixed")
    second = [r for r in caplog.records if r.message == "hardware.detect"][-1]
    assert second.payload == {"tier": "HOPPER", "selected_precision": "bf16-mixed"}
