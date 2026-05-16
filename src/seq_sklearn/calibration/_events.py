"""Shared ``calibration.*`` structured-log emission (F11).

Both calibrator families emit the same two F11 records on ``fit``: a
``calibration.small_set`` WARNING when the fold is under
:data:`_MIN_RECOMMENDED_CAL_SIZE` rows, then a ``calibration.fit`` INFO
carrying the family-specific quality scalars. Centralised here so the
WARNING-then-INFO ordering and the threshold are pinned in one place.
"""

import logging
from typing import Any, cast

from seq_sklearn.logging import Event, emit

__all__ = ["emit_calibration_fit"]

# requirements.md F11: calibration.small_set fires below 100 rows.
_MIN_RECOMMENDED_CAL_SIZE = 100


def emit_calibration_fit(
    logger: logging.Logger,
    strategy: str,
    cal_size: int,
    **metrics: float,
) -> None:
    """Emit ``calibration.small_set`` (if applicable) then ``calibration.fit``.

    ``metrics`` is the family-specific scalar payload (``pre_ece`` /
    ``post_ece`` for classifiers, ``pre_coverage`` / ``post_coverage``
    for regressors). The small-set WARNING is emitted first so a log
    consumer sees the caveat before the fit result.
    """
    if cal_size < _MIN_RECOMMENDED_CAL_SIZE:
        emit(
            logger,
            Event.CALIBRATION_SMALL_SET,
            level=logging.WARNING,
            strategy=strategy,
            cal_size=cal_size,
            min_recommended=_MIN_RECOMMENDED_CAL_SIZE,
        )
    emit(
        logger,
        Event.CALIBRATION_FIT,
        **cast(dict[str, Any], {"strategy": strategy, "cal_size": cal_size, **metrics}),
    )
