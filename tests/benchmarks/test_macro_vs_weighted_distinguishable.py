"""Phase B4 macro-vs-weighted separation test (B5 delta R4).

Proves the multiclass record's `*_macro_zd0` and `*_weighted_zd0` are
computed by separate code paths (not silently collapsed to the same
sklearn call) by asserting they differ on an unequal-counts fixture
where macro and weighted are mathematically distinguishable.
"""

import numpy as np
import pytest
from benchmarks.metrics import MulticlassMetrics, compute_all


def test_macro_and_weighted_f1_are_distinguishable_on_unequal_supports() -> None:
    # supports per class: (3, 1, 1); class 2 fully omitted from y_pred
    # so per-class metrics differ across classes, and the weighted
    # average (by support) differs from the unweighted macro mean.
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.3, 0.6, 0.1],
            [0.2, 0.7, 0.1],
            [0.4, 0.5, 0.1],
        ]
    )

    rec = compute_all(
        task_type="multiclass",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=np.array([0, 1, 2]),
    )
    assert isinstance(rec, MulticlassMetrics)

    # The point of this test: macro and weighted must NOT be equal here.
    # If a refactor accidentally routes both through the same average,
    # this assertion catches it.
    assert rec.precision_macro_zd0 != pytest.approx(rec.precision_weighted_zd0)
    assert rec.recall_macro_zd0 != pytest.approx(rec.recall_weighted_zd0)
    assert rec.f1_macro_zd0 != pytest.approx(rec.f1_weighted_zd0)
