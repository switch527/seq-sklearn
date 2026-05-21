"""Phase B8 Friedman + Holm correction tests.

The helpers in `benchmarks/stats/friedman.py` are pure numerics
with known oracles:

- Holm-Bonferroni adjustment matches the classical step-down
  procedure across hand-crafted families.
- Friedman omnibus matches `scipy.stats.friedmanchisquare` exactly
  (the module wraps it; the test pins the round-trip).
- Pairwise Wilcoxon matches `scipy.stats.wilcoxon` per pair.
"""

import pandas as pd
import pytest
from benchmarks.stats.friedman import (
    FriedmanHolmResult,
    HolmAdjustedPValue,
    friedman_holm,
    holm_correction,
)
from scipy import stats

# --- holm_correction ---------------------------------------------------------


def test_holm_correction_empty_input_returns_empty_list() -> None:
    assert holm_correction([]) == []


def test_holm_correction_single_value_unchanged() -> None:
    # A family of size 1 has nothing to correct: the largest scaled
    # p-value (* 1) equals the input.
    assert holm_correction([0.05]) == pytest.approx([0.05])


def test_holm_correction_three_value_oracle() -> None:
    # Hand oracle: sorted [0.01, 0.02, 0.04] scaled by [3, 2, 1] =
    # [0.03, 0.04, 0.04]. After monotone-max + clamp: [0.03, 0.04,
    # 0.04]. Unsort back to the input order.
    inputs = [0.04, 0.01, 0.02]
    adjusted = holm_correction(inputs)
    assert adjusted == pytest.approx([0.04, 0.03, 0.04])


def test_holm_correction_running_max_pins_monotonicity() -> None:
    # Input [0.4, 0.001]: sorted [0.001, 0.4] scaled [0.002, 0.4].
    # After running max: [0.002, 0.4]. The smaller p-value still
    # gets adjusted up, but the larger one stays >= the smaller.
    adjusted = holm_correction([0.4, 0.001])
    assert adjusted[1] == pytest.approx(0.002)
    assert adjusted[0] == pytest.approx(0.4)
    assert adjusted[0] >= adjusted[1]


def test_holm_correction_clamps_to_one() -> None:
    # 4 raw values scaled by [4, 3, 2, 1] sorted ascending; the
    # smallest scaled (0.5 * 4 = 2.0) is clamped to 1.0.
    inputs = [0.5, 0.6, 0.7, 0.9]
    adjusted = holm_correction(inputs)
    assert all(0.0 <= a <= 1.0 for a in adjusted)


def test_holm_correction_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        holm_correction([0.1, float("nan"), 0.2])


def test_holm_correction_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        holm_correction([0.1, 1.5])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        holm_correction([-0.1, 0.5])


# --- friedman_holm: degenerate matrices --------------------------------------


def test_friedman_holm_empty_matrix_returns_null_result() -> None:
    result = friedman_holm(pd.DataFrame(), reference_model="m_x")
    assert isinstance(result, FriedmanHolmResult)
    assert result.n_models == 0
    assert result.n_datasets == 0
    assert result.friedman_statistic is None
    assert result.friedman_p_value is None
    assert result.pairwise == ()
    assert result.family_size == 0


def test_friedman_holm_single_model_returns_null_result() -> None:
    matrix = pd.DataFrame({"ds_a": [0.5], "ds_b": [0.4]}, index=pd.Index(["m_x"]))
    result = friedman_holm(matrix, reference_model="m_x")
    assert result.n_models == 1
    assert result.friedman_statistic is None
    assert result.pairwise == ()


def test_friedman_holm_single_dataset_returns_null_result() -> None:
    matrix = pd.DataFrame({"ds_a": [0.5, 0.4]}, index=pd.Index(["m_x", "m_y"]))
    result = friedman_holm(matrix, reference_model="m_x")
    assert result.n_datasets == 1
    assert result.friedman_statistic is None


def test_friedman_holm_reference_missing_raises() -> None:
    matrix = pd.DataFrame(
        {"ds_a": [0.5, 0.4], "ds_b": [0.6, 0.3]},
        index=pd.Index(["m_x", "m_y"]),
    )
    with pytest.raises(ValueError, match="reference_model"):
        friedman_holm(matrix, reference_model="m_z")


# --- friedman_holm: oracle round-trips ---------------------------------------


def test_friedman_holm_friedman_statistic_matches_scipy() -> None:
    matrix = pd.DataFrame(
        {
            "ds_a": [0.50, 0.55, 0.45],
            "ds_b": [0.40, 0.42, 0.35],
            "ds_c": [0.60, 0.62, 0.55],
            "ds_d": [0.30, 0.32, 0.28],
        },
        index=pd.Index(["m_ref", "m_a", "m_b"]),
    )
    result = friedman_holm(matrix, reference_model="m_ref")
    assert result.friedman_statistic is not None
    assert result.friedman_p_value is not None
    # The driver sorts the model index ascending; replicate.
    sorted_rows = [matrix.loc[m].to_numpy() for m in sorted(matrix.index)]
    expected = stats.friedmanchisquare(*sorted_rows)
    assert result.friedman_statistic == pytest.approx(float(expected.statistic))
    assert result.friedman_p_value == pytest.approx(float(expected.pvalue))


def test_friedman_holm_pairwise_wilcoxon_matches_scipy() -> None:
    matrix = pd.DataFrame(
        {
            "ds_a": [0.50, 0.55, 0.45],
            "ds_b": [0.40, 0.42, 0.35],
            "ds_c": [0.60, 0.62, 0.55],
            "ds_d": [0.30, 0.32, 0.28],
            "ds_e": [0.20, 0.22, 0.18],
        },
        index=pd.Index(["m_ref", "m_a", "m_b"]),
    )
    result = friedman_holm(matrix, reference_model="m_ref")
    # 2 other models -> 2 pairwise comparisons.
    assert len(result.pairwise) == 2
    assert result.family_size == 2
    pairwise_by_name = {p.comparison_name: p for p in result.pairwise}
    # Replicate scipy oracle for each (ref - other) pair.
    for other in ("m_a", "m_b"):
        diffs = matrix.loc["m_ref"].to_numpy() - matrix.loc[other].to_numpy()
        expected = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        label = f"m_ref vs {other}"
        actual = pairwise_by_name[label]
        assert actual.wilcoxon_statistic == pytest.approx(float(expected.statistic))
        assert actual.raw_p_value == pytest.approx(float(expected.pvalue))


def test_friedman_holm_drops_nan_dataset_columns() -> None:
    matrix = pd.DataFrame(
        {
            "ds_a": [0.50, 0.45, 0.55],
            "ds_b": [0.40, 0.35, 0.45],
            "ds_c_bad": [float("nan"), 0.40, 0.50],
        },
        index=pd.Index(["m_ref", "m_a", "m_b"]),
    )
    result = friedman_holm(matrix, reference_model="m_ref")
    # ds_c_bad is dropped; n_datasets reflects the kept count.
    assert result.n_datasets == 2
    assert result.friedman_statistic is not None


def test_friedman_holm_all_zero_diffs_yield_neutral_pvalue() -> None:
    # ref equals m_a on every dataset; the Wilcoxon helper should
    # surface 1.0 (neutral) instead of raising.
    matrix = pd.DataFrame(
        {
            "ds_a": [0.50, 0.50, 0.45],
            "ds_b": [0.40, 0.40, 0.35],
            "ds_c": [0.30, 0.30, 0.20],
        },
        index=pd.Index(["m_ref", "m_a", "m_b"]),
    )
    result = friedman_holm(matrix, reference_model="m_ref")
    pairwise_by_name = {p.comparison_name: p for p in result.pairwise}
    assert pairwise_by_name["m_ref vs m_a"].raw_p_value == pytest.approx(1.0)
    assert pairwise_by_name["m_ref vs m_a"].wilcoxon_statistic == pytest.approx(0.0)


# --- structural --------------------------------------------------------------


def test_friedman_holm_result_is_frozen_and_extra_forbid() -> None:
    from pydantic import ValidationError

    result = FriedmanHolmResult(
        n_models=2,
        n_datasets=3,
        friedman_statistic=1.0,
        friedman_p_value=0.5,
        pairwise=(),
        family_size=0,
    )
    with pytest.raises(ValidationError):
        result.n_models = 5  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError):
        FriedmanHolmResult(
            n_models=1,
            n_datasets=1,
            family_size=0,
            stray_field=1,  # pyright: ignore[reportCallIssue]
        )


def test_holm_adjusted_p_value_is_frozen_and_extra_forbid() -> None:
    from pydantic import ValidationError

    entry = HolmAdjustedPValue(
        comparison_name="m_ref vs m_a",
        raw_p_value=0.01,
        holm_adjusted_p_value=0.02,
        wilcoxon_statistic=5.0,
        n_datasets=4,
    )
    with pytest.raises(ValidationError):
        entry.raw_p_value = 0.5  # pyright: ignore[reportAttributeAccessIssue]
