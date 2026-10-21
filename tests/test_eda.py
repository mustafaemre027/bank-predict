import pandas as pd
import pytest

from src.eda import (
    acceptance_rate_by,
    chi_square_tests,
    iqr_outlier_summary,
    mann_whitney_tests,
    numeric_summary,
    pdays_poutcome_crosstab,
)


def make_frame(**columns: list[object]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_acceptance_rate_by_computes_rate_and_lift_per_category() -> None:
    df = make_frame(
        group=["a", "a", "a", "a", "b", "b", "b", "b"],
        y=["yes", "yes", "no", "no", "no", "no", "no", "no"],
    )

    table = acceptance_rate_by(df, "group")

    assert table.loc["a", "count"] == 4
    assert table.loc["a", "yes_count"] == 2
    assert table.loc["a", "acceptance_rate"] == pytest.approx(0.5)
    assert table.loc["a", "lift"] == pytest.approx(2.0)
    assert table.loc["b", "acceptance_rate"] == pytest.approx(0.0)


def test_acceptance_rate_by_sorts_categories_from_highest_rate() -> None:
    df = make_frame(group=["low", "high", "high", "low"], y=["no", "yes", "yes", "no"])

    table = acceptance_rate_by(df, "group")

    assert list(table.index) == ["high", "low"]


def test_numeric_summary_reports_positive_skewness_for_right_tailed_values() -> None:
    df = make_frame(balance=[1, 1, 1, 1, 50])

    summary = numeric_summary(df, ["balance"])

    assert summary.loc["balance", "skewness"] > 0


def test_iqr_outlier_summary_flags_only_values_beyond_bounds() -> None:
    df = make_frame(value=[1, 2, 3, 4, 100])

    summary = iqr_outlier_summary(df, ["value"])

    assert summary.loc["value", "lower_bound"] == pytest.approx(-1.0)
    assert summary.loc["value", "upper_bound"] == pytest.approx(7.0)
    assert summary.loc["value", "outlier_count"] == 1
    assert summary.loc["value", "outlier_pct"] == pytest.approx(20.0)


def test_iqr_outlier_summary_flags_every_nonzero_value_when_iqr_is_zero() -> None:
    df = make_frame(previous=[0] * 10 + [3, 5])

    summary = iqr_outlier_summary(df, ["previous"])

    assert summary.loc["previous", "iqr"] == 0
    assert summary.loc["previous", "outlier_count"] == 2


def test_chi_square_tests_gives_cramers_v_of_one_for_perfect_association() -> None:
    df = make_frame(signal=["a"] * 10 + ["b"] * 10, y=["yes"] * 10 + ["no"] * 10)

    result = chi_square_tests(df, ["signal"])

    assert result.loc["signal", "cramers_v"] == pytest.approx(1.0)


def test_chi_square_tests_gives_zero_effect_for_independent_variable() -> None:
    df = make_frame(noise=["a", "a", "b", "b"] * 5, y=["yes", "no", "yes", "no"] * 5)

    result = chi_square_tests(df, ["noise"])

    assert result.loc["noise", "cramers_v"] == pytest.approx(0.0)
    assert result.loc["noise", "p_value"] == pytest.approx(1.0)


def test_chi_square_tests_sorts_columns_by_effect_size() -> None:
    df = make_frame(
        noise=["a", "a", "b", "b"] * 5,
        signal=["a", "b", "a", "b"] * 5,
        y=["yes", "no", "yes", "no"] * 5,
    )

    result = chi_square_tests(df, ["noise", "signal"])

    assert list(result.index) == ["signal", "noise"]


def test_chi_square_tests_counts_cells_with_low_expected_frequency() -> None:
    df = make_frame(rare=["a", "a", "b", "b"], y=["yes", "no", "yes", "no"])

    result = chi_square_tests(df, ["rare"])

    assert result.loc["rare", "low_expected_cells"] == 4


def test_mann_whitney_tests_gives_full_effect_when_groups_do_not_overlap() -> None:
    df = make_frame(balance=[10, 11, 12, 1, 2, 3], y=["yes", "yes", "yes", "no", "no", "no"])

    result = mann_whitney_tests(df, ["balance"])

    assert result.loc["balance", "rank_biserial"] == pytest.approx(1.0)
    assert result.loc["balance", "median_yes"] == pytest.approx(11.0)
    assert result.loc["balance", "median_no"] == pytest.approx(2.0)


def test_mann_whitney_tests_gives_zero_effect_for_identical_groups() -> None:
    df = make_frame(age=[30, 40, 30, 40], y=["yes", "yes", "no", "no"])

    result = mann_whitney_tests(df, ["age"])

    assert result.loc["age", "rank_biserial"] == pytest.approx(0.0)


def test_mann_whitney_tests_sorts_by_absolute_effect_size() -> None:
    df = make_frame(
        weak=[1, 2, 3, 4, 1, 2, 3, 5],
        strong_negative=[1, 2, 3, 4, 10, 11, 12, 13],
        y=["yes"] * 4 + ["no"] * 4,
    )

    result = mann_whitney_tests(df, ["weak", "strong_negative"])

    assert list(result.index) == ["strong_negative", "weak"]
    assert result.loc["strong_negative", "rank_biserial"] == pytest.approx(-1.0)


def test_rate_by_bins_rejects_empty_bins_instead_of_dropping_them() -> None:
    from notebooks.build_kesifsel_analiz import rate_by_bins

    df = make_frame(age=[20, 22, 70], y=["yes", "no", "yes"])

    with pytest.raises(ValueError, match="boş aralıklar"):
        rate_by_bins(df, "age", [-float("inf"), 25, 60, float("inf")], ["genç", "orta", "yaşlı"])


def test_pdays_poutcome_crosstab_rows_are_readable_by_name_with_totals() -> None:
    df = make_frame(
        pdays=[-1, -1, -1, 10, 20],
        poutcome=["unknown", "unknown", "unknown", "failure", "success"],
    )

    table = pdays_poutcome_crosstab(df)

    assert table.loc["pdays_equal_minus_one", "unknown"] == 3
    assert table.loc["pdays_not_minus_one", "failure"] == 1
    assert table.loc["All", "All"] == 5
