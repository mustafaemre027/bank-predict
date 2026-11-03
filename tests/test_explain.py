import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.explain import (
    OTHER_PERIOD_LABEL,
    PEAK_PERIOD_LABEL,
    error_profile,
    logistic_coefficients,
    metrics_by_group,
    permutation_importance_table,
    rank_variables_by_coefficient,
    source_variable,
    top_overlap,
    with_campaign_period,
)
from src.modeling import build_logistic_regression, build_random_forest

SAMPLE_SIZE = 600


def build_synthetic_frame(seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Hedefi yalnızca `poutcome` belirleyen ham tablo: success ise kabul, değilse ret (az miktarda gürültüyle)."""
    rng = np.random.default_rng(seed)
    poutcome = rng.choice(["success", "failure", "unknown"], size=SAMPLE_SIZE, p=[0.3, 0.3, 0.4])
    frame = pd.DataFrame(
        {
            "age": rng.integers(20, 70, SAMPLE_SIZE),
            "job": rng.choice(["admin.", "technician", "retired"], SAMPLE_SIZE),
            "marital": rng.choice(["married", "single"], SAMPLE_SIZE),
            "education": rng.choice(["primary", "secondary", "tertiary"], SAMPLE_SIZE),
            "default": rng.choice(["yes", "no"], SAMPLE_SIZE),
            "balance": rng.integers(-500, 5000, SAMPLE_SIZE),
            "housing": rng.choice(["yes", "no"], SAMPLE_SIZE),
            "loan": rng.choice(["yes", "no"], SAMPLE_SIZE),
            "contact": rng.choice(["cellular", "telephone", "unknown"], SAMPLE_SIZE),
            "day": rng.integers(1, 29, SAMPLE_SIZE),
            "month": rng.choice(["may", "jun", "aug", "nov"], SAMPLE_SIZE),
            "campaign": rng.integers(1, 10, SAMPLE_SIZE),
            "pdays": rng.integers(-1, 300, SAMPLE_SIZE),
            "previous": rng.integers(0, 5, SAMPLE_SIZE),
            "poutcome": poutcome,
        }
    )
    target = pd.Series((poutcome == "success").astype(int), index=frame.index)
    flip = rng.random(SAMPLE_SIZE) < 0.03
    return frame, target.where(~flip, 1 - target)


def test_permutation_importance_puts_the_only_informative_feature_first():
    X, y = build_synthetic_frame()
    model = build_random_forest().fit(X, y)

    table = permutation_importance_table(model, X, y, n_repeats=3)

    assert table.loc[0, "feature"] == "poutcome"
    assert set(table["feature"]) == set(X.columns)
    assert table["importance_mean"].is_monotonic_decreasing


def test_permutation_importance_is_reproducible():
    X, y = build_synthetic_frame()
    model = build_random_forest().fit(X, y)

    first = permutation_importance_table(model, X, y, n_repeats=3)
    second = permutation_importance_table(model, X, y, n_repeats=3)

    pd.testing.assert_frame_equal(first, second)


def test_logistic_coefficients_rank_the_informative_category_first_by_absolute_value():
    X, y = build_synthetic_frame()
    model = build_logistic_regression().fit(X, y)

    table = logistic_coefficients(model)

    assert table.loc[0, "feature"] == "nominal__poutcome_success"
    assert table.loc[0, "coefficient"] > 0
    assert table["coefficient"].abs().is_monotonic_decreasing


def test_logistic_coefficients_rejects_a_pipeline_that_is_not_fitted():
    with pytest.raises(Exception):
        logistic_coefficients(build_logistic_regression())


def test_top_overlap_keeps_first_list_order_and_only_counts_the_top_n():
    first = ["a", "b", "c", "d"]
    second = ["c", "x", "a", "b"]

    assert top_overlap(first, second, 3) == ["a", "c"]


def test_error_profile_compares_false_negatives_with_true_positives():
    X = pd.DataFrame({"contact": ["cellular", "cellular", "telephone", "unknown", "cellular", "unknown"]})
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])

    profile = error_profile(X, y_true, y_pred, ["contact"])

    rows = {row["category"]: row for row in profile["contact"]}
    assert rows["cellular"]["true_positive_share"] == 1.0
    assert rows["cellular"]["false_negative_count"] == 0
    assert rows["telephone"]["false_negative_share"] == 0.5
    assert rows["unknown"]["false_negative_count"] == 1
    assert rows["unknown"]["true_positive_count"] == 0


def test_error_profile_gives_zero_shares_when_a_group_is_empty():
    X = pd.DataFrame({"contact": ["cellular", "unknown"]})

    profile = error_profile(X, np.array([1, 1]), np.array([1, 1]), ["contact"])

    assert all(row["false_negative_share"] == 0.0 for row in profile["contact"])


def test_campaign_period_marks_may_and_june_as_the_peak():
    X = pd.DataFrame({"month": ["may", "jun", "aug", "nov"]})

    assert list(with_campaign_period(X)["period"]) == [PEAK_PERIOD_LABEL, PEAK_PERIOD_LABEL, OTHER_PERIOD_LABEL, OTHER_PERIOD_LABEL]


def test_metrics_by_group_reports_count_rate_and_pr_auc_per_group():
    X = pd.DataFrame({"contact": ["a", "a", "a", "a", "b", "b"]}, index=[10, 11, 12, 13, 20, 21])
    y_true = np.array([1, 0, 1, 0, 0, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.4])

    table = metrics_by_group(X, y_true, proba, "contact").set_index("group")

    assert table.loc["a", "count"] == 4
    assert table.loc["a", "positive_rate"] == 0.5
    assert table.loc["a", "pr_auc"] == pytest.approx(1.0)
    assert table.loc["b", "positive_rate"] == 0.0
    assert np.isnan(table.loc["b", "pr_auc"])


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("nominal__job_retired", "job"),
        ("month__month_may", "month"),
        ("signed_log__balance", "balance"),
        ("log__campaign", "campaign"),
        ("binary__housing", "housing"),
        ("flag__was_contacted_before", "pdays"),
        ("nominal__age_group_41-50", "age"),
        ("scaled__age", "age"),
    ],
)
def test_source_variable_maps_preprocessed_names_to_raw_columns(feature: str, expected: str) -> None:
    assert source_variable(feature, build_synthetic_frame()[0].columns) == expected


def test_source_variable_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="eşleşmedi"):
        source_variable("nominal__zodiac_leo", ["age", "job"])


def test_variable_ranking_takes_the_largest_absolute_coefficient_per_variable():
    coefficients = pd.DataFrame(
        {"feature": ["nominal__job_retired", "nominal__job_admin.", "scaled__age", "nominal__age_group_61+"], "coefficient": [-0.9, 0.2, 0.3, 0.5]}
    )

    ranking = rank_variables_by_coefficient(coefficients, ["job", "age"])

    assert list(ranking["variable"]) == ["job", "age"]
    assert list(ranking["strength"]) == [0.9, 0.5]


def test_metrics_by_group_rejects_mismatched_lengths():
    X = pd.DataFrame({"contact": ["a", "b", "a"]})

    with pytest.raises(ValueError, match="aynı olmalı"):
        metrics_by_group(X, np.array([1, 0, 1]), np.array([0.5, 0.4]), "contact")
