import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.evaluate import (
    build_comparison_table,
    compare_folds,
    compute_metrics,
    fold_composition_for_json,
    fold_metrics_at_threshold,
    is_ablation_candidate,
    min_folds_improved_for_candidate,
    summarize_cv,
)


def test_compute_metrics_basic_example():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.2, 0.1])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    expected = {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "roc_auc": 0.75,
        "pr_auc": 5 / 6,
    }

    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)


def test_summarize_cv_reports_mean_and_sample_std_per_metric():
    folds = [{"pr_auc": 0.4, "f1": 0.2}, {"pr_auc": 0.6, "f1": 0.4}]

    summary = summarize_cv(folds)

    assert summary.loc["pr_auc", "mean"] == pytest.approx(0.5)
    assert summary.loc["pr_auc", "std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert list(summary.index) == ["pr_auc", "f1"]


def test_compare_folds_measures_variant_minus_base_per_fold():
    base = [{"pr_auc": 0.40}, {"pr_auc": 0.50}, {"pr_auc": 0.45}]
    variant = [{"pr_auc": 0.42}, {"pr_auc": 0.49}, {"pr_auc": 0.48}]

    comparison = compare_folds(base, variant, "pr_auc")

    assert comparison["mean_difference"] == pytest.approx(0.04 / 3)
    assert comparison["folds_improved"] == 2
    assert comparison["fold_count"] == 3


def test_compare_folds_rejects_different_fold_counts():
    with pytest.raises(ValueError, match="Katman sayıları farklı"):
        compare_folds([{"pr_auc": 0.4}], [{"pr_auc": 0.4}, {"pr_auc": 0.5}], "pr_auc")


def test_summaries_reject_single_fold_where_std_is_undefined():
    with pytest.raises(ValueError, match="En az 2 katman"):
        summarize_cv([{"pr_auc": 0.4}])
    with pytest.raises(ValueError, match="En az 2 katman"):
        compare_folds([{"pr_auc": 0.4}], [{"pr_auc": 0.5}], "pr_auc")


def test_build_comparison_table_has_one_row_per_model_with_mean_and_std_per_metric():
    summaries = {
        "a": summarize_cv([{"pr_auc": 0.4, "roc_auc": 0.7, "f1": 0.1}, {"pr_auc": 0.6, "roc_auc": 0.9, "f1": 0.3}]),
        "b": summarize_cv([{"pr_auc": 0.2, "roc_auc": 0.5, "f1": 0.0}, {"pr_auc": 0.2, "roc_auc": 0.5, "f1": 0.0}]),
    }

    table = build_comparison_table(summaries, ["pr_auc", "roc_auc"])

    assert list(table.index) == ["a", "b"]
    assert table.loc["a", ("pr_auc", "mean")] == pytest.approx(0.5)
    assert table.loc["a", ("roc_auc", "std")] == pytest.approx(np.std([0.7, 0.9], ddof=1))
    assert table.loc["b", ("pr_auc", "std")] == 0
    assert ("f1", "mean") not in table.columns


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        ({"mean_difference": 0.02, "std_difference": 0.005, "folds_improved": 5, "fold_count": 5}, True),
        ({"mean_difference": 0.02, "std_difference": 0.005, "folds_improved": 4, "fold_count": 5}, True),
        ({"mean_difference": 0.02, "std_difference": 0.005, "folds_improved": 3, "fold_count": 5}, False),
        ({"mean_difference": 0.003, "std_difference": 0.005, "folds_improved": 5, "fold_count": 5}, False),
        ({"mean_difference": -0.02, "std_difference": 0.005, "folds_improved": 0, "fold_count": 5}, False),
        ({"mean_difference": 0.0, "std_difference": 0.0, "folds_improved": 0, "fold_count": 5}, False),
    ],
)
def test_ablation_candidate_needs_gain_above_noise_in_most_folds(comparison, expected):
    assert is_ablation_candidate(comparison) is expected


@pytest.mark.parametrize(("fold_count", "expected"), [(5, 4), (10, 8), (3, 3)])
def test_required_improving_folds_scale_with_fold_count(fold_count, expected):
    assert min_folds_improved_for_candidate(fold_count) == expected


def test_fold_composition_lists_sizes_and_positive_rate_per_fold():
    X = pd.DataFrame({"a": range(20)})
    y = pd.Series([1, 0, 0, 0] * 5)

    composition = fold_composition_for_json(X, y, StratifiedKFold(n_splits=5))

    assert len(composition) == 5
    assert all(fold["train_rows"] == 16 and fold["validation_rows"] == 4 for fold in composition)
    assert all(fold["validation_positive_rate"] == 0.25 for fold in composition)


def test_fold_metrics_at_threshold_scores_each_fold_with_the_given_threshold():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.1, 0.4, 0.2, 0.8, 0.3, 0.6, 0.7])
    fold_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    folds = fold_metrics_at_threshold(y_true, proba, fold_ids, threshold=0.5)

    assert [fold["recall"] for fold in folds] == [0.5, 1.0]
    assert [fold["precision"] for fold in folds] == [1.0, 2 / 3]
    assert folds[0]["roc_auc"] == 1.0
