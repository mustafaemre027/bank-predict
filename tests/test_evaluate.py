import numpy as np
import pytest

from src.evaluate import compare_folds, compute_metrics, summarize_cv


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
