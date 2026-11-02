import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model_plots import (
    plot_calibration_curves,
    plot_model_comparison,
    plot_pr_curves_overlay,
    plot_pr_roc_curves,
    plot_search_results,
    plot_strategy_comparison,
    plot_threshold_curves,
)
from src.threshold import select_threshold_for_recall, select_threshold_max_f1

FOLD_COUNT = 3


def test_plot_pr_roc_curves_draws_each_fold_plus_pooled_and_reference_lines():
    rng = np.random.default_rng(42)
    y_true = np.tile([0, 0, 0, 1], 30)
    proba = np.clip(y_true * 0.3 + rng.random(len(y_true)) * 0.7, 0, 1)
    fold_ids = np.arange(len(y_true)) % FOLD_COUNT

    fig = plot_pr_roc_curves(y_true, proba, fold_ids, "Model")

    pr_ax, roc_ax = fig.axes
    assert len(pr_ax.lines) == FOLD_COUNT + 2
    assert len(roc_ax.lines) == FOLD_COUNT + 2
    reference_level = pr_ax.lines[-1].get_ydata()[0]
    assert reference_level == y_true.mean()
    plt.close(fig)


LABELS = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGB"}


def test_plot_model_comparison_draws_one_bar_per_model_in_every_metric_panel():
    summary = {metric: {"mean": 0.4, "std": 0.01} for metric in ("pr_auc", "roc_auc")}
    summary_by_model = {model: summary for model in LABELS}

    fig = plot_model_comparison(summary_by_model, LABELS, reference_summary={"pr_auc": {"mean": 0.117}, "roc_auc": {"mean": 0.5}})

    assert len(fig.axes) == 2
    assert all(len(ax.patches) == len(LABELS) for ax in fig.axes)
    assert [text.get_text() for text in fig.axes[0].get_xticklabels()] == list(LABELS.values())
    plt.close(fig)


def test_plot_pr_curves_overlay_draws_every_model_plus_reference_line():
    rng = np.random.default_rng(42)
    y_true = np.tile([0, 0, 0, 1], 30)
    oof_by_model = {model: np.clip(y_true * 0.3 + rng.random(len(y_true)) * 0.7, 0, 1) for model in LABELS}

    fig = plot_pr_curves_overlay(y_true, oof_by_model, LABELS)

    (ax,) = fig.axes
    assert len(ax.lines) == len(LABELS) + 1
    assert ax.lines[-1].get_ydata()[0] == y_true.mean()
    plt.close(fig)


STRATEGY_LABELS = {"none": "Ağırlıksız", "class_weight": "Sınıf ağırlığı", "smote": "SMOTE"}


def test_plot_strategy_comparison_draws_one_bar_per_model_and_strategy_in_every_panel():
    values = {model: {strategy: {"mean": 0.4, "std": 0.01} for strategy in STRATEGY_LABELS} for model in LABELS}

    fig = plot_strategy_comparison([("PR-AUC", values), ("F1", values)], LABELS, STRATEGY_LABELS)

    assert len(fig.axes) == 2
    assert all(len(ax.patches) == len(LABELS) * len(STRATEGY_LABELS) for ax in fig.axes)
    assert [text.get_text() for text in fig.axes[0].get_xticklabels()] == list(STRATEGY_LABELS.values())
    plt.close(fig)


def test_plot_threshold_curves_draws_three_metric_curves_and_marks_both_selected_thresholds():
    rng = np.random.default_rng(42)
    y_true = np.tile([0, 0, 0, 1], 50)
    proba = np.clip(y_true * 0.3 + rng.random(len(y_true)) * 0.7, 0, 1)

    fig = plot_threshold_curves(y_true, proba, select_threshold_max_f1(y_true, proba), select_threshold_for_recall(y_true, proba), "Eşik")

    (ax,) = fig.axes
    curves = [line for line in ax.lines if line.get_linestyle() != "--"]
    markers = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(curves) == 3
    assert len({line.get_linestyle() for line in curves}) == 2
    assert len(markers) == 2
    plt.close(fig)


def test_plot_calibration_curves_draws_each_curve_plus_the_perfect_calibration_line():
    bins = [{"mean_predicted": 0.1 * step, "observed_rate": 0.09 * step, "count": 10} for step in range(1, 6)]

    fig = plot_calibration_curves([("XGBoost", "xgboost", {"Kalibrasyonsuz": bins, "Kalibre edilmiş": bins}), ("Rastgele Orman", "random_forest", {"Kalibrasyonsuz": bins})])

    first, second = fig.axes
    assert len(first.lines) == 3
    assert len(second.lines) == 2
    assert first.lines[0].get_linestyle() != first.lines[1].get_linestyle()
    plt.close(fig)


def test_plot_search_results_draws_one_point_per_configuration_and_marks_the_best():
    cv_results = pd.DataFrame(
        {
            "param_model__learning_rate": [0.01, 0.05, 0.2, 0.1],
            "param_model__max_depth": [3, 5, 7, 4],
            "mean_test_score": [0.40, 0.45, 0.43, 0.44],
            "rank_test_score": [4, 1, 3, 2],
        }
    )

    fig = plot_search_results([("xgboost", "XGBoost", cv_results, ["model__learning_rate", "model__max_depth"])])

    assert len(fig.axes) == 2
    assert all(len(ax.collections) == 2 for ax in fig.axes)
    assert [len(collection.get_offsets()) for collection in fig.axes[0].collections] == [4, 1]
    plt.close(fig)
