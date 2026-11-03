"""Model değerlendirme grafikleri: çapraz doğrulama out-of-fold tahminlerinden PR ve ROC eğrileri."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from src.eda_plots import COLOR_GRID, COLOR_NO, COLOR_REFERENCE, COLOR_TEXT, COLOR_YES
from src.formatting import format_number
from src.modeling import LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST

FOLD_LINE_ALPHA = 0.35
FOLD_LINE_WIDTH = 0.8
MAIN_LINE_WIDTH = 2
BAR_WIDTH = 0.5
ERROR_BAR_CAPSIZE = 4
# Modelin rengi kimliğine bağlıdır, sırasına değil: hangi grafikte hangi model çizilirse çizilsin aynı kalır.
# Üçüncü renk (turkuaz) açık yüzeyde 3:1 kontrastın altında kaldığı için değerler çubuğun üstüne yazılır ve rapora tablo da girer.
MODEL_COLORS = {LOGISTIC_REGRESSION: COLOR_NO, RANDOM_FOREST: COLOR_YES, XGBOOST: "#1baf7a"}
COMPARISON_METRIC_TITLES = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}
HEADROOM_RATIO = 1.25


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=COLOR_TEXT)
    ax.grid(color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_pr_roc_curves(y_true: pd.Series | np.ndarray, oof_proba: np.ndarray, fold_ids: np.ndarray, model_label: str) -> Figure:
    """Her CV katmanının eğrisini ince, tüm OOF tahminlerinden çıkan eğriyi kalın çizer.

    Referans model herkese aynı olasılığı verdiği için PR'de pozitif oran düzeyinde yatay, ROC'ta köşegen bir çizgidir.
    """
    y_array = np.asarray(y_true)
    positive_rate = float(y_array.mean())
    fig, (pr_ax, roc_ax) = plt.subplots(1, 2, figsize=(9, 4))
    for fold in np.unique(fold_ids):
        in_fold = fold_ids == fold
        precision, recall, _ = precision_recall_curve(y_array[in_fold], oof_proba[in_fold])
        false_positive, true_positive, _ = roc_curve(y_array[in_fold], oof_proba[in_fold])
        fold_style = {"color": COLOR_NO, "alpha": FOLD_LINE_ALPHA, "linewidth": FOLD_LINE_WIDTH}
        pr_ax.plot(recall, precision, **fold_style)
        roc_ax.plot(false_positive, true_positive, **fold_style)

    precision, recall, _ = precision_recall_curve(y_array, oof_proba)
    false_positive, true_positive, _ = roc_curve(y_array, oof_proba)
    pr_ax.plot(recall, precision, color=COLOR_NO, linewidth=MAIN_LINE_WIDTH, label=model_label)
    roc_ax.plot(false_positive, true_positive, color=COLOR_NO, linewidth=MAIN_LINE_WIDTH, label=model_label)
    reference_style = {"color": COLOR_REFERENCE, "linestyle": "--", "linewidth": 1, "label": "Referans model"}
    pr_ax.axhline(positive_rate, **reference_style)
    roc_ax.plot([0, 1], [0, 1], **reference_style)

    _style_axes(pr_ax, "Precision-Recall eğrisi", "Recall", "Precision")
    _style_axes(roc_ax, "ROC eğrisi", "Yanlış pozitif oranı", "Doğru pozitif oranı (Recall)")
    fold_handle = plt.Line2D([], [], color=COLOR_NO, alpha=FOLD_LINE_ALPHA, linewidth=FOLD_LINE_WIDTH, label="Tek tek CV katmanları")
    handles, labels = pr_ax.get_legend_handles_labels()
    fig.legend([*handles, fold_handle], [*labels, fold_handle.get_label()], loc="upper center", ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_model_comparison(
    summary_by_model: dict[str, dict[str, dict[str, float]]],
    labels: dict[str, str],
    reference_summary: dict[str, dict[str, float]],
    metrics: tuple[str, ...] = ("pr_auc", "roc_auc"),
) -> Figure:
    """Modellerin CV ortalamasını çubukla, katmanlar arası standart sapmayı hata çubuğuyla gösterir.

    summary_by_model[model][metric] = {"mean", "std"}. Referans model her panelde kesikli yatay çizgidir.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    models = list(summary_by_model)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        means = [summary_by_model[model][metric]["mean"] for model in models]
        stds = [summary_by_model[model][metric]["std"] for model in models]
        positions = np.arange(len(models))
        ax.bar(positions, means, width=BAR_WIDTH, color=[MODEL_COLORS[model] for model in models], zorder=2)
        ax.errorbar(positions, means, yerr=stds, fmt="none", ecolor=COLOR_TEXT, elinewidth=1, capsize=ERROR_BAR_CAPSIZE, zorder=3)
        for position, mean, std in zip(positions, means, stds):
            ax.text(position, mean + std, format_number(mean, 3), ha="center", va="bottom", color=COLOR_TEXT, fontsize=9)
        ax.axhline(reference_summary[metric]["mean"], color=COLOR_REFERENCE, linestyle="--", linewidth=1, label="Referans model", zorder=1)
        ax.set_xticks(positions, [labels[model] for model in models])
        ax.set_ylim(0, max(mean + std for mean, std in zip(means, stds)) * HEADROOM_RATIO)
        ax.set_title(f"{COMPARISON_METRIC_TITLES[metric]} (5 katlı CV)", loc="left", color=COLOR_TEXT)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    handles, legend_labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_pr_curves_overlay(
    y_true: pd.Series | np.ndarray,
    oof_by_model: dict[str, np.ndarray],
    labels: dict[str, str],
) -> Figure:
    """Modellerin tüm OOF tahminlerinden çıkan PR eğrilerini tek grafikte çizer; referans çizgisi kabul oranıdır."""
    y_array = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for model, oof_proba in oof_by_model.items():
        precision, recall, _ = precision_recall_curve(y_array, oof_proba)
        ax.plot(recall, precision, color=MODEL_COLORS[model], linewidth=MAIN_LINE_WIDTH, label=labels[model])
    ax.axhline(float(y_array.mean()), color=COLOR_REFERENCE, linestyle="--", linewidth=1, label="Referans model")
    _style_axes(ax, "Precision-Recall eğrileri (OOF)", "Recall", "Precision")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


__all__ = ["MODEL_COLORS", "plot_model_comparison", "plot_pr_curves_overlay", "plot_pr_roc_curves"]
