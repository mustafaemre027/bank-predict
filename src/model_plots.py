"""Model değerlendirme grafikleri: çapraz doğrulama out-of-fold tahminlerinden PR ve ROC eğrileri."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from src.eda_plots import COLOR_GRID, COLOR_NO, COLOR_REFERENCE, COLOR_TEXT

FOLD_LINE_ALPHA = 0.35
FOLD_LINE_WIDTH = 0.8
MAIN_LINE_WIDTH = 2


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


__all__ = ["plot_pr_roc_curves"]
