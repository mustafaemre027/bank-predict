"""Model değerlendirme grafikleri: çapraz doğrulama out-of-fold tahminlerinden PR ve ROC eğrileri."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from src.eda_plots import COLOR_GRID, COLOR_MUTED, COLOR_NO, COLOR_REFERENCE, COLOR_TEXT, COLOR_YES
from src.formatting import format_number, format_percent
from src.modeling import LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST
from src.threshold import ThresholdChoice, threshold_curves

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
LOG_AXIS_MIN_RATIO = 50
GROUP_WIDTH = 0.8
IMPORTANCE_TOP_N = 10
# Koyu hücrede yazı beyaz olur, açık hücrede siyah kalır.
CONFUSION_TEXT_SWITCH = 0.5
# Eşik grafiğindeki eğriler model değil metrik olduğu için model renklerini kullanmaz. Eğriler ağırlık ve çizgi stiliyle ayrılır:
# optimize edilen F1 koyu ve kalın, Precision ve Recall gri tonunda (düz ve noktalı çizgi).
THRESHOLD_CURVE_STYLES = {
    "f1": (COLOR_TEXT, "-", 2.5, "F1-Score"),
    "precision": (COLOR_MUTED, "-", 1.6, "Precision"),
    "recall": (COLOR_MUTED, "-.", 1.6, "Recall"),
}


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=COLOR_TEXT)
    ax.grid(color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_pr_roc_curves(y_true: pd.Series | np.ndarray, oof_proba: np.ndarray, fold_ids: np.ndarray | None, model_label: str) -> Figure:
    """Her CV katmanının eğrisini ince, tüm tahminlerden çıkan eğriyi kalın çizer.

    fold_ids None ise (ör. test seti) tek eğri çizilir ve katman açıklaması eklenmez.

    Referans model herkese aynı olasılığı verdiği için PR'de pozitif oran düzeyinde yatay, ROC'ta köşegen bir çizgidir.
    """
    y_array = np.asarray(y_true)
    positive_rate = float(y_array.mean())
    fig, (pr_ax, roc_ax) = plt.subplots(1, 2, figsize=(9, 4))
    for fold in np.unique(fold_ids) if fold_ids is not None else []:
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
    handles, labels = pr_ax.get_legend_handles_labels()
    if fold_ids is not None:
        fold_handle = plt.Line2D([], [], color=COLOR_NO, alpha=FOLD_LINE_ALPHA, linewidth=FOLD_LINE_WIDTH, label="Tek tek CV katmanları")
        handles, labels = [*handles, fold_handle], [*labels, fold_handle.get_label()]
    fig.legend(handles, labels, loc="upper center", ncols=3)
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


def plot_strategy_comparison(
    panels: list[tuple[str, dict[str, dict[str, dict[str, float]]]]],
    model_labels: dict[str, str],
    strategy_labels: dict[str, str],
) -> Figure:
    """Her panelde stratejiler gruplanır, gruptaki çubuklar modelleri (sabit renkle) gösterir.

    panels: (başlık, {model: {strateji: {"mean", "std"}}}) listesi.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4))
    for ax, (title, values) in zip(np.atleast_1d(axes), panels):
        models = list(values)
        strategies = list(values[models[0]])
        bar_width = GROUP_WIDTH / len(models)
        peak = 0.0
        for model_index, model in enumerate(models):
            positions = np.arange(len(strategies)) + (model_index - (len(models) - 1) / 2) * bar_width
            means = [values[model][strategy]["mean"] for strategy in strategies]
            stds = [values[model][strategy]["std"] for strategy in strategies]
            ax.bar(positions, means, width=bar_width * 0.9, color=MODEL_COLORS[model], label=model_labels[model], zorder=2)
            ax.errorbar(positions, means, yerr=stds, fmt="none", ecolor=COLOR_TEXT, elinewidth=1, capsize=3, zorder=3)
            peak = max(peak, *(mean + std for mean, std in zip(means, stds)))
        ax.set_xticks(np.arange(len(strategies)), [strategy_labels[strategy] for strategy in strategies])
        ax.set_ylim(0, peak * HEADROOM_RATIO)
        ax.set_title(title, loc="left", color=COLOR_TEXT)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_threshold_curves(
    y_true: pd.Series | np.ndarray,
    oof_proba: np.ndarray,
    f1_choice: ThresholdChoice,
    recall_choice: ThresholdChoice,
    title: str,
) -> Figure:
    """OOF tahminlerinde eşiğe göre Precision, Recall ve F1 eğrileri; seçilen iki eşik dikey çizgiyle işaretlenir."""
    curves = threshold_curves(y_true, oof_proba)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for key in THRESHOLD_CURVE_STYLES:
        color, linestyle, linewidth, label = THRESHOLD_CURVE_STYLES[key]
        ax.plot(curves.thresholds, getattr(curves, key), color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    # İki eşik birbirine yakın olabildiğinden soldaki etiket çizginin soluna, sağdaki sağına yaslanır ve etiketler üst üste binmez.
    marked = sorted([(f1_choice, "F1 en yüksek"), (recall_choice, "Hedef Recall")], key=lambda item: item[0].threshold)
    for (choice, label), alignment in zip(marked, ("right", "left")):
        ax.axvline(choice.threshold, color=COLOR_REFERENCE, linestyle="--", linewidth=1)
        ax.text(choice.threshold, 1.015, f"{label}: {format_number(choice.threshold, 2)}", ha=alignment, va="bottom", color=COLOR_TEXT, fontsize=8)
    _style_axes(ax, title, "Karar eşiği", "Metrik değeri")
    ax.set_title(title, loc="left", color=COLOR_TEXT, pad=22)
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def plot_calibration_curves(panels: list[tuple[str, str, dict[str, list[dict[str, float]]]]]) -> Figure:
    """Kalibrasyon eğrileri: ortalama tahmin ve gerçek kabul oranı. Köşegen, mükemmel kalibrasyondur.

    panels: (başlık, model, {eğri adı: calibration_summary bins}) listesi. İlk eğri modelin renginde düz çizgi, diğerleri koyu kesikli
    çizgi olarak çizilir, bu yüzden renge bakmadan da ayırt edilirler.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2))
    for ax, (title, model, curves) in zip(np.atleast_1d(axes), panels):
        peak = 0.0
        for index, (curve_name, bins) in enumerate(curves.items()):
            predicted = [entry["mean_predicted"] for entry in bins]
            observed = [entry["observed_rate"] for entry in bins]
            peak = max(peak, *predicted, *observed)
            style = {"color": MODEL_COLORS[model], "linestyle": "-", "marker": "o"} if index == 0 else {"color": COLOR_TEXT, "linestyle": "--", "marker": "s"}
            ax.plot(predicted, observed, linewidth=MAIN_LINE_WIDTH, markersize=4, label=curve_name, **style)
        ax.plot([0, peak], [0, peak], color=COLOR_REFERENCE, linestyle=":", linewidth=1, label="Mükemmel kalibrasyon")
        ax.set_xlim(0, peak * 1.05)
        ax.set_ylim(0, peak * 1.05)
        ax.set_xlabel("Ortalama tahmin edilen olasılık")
        ax.set_ylabel("Gerçek kabul oranı")
        ax.set_title(title, loc="left", color=COLOR_TEXT)
        ax.grid(color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_search_results(searches: list[tuple[str, str, pd.DataFrame, list[str]]]) -> Figure:
    """Her aday için en etkili parametrelerin değeri ile CV skoru: her nokta bir yapılandırma, en iyisi kenarlıklı.

    searches: (model, model adı, cv_results, parametre adları) listesi.
    """
    column_count = max(len(params) for *_, params in searches)
    fig, axes = plt.subplots(len(searches), column_count, figsize=(4.4 * column_count, 3.6 * len(searches)), squeeze=False)
    for row, (model, label, cv_results, params) in enumerate(searches):
        best = cv_results["rank_test_score"] == 1
        for column, param in enumerate(params):
            ax = axes[row][column]
            values = pd.to_numeric(cv_results[f"param_{param}"])
            ax.scatter(values, cv_results["mean_test_score"], color=MODEL_COLORS[model], s=28, zorder=3)
            ax.scatter(values[best], cv_results.loc[best, "mean_test_score"], color=MODEL_COLORS[model], edgecolor=COLOR_TEXT, linewidth=1.5, s=70, zorder=4)
            if values.min() > 0 and values.max() / values.min() >= LOG_AXIS_MIN_RATIO:
                ax.set_xscale("log")
            ax.set_xlabel(param.removeprefix("model__"))
            ax.set_ylabel("PR-AUC (CV)")
            ax.set_title(label if column == 0 else "", loc="left", color=COLOR_TEXT)
            ax.grid(color=COLOR_GRID, linewidth=0.6)
            ax.set_axisbelow(True)
        for column in range(len(params), column_count):
            axes[row][column].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(counts: dict[str, int]) -> Figure:
    """2x2 karmaşıklık matrisi: satırlar gerçek, sütunlar tahmin edilen sınıf. Hücreye sayı ve satır içi pay yazılır."""
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    row_shares = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(row_shares, cmap=LinearSegmentedColormap.from_list("share", ["#ffffff", COLOR_NO]), vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            color = "#ffffff" if row_shares[row, column] > CONFUSION_TEXT_SWITCH else COLOR_TEXT
            ax.text(column, row, f"{format_number(matrix[row, column])}\n{format_percent(row_shares[row, column], 1)}", ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks([0, 1], ["Kabul etmez", "Kabul eder"])
    ax.set_yticks([0, 1], ["Kabul etmez", "Kabul eder"])
    ax.set_xlabel("Tahmin edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title("Karmaşıklık matrisi (test)", loc="left", color=COLOR_TEXT)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return fig


def plot_gain_curve(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> Figure:
    """Kümülatif kazanç eğrisi: müşteriler olasılığa göre sıralanıp arandıkça yakalanan kabul edenlerin payı. Rastgele arama köşegendir."""
    labels = np.asarray(y_true)
    sorted_labels = labels[np.argsort(-np.asarray(proba), kind="stable")]
    contacted_share = np.arange(len(labels) + 1) / len(labels)
    captured_share = np.concatenate([[0.0], np.cumsum(sorted_labels) / labels.sum()])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(contacted_share, captured_share, color=COLOR_YES, linewidth=MAIN_LINE_WIDTH, label="Model")
    ax.plot([0, 1], [0, 1], color=COLOR_REFERENCE, linestyle="--", linewidth=1, label="Rastgele arama")
    _style_axes(ax, "Kümülatif kazanç eğrisi (test)", "Aranan müşteri payı", "Yakalanan kabul eden payı")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _horizontal_bars(ax: plt.Axes, names: list[str], values: np.ndarray, colors: list[str], errors: np.ndarray | None = None) -> None:
    positions = np.arange(len(names))
    ax.barh(positions, values, color=colors, height=0.7, xerr=errors, ecolor=COLOR_TEXT, error_kw={"elinewidth": 1, "capsize": 3}, zorder=2)
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_permutation_importance(table: pd.DataFrame, top_n: int = IMPORTANCE_TOP_N) -> Figure:
    """İlk `top_n` değişkenin permutation importance değeri (PR-AUC düşüşü); çizgi tekrarlar arası standart sapmadır."""
    top = table.head(top_n)
    fig, ax = plt.subplots(figsize=(6, 0.42 * len(top) + 1.2))
    _horizontal_bars(ax, list(top["feature"]), top["importance_mean"].to_numpy(), [COLOR_YES] * len(top), top["importance_std"].to_numpy())
    ax.set_xlabel("PR-AUC'deki düşüş (değişken karıştırılınca)")
    ax.set_title(f"Permutation importance, ilk {len(top)} değişken (test)", loc="left", color=COLOR_TEXT)
    fig.tight_layout()
    return fig


def plot_lr_coefficients(table: pd.DataFrame, top_n: int = IMPORTANCE_TOP_N) -> Figure:
    """Mutlak değere göre ilk `top_n` LR katsayısı. Kabul olasılığını artıranlar turuncu, azaltanlar mavi."""
    top = table.head(top_n)
    colors = [COLOR_YES if coefficient > 0 else COLOR_NO for coefficient in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(top) + 1.8))
    # Dönüştürücü öneki (nominal__, month__ ...) okumayı zorlaştırdığı için etiketten atılır.
    _horizontal_bars(ax, [name.split("__", 1)[-1] for name in top["feature"]], top["coefficient"].to_numpy(), colors)
    ax.axvline(0, color=COLOR_REFERENCE, linewidth=1)
    ax.set_xlabel("Katsayı (log-odds)")
    ax.set_title(f"Lojistik Regresyon katsayıları, ilk {len(top)} özellik", loc="left", color=COLOR_TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_YES), plt.Rectangle((0, 0), 1, 1, color=COLOR_NO)]
    ax.legend(handles, ["Kabul olasılığını artırır", "Kabul olasılığını azaltır"], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2, fontsize=8)
    fig.tight_layout()
    return fig


__all__ = [
    "MODEL_COLORS",
    "plot_calibration_curves",
    "plot_confusion_matrix",
    "plot_gain_curve",
    "plot_lr_coefficients",
    "plot_model_comparison",
    "plot_permutation_importance",
    "plot_pr_curves_overlay",
    "plot_pr_roc_curves",
    "plot_search_results",
    "plot_strategy_comparison",
    "plot_threshold_curves",
]
