from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src.config import TARGET
from src.eda import POSITIVE_LABEL
from src.formatting import format_number, format_percent

# Renkler, 2. hafta grafikleriyle tutarlı olacak şekilde renk körlüğü kontrolünden geçmiş paletten seçildi.
COLOR_NO = "#2a78d6"
COLOR_YES = "#eb6834"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_REFERENCE = "#898781"
DIVERGING_COLORS = ["#2a78d6", "#f0efec", "#e34948"]
FIGURE_DPI = 150
PERCENT = 100
SKEWNESS_DISPLAY_LIMIT = 1.0
MAX_INTEGER_BINS = 40
DEFAULT_BIN_COUNT = 30
LABEL_OFFSET_RATIO = 0.012

plt.rcParams.update(
    {
        "axes.edgecolor": COLOR_GRID,
        "axes.labelcolor": COLOR_MUTED,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLOR_TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def _reference_note(overall_rate: float) -> str:
    return f"Kesikli çizgi: genel kabul oranı ({format_percent(overall_rate)})"


def _draw_rate_bars(ax: plt.Axes, table: pd.DataFrame, overall_rate: float, title: str) -> None:
    rates = table["acceptance_rate"] * PERCENT
    positions = np.arange(len(table))
    ax.barh(positions, rates, color=COLOR_YES, height=0.7)
    ax.set_yticks(positions, [f"{label}  (n={format_number(count)})" for label, count in table["count"].items()])
    ax.invert_yaxis()
    ax.axvline(overall_rate * PERCENT, color=COLOR_REFERENCE, linestyle="--", linewidth=1)
    for position, (rate, ratio) in enumerate(zip(rates, table["acceptance_rate"])):
        ax.text(rate + rates.max() * LABEL_OFFSET_RATIO, position, format_percent(ratio, 1), va="center", color=COLOR_TEXT, fontsize=8)
    ax.set_xlim(0, rates.max() * 1.18)
    ax.set_xlabel("Kabul oranı (%)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_acceptance_rates(tables: dict[str, pd.DataFrame], overall_rate: float, titles: dict[str, str]) -> Figure:
    """Her değişken için kategori bazlı kabul oranlarını, genel oran referans çizgisiyle yan yana çizer."""
    heights = [max(2.2, 0.32 * len(table) + 0.9) for table in tables.values()]
    fig, axes = plt.subplots(len(tables), 1, figsize=(7.5, sum(heights)), gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    for ax, (column, table) in zip(axes, tables.items()):
        _draw_rate_bars(ax, table, overall_rate, titles[column])
    fig.text(0.99, 0.002, _reference_note(overall_rate), ha="right", va="bottom", color=COLOR_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def _visible_range(values: pd.Series, tail_quantile: float) -> tuple[float, float]:
    # Yalnızca çok çarpık değişkenlerde uç kuyruk görselden çıkarılır; analizler tüm veriyle yapılır.
    if abs(values.skew()) <= SKEWNESS_DISPLAY_LIMIT:
        return float(values.min()), float(values.max())
    return float(values.quantile(tail_quantile)), float(values.quantile(1 - tail_quantile))


def _histogram_bins(values: pd.Series, lower: float, upper: float) -> np.ndarray:
    if pd.api.types.is_integer_dtype(values) and upper - lower <= MAX_INTEGER_BINS:
        # Kantil sınırları ondalıklı çıkabilir; kutular tamsayıların tam ortasına gelsin diye sınırlar yuvarlanır.
        return np.arange(np.floor(lower) - 0.5, np.ceil(upper) + 1.5)
    return np.linspace(lower, upper, DEFAULT_BIN_COUNT + 1)


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str], tail_quantile: float, target: str = TARGET) -> Figure:
    """Sayısal değişkenlerin kabul eden/etmeyen gruplardaki dağılımlarını normalize çizgi histogramlarla karşılaştırır."""
    fig, axes = plt.subplots(2, int(np.ceil(len(columns) / 2)), figsize=(9, 5.5))
    is_positive = df[target] == POSITIVE_LABEL
    for ax, column in zip(axes.flat, columns):
        lower, upper = _visible_range(df[column], tail_quantile)
        bins = _histogram_bins(df[column], lower, upper)
        in_range = df[column].between(bins[0], bins[-1])
        for mask, color, label in [(~is_positive, COLOR_NO, "Kabul etmedi (no)"), (is_positive, COLOR_YES, "Kabul etti (yes)")]:
            ax.hist(df.loc[mask & in_range, column], bins=bins, density=True, histtype="step", linewidth=2, color=color, label=label)
        ax.set_title(column, loc="left")
        ax.set_yticks([])
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_rate_by_bins(table: pd.DataFrame, overall_rate: float, title: str, xlabel: str) -> Figure:
    """Sayısal bir değişkenin aralıklarına göre kabul oranını sütun grafikle gösterir."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    rates = table["acceptance_rate"] * PERCENT
    positions = np.arange(len(table))
    ax.bar(positions, rates, color=COLOR_YES, width=0.7)
    ax.set_xticks(positions, [str(label) for label in table.index])
    ax.axhline(overall_rate * PERCENT, color=COLOR_REFERENCE, linestyle="--", linewidth=1)
    for position, (rate, ratio) in enumerate(zip(rates, table["acceptance_rate"])):
        ax.text(
            position, rate + rates.max() * LABEL_OFFSET_RATIO, format_percent(ratio, 1),
            ha="center", va="bottom", color=COLOR_TEXT, fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    ax.set_ylim(0, rates.max() * 1.18)
    ax.set_ylabel("Kabul oranı (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.99, 0.002, _reference_note(overall_rate), ha="right", va="bottom", color=COLOR_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_correlation_heatmap(correlation: pd.DataFrame, title: str) -> Figure:
    """Korelasyon matrisini mavi-gri-kırmızı ıraksak renk ölçeğiyle, hücre değerleri yazılı olarak çizer."""
    colormap = LinearSegmentedColormap.from_list("diverging", DIVERGING_COLORS)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    image = ax.imshow(correlation.to_numpy(), cmap=colormap, vmin=-1, vmax=1)
    labels = list(correlation.columns)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = correlation.iat[row, col]
            ax.text(col, row, format_number(value, 2), ha="center", va="center", fontsize=8, color=COLOR_TEXT)
    ax.spines[:].set_visible(False)
    ax.set_title(title, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_effect_sizes(cramers_v: pd.Series, rank_biserial: pd.Series) -> Figure:
    """Kategorik (Cramér's V) ve sayısal (|rank-biserial r|) etki büyüklüklerini iki ayrı panelde sıralı gösterir."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
    panels = [
        (left, cramers_v.sort_values(), "Kategorik değişkenler: Cramér's V"),
        (right, rank_biserial.abs().sort_values(), "Sayısal değişkenler: |rank-biserial r|"),
    ]
    for ax, values, title in panels:
        ax.barh(values.index, values.to_numpy(), color=COLOR_NO, height=0.65)
        for position, value in enumerate(values.to_numpy()):
            ax.text(value + values.max() * LABEL_OFFSET_RATIO, position, format_number(value, 3), va="center", color=COLOR_TEXT, fontsize=8)
        ax.set_xlim(0, values.max() * 1.25)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


__all__ = [
    "plot_acceptance_rates",
    "plot_correlation_heatmap",
    "plot_effect_sizes",
    "plot_numeric_distributions",
    "plot_rate_by_bins",
    "save_figure",
]
