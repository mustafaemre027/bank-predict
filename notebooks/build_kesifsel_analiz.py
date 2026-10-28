"""Hafta 3 keşifsel veri analizini çalıştırır: özet JSON'u, rapor grafiklerini ve notebook'u üretir.

Kullanım: python notebooks/build_kesifsel_analiz.py
Rapora giren her sayı bu betiğin ürettiği reports/haftalik/hafta3_ozet.json dosyasından gelir.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AGE_BINS, AGE_LABELS, CATEGORICAL_FEATURES, MONTH_ORDER, NOT_PREVIOUSLY_CONTACTED, NUMERIC_FEATURES, TARGET
from src.data_loader import load_raw, validate_raw
from src.eda import (
    POSITIVE_LABEL,
    acceptance_rate_by,
    chi_square_tests,
    iqr_outlier_summary,
    mann_whitney_tests,
    numeric_summary,
)
from src.eda_plots import (
    plot_acceptance_rates,
    plot_correlation_heatmap,
    plot_effect_sizes,
    plot_numeric_distributions,
    plot_rate_by_bins,
    save_figure,
)

NOTEBOOK_PATH = ROOT / "notebooks" / "02_kesifsel_analiz.ipynb"
FIGURES_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta3_ozet.json"

# Uç aralıklar açık uçludur; böylece beklenmeyen bir değer hiçbir aralığın dışında kalıp sessizce kaybolmaz.
CAMPAIGN_BINS = [-np.inf, 1, 2, 3, 4, 5, 10, np.inf]
CAMPAIGN_LABELS = ["1", "2", "3", "4", "5", "6-10", "11+"]
OUTLIER_COLUMNS = ["age", "balance", "campaign", "previous"]
CONTACTED_ONLY_COLUMNS = ["pdays", "previous"]
DISTRIBUTION_COLUMNS = ["age", "balance", "day", "campaign"]
DISTRIBUTION_TAIL_QUANTILE = 0.01
STRONG_CORRELATION = 0.5
DECIMALS = 4

RATE_TITLES = {
    "job": "Mesleğe göre kabul oranı",
    "month": "Son iletişim ayına göre kabul oranı",
    "poutcome": "Önceki kampanya sonucuna göre kabul oranı",
    "contact": "İletişim türüne göre kabul oranı",
    "education": "Eğitim düzeyine göre kabul oranı",
}


def table_to_dict(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {str(index): row for index, row in table.round(DECIMALS).to_dict(orient="index").items()}


def rate_by_bins(df: pd.DataFrame, column: str, bins: list[float], labels: list[str]) -> pd.DataFrame:
    binned = df.assign(**{column: pd.cut(df[column], bins=bins, labels=labels).astype(str)})
    table = acceptance_rate_by(binned, column).reindex(labels)
    empty_bins = table.index[table["count"].isna()].tolist()
    if empty_bins:
        raise ValueError(f"'{column}' için boş aralıklar var: {empty_bins}. Aralık sınırlarını gözden geçirin.")
    return table


def outlier_acceptance(df: pd.DataFrame, outliers: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Aykırı değerlerin silinip silinmeyeceğine karar vermek için aykırı/normal gruplarda kabul oranını karşılaştırır."""
    is_positive = df[TARGET] == POSITIVE_LABEL
    result: dict[str, dict[str, float]] = {}
    for column in columns:
        bounds = outliers.loc[column]
        is_outlier = (df[column] < bounds["lower_bound"]) | (df[column] > bounds["upper_bound"])
        result[column] = {
            "outlier_rate": round(float(is_positive[is_outlier].mean()), DECIMALS),
            "inlier_rate": round(float(is_positive[~is_outlier].mean()), DECIMALS),
        }
    return result


def strong_correlations(correlation: pd.DataFrame) -> list[dict[str, float | str]]:
    """Bağımsız değişkenler arasındaki güçlü (|r| >= eşik) ilişkileri, mutlak değere göre azalan sırada döndürür."""
    pairs: list[dict[str, float | str]] = []
    columns = [column for column in correlation.columns if column != TARGET]
    for row_index, first in enumerate(columns):
        for second in columns[row_index + 1:]:
            value = float(correlation.loc[first, second])
            if abs(value) >= STRONG_CORRELATION:
                pairs.append({"first": first, "second": second, "spearman": round(value, DECIMALS)})
    return sorted(pairs, key=lambda pair: abs(float(pair["spearman"])), reverse=True)


def target_correlation_ranking(correlation: pd.DataFrame) -> list[dict[str, float | str]]:
    """Değişkenleri hedefle Spearman korelasyonunun mutlak değerine göre azalan sırada döndürür."""
    with_target = correlation[TARGET].drop(TARGET)
    ordered = with_target.reindex(with_target.abs().sort_values(ascending=False).index)
    return [{"column": column, "spearman": round(float(value), DECIMALS)} for column, value in ordered.items()]


def build_summary(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    is_positive = df[TARGET] == POSITIVE_LABEL
    overall_rate = float(is_positive.mean())
    was_contacted = df["pdays"] != NOT_PREVIOUSLY_CONTACTED

    rate_tables = {column: acceptance_rate_by(df, column) for column in CATEGORICAL_FEATURES}
    rate_tables["month"] = rate_tables["month"].reindex(MONTH_ORDER)
    age_rates = rate_by_bins(df, "age", AGE_BINS, AGE_LABELS)
    campaign_rates = rate_by_bins(df, "campaign", CAMPAIGN_BINS, CAMPAIGN_LABELS)

    outliers = iqr_outlier_summary(df, OUTLIER_COLUMNS)
    contacted_outliers = iqr_outlier_summary(df[was_contacted], CONTACTED_ONLY_COLUMNS)
    correlation = df[NUMERIC_FEATURES].assign(**{TARGET: is_positive.astype(int)}).corr(method="spearman")
    chi_square = chi_square_tests(df, CATEGORICAL_FEATURES)
    mann_whitney = mann_whitney_tests(df, NUMERIC_FEATURES)
    contact_unknown_by_month = (df["contact"] == "unknown").groupby(df["month"]).mean().reindex(MONTH_ORDER)

    summary = {
        "row_count": int(len(df)),
        "yes_count": int(is_positive.sum()),
        "overall_acceptance_rate": round(overall_rate, DECIMALS),
        "acceptance_rates": {column: table_to_dict(table) for column, table in rate_tables.items()},
        "age_bin_rates": table_to_dict(age_rates),
        "campaign_bin_rates": table_to_dict(campaign_rates),
        "previously_contacted_rates": {
            "contacted": round(float(is_positive[was_contacted].mean()), DECIMALS),
            "not_contacted": round(float(is_positive[~was_contacted].mean()), DECIMALS),
            "contacted_count": int(was_contacted.sum()),
        },
        "contact_unknown_share_by_month": {month: round(float(share), DECIMALS) for month, share in contact_unknown_by_month.items()},
        "numeric_summary": table_to_dict(numeric_summary(df, NUMERIC_FEATURES)),
        "iqr_outliers": table_to_dict(outliers),
        "iqr_outliers_contacted_only": table_to_dict(contacted_outliers),
        "outlier_acceptance": outlier_acceptance(df, outliers, ["balance", "campaign"]),
        "spearman_correlation": table_to_dict(correlation),
        "strong_correlations": strong_correlations(correlation),
        "target_correlation_ranking": target_correlation_ranking(correlation),
        "chi_square": table_to_dict(chi_square),
        "mann_whitney": table_to_dict(mann_whitney),
    }
    tables = {
        "rates": rate_tables,
        "age": age_rates,
        "campaign": campaign_rates,
        "correlation": correlation,
        "chi_square": chi_square,
        "mann_whitney": mann_whitney,
    }
    return summary, tables


def save_figures(df: pd.DataFrame, overall_rate: float, tables: dict[str, Any]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rates = tables["rates"]
    save_figure(
        plot_acceptance_rates({"job": rates["job"]}, overall_rate, RATE_TITLES),
        FIGURES_DIR / "h3_kabul_meslek.png",
    )
    save_figure(
        plot_acceptance_rates({"month": rates["month"]}, overall_rate, RATE_TITLES),
        FIGURES_DIR / "h3_kabul_ay.png",
    )
    save_figure(
        plot_acceptance_rates({column: rates[column] for column in ["poutcome", "contact", "education"]}, overall_rate, RATE_TITLES),
        FIGURES_DIR / "h3_kabul_poutcome_contact_education.png",
    )
    save_figure(
        plot_rate_by_bins(tables["age"], overall_rate, "Yaş grubuna göre kabul oranı", "Yaş grubu"),
        FIGURES_DIR / "h3_kabul_yas.png",
    )
    save_figure(
        plot_rate_by_bins(tables["campaign"], overall_rate, "Bu kampanyadaki arama sayısına göre kabul oranı", "Arama sayısı (campaign)"),
        FIGURES_DIR / "h3_kabul_campaign.png",
    )
    save_figure(
        plot_numeric_distributions(df, DISTRIBUTION_COLUMNS, DISTRIBUTION_TAIL_QUANTILE),
        FIGURES_DIR / "h3_sayisal_dagilimlar.png",
    )
    save_figure(
        plot_correlation_heatmap(tables["correlation"], "Sayısal değişkenler ve hedef: Spearman korelasyonu"),
        FIGURES_DIR / "h3_korelasyon.png",
    )
    save_figure(
        plot_effect_sizes(tables["chi_square"]["cramers_v"], tables["mann_whitney"]["rank_biserial"]),
        FIGURES_DIR / "h3_etki_buyuklukleri.png",
    )


def create_notebook() -> None:
    # Hücrelerdeki sütun listeleri betikteki sabitlerden üretilir; notebook ile özet JSON aynı tanımları kullanır.
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Hafta 3 — Keşifsel Veri Analizi\n\n"
            "Hesaplamalar `src/eda.py`, grafikler `src/eda_plots.py` içindedir. "
            "Rapordaki sayıların kaynağı `notebooks/build_kesifsel_analiz.py` ile üretilen `hafta3_ozet.json` dosyasıdır."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            "import pandas as pd\n"
            "from src.config import CATEGORICAL_FEATURES, NOT_PREVIOUSLY_CONTACTED, NUMERIC_FEATURES, TARGET\n"
            "from src.data_loader import load_raw\n"
            "from src.eda import POSITIVE_LABEL, acceptance_rate_by, chi_square_tests, iqr_outlier_summary, mann_whitney_tests, numeric_summary\n"
            "from src.eda_plots import plot_acceptance_rates, plot_correlation_heatmap\n"
            "\n"
            "df = load_raw()\n"
            "is_positive = df[TARGET] == POSITIVE_LABEL\n"
            "overall_rate = is_positive.mean()\n"
            "print(df.shape, f'genel kabul oranı: {overall_rate:.4f}')"
        ),
        nbf.v4.new_markdown_cell("## 1. Kategorik değişkenlere göre kabul oranları"),
        nbf.v4.new_code_cell(
            f"for column in {list(RATE_TITLES)!r}:\n"
            "    display(acceptance_rate_by(df, column).round(4))"
        ),
        nbf.v4.new_code_cell(
            "titles = {'poutcome': 'Önceki kampanya sonucu', 'contact': 'İletişim türü'}\n"
            "plot_acceptance_rates({c: acceptance_rate_by(df, c) for c in titles}, overall_rate, titles);"
        ),
        nbf.v4.new_markdown_cell("## 2. Sayısal değişkenler: dağılım ve IQR aykırı değerleri"),
        nbf.v4.new_code_cell(
            "display(numeric_summary(df, NUMERIC_FEATURES).round(2))\n"
            f"display(iqr_outlier_summary(df, {OUTLIER_COLUMNS!r}).round(2))\n"
            f"display(iqr_outlier_summary(df[df['pdays'] != NOT_PREVIOUSLY_CONTACTED], {CONTACTED_ONLY_COLUMNS!r}).round(2))"
        ),
        nbf.v4.new_markdown_cell("## 3. Spearman korelasyonu"),
        nbf.v4.new_code_cell(
            "correlation = df[NUMERIC_FEATURES].assign(**{TARGET: is_positive.astype(int)}).corr(method='spearman')\n"
            "plot_correlation_heatmap(correlation, 'Spearman korelasyonu');"
        ),
        nbf.v4.new_markdown_cell("## 4. İstatistiksel testler"),
        nbf.v4.new_code_cell(
            "display(chi_square_tests(df, CATEGORICAL_FEATURES).round(4))\n"
            "display(mann_whitney_tests(df, NUMERIC_FEATURES).round(4))"
        ),
    ]
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as file:
        nbf.write(nb, file)


def execute_notebook() -> None:
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NOTEBOOK_PATH)],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    df = load_raw()
    validate_raw(df)
    summary, tables = build_summary(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_figures(df, summary["overall_acceptance_rate"], tables)
    create_notebook()
    execute_notebook()
    print(f"Özet: {SUMMARY_PATH}\nGrafikler: {FIGURES_DIR}\nNotebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
