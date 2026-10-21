from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import TARGET

POSITIVE_LABEL = "yes"
IQR_MULTIPLIER = 1.5
# Ki-kare testinin güvenilirliği için beklenen frekansların en az 5 olması önerilir (Cochran kuralı).
MIN_EXPECTED_FREQUENCY = 5


def acceptance_rate_by(df: pd.DataFrame, column: str, target: str = TARGET) -> pd.DataFrame:
    """Bir kategorik değişkenin her kategorisi için kayıt sayısı, kabul sayısı, kabul oranı ve lift değerini döndürür."""
    is_positive = df[target] == POSITIVE_LABEL
    table = is_positive.groupby(df[column]).agg(count="size", yes_count="sum")
    table["acceptance_rate"] = table["yes_count"] / table["count"]
    table["lift"] = table["acceptance_rate"] / is_positive.mean()
    return table.sort_values("acceptance_rate", ascending=False)


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sayısal değişkenlerin betimsel istatistiklerini çarpıklık katsayısıyla birlikte döndürür."""
    summary = df[columns].describe().T
    summary["skewness"] = df[columns].skew()
    return summary


def iqr_outlier_summary(
    df: pd.DataFrame, columns: list[str], multiplier: float = IQR_MULTIPLIER
) -> pd.DataFrame:
    """Her değişken için IQR yöntemiyle alt/üst sınırları ve aykırı değer sayısını/oranını döndürür."""
    rows = []
    for column in columns:
        values = df[column]
        first_quartile, third_quartile = values.quantile([0.25, 0.75])
        iqr = third_quartile - first_quartile
        lower_bound = first_quartile - multiplier * iqr
        upper_bound = third_quartile + multiplier * iqr
        is_outlier = (values < lower_bound) | (values > upper_bound)
        rows.append(
            {
                "column": column,
                "q1": float(first_quartile),
                "q3": float(third_quartile),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": int(is_outlier.sum()),
                "outlier_pct": float(is_outlier.mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def chi_square_tests(df: pd.DataFrame, columns: list[str], target: str = TARGET) -> pd.DataFrame:
    """Kategorik değişkenlerle hedef arasında ki-kare bağımsızlık testi ve Cramér's V etki büyüklüğünü hesaplar."""
    rows = []
    for column in columns:
        contingency = pd.crosstab(df[column], df[target])
        # Yates düzeltmesi kapalı: Cramér's V tüm tablo boyutlarında aynı istatistikle hesaplansın diye.
        result = stats.chi2_contingency(contingency, correction=False)
        sample_size = contingency.to_numpy().sum()
        min_dimension = min(contingency.shape) - 1
        rows.append(
            {
                "column": column,
                "chi2": float(result.statistic),
                "p_value": float(result.pvalue),
                "dof": int(result.dof),
                "cramers_v": float(np.sqrt(result.statistic / (sample_size * min_dimension))),
                "low_expected_cells": int((result.expected_freq < MIN_EXPECTED_FREQUENCY).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("column").sort_values("cramers_v", ascending=False)


def mann_whitney_tests(df: pd.DataFrame, columns: list[str], target: str = TARGET) -> pd.DataFrame:
    """Sayısal değişkenlerin kabul eden/etmeyen gruplarda farklı dağılıp dağılmadığını Mann-Whitney U ile test eder."""
    is_positive = df[target] == POSITIVE_LABEL
    rows = []
    for column in columns:
        positive_values = df.loc[is_positive, column]
        negative_values = df.loc[~is_positive, column]
        result = stats.mannwhitneyu(positive_values, negative_values, alternative="two-sided")
        # 45 bin kayıtta p-değeri neredeyse her farkta küçük çıkar; farkın büyüklüğünü rank-biserial korelasyon verir.
        rank_biserial = 2 * result.statistic / (len(positive_values) * len(negative_values)) - 1
        rows.append(
            {
                "column": column,
                "u_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "median_yes": float(positive_values.median()),
                "median_no": float(negative_values.median()),
                "rank_biserial": float(rank_biserial),
            }
        )
    table = pd.DataFrame(rows).set_index("column")
    return table.reindex(table["rank_biserial"].abs().sort_values(ascending=False).index)


__all__ = [
    "IQR_MULTIPLIER",
    "MIN_EXPECTED_FREQUENCY",
    "POSITIVE_LABEL",
    "acceptance_rate_by",
    "chi_square_tests",
    "iqr_outlier_summary",
    "mann_whitney_tests",
    "numeric_summary",
]
