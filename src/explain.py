"""Açıklanabilirlik ve hata analizi: permutation importance, Lojistik Regresyon katsayıları, yanlış negatif profili ve grup bazlı metrikler.

SHAP kullanılmaz. Genel önem için ham değişkenler üzerinde permutation importance (Fisher vd., 2019) ve ayrı bir LR modelinin
katsayıları kullanılır.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE

PERMUTATION_SCORING = "average_precision"
DEFAULT_REPEATS = 10
PERIOD_FEATURE = "period"
PEAK_PERIOD_MONTHS = ("may", "jun")
PEAK_PERIOD_LABEL = "mayıs-haziran"
OTHER_PERIOD_LABEL = "diğer aylar"
FALSE_NEGATIVE = "false_negative"
TRUE_POSITIVE = "true_positive"


def permutation_importance_table(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series | np.ndarray, n_repeats: int = DEFAULT_REPEATS) -> pd.DataFrame:
    """Her ham değişkenin sütunu karıştırıldığında PR-AUC'deki düşüş (ortalama ve std), büyükten küçüğe sıralı.

    Tahmin pipeline'ının tamamı üzerinden yapıldığı için sonuç One-Hot sütunları yerine ham değişken adlarıyla okunur.
    estimator fit edilmiş olmalıdır.
    """
    result = permutation_importance(
        estimator, X, y, scoring=PERMUTATION_SCORING, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1
    )
    table = pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
    return table.sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)


def logistic_coefficients(fitted_lr_pipeline: Pipeline) -> pd.DataFrame:
    """Fit edilmiş LR pipeline'ının katsayıları (özellik adı + katsayı), mutlak değere göre büyükten küçüğe sıralı.

    Sayısal değişkenler standartlaştırıldığı için katsayılar birbiriyle kıyaslanabilir. One-Hot sütunlarında referans kategori yoktur,
    bu yüzden katsayı o kategorinin ortalama etkiye göre konumunu gösterir.
    """
    names = fitted_lr_pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = fitted_lr_pipeline.named_steps["model"].coef_.ravel()
    if len(names) != len(coefficients):
        raise ValueError(f"Özellik adı sayısı ({len(names)}) katsayı sayısından ({len(coefficients)}) farklı.")
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    order = np.argsort(-np.abs(table["coefficient"].to_numpy()), kind="stable")
    return table.iloc[order].reset_index(drop=True)


# Üretilen değişkenin hangi ham değişkenden türediği; permutation importance ham değişkenlerde hesaplandığı için ikisi bu eşlemeyle karşılaştırılır.
DERIVED_FEATURE_SOURCES = {"age_group": "age", "was_contacted_before": "pdays"}


def source_variable(feature_name: str, raw_columns: Sequence[str]) -> str:
    """Ön işlemeden çıkan özellik adının (ör. `nominal__job_retired`) türediği ham değişken (`job`)."""
    stripped = feature_name.split("__", 1)[-1]
    candidates = [*raw_columns, *DERIVED_FEATURE_SOURCES]
    # Önek eşleşmesinde en uzun ad seçilir: `age_group_41-50`, `age` ile değil `age_group` ile eşleşmeli.
    matches = [name for name in candidates if stripped == name or stripped.startswith(f"{name}_")]
    if not matches:
        raise ValueError(f"Özellik adı hiçbir ham değişkenle eşleşmedi: {feature_name}.")
    longest = max(matches, key=len)
    return DERIVED_FEATURE_SOURCES.get(longest, longest)


def rank_variables_by_coefficient(coefficients: pd.DataFrame, raw_columns: Sequence[str]) -> pd.DataFrame:
    """LR katsayılarını ham değişkene toplar: değişkenin en büyük mutlak katsayısı alınır, büyükten küçüğe sıralanır.

    Toplam yerine en büyük değer alınır, çünkü çok kategorili bir değişkenin sütun sayısı katsayı toplamını yapay olarak büyütür.
    """
    ranked = coefficients.assign(
        variable=[source_variable(name, raw_columns) for name in coefficients["feature"]],
        strength=coefficients["coefficient"].abs(),
    )
    table = ranked.groupby("variable", as_index=False)["strength"].max()
    return table.sort_values("strength", ascending=False, kind="stable").reset_index(drop=True)


def top_overlap(first: Sequence[str], second: Sequence[str], top_n: int) -> list[str]:
    """İki sıralı listenin ilk `top_n` elemanında ortak olanlar (birinci listedeki sırayla)."""
    shared = set(second[:top_n])
    return [name for name in first[:top_n] if name in shared]


def error_profile(X: pd.DataFrame, y_true: pd.Series | np.ndarray, y_pred: np.ndarray, columns: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """Yanlış negatiflerin (kabul eden ama aranmayacak) ve doğru pozitiflerin seçili değişkenlerdeki dağılımı.

    Her değişken için kategori başına iki grubun içindeki payı ve sayısı verilir. Pay, grubun kendi toplamına bölünür, böylece
    iki grup farklı büyüklükte olsa da karşılaştırılabilir.
    """
    labels, predictions = np.asarray(y_true), np.asarray(y_pred)
    masks = {FALSE_NEGATIVE: (labels == 1) & (predictions == 0), TRUE_POSITIVE: (labels == 1) & (predictions == 1)}
    profile: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        counts = {group: X.loc[mask, column].astype(str).value_counts() for group, mask in masks.items()}
        categories = sorted(set(counts[FALSE_NEGATIVE].index) | set(counts[TRUE_POSITIVE].index))
        profile[column] = [
            {
                "category": category,
                **{
                    f"{group}_{key}": value
                    for group, group_counts in counts.items()
                    for key, value in (
                        ("count", int(group_counts.get(category, 0))),
                        ("share", float(group_counts.get(category, 0) / group_counts.sum()) if group_counts.sum() else 0.0),
                    )
                },
            }
            for category in categories
        ]
    return profile


def with_campaign_period(X: pd.DataFrame) -> pd.DataFrame:
    """Aranma ayına göre iki dönem sütunu ekler: kampanyanın yoğun olduğu mayıs-haziran ve diğer aylar."""
    in_peak = X["month"].isin(PEAK_PERIOD_MONTHS)
    return X.assign(**{PERIOD_FEATURE: np.where(in_peak, PEAK_PERIOD_LABEL, OTHER_PERIOD_LABEL)})


def metrics_by_group(X: pd.DataFrame, y_true: pd.Series | np.ndarray, proba: np.ndarray, column: str) -> pd.DataFrame:
    """Bir değişkenin her değeri için kayıt sayısı, kabul oranı ve PR-AUC. Tek sınıflı grupta PR-AUC tanımsızdır ve NaN yazılır."""
    labels = np.asarray(y_true)
    rows = []
    for group, indices in X.groupby(column, sort=True).indices.items():
        group_labels = labels[indices]
        has_both_classes = 0 < group_labels.sum() < len(group_labels)
        rows.append(
            {
                "group": str(group),
                "count": len(indices),
                "positive_rate": float(group_labels.mean()),
                "pr_auc": float(average_precision_score(group_labels, proba[indices])) if has_both_classes else float("nan"),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "FALSE_NEGATIVE",
    "OTHER_PERIOD_LABEL",
    "PEAK_PERIOD_LABEL",
    "PERIOD_FEATURE",
    "TRUE_POSITIVE",
    "rank_variables_by_coefficient",
    "source_variable",
    "error_profile",
    "logistic_coefficients",
    "metrics_by_group",
    "permutation_importance_table",
    "top_overlap",
    "with_campaign_period",
]
