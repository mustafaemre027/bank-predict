from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Özet JSON'a yazılan sayılar bu basamağa yuvarlanır; iki çalıştırmanın JSON'ları bu sayede birebir karşılaştırılabilir.
JSON_DECIMALS = 4
# Ablasyon varyantının varsayılan yapılmaya aday olması için iyileşmesi gereken katman payı (5 katmanın 4'ü).
MIN_IMPROVED_FOLD_SHARE = Fraction(4, 5)


def round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(value, JSON_DECIMALS) for name, value in metrics.items()}


def compute_metrics(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray, y_proba: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Compute classification metrics for binary targets."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_proba_array = np.asarray(y_proba)

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, y_proba_array)),
        "pr_auc": float(average_precision_score(y_true_array, y_proba_array)),
    }


def _require_multiple_folds(fold_count: int) -> None:
    # Örneklem standart sapması (ddof=1) tek katmanda tanımsızdır; NaN sessizce JSON'a ve rapora geçmesin.
    if fold_count < 2:
        raise ValueError(f"En az 2 katman gerekli, {fold_count} katman verildi.")


def summarize_cv(fold_metrics: list[dict[str, float]]) -> pd.DataFrame:
    """Katman metriklerini metrik başına ortalama ve standart sapma (örneklem, ddof=1) tablosuna toplar."""
    _require_multiple_folds(len(fold_metrics))
    table = pd.DataFrame(fold_metrics)
    return pd.DataFrame({"mean": table.mean(), "std": table.std(ddof=1)})


def compare_folds(base: list[dict[str, float]], variant: list[dict[str, float]], metric: str) -> dict[str, float]:
    """Aynı katmanlarda eğitilmiş iki modeli bir metrikte katman katman karşılaştırır (variant - base)."""
    if len(base) != len(variant):
        raise ValueError(f"Katman sayıları farklı: {len(base)} ve {len(variant)}. Modeller aynı CV düzeniyle eğitilmeli.")
    _require_multiple_folds(len(base))
    differences = np.array([after[metric] - before[metric] for before, after in zip(base, variant)])
    return {
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std(ddof=1)),
        "folds_improved": int((differences > 0).sum()),
        "fold_count": len(differences),
    }


def build_comparison_table(summaries: dict[str, pd.DataFrame], metrics: list[str]) -> pd.DataFrame:
    """Modelleri satır, (metrik, mean/std) çiftlerini sütun yapan karşılaştırma tablosu. Girdi summarize_cv çıktısıdır."""
    rows = {
        model: {(metric, statistic): float(summary.loc[metric, statistic]) for metric in metrics for statistic in ("mean", "std")}
        for model, summary in summaries.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize_for_json(fold_metrics: list[dict[str, float]]) -> dict[str, object]:
    """Bir modelin katman metriklerini ve ortalama ± std özetini özet JSON'a yazılacak biçimde yuvarlar."""
    return {
        "folds": [round_metrics(metrics) for metrics in fold_metrics],
        "summary": {metric: round_metrics(values) for metric, values in summarize_cv(fold_metrics).to_dict(orient="index").items()},
    }


def min_folds_improved_for_candidate(fold_count: int) -> int:
    return math.ceil(MIN_IMPROVED_FOLD_SHARE * fold_count)


def is_ablation_candidate(comparison: dict[str, float]) -> bool:
    """Bir varyantın varsayılanı değiştirmeye aday olması için ortalama artışı katmanlar arası std'yi aşmalı ve katmanların çoğunda iyileşme olmalı."""
    return (
        comparison["mean_difference"] > 0
        and comparison["mean_difference"] > comparison["std_difference"]
        and comparison["folds_improved"] >= min_folds_improved_for_candidate(comparison["fold_count"])
    )


def fold_metrics_at_threshold(
    y_true: Sequence[int] | np.ndarray,
    oof_proba: np.ndarray,
    fold_ids: np.ndarray,
    threshold: float,
) -> list[dict[str, float]]:
    """OOF olasılıklarından, her CV katmanının metriklerini verilen eşikte hesaplar (eşikten bağımsız ROC-AUC ve PR-AUC dahil)."""
    y_array = np.asarray(y_true)
    metrics = []
    for fold in np.unique(fold_ids):
        in_fold = fold_ids == fold
        proba = oof_proba[in_fold]
        metrics.append(compute_metrics(y_array[in_fold], (proba >= threshold).astype(int), proba))
    return metrics


def fold_composition_for_json(X: pd.DataFrame, y: pd.Series, cv: Any) -> list[dict[str, Any]]:
    """Katmanlamanın her katmanda hedef oranını koruduğunu göstermek için katman boyutları ve pozitif oranları."""
    return [
        {
            "train_rows": len(train_index),
            "validation_rows": len(validation_index),
            "validation_positive_rate": round(float(y.iloc[validation_index].mean()), JSON_DECIMALS),
        }
        for train_index, validation_index in cv.split(X, y)
    ]
