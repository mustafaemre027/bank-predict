from __future__ import annotations

from typing import Sequence

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
