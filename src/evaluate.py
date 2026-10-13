from __future__ import annotations

from typing import Sequence

import numpy as np
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
