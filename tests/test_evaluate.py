import numpy as np
import pytest

from src.evaluate import compute_metrics


def test_compute_metrics_basic_example():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.2, 0.1])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    expected = {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "roc_auc": 0.75,
        "pr_auc": 5 / 6,
    }

    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)
