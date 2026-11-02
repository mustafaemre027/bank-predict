"""Karar eşiğinin çapraz doğrulama dışı (OOF) tahminlerden seçilmesi.

Fonksiyonlar yalnızca gerçek etiket ve OOF olasılığı alır; test verisi parametre olarak bile verilemez (FG-9).
Bir müşteri olasılığı eşiğe eşit ya da eşiğin üstündeyse "kabul eder" sayılır.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config import TARGET_RECALL

CRITERION_MAX_F1 = "max_f1"
CRITERION_TARGET_RECALL = "target_recall"
# F1 değerleri kayan noktayla hesaplandığından eşitlik bu toleransla kontrol edilir.
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float
    f1: float
    criterion: str


@dataclass(frozen=True)
class ThresholdCurves:
    """Artan eşik sırasıyla her eşikteki Precision, Recall ve F1 değerleri."""

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray


def threshold_curves(y_true: np.ndarray, oof_proba: np.ndarray) -> ThresholdCurves:
    """Eşiğe göre Precision, Recall ve F1; eğrinin eşiği olmayan son noktası atılır."""
    y_array, proba = np.asarray(y_true), np.asarray(oof_proba)
    if len(y_array) != len(proba):
        raise ValueError(f"Etiket ve olasılık sayıları farklı: {len(y_array)} ve {len(proba)}.")
    if len(np.unique(y_array)) < 2:
        raise ValueError("Eşik seçimi için etiketlerde iki sınıf da bulunmalı.")
    precision, recall, thresholds = precision_recall_curve(y_array, proba)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return ThresholdCurves(thresholds, precision, recall, f1)


def _choice(index: int, curves: ThresholdCurves, criterion: str) -> ThresholdChoice:
    return ThresholdChoice(
        float(curves.thresholds[index]), float(curves.precision[index]), float(curves.recall[index]), float(curves.f1[index]), criterion
    )


def select_threshold_max_f1(y_true: np.ndarray, oof_proba: np.ndarray) -> ThresholdChoice:
    """F1'i en büyük yapan eşik. Eşitlikte daha yüksek eşik seçilir, böylece daha az müşteri aranır."""
    curves = threshold_curves(y_true, oof_proba)
    best_indices = np.flatnonzero(curves.f1 >= curves.f1.max() - TIE_TOLERANCE)
    return _choice(int(best_indices[-1]), curves, CRITERION_MAX_F1)


def select_threshold_for_recall(y_true: np.ndarray, oof_proba: np.ndarray, target_recall: float = TARGET_RECALL) -> ThresholdChoice:
    """Recall'u hedefe ulaştıran en yüksek eşik. En düşük eşikte Recall 1 olduğundan hedef 1'i aşmadıkça her zaman bulunur."""
    if not 0 < target_recall <= 1:
        raise ValueError(f"Hedef Recall 0 ile 1 arasında olmalı, {target_recall} verildi.")
    curves = threshold_curves(y_true, oof_proba)
    reaching_target = np.flatnonzero(curves.recall >= target_recall)
    return _choice(int(reaching_target[-1]), curves, CRITERION_TARGET_RECALL)


__all__ = [
    "CRITERION_MAX_F1",
    "CRITERION_TARGET_RECALL",
    "ThresholdChoice",
    "ThresholdCurves",
    "select_threshold_for_recall",
    "select_threshold_max_f1",
    "threshold_curves",
]
