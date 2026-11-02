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


def _precision_recall_by_threshold(y_true: np.ndarray, oof_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Artan eşik sırasıyla (eşik, precision, recall) dizileri; eğrinin eşiği olmayan son noktası atılır."""
    if len(y_true) != len(oof_proba):
        raise ValueError(f"Etiket ve olasılık sayıları farklı: {len(y_true)} ve {len(oof_proba)}.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Eşik seçimi için etiketlerde iki sınıf da bulunmalı.")
    precision, recall, thresholds = precision_recall_curve(y_true, oof_proba)
    return thresholds, precision[:-1], recall[:-1]


def _f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    total = precision + recall
    return np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)


def _choice(index: int, thresholds: np.ndarray, precision: np.ndarray, recall: np.ndarray, criterion: str) -> ThresholdChoice:
    f1 = _f1(precision, recall)
    return ThresholdChoice(float(thresholds[index]), float(precision[index]), float(recall[index]), float(f1[index]), criterion)


def select_threshold_max_f1(y_true: np.ndarray, oof_proba: np.ndarray) -> ThresholdChoice:
    """F1'i en büyük yapan eşik. Eşitlikte daha yüksek eşik seçilir, böylece daha az müşteri aranır."""
    thresholds, precision, recall = _precision_recall_by_threshold(np.asarray(y_true), np.asarray(oof_proba))
    f1 = _f1(precision, recall)
    best_indices = np.flatnonzero(f1 >= f1.max() - TIE_TOLERANCE)
    return _choice(int(best_indices[-1]), thresholds, precision, recall, CRITERION_MAX_F1)


def select_threshold_for_recall(y_true: np.ndarray, oof_proba: np.ndarray, target_recall: float = TARGET_RECALL) -> ThresholdChoice:
    """Recall'u hedefe ulaştıran en yüksek eşik. En düşük eşikte Recall 1 olduğundan hedef 1'i aşmadıkça her zaman bulunur."""
    if not 0 < target_recall <= 1:
        raise ValueError(f"Hedef Recall 0 ile 1 arasında olmalı, {target_recall} verildi.")
    thresholds, precision, recall = _precision_recall_by_threshold(np.asarray(y_true), np.asarray(oof_proba))
    reaching_target = np.flatnonzero(recall >= target_recall)
    return _choice(int(reaching_target[-1]), thresholds, precision, recall, CRITERION_TARGET_RECALL)


__all__ = [
    "CRITERION_MAX_F1",
    "CRITERION_TARGET_RECALL",
    "ThresholdChoice",
    "select_threshold_for_recall",
    "select_threshold_max_f1",
]
