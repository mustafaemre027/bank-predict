"""Model pipeline'ları ve 5 katlı katmanlı çapraz doğrulama düzeni.

Model seçimi ve karşılaştırmalar yalnızca eğitim verisi içinde bu CV düzeniyle yapılır; test seti Hafta 8'e kadar kullanılmaz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_STATE
from src.evaluate import compute_metrics
from src.preprocessing import build_linear_feature_pipeline

# Karar eşiği Hafta 6'da OOF tahminlerinden seçilecek; o zamana kadar etiket bazlı metrikler varsayılan eşikle verilir.
DEFAULT_THRESHOLD = 0.5
# Ölçeklenmiş veride lbfgs varsayılan 100 iterasyonda yakınsamayabildiği için üst sınır yükseltildi.
LR_MAX_ITER = 1000


@dataclass(frozen=True)
class CrossValidationResult:
    fold_metrics: list[dict[str, float]]
    oof_proba: np.ndarray
    fold_ids: np.ndarray


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_baseline() -> DummyClassifier:
    """Hafta 2'deki referans model: herkese çoğunluk sınıfını ("kabul etmez") tahmin eder."""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression(include_age_group: bool = True, rare_month_min_frequency: float | None = None) -> Pipeline:
    """Ön işleme + Lojistik Regresyon. Varsayılan düzenlileştirme L2'dir (C = 1)."""
    feature_pipeline = build_linear_feature_pipeline(include_age_group, rare_month_min_frequency)
    return Pipeline([*feature_pipeline.steps, ("model", LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE))])


def cross_validate_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    threshold: float = DEFAULT_THRESHOLD,
) -> CrossValidationResult:
    """Her katmanda modelin yeni bir kopyasını yalnızca eğitim katmanına fit eder.

    Katman metriklerinin yanında her kaydın, kendisini görmemiş modelden aldığı out-of-fold (OOF) olasılığını döndürür.
    """
    oof_proba = np.zeros(len(y))
    fold_ids = np.zeros(len(y), dtype=int)
    fold_metrics: list[dict[str, float]] = []
    for fold, (train_index, validation_index) in enumerate(cv.split(X, y)):
        model = clone(estimator).fit(X.iloc[train_index], y.iloc[train_index])
        proba = model.predict_proba(X.iloc[validation_index])[:, 1]
        y_validation = y.iloc[validation_index]
        fold_metrics.append(compute_metrics(y_validation, (proba >= threshold).astype(int), proba))
        oof_proba[validation_index] = proba
        fold_ids[validation_index] = fold
    return CrossValidationResult(fold_metrics, oof_proba, fold_ids)


__all__ = [
    "DEFAULT_THRESHOLD",
    "CrossValidationResult",
    "build_baseline",
    "build_logistic_regression",
    "cross_validate_model",
    "make_cv",
]
