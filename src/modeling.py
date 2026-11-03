"""Model pipeline'ları ve 5 katlı katmanlı çapraz doğrulama düzeni.

Model seçimi ve karşılaştırmalar yalnızca eğitim verisi içinde bu CV düzeniyle yapılır; test seti Hafta 8'e kadar kullanılmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import CV_FOLDS, RANDOM_STATE
from src.evaluate import compute_metrics, fold_metrics_at_nested_threshold, summarize_for_json
from src.preprocessing import build_linear_feature_pipeline, build_tree_feature_pipeline

# Karar eşiği Hafta 6'da OOF tahminlerinden seçilecek; o zamana kadar etiket bazlı metrikler varsayılan eşikle verilir.
DEFAULT_THRESHOLD = 0.5
# Ölçeklenmiş veride lbfgs varsayılan 100 iterasyonda yakınsamayabildiği için üst sınır yükseltildi.
LR_MAX_ITER = 1000
# Ağaç sayısı ve öğrenme hızı gibi ayarlar Hafta 7'de aranacak; bu hafta makul sabit değerlerle üç model karşılaştırılır.
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 5, "max_features": "sqrt"}
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
}

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
MODEL_LABELS = {
    LOGISTIC_REGRESSION: "Lojistik Regresyon",
    RANDOM_FOREST: "Rastgele Orman",
    XGBOOST: "XGBoost",
}

# Sınıf dengesizliği stratejileri. Eşik ayarlama ayrı bir pipeline gerektirmez: "none" modeli ve OOF'tan seçilen eşik kullanılır.
IMBALANCE_NONE = "none"
IMBALANCE_CLASS_WEIGHT = "class_weight"
IMBALANCE_SMOTE = "smote"
IMBALANCE_STRATEGIES = (IMBALANCE_NONE, IMBALANCE_CLASS_WEIGHT, IMBALANCE_SMOTE)
STRATEGY_LABELS = {
    IMBALANCE_NONE: "Ağırlıksız",
    IMBALANCE_CLASS_WEIGHT: "Sınıf ağırlığı",
    IMBALANCE_SMOTE: "SMOTE",
}
# Rastgele Orman'da "balanced_subsample" ağırlığı her bootstrap örneklemi için yeniden hesaplar.
SKLEARN_CLASS_WEIGHT = {LOGISTIC_REGRESSION: "balanced", RANDOM_FOREST: "balanced_subsample"}


@dataclass(frozen=True)
class CrossValidationResult:
    fold_metrics: list[dict[str, float]]
    oof_proba: np.ndarray
    fold_ids: np.ndarray


def nested_threshold_summary(
    y_true: pd.Series | np.ndarray,
    result: CrossValidationResult,
    select_threshold: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, dict[str, float]]:
    """Eşik her katman için diğer katmanların OOF tahminlerinden seçilip o katmanda ölçülür, sonuç ortalama ± std olarak döner."""
    folds = fold_metrics_at_nested_threshold(y_true, result.oof_proba, result.fold_ids, select_threshold)
    return summarize_for_json(folds)["summary"]


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_baseline() -> DummyClassifier:
    """Hafta 2'deki referans model: herkese çoğunluk sınıfını ("kabul etmez") tahmin eder."""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression(
    include_age_group: bool = True,
    rare_month_min_frequency: float | None = None,
    include_day: bool = True,
    use_log: bool = True,
) -> Pipeline:
    """Ön işleme + Lojistik Regresyon. Varsayılan düzenlileştirme L2'dir (C = 1)."""
    feature_pipeline = build_linear_feature_pipeline(include_age_group, rare_month_min_frequency, include_day, use_log)
    return Pipeline([*feature_pipeline.steps, ("model", LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE))])


def build_random_forest(include_age_group: bool = False, include_day: bool = True) -> Pipeline:
    """Ölçeklemesiz ön işleme + Rastgele Orman."""
    feature_pipeline = build_tree_feature_pipeline(include_age_group, include_day)
    # n_jobs=1: çok iş parçacığında ağaç olasılıkları değişken sırada toplandığı için sonuç 1e-16 düzeyinde oynuyor.
    # Tekrar üretilebilirlik (FOG-2) için model tek iş parçacığında çalışır, paralellik CV ve arama düzeyinde sağlanır.
    forest = RandomForestClassifier(**RF_PARAMS, n_jobs=1, random_state=RANDOM_STATE)
    return Pipeline([*feature_pipeline.steps, ("model", forest)])


def build_xgboost(include_age_group: bool = False, include_day: bool = True) -> Pipeline:
    """Ölçeklemesiz ön işleme + XGBoost."""
    feature_pipeline = build_tree_feature_pipeline(include_age_group, include_day)
    booster = XGBClassifier(**XGB_PARAMS, n_jobs=-1, random_state=RANDOM_STATE)
    return Pipeline([*feature_pipeline.steps, ("model", booster)])


class ClassWeightedXGBClassifier(XGBClassifier):
    """Pozitif sınıfa, her fit çağrısında o fit'in eğitim etiketlerinden hesaplanan negatif/pozitif oranı kadar ağırlık veren XGBoost.

    Ağırlık hiperparametre olarak değil sample_weight olarak verilir, bu yüzden fit modelin kurucu parametrelerini değiştirmez.
    CV'de her katman kendi eğitim oranını kullanır, doğrulama katmanının etiketlerinden bilgi sızmaz. Etiketler 0/1 olmalıdır.
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "ClassWeightedXGBClassifier":
        labels = np.asarray(y)
        positives = int((labels == 1).sum())
        negatives = len(labels) - positives
        if positives == 0 or negatives == 0:
            raise ValueError("Sınıf ağırlığı hesaplamak için eğitim etiketlerinde iki sınıf da bulunmalı.")
        sample_weight = np.where(labels == 1, negatives / positives, 1.0)
        return super().fit(X, y, sample_weight=sample_weight)


def add_smote_step(pipeline: Pipeline) -> ImbPipeline:
    """SMOTE'u ön işlemeden sonra, model adımından önce ekler.

    imblearn Pipeline'ı örneklemeyi yalnızca fit sırasında uygular. Böylece CV'de yalnızca eğitim katmanı çoğaltılır,
    doğrulama katmanı tahmin sırasında olduğu gibi kalır.
    """
    *preparation_steps, model_step = pipeline.steps
    return ImbPipeline([*preparation_steps, ("smote", SMOTE(random_state=RANDOM_STATE)), model_step])


def _add_class_weight(model_name: str, pipeline: Pipeline) -> Pipeline:
    if model_name == XGBOOST:
        booster = ClassWeightedXGBClassifier(**XGB_PARAMS, n_jobs=-1, random_state=RANDOM_STATE)
        return Pipeline([*pipeline.steps[:-1], ("model", booster)])
    return pipeline.set_params(model__class_weight=SKLEARN_CLASS_WEIGHT[model_name])


def build_model(model_name: str, strategy: str = IMBALANCE_NONE) -> Pipeline | ImbPipeline:
    """Model adı ve dengesizlik stratejisinden, varsayılan ön işleme seçenekleriyle fit edilmemiş pipeline kurar."""
    builders = {LOGISTIC_REGRESSION: build_logistic_regression, RANDOM_FOREST: build_random_forest, XGBOOST: build_xgboost}
    if model_name not in builders:
        raise ValueError(f"Bilinmeyen model: {model_name}. Geçerli modeller: {', '.join(builders)}.")
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Bilinmeyen strateji: {strategy}. Geçerli stratejiler: {', '.join(IMBALANCE_STRATEGIES)}.")
    pipeline = builders[model_name]()
    if strategy == IMBALANCE_CLASS_WEIGHT:
        return _add_class_weight(model_name, pipeline)
    if strategy == IMBALANCE_SMOTE:
        return add_smote_step(pipeline)
    return pipeline


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
    "IMBALANCE_CLASS_WEIGHT",
    "IMBALANCE_NONE",
    "IMBALANCE_SMOTE",
    "IMBALANCE_STRATEGIES",
    "STRATEGY_LABELS",
    "ClassWeightedXGBClassifier",
    "add_smote_step",
    "build_model",
    "LOGISTIC_REGRESSION",
    "MODEL_LABELS",
    "RANDOM_FOREST",
    "XGBOOST",
    "CrossValidationResult",
    "build_baseline",
    "build_logistic_regression",
    "build_random_forest",
    "build_xgboost",
    "cross_validate_model",
    "nested_threshold_summary",
    "make_cv",
]
