"""Hiperparametre araması: sabit iterasyon bütçesiyle rastgele arama, PR-AUC'ye (average_precision) göre.

Arama yalnızca eğitim verisinde ve modelleri karşılaştırmakta kullanılan CV katmanlarıyla yapılır, test seti kullanılmaz.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RandomizedSearchCV

from src.config import RANDOM_STATE
from src.modeling import LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, make_cv

# Hafta 7'de her model için denenecek sabit yapılandırma sayısı (FOG-5: arama dizüstü bilgisayarda 30 dakikayı aşmamalı).
N_ITER = 30
SCORING = "average_precision"

SPEC_LOGUNIFORM = "loguniform"
SPEC_RANDINT = "randint"
SPEC_UNIFORM = "uniform"
SPEC_CHOICE = "choice"
# Arama uzayı: parametre adı -> (dağılım türü, bağımsız değişkenler). randint üst sınırı dahil değildir, uniform [alt, üst] aralığıdır.
PARAM_SPECS: dict[str, dict[str, tuple[Any, ...]]] = {
    LOGISTIC_REGRESSION: {
        "model__C": (SPEC_LOGUNIFORM, 1e-3, 1e2),
    },
    RANDOM_FOREST: {
        "model__n_estimators": (SPEC_RANDINT, 200, 600),
        "model__max_depth": (SPEC_CHOICE, [None, 8, 12, 16, 24]),
        "model__min_samples_leaf": (SPEC_RANDINT, 1, 20),
        "model__max_features": (SPEC_CHOICE, ["sqrt", 0.3, 0.5]),
    },
    XGBOOST: {
        "model__n_estimators": (SPEC_RANDINT, 200, 800),
        "model__learning_rate": (SPEC_LOGUNIFORM, 0.01, 0.2),
        "model__max_depth": (SPEC_RANDINT, 3, 9),
        "model__min_child_weight": (SPEC_RANDINT, 1, 10),
        "model__subsample": (SPEC_UNIFORM, 0.6, 1.0),
        "model__colsample_bytree": (SPEC_UNIFORM, 0.5, 1.0),
        "model__reg_lambda": (SPEC_LOGUNIFORM, 0.1, 10),
    },
}


@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    seconds: float


def build_distribution(spec: tuple[Any, ...]) -> Any:
    kind, *arguments = spec
    if kind == SPEC_LOGUNIFORM:
        return loguniform(*arguments)
    if kind == SPEC_RANDINT:
        return randint(*arguments)
    if kind == SPEC_UNIFORM:
        low, high = arguments
        return uniform(low, high - low)
    if kind == SPEC_CHOICE:
        return arguments[0]
    raise ValueError(f"Bilinmeyen dağılım türü: {kind}")


def describe_spec(spec: tuple[Any, ...]) -> str:
    """Arama uzayını rapora ve JSON'a yazılacak okunur metne çevirir."""
    kind, *arguments = spec
    if kind == SPEC_CHOICE:
        return "{" + ", ".join(str(value) for value in arguments[0]) + "}"
    low, high = arguments
    description = {
        SPEC_LOGUNIFORM: "logaritmik düzgün",
        SPEC_RANDINT: "tamsayı",
        SPEC_UNIFORM: "düzgün",
    }[kind]
    upper_bound = f"{high} (hariç)" if kind == SPEC_RANDINT else f"{high}"
    return f"{description} [{low}, {upper_bound}]"


def to_python(value: Any) -> Any:
    """numpy skalerlerini JSON'a yazılabilir Python tiplerine çevirir."""
    return value.item() if isinstance(value, np.generic) else value


def most_influential_params(cv_results: pd.DataFrame, count: int = 2) -> list[str]:
    """Skorla Spearman korelasyonu en yüksek (mutlak değerce) sayısal parametreler. Metin içeren ya da sabit parametreler atlanır."""
    scores = cv_results["mean_test_score"]
    correlations = {}
    for column in cv_results.columns:
        if not column.startswith("param_"):
            continue
        values = pd.to_numeric(cv_results[column], errors="coerce")
        if values.isna().any() or values.nunique() < 3:
            continue
        correlations[column.removeprefix("param_")] = abs(spearmanr(values, scores).statistic)
    return sorted(correlations, key=correlations.__getitem__, reverse=True)[:count]


def tune_model(
    estimator: BaseEstimator,
    param_specs: dict[str, tuple[Any, ...]],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = N_ITER,
) -> TuningResult:
    """Rastgele arama yapar. Paralellik yapılandırma/katman düzeyindedir, bu yüzden modelin kendi iş parçacığı 1'e çekilir."""
    search_estimator = clone(estimator)
    if "model__n_jobs" in search_estimator.get_params():
        search_estimator.set_params(model__n_jobs=1)
    search = RandomizedSearchCV(
        search_estimator,
        {name: build_distribution(spec) for name, spec in param_specs.items()},
        n_iter=n_iter,
        scoring=SCORING,
        cv=make_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
    )
    started = time.perf_counter()
    search.fit(X, y)
    seconds = time.perf_counter() - started
    return TuningResult(
        best_params={name: to_python(value) for name, value in search.best_params_.items()},
        best_score=float(search.best_score_),
        cv_results=pd.DataFrame(search.cv_results_),
        seconds=seconds,
    )


__all__ = [
    "N_ITER",
    "PARAM_SPECS",
    "SCORING",
    "TuningResult",
    "build_distribution",
    "describe_spec",
    "most_influential_params",
    "to_python",
    "tune_model",
]
