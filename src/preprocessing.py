"""Ön işleme kararlarının (docs/on_isleme_kararlari.md) kod karşılığı: özellik üretimi, eğitim/test ayrımı ve dönüşümler.

Bütün dönüşümler scikit-learn Pipeline/ColumnTransformer içinde tanımlanır. Böylece ölçekleme gibi veriden öğrenilen
adımlar çapraz doğrulamada yalnızca eğitim katmanına fit edilir ve doğrulama katmanına sızıntı olmaz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    AGE_BINS,
    AGE_LABELS,
    BINARY_FEATURES,
    DROP_COLS,
    NOMINAL_FEATURES,
    NOT_PREVIOUSLY_CONTACTED,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

WAS_CONTACTED_FEATURE = "was_contacted_before"
AGE_GROUP_FEATURE = "age_group"
# Aranmamış müşterilerde -1 sayısal olarak anlamsız; bilgi was_contacted_before bayrağıyla taşınır.
NOT_CONTACTED_PDAYS_VALUE = 0
SIGNED_LOG_FEATURES = ["balance"]
LOG_FEATURES = ["campaign", "previous"]
SCALED_ONLY_FEATURES = ["age", "day", "pdays"]
MONTH_FEATURE = "month"


@dataclass(frozen=True)
class TrainTestSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def encode_target(target: pd.Series) -> pd.Series:
    """yes/no hedefini 1/0 tamsayıya çevirir."""
    return (target == POSITIVE_LABEL).astype(int)


def split_train_test(df: pd.DataFrame) -> TrainTestSplit:
    """duration ve hedefi çıkarıp veriyi hedef oranını koruyarak (stratified) eğitim ve test olarak ayırır."""
    features = df.drop(columns=[*DROP_COLS, TARGET])
    target = encode_target(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    return TrainTestSplit(X_train, X_test, y_train, y_test)


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """was_contacted_before ve yaş grubu özelliklerini ekler, aranmamış müşterilerde pdays değerini 0'a çeker.

    Bu adım satır bazlıdır ve veriden hiçbir şey öğrenmez, bu yüzden sızıntı riski taşımaz.
    """
    was_contacted = features["pdays"] != NOT_PREVIOUSLY_CONTACTED
    return features.assign(
        **{
            WAS_CONTACTED_FEATURE: was_contacted.astype(int),
            "pdays": features["pdays"].where(was_contacted, NOT_CONTACTED_PDAYS_VALUE),
            AGE_GROUP_FEATURE: pd.cut(features["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str),
        }
    )


def signed_log1p(values: pd.DataFrame) -> pd.DataFrame:
    """Negatif değer alabilen çarpık değişkenler için işareti koruyan log dönüşümü: sign(x)·log1p(|x|)."""
    return np.sign(values) * np.log1p(np.abs(values))


def encode_binary(values: pd.DataFrame) -> pd.DataFrame:
    return (values == POSITIVE_LABEL).astype(int)


def _one_hot(min_frequency: float | None = None) -> OneHotEncoder:
    # Eğitimde görülmeyen bir kategori tahmin anında hata vermez, tüm sütunları 0 olarak kodlanır.
    return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=False)


def build_linear_preprocessor(include_age_group: bool = True, rare_month_min_frequency: float | None = None) -> ColumnTransformer:
    """Lojistik Regresyon için log dönüşümü, ölçekleme ve kodlama adımlarını içeren ColumnTransformer kurar.

    rare_month_min_frequency verilirse eğitim verisindeki payı bu oranın altında kalan aylar tek bir kategoride birleştirilir.
    """
    nominal_features = [column for column in NOMINAL_FEATURES if column != MONTH_FEATURE]
    if include_age_group:
        nominal_features.append(AGE_GROUP_FEATURE)
    return ColumnTransformer(
        [
            ("signed_log", make_pipeline(FunctionTransformer(signed_log1p, feature_names_out="one-to-one"), StandardScaler()), SIGNED_LOG_FEATURES),
            ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), LOG_FEATURES),
            ("scaled", StandardScaler(), SCALED_ONLY_FEATURES),
            ("binary", FunctionTransformer(encode_binary, feature_names_out="one-to-one"), BINARY_FEATURES),
            ("flag", "passthrough", [WAS_CONTACTED_FEATURE]),
            ("month", _one_hot(rare_month_min_frequency), [MONTH_FEATURE]),
            ("nominal", _one_hot(), nominal_features),
        ],
        remainder="drop",
    )


def build_linear_feature_pipeline(include_age_group: bool = True, rare_month_min_frequency: float | None = None) -> Pipeline:
    """Ham özellik tablosunu Lojistik Regresyon'a hazır sayısal matrise çeviren tam ön işleme pipeline'ı."""
    return Pipeline(
        [
            ("features", FunctionTransformer(add_engineered_features)),
            ("preprocess", build_linear_preprocessor(include_age_group, rare_month_min_frequency)),
        ]
    )


__all__ = [
    "AGE_GROUP_FEATURE",
    "MONTH_FEATURE",
    "WAS_CONTACTED_FEATURE",
    "TrainTestSplit",
    "add_engineered_features",
    "build_linear_feature_pipeline",
    "build_linear_preprocessor",
    "encode_binary",
    "encode_target",
    "signed_log1p",
    "split_train_test",
]
