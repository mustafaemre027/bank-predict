"""Tahmin servisi: tek müşteri ve liste için kabul olasılığı, olasılık bandı ve kararı en çok etkileyen ilk 3 etken.

Girdi her zaman src/schemas.py ile doğrulanır. Arayüz bu modülü ve şemayı çağırır, iş mantığı içermez.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.config import DROP_COLS, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from src.model_store import ModelBundle
from src.schemas import CustomerInput, format_validation_errors

BAND_HIGH = "yüksek"
BAND_MEDIUM = "orta"
BAND_LOW = "düşük"
DIRECTION_INCREASES = "artırıyor"
DIRECTION_DECREASES = "azaltıyor"
TOP_FACTOR_COUNT = 3
REFERENCE_KEY = "reference_values"
# Liste sıralamasında geçersiz satırlar sonda kalır ve olasılık sütunu boştur.
RESULT_COLUMNS = ["probability", "will_accept", "band", "error"]


@dataclass(frozen=True)
class Factor:
    feature: str
    direction: str
    effect: float


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    will_accept: bool
    band: str
    top_factors: list[Factor]


def probability_band(probability: float, threshold: float, base_rate: float) -> str:
    """Yüksek: p ≥ eşik. Orta: eğitim kabul oranı ≤ p < eşik. Düşük: p < kabul oranı. Eşik ≤ kabul oranı ise bant tanımsızdır."""
    if threshold <= base_rate:
        raise ValueError(f"Eşik ({threshold:.4f}) eğitim kabul oranından ({base_rate:.4f}) büyük olmalı. Olasılık bandı tanımlanamaz.")
    if probability >= threshold:
        return BAND_HIGH
    return BAND_MEDIUM if probability >= base_rate else BAND_LOW


def build_reference_values(X_train: pd.DataFrame) -> dict[str, Any]:
    """Etken analizinde karşılaştırma değeri: kategoriklerde eğitim verisinin modu, sayısallarda medyanı."""
    reference: dict[str, Any] = {}
    for column in FEATURE_COLUMNS:
        reference[column] = int(X_train[column].median()) if column in NUMERIC_FEATURES else str(X_train[column].mode().iloc[0])
    return reference


def _predict_proba(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    return bundle.pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]


def top_factors(customer: dict[str, Any], bundle: ModelBundle, probability: float | None = None) -> list[Factor]:
    """Her değişken için müşterinin değeri referans değerle değiştirilir, olasılık farkı etkidir (orijinal - değiştirilmiş).

    Pozitif etki değişkenin olasılığı artırdığını gösterir. Mutlak etkisi en büyük ilk 3 değişken döner, etkisi 0 olanlar alınmaz.
    Tüm değişkenler tek tahmin çağrısında hesaplanır (orijinal dahil 16 satır).
    """
    reference = bundle.metadata[REFERENCE_KEY]
    variants = [{**customer, column: reference[column]} for column in FEATURE_COLUMNS]
    probabilities = _predict_proba(bundle, pd.DataFrame([customer, *variants]))
    original = float(probabilities[0]) if probability is None else probability
    ranked = []
    for column, replaced in zip(FEATURE_COLUMNS, probabilities[1:]):
        effect = original - float(replaced)
        if effect != 0:
            ranked.append(Factor(column, DIRECTION_INCREASES if effect > 0 else DIRECTION_DECREASES, effect))
    return sorted(ranked, key=lambda factor: abs(factor.effect), reverse=True)[:TOP_FACTOR_COUNT]


def predict_customer(raw: dict[str, Any], bundle: ModelBundle) -> PredictionResult:
    """Girdiyi doğrular (geçersizse pydantic.ValidationError), olasılığı ve ilk 3 etkeni hesaplar."""
    customer = CustomerInput.model_validate(raw).model_dump()
    probability = float(_predict_proba(bundle, pd.DataFrame([customer]))[0])
    return PredictionResult(
        probability=probability,
        will_accept=probability >= bundle.threshold,
        band=probability_band(probability, bundle.threshold, bundle.base_rate),
        top_factors=top_factors(customer, bundle, probability),
    )


def _validation_error_text(row: dict[str, Any]) -> str | None:
    try:
        CustomerInput.model_validate(row)
    except ValidationError as error:
        return " ".join(format_validation_errors(error))
    return None


def predict_batch(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Her satırı doğrular. Geçerliler olasılığa göre azalan sıralanır, geçersizler nedenleriyle `error` sütununda sonda döner.

    bank-full.csv düzeni desteklenir: `duration` ve hedef sütunu model girdisi olmadığı için sessizce yok sayılır.
    """
    inputs = df.drop(columns=[column for column in [*DROP_COLS, TARGET] if column in df.columns])
    records = inputs.astype(object).to_dict(orient="records")
    errors = [_validation_error_text(record) for record in records]
    is_valid = np.array([error is None for error in errors])
    result = inputs.assign(probability=np.nan, will_accept=pd.NA, band=pd.NA, error=pd.Series(errors, index=inputs.index, dtype=object))
    if is_valid.any():
        valid_frame = inputs.loc[is_valid, FEATURE_COLUMNS]
        probabilities = _predict_proba(bundle, valid_frame)
        result.loc[is_valid, "probability"] = probabilities
        result.loc[is_valid, "will_accept"] = probabilities >= bundle.threshold
        result.loc[is_valid, "band"] = [probability_band(value, bundle.threshold, bundle.base_rate) for value in probabilities]
    return result.sort_values("probability", ascending=False, kind="stable", na_position="last")


__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BAND_MEDIUM",
    "DIRECTION_DECREASES",
    "DIRECTION_INCREASES",
    "REFERENCE_KEY",
    "RESULT_COLUMNS",
    "Factor",
    "PredictionResult",
    "build_reference_values",
    "predict_batch",
    "predict_customer",
    "probability_band",
    "top_factors",
]
