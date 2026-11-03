import time

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.model_store import ModelBundle
from src.modeling import build_logistic_regression
from src.predict_service import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    DIRECTION_INCREASES,
    REFERENCE_KEY,
    build_reference_values,
    predict_batch,
    predict_customer,
    probability_band,
    top_factors,
)
from tests.test_explain import build_synthetic_frame

THRESHOLD = 0.5
BASE_RATE = 0.12
MAX_AVERAGE_SECONDS = 1.0
TIMING_REPEATS = 5

PREDICT_CUSTOMER = {
    "age": 41, "job": "technician", "marital": "married", "education": "secondary", "default": "no", "balance": 1200,
    "housing": "yes", "loan": "no", "contact": "cellular", "day": 15, "month": "may", "campaign": 2, "pdays": -1,
    "previous": 0, "poutcome": "success",
}


def make_bundle() -> ModelBundle:
    """Hedefi yalnızca poutcome=success belirleyen sentetik veriyle eğitilmiş küçük bir paket."""
    X, y = build_synthetic_frame()
    pipeline = build_logistic_regression().fit(X, y)
    return ModelBundle(pipeline, THRESHOLD, BASE_RATE, {REFERENCE_KEY: build_reference_values(X)})


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return make_bundle()


def test_probability_is_between_zero_and_one_and_decision_uses_the_threshold(bundle) -> None:
    accepting = predict_customer(PREDICT_CUSTOMER, bundle)
    declining = predict_customer({**PREDICT_CUSTOMER, "poutcome": "failure"}, bundle)

    assert 0 <= accepting.probability <= 1 and 0 <= declining.probability <= 1
    assert accepting.will_accept is True and accepting.probability >= THRESHOLD
    assert declining.will_accept is False


def test_invalid_input_raises_a_validation_error(bundle) -> None:
    with pytest.raises(ValidationError):
        predict_customer({**PREDICT_CUSTOMER, "job": "astronaut"}, bundle)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.60, BAND_HIGH), (0.50, BAND_HIGH), (0.30, BAND_MEDIUM), (0.12, BAND_MEDIUM), (0.05, BAND_LOW)],
)
def test_probability_band_follows_the_three_rules(probability, expected) -> None:
    assert probability_band(probability, THRESHOLD, BASE_RATE) == expected


@pytest.mark.parametrize("threshold", [0.12, 0.05])
def test_band_is_undefined_when_the_threshold_is_not_above_the_base_rate(threshold) -> None:
    with pytest.raises(ValueError, match="büyük olmalı"):
        probability_band(0.3, threshold, BASE_RATE)


def test_the_only_informative_variable_is_the_top_factor_and_increases_the_probability(bundle) -> None:
    result = predict_customer(PREDICT_CUSTOMER, bundle)

    assert result.top_factors[0].feature == "poutcome"
    assert result.top_factors[0].direction == DIRECTION_INCREASES
    assert result.top_factors[0].effect > 0
    assert len(result.top_factors) <= 3
    assert [abs(factor.effect) for factor in result.top_factors] == sorted((abs(factor.effect) for factor in result.top_factors), reverse=True)


def test_variable_equal_to_the_reference_has_no_effect(bundle) -> None:
    reference = bundle.metadata[REFERENCE_KEY]
    customer = {**PREDICT_CUSTOMER, **reference}

    assert top_factors(customer, bundle) == []


def test_batch_ranks_valid_rows_and_reports_invalid_ones_at_the_end(bundle) -> None:
    rows = pd.DataFrame(
        [
            {**PREDICT_CUSTOMER, "poutcome": "failure"},
            {**PREDICT_CUSTOMER, "job": "astronaut"},
            PREDICT_CUSTOMER,
        ]
    )

    result = predict_batch(rows, bundle)

    assert list(result.index) == [2, 0, 1]
    assert result["probability"].iloc[0] > result["probability"].iloc[1]
    assert pd.isna(result["probability"].iloc[2])
    assert "`job` alanı geçersiz" in result["error"].iloc[2]
    assert result["error"].iloc[:2].isna().all()


def test_batch_ignores_duration_and_target_columns_of_the_raw_file_layout(bundle) -> None:
    rows = pd.DataFrame([{**PREDICT_CUSTOMER, "duration": 300, "y": "yes"}])

    result = predict_batch(rows, bundle)

    assert result["error"].isna().all() and result["probability"].notna().all()


def test_batch_marks_missing_values_as_invalid(bundle) -> None:
    rows = pd.DataFrame([{**PREDICT_CUSTOMER, "balance": np.nan}])

    result = predict_batch(rows, bundle)

    assert pd.isna(result["probability"].iloc[0]) and "`balance`" in result["error"].iloc[0]


def test_batch_with_only_invalid_rows_does_not_call_the_model(bundle) -> None:
    result = predict_batch(pd.DataFrame([{**PREDICT_CUSTOMER, "age": 5}]), bundle)

    assert result["probability"].isna().all()


def test_single_prediction_stays_under_one_second_on_average(bundle) -> None:
    """FOG-5: küçük bir pipeline ile ölçülür. Gerçek model için ayrı ölçüm build_model_kaydi.py içinde yapılır."""
    started = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        predict_customer(PREDICT_CUSTOMER, bundle)

    assert (time.perf_counter() - started) / TIMING_REPEATS < MAX_AVERAGE_SECONDS


def test_real_model_single_prediction_stays_under_one_second_on_average() -> None:
    """FOG-5: kaydedilmiş gerçek model varsa ek ölçüm yapılır."""
    from src.config import MODEL_PATH
    from src.model_store import load_bundle

    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH} yok, gerçek model ölçümü atlandı. python notebooks/build_model_kaydi.py ile üretilir.")
    real_bundle = load_bundle(MODEL_PATH)
    predict_customer(PREDICT_CUSTOMER, real_bundle)

    started = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        predict_customer(PREDICT_CUSTOMER, real_bundle)

    assert (time.perf_counter() - started) / TIMING_REPEATS < MAX_AVERAGE_SECONDS
