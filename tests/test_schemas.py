import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import CATEGORICAL_FEATURES, CATEGORY_VALUES, FEATURE_COLUMNS, NUMERIC_FEATURES, NUMERIC_RANGES
from src.data_loader import load_raw
from src.schemas import CustomerInput, format_validation_errors

VALID_CUSTOMER = {
    "age": 41, "job": "technician", "marital": "married", "education": "secondary", "default": "no", "balance": 1200,
    "housing": "yes", "loan": "no", "contact": "cellular", "day": 15, "month": "may", "campaign": 2, "pdays": -1,
    "previous": 0, "poutcome": "unknown",
}


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


def test_category_values_match_the_raw_data_exactly(raw) -> None:
    for column in CATEGORICAL_FEATURES:
        assert CATEGORY_VALUES[column] == sorted(raw[column].unique()), column
    assert set(CATEGORY_VALUES) == set(CATEGORICAL_FEATURES)


def test_numeric_ranges_match_the_raw_data_exactly(raw) -> None:
    for column in NUMERIC_FEATURES:
        assert NUMERIC_RANGES[column] == (int(raw[column].min()), int(raw[column].max())), column
    assert set(NUMERIC_RANGES) == set(NUMERIC_FEATURES)


def test_feature_columns_follow_the_raw_file_order_without_duration(raw) -> None:
    assert FEATURE_COLUMNS == [column for column in raw.columns if column not in ("duration", "y")]
    assert list(CustomerInput.model_fields) == FEATURE_COLUMNS


def test_valid_customer_is_accepted() -> None:
    assert CustomerInput.model_validate(VALID_CUSTOMER).job == "technician"


def test_every_row_of_the_raw_data_is_accepted(raw) -> None:
    rows = raw[FEATURE_COLUMNS].astype(object).to_dict(orient="records")

    for row in rows[:: len(rows) // 500]:
        CustomerInput.model_validate(row)


def validation_messages(**changes: object) -> list[str]:
    with pytest.raises(ValidationError) as excinfo:
        CustomerInput.model_validate({**VALID_CUSTOMER, **changes})
    return format_validation_errors(excinfo.value)


def test_invalid_category_names_the_field_and_lists_valid_values() -> None:
    (message,) = validation_messages(job="astronaut")

    assert "`job` alanı geçersiz" in message
    assert "admin., blue-collar" in message and "unknown" in message


@pytest.mark.parametrize(("field", "value", "expected"), [("age", 17, "18 ile 95"), ("day", 32, "1 ile 31"), ("pdays", -2, "-1 ile 871")])
def test_out_of_range_numbers_are_rejected_with_the_allowed_range(field, value, expected) -> None:
    (message,) = validation_messages(**{field: value})

    assert f"`{field}` alanı" in message and expected in message


def test_non_integer_number_is_rejected_in_plain_turkish() -> None:
    (message,) = validation_messages(age="otuz")

    assert message == "`age` alanı tam sayı olmalı."


def test_missing_field_is_reported() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CustomerInput.model_validate({key: value for key, value in VALID_CUSTOMER.items() if key != "balance"})

    assert format_validation_errors(excinfo.value) == ["`balance` alanı eksik."]


def test_duration_and_other_extra_fields_are_rejected() -> None:
    (message,) = validation_messages(duration=300)

    assert "`duration` alanı kabul edilmiyor" in message


def test_error_messages_contain_no_technical_details() -> None:
    messages = validation_messages(job="x", age=500, duration=1)

    for message in messages:
        assert "pydantic" not in message.lower() and "Traceback" not in message and "type=" not in message
