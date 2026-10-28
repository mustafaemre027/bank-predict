import numpy as np
import pandas as pd
import pytest

from src.config import DROP_COLS, NOT_PREVIOUSLY_CONTACTED, TARGET
from src.data_loader import load_raw
from src.preprocessing import (
    AGE_GROUP_FEATURE,
    WAS_CONTACTED_FEATURE,
    add_engineered_features,
    build_linear_feature_pipeline,
    encode_binary,
    encode_target,
    signed_log1p,
    split_train_test,
)

JOBS = ["admin.", "blue-collar", "student", "retired"]
MONTHS = ["may", "jun", "jul", "aug", "dec"]


def make_customers(row_count: int = 40) -> pd.DataFrame:
    """Modelin beklediği tüm sütunları içeren, deterministik küçük bir müşteri tablosu."""
    rows = np.arange(row_count)
    return pd.DataFrame(
        {
            "age": 20 + rows % 50,
            "job": [JOBS[i % len(JOBS)] for i in rows],
            "marital": np.where(rows % 2 == 0, "married", "single"),
            "education": np.where(rows % 3 == 0, "tertiary", "secondary"),
            "default": np.where(rows % 7 == 0, "yes", "no"),
            "balance": rows * 100 - 500,
            "housing": np.where(rows % 2 == 0, "yes", "no"),
            "loan": np.where(rows % 5 == 0, "yes", "no"),
            "contact": np.where(rows % 4 == 0, "unknown", "cellular"),
            "day": 1 + rows % 28,
            # Aralık bilerek tek kayıtla temsil edilir; az gözlemli ay birleştirme testi buna dayanır.
            "month": ["dec" if i == 0 else MONTHS[i % (len(MONTHS) - 1)] for i in rows],
            "campaign": 1 + rows % 6,
            "pdays": np.where(rows % 3 == 0, rows * 2, NOT_PREVIOUSLY_CONTACTED),
            "previous": np.where(rows % 3 == 0, 1 + rows % 4, 0),
            "poutcome": np.where(rows % 3 == 0, "failure", "unknown"),
        }
    )


@pytest.fixture(scope="module")
def raw_split():
    df = load_raw()
    return df, split_train_test(df)


def test_split_removes_duration_and_target_from_features(raw_split) -> None:
    _, split = raw_split

    for column in [*DROP_COLS, TARGET]:
        assert column not in split.X_train.columns
        assert column not in split.X_test.columns


def test_split_keeps_train_and_test_disjoint_and_covers_all_rows(raw_split) -> None:
    df, split = raw_split

    assert split.X_train.index.intersection(split.X_test.index).empty
    assert len(split.X_train) + len(split.X_test) == len(df)
    assert len(split.X_test) == pytest.approx(0.2 * len(df), abs=1)


def test_split_preserves_target_rate_in_both_sets(raw_split) -> None:
    df, split = raw_split
    overall_rate = encode_target(df[TARGET]).mean()

    assert split.y_train.mean() == pytest.approx(overall_rate, abs=1e-3)
    assert split.y_test.mean() == pytest.approx(overall_rate, abs=1e-3)


def test_split_is_reproducible_with_fixed_random_state(raw_split) -> None:
    df, split = raw_split

    again = split_train_test(df)

    assert split.X_test.index.equals(again.X_test.index)


def test_encode_target_maps_yes_to_one_and_no_to_zero() -> None:
    assert encode_target(pd.Series(["yes", "no", "no"])).tolist() == [1, 0, 0]


def test_engineered_features_flag_previous_contact_and_zero_pdays_for_new_customers() -> None:
    customers = make_customers().iloc[[0, 1]].assign(pdays=[120, NOT_PREVIOUSLY_CONTACTED])

    features = add_engineered_features(customers)

    assert features[WAS_CONTACTED_FEATURE].tolist() == [1, 0]
    assert features["pdays"].tolist() == [120, 0]


def test_engineered_features_do_not_modify_input() -> None:
    customers = make_customers()
    original = customers.copy()

    add_engineered_features(customers)

    pd.testing.assert_frame_equal(customers, original)


@pytest.mark.parametrize(
    ("age", "expected_group"),
    [(18, "18-25"), (25, "18-25"), (26, "26-30"), (50, "41-50"), (60, "51-60"), (61, "61+"), (95, "61+")],
)
def test_age_group_bins_include_upper_bound(age: int, expected_group: str) -> None:
    customers = make_customers().iloc[[0]].assign(age=[age])

    assert add_engineered_features(customers)[AGE_GROUP_FEATURE].iloc[0] == expected_group


def test_signed_log1p_keeps_sign_and_maps_zero_to_zero() -> None:
    values = pd.DataFrame({"balance": [-100, 0, 100]})

    transformed = signed_log1p(values)["balance"].to_numpy()

    assert transformed[1] == 0
    assert transformed[0] == pytest.approx(-transformed[2])
    assert transformed[2] == pytest.approx(np.log1p(100))


def test_encode_binary_maps_yes_to_one() -> None:
    values = pd.DataFrame({"loan": ["yes", "no"], "housing": ["no", "no"]})

    assert encode_binary(values).to_numpy().tolist() == [[1, 0], [0, 0]]


def test_feature_pipeline_output_has_no_missing_values_and_no_duration() -> None:
    pipeline = build_linear_feature_pipeline().fit(make_customers())

    output = pipeline.transform(make_customers())
    names = pipeline.named_steps["preprocess"].get_feature_names_out()

    assert not np.isnan(output).any()
    assert not any("duration" in name for name in names)
    assert any(AGE_GROUP_FEATURE in name for name in names)


def test_feature_pipeline_without_age_group_drops_age_group_columns() -> None:
    pipeline = build_linear_feature_pipeline(include_age_group=False).fit(make_customers())

    names = pipeline.named_steps["preprocess"].get_feature_names_out()

    assert not any(AGE_GROUP_FEATURE in name for name in names)


def test_scaler_statistics_are_learned_from_training_data_only() -> None:
    train, other = make_customers(), make_customers().assign(campaign=50)
    pipeline = build_linear_feature_pipeline().fit(train)
    log_scaler = pipeline.named_steps["preprocess"].named_transformers_["log"][-1]

    pipeline.transform(other)

    assert log_scaler.mean_[0] == pytest.approx(np.log1p(train["campaign"]).mean())


def test_transform_does_not_depend_on_other_rows_in_the_batch() -> None:
    # Dönüşüm transform sırasında yeniden fit edilseydi aynı satırın çıktısı birlikte gelen satırlara göre değişirdi.
    train = make_customers()
    pipeline = build_linear_feature_pipeline().fit(train)
    extreme = make_customers(5).assign(balance=1_000_000, campaign=60, age=90)

    alone = pipeline.transform(train.iloc[[3]])
    with_extremes = pipeline.transform(pd.concat([train.iloc[[3]], extreme]))

    np.testing.assert_allclose(alone[0], with_extremes[0])


def test_unseen_category_is_encoded_as_all_zeros_instead_of_failing() -> None:
    pipeline = build_linear_feature_pipeline().fit(make_customers())
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    job_columns = [index for index, name in enumerate(names) if name.startswith("nominal__job_")]

    output = pipeline.transform(make_customers().iloc[[0]].assign(job=["astronaut"]))

    assert output[0, job_columns].sum() == 0


def test_rare_months_are_merged_into_one_category() -> None:
    pipeline = build_linear_feature_pipeline(rare_month_min_frequency=0.05).fit(make_customers())

    month_encoder = pipeline.named_steps["preprocess"].named_transformers_["month"]

    assert month_encoder.infrequent_categories_[0].tolist() == ["dec"]
