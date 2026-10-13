from src.data_loader import load_raw, validate_raw


def test_bank_raw_dataset_matches_expected_shape_and_target_values():
    df = load_raw()
    validate_raw(df)

    assert df.shape == (45211, 17)
    expected_columns = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    ]
    assert list(df.columns) == expected_columns
    assert set(df["y"].unique()) <= {"yes", "no"}
    assert df["y"].value_counts()["yes"] == 5289
