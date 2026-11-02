import numpy as np
import pandas as pd
import pytest

from src.modeling import RANDOM_FOREST, XGBOOST, build_model
from src.tuning import PARAM_SPECS, SPEC_CHOICE, build_distribution, describe_spec, most_influential_params, to_python, tune_model
from tests.test_preprocessing import make_customers

SMALL_ITER = 3
FAST_RF_SPECS = {"model__n_estimators": ("randint", 5, 15), "model__min_samples_leaf": ("randint", 1, 5)}


@pytest.fixture(scope="module")
def labelled_customers() -> tuple[pd.DataFrame, pd.Series]:
    customers = make_customers(120)
    return customers, pd.Series(np.arange(120) % 4 == 0, index=customers.index).astype(int)


def test_loguniform_spec_samples_inside_its_bounds() -> None:
    distribution = build_distribution(("loguniform", 0.01, 0.2))

    samples = distribution.rvs(200, random_state=42)

    assert samples.min() >= 0.01 and samples.max() <= 0.2
    assert np.median(samples) < 0.1  # logaritmik dağılım küçük değerlere yığılır, düzgün dağılımın medyanı 0,105 olurdu


def test_uniform_spec_covers_the_given_interval_not_loc_and_scale() -> None:
    samples = build_distribution(("uniform", 0.6, 1.0)).rvs(500, random_state=42)

    assert samples.min() >= 0.6 and samples.max() <= 1.0
    assert samples.max() > 0.95


def test_randint_spec_excludes_the_upper_bound() -> None:
    samples = build_distribution(("randint", 3, 9)).rvs(500, random_state=42)

    assert set(np.unique(samples)) == set(range(3, 9))


def test_choice_spec_returns_the_options_for_scikit_learn_to_sample_from() -> None:
    assert build_distribution((SPEC_CHOICE, [None, 8, 12])) == [None, 8, 12]


def test_unknown_spec_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen dağılım"):
        build_distribution(("gamma", 1, 2))


def test_every_search_space_parameter_exists_on_the_model_pipeline() -> None:
    for model_name, specs in PARAM_SPECS.items():
        valid_parameters = build_model(model_name).get_params()

        assert [name for name in specs if name not in valid_parameters] == []


def test_spec_descriptions_are_readable_text() -> None:
    assert describe_spec(("randint", 3, 9)) == "tamsayı [3, 9 (hariç)]"
    assert describe_spec(("uniform", 0.6, 1.0)) == "düzgün [0.6, 1.0]"
    assert describe_spec((SPEC_CHOICE, [None, 8])) == "{None, 8}"


def test_to_python_turns_numpy_scalars_into_json_serializable_values() -> None:
    assert type(to_python(np.int64(3))) is int
    assert type(to_python(np.float64(0.5))) is float
    assert to_python("sqrt") == "sqrt"


def test_tuning_returns_the_best_of_the_sampled_configurations(labelled_customers) -> None:
    customers, target = labelled_customers

    result = tune_model(build_model(RANDOM_FOREST), FAST_RF_SPECS, customers, target, n_iter=SMALL_ITER)

    assert len(result.cv_results) == SMALL_ITER
    assert result.best_score == pytest.approx(result.cv_results["mean_test_score"].max())
    assert set(result.best_params) == set(FAST_RF_SPECS)
    assert result.seconds > 0


def test_tuning_gives_identical_results_for_the_same_seed(labelled_customers) -> None:
    customers, target = labelled_customers

    first = tune_model(build_model(RANDOM_FOREST), FAST_RF_SPECS, customers, target, n_iter=SMALL_ITER)
    second = tune_model(build_model(RANDOM_FOREST), FAST_RF_SPECS, customers, target, n_iter=SMALL_ITER)

    assert first.best_params == second.best_params
    assert first.best_score == second.best_score


def test_tuning_does_not_modify_the_estimator_it_was_given(labelled_customers) -> None:
    customers, target = labelled_customers
    estimator = build_model(XGBOOST)
    parameters_before = estimator.get_params()["model__n_jobs"]

    tune_model(estimator, {"model__n_estimators": ("randint", 5, 10)}, customers, target, n_iter=2)

    assert estimator.get_params()["model__n_jobs"] == parameters_before


def test_most_influential_params_ranks_numeric_parameters_by_rank_correlation_with_the_score() -> None:
    cv_results = pd.DataFrame(
        {
            "param_model__depth": [1, 2, 3, 4, 5, 6],
            "param_model__noise": [3, 1, 4, 2, 6, 5],
            "param_model__constant": [7, 7, 7, 7, 7, 7],
            "param_model__max_features": ["sqrt", 0.3, "sqrt", 0.5, 0.3, "sqrt"],
            "mean_test_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    assert most_influential_params(cv_results, count=2) == ["model__depth", "model__noise"]


def test_most_influential_params_skips_parameters_with_missing_numeric_values() -> None:
    cv_results = pd.DataFrame(
        {"param_model__max_depth": [None, 8, 12, 16], "param_model__leaf": [1, 2, 3, 4], "mean_test_score": [0.1, 0.2, 0.3, 0.4]}
    )

    assert most_influential_params(cv_results) == ["model__leaf"]
