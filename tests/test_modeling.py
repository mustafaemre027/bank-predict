import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import CV_FOLDS, RANDOM_STATE
from src.data_loader import load_raw
from src.modeling import (
    build_baseline,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    cross_validate_model,
    make_cv,
)
from src.preprocessing import split_train_test
from tests.test_preprocessing import make_customers


class FitRecorder(ClassifierMixin, BaseEstimator):
    """Hangi satırlarla fit edildiğini kaydeden ve pozitif olasılık olarak sabit 0,5 döndüren sahte model."""

    fitted_indices: list[pd.Index] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FitRecorder":
        FitRecorder.fitted_indices.append(X.index)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.5, 0.5], (len(X), 1))


@pytest.fixture
def labelled_customers() -> tuple[pd.DataFrame, pd.Series]:
    customers = make_customers(100)
    target = pd.Series(np.arange(100) % 5 == 0, index=customers.index).astype(int)
    return customers, target


def test_make_cv_uses_five_shuffled_stratified_folds_with_fixed_seed(labelled_customers) -> None:
    customers, target = labelled_customers
    cv = make_cv()

    first = [validation.tolist() for _, validation in cv.split(customers, target)]
    second = [validation.tolist() for _, validation in make_cv().split(customers, target)]

    assert cv.get_n_splits() == CV_FOLDS
    assert first == second
    for validation in first:
        assert target.iloc[validation].mean() == pytest.approx(target.mean())


def test_cross_validation_fits_each_fold_only_on_its_training_rows(labelled_customers) -> None:
    customers, target = labelled_customers
    FitRecorder.fitted_indices = []

    result = cross_validate_model(FitRecorder(), customers, target, make_cv())

    assert len(FitRecorder.fitted_indices) == CV_FOLDS
    for fold, fitted_index in enumerate(FitRecorder.fitted_indices):
        validation_index = customers.index[result.fold_ids == fold]
        assert fitted_index.intersection(validation_index).empty
        assert len(fitted_index) + len(validation_index) == len(customers)


def test_cross_validation_returns_one_out_of_fold_prediction_per_row(labelled_customers) -> None:
    customers, target = labelled_customers

    result = cross_validate_model(build_logistic_regression(), customers, target, make_cv())

    assert result.oof_proba.shape == (len(customers),)
    assert ((result.oof_proba >= 0) & (result.oof_proba <= 1)).all()
    assert sorted(np.unique(result.fold_ids).tolist()) == list(range(CV_FOLDS))
    assert len(result.fold_metrics) == CV_FOLDS


def test_baseline_has_chance_level_ranking_metrics(labelled_customers) -> None:
    customers, target = labelled_customers

    result = cross_validate_model(build_baseline(), customers, target, make_cv())

    for metrics in result.fold_metrics:
        assert metrics["roc_auc"] == pytest.approx(0.5)
        assert metrics["recall"] == 0


TREE_BUILDERS = [build_random_forest, build_xgboost]
REPRODUCIBILITY_SAMPLE_SIZE = 3000


@pytest.fixture(scope="module")
def real_training_sample() -> tuple[pd.DataFrame, pd.Series]:
    split = split_train_test(load_raw())
    sample = split.X_train.sample(REPRODUCIBILITY_SAMPLE_SIZE, random_state=RANDOM_STATE)
    return sample, split.y_train.loc[sample.index]


@pytest.mark.parametrize("build_model", TREE_BUILDERS)
def test_tree_models_return_probabilities_between_zero_and_one(build_model, labelled_customers) -> None:
    customers, target = labelled_customers

    proba = build_model().fit(customers, target).predict_proba(customers)[:, 1]

    assert proba.shape == (len(customers),)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("build_model", TREE_BUILDERS)
def test_tree_models_ignore_unseen_category_at_prediction_time(build_model, labelled_customers) -> None:
    customers, target = labelled_customers
    model = build_model().fit(customers, target)

    proba = model.predict_proba(customers.iloc[[0]].assign(job=["astronaut"]))[:, 1]

    assert 0 <= proba[0] <= 1


@pytest.mark.parametrize("build_model", [build_logistic_regression, *TREE_BUILDERS])
def test_models_built_without_day_do_not_use_day_column(build_model, labelled_customers) -> None:
    customers, target = labelled_customers

    model = build_model(include_day=False).fit(customers, target)

    names = model.named_steps["preprocess"].get_feature_names_out()
    assert not any(name.endswith("__day") for name in names)


@pytest.mark.parametrize("build_model", [build_random_forest, build_xgboost, build_logistic_regression])
def test_same_seed_gives_identical_out_of_fold_predictions(build_model, real_training_sample) -> None:
    X, y = real_training_sample

    first = cross_validate_model(build_model(), X, y, make_cv())
    second = cross_validate_model(build_model(), X, y, make_cv())

    assert np.array_equal(first.oof_proba, second.oof_proba)
