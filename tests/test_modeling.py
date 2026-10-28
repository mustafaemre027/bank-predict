import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import CV_FOLDS
from src.modeling import build_baseline, build_logistic_regression, cross_validate_model, make_cv
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
