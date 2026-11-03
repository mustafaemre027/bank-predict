import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import CV_FOLDS, RANDOM_STATE
from src.data_loader import load_raw
from src.modeling import (
    IMBALANCE_CLASS_WEIGHT,
    IMBALANCE_NONE,
    IMBALANCE_SMOTE,
    IMBALANCE_STRATEGIES,
    ClassWeightedXGBClassifier,
    add_smote_step,
    build_baseline,
    build_model,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    cross_validate_model,
    make_cv,
)
from src.preprocessing import build_tree_feature_pipeline, split_train_test
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


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Fit'te gördüğü satır sayısı ve pozitif oranı ile predict_proba'da gördüğü satır sayısını kaydeden sahte model."""

    fit_rows: list[int] = []
    fit_positive_rates: list[float] = []
    predict_rows: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpyClassifier":
        SpyClassifier.fit_rows.append(len(y))
        SpyClassifier.fit_positive_rates.append(float(np.mean(y)))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        SpyClassifier.predict_rows.append(len(X))
        return np.tile([0.5, 0.5], (len(X), 1))


@pytest.fixture
def spy_with_smote(labelled_customers):
    SpyClassifier.fit_rows, SpyClassifier.fit_positive_rates, SpyClassifier.predict_rows = [], [], []
    spy_pipeline = Pipeline([*build_tree_feature_pipeline().steps, ("model", SpyClassifier())])
    return add_smote_step(spy_pipeline)


def test_smote_oversamples_only_the_training_fold(spy_with_smote, labelled_customers) -> None:
    customers, target = labelled_customers
    cv = make_cv()
    train_sizes = [len(train_index) for train_index, _ in cv.split(customers, target)]

    cross_validate_model(spy_with_smote, customers, target, cv)

    assert all(fitted > original for fitted, original in zip(SpyClassifier.fit_rows, train_sizes))
    assert SpyClassifier.fit_positive_rates == pytest.approx([0.5] * CV_FOLDS)


def test_smote_does_not_change_the_size_of_the_validation_fold(spy_with_smote, labelled_customers) -> None:
    customers, target = labelled_customers
    cv = make_cv()
    validation_sizes = [len(validation_index) for _, validation_index in cv.split(customers, target)]

    cross_validate_model(spy_with_smote, customers, target, cv)

    assert SpyClassifier.predict_rows == validation_sizes


def test_build_model_rejects_unknown_model_and_strategy() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen model"):
        build_model("svm")
    with pytest.raises(ValueError, match="Bilinmeyen strateji"):
        build_model("xgboost", "undersampling")


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost"])
def test_every_strategy_builds_a_model_that_returns_probabilities(model_name, labelled_customers) -> None:
    customers, target = labelled_customers

    for strategy in IMBALANCE_STRATEGIES:
        proba = clone(build_model(model_name, strategy)).fit(customers, target).predict_proba(customers)[:, 1]

        assert ((proba >= 0) & (proba <= 1)).all()


def test_only_smote_strategy_adds_a_resampling_step() -> None:
    steps_by_strategy = {strategy: [name for name, _ in build_model("random_forest", strategy).steps] for strategy in IMBALANCE_STRATEGIES}

    assert steps_by_strategy[IMBALANCE_SMOTE] == ["features", "preprocess", "smote", "model"]
    assert steps_by_strategy[IMBALANCE_NONE] == steps_by_strategy[IMBALANCE_CLASS_WEIGHT] == ["features", "preprocess", "model"]


def test_class_weight_strategy_sets_balanced_weights_on_sklearn_models() -> None:
    logistic = build_model("logistic_regression", IMBALANCE_CLASS_WEIGHT).named_steps["model"]
    forest = build_model("random_forest", IMBALANCE_CLASS_WEIGHT).named_steps["model"]

    assert logistic.class_weight == "balanced"
    assert forest.class_weight == "balanced_subsample"


def test_class_weighted_xgboost_weights_positives_by_the_negative_to_positive_ratio_of_each_fit() -> None:
    X = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float) % 7})
    one_in_five = pd.Series(np.arange(100) % 5 == 0).astype(int)
    weighted = ClassWeightedXGBClassifier(n_estimators=20, random_state=RANDOM_STATE).fit(X, one_in_five)
    manual = XGBClassifier(n_estimators=20, random_state=RANDOM_STATE).fit(X, one_in_five, sample_weight=np.where(one_in_five == 1, 80 / 20, 1.0))

    assert np.allclose(weighted.predict_proba(X), manual.predict_proba(X))


def test_class_weighted_xgboost_fit_does_not_change_constructor_parameters() -> None:
    X = pd.DataFrame({"a": np.arange(100, dtype=float)})
    y = pd.Series(np.arange(100) % 5 == 0).astype(int)
    booster = ClassWeightedXGBClassifier(n_estimators=5)

    parameters_before = booster.get_params()
    booster.fit(X, y)

    assert booster.get_params() == parameters_before


def test_class_weighted_xgboost_rejects_single_class_labels() -> None:
    X = pd.DataFrame({"a": np.arange(10, dtype=float)})

    with pytest.raises(ValueError, match="iki sınıf"):
        ClassWeightedXGBClassifier(n_estimators=5).fit(X, pd.Series(np.zeros(10, dtype=int)))
