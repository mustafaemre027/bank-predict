import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.final_model import (
    CALIBRATION_METHOD,
    FinalModelConfig,
    build_configured_estimator,
    build_final_pipeline,
    decide_calibration,
    load_selected_config,
    select_final_candidate,
)
from src.modeling import IMBALANCE_CLASS_WEIGHT, IMBALANCE_NONE, IMBALANCE_SMOTE, LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST
from tests.test_preprocessing import make_customers


def candidate(model: str, pr_auc_mean: float) -> dict:
    return {"model": model, "pr_auc_mean": pr_auc_mean}


def never_close(best: dict, other: dict) -> bool:
    return False


def always_close(best: dict, other: dict) -> bool:
    return True


def test_best_pr_auc_candidate_is_selected_when_the_simpler_one_is_clearly_worse() -> None:
    chosen = select_final_candidate([candidate(RANDOM_FOREST, 0.44), candidate(XGBOOST, 0.46)], never_close)

    assert chosen["model"] == XGBOOST


def test_simpler_candidate_is_selected_when_it_is_within_noise_of_the_best() -> None:
    chosen = select_final_candidate([candidate(RANDOM_FOREST, 0.455), candidate(XGBOOST, 0.457)], always_close)

    assert chosen["model"] == RANDOM_FOREST


def test_simplest_candidate_within_noise_wins_among_three() -> None:
    candidates = [candidate(XGBOOST, 0.46), candidate(LOGISTIC_REGRESSION, 0.455), candidate(RANDOM_FOREST, 0.458)]

    assert select_final_candidate(candidates, always_close)["model"] == LOGISTIC_REGRESSION


def test_best_candidate_is_kept_when_it_is_already_the_simplest() -> None:
    chosen = select_final_candidate([candidate(RANDOM_FOREST, 0.46), candidate(XGBOOST, 0.44)], always_close)

    assert chosen["model"] == RANDOM_FOREST


def test_noise_check_is_asked_about_the_best_candidate_and_each_simpler_one() -> None:
    asked = []

    def record(best: dict, other: dict) -> bool:
        asked.append((best["model"], other["model"]))
        return False

    select_final_candidate([candidate(XGBOOST, 0.46), candidate(RANDOM_FOREST, 0.45)], record)

    assert asked == [(XGBOOST, RANDOM_FOREST)]


@pytest.mark.parametrize("strategy", [IMBALANCE_CLASS_WEIGHT, IMBALANCE_SMOTE])
def test_calibration_is_mandatory_for_weighted_and_oversampled_models(strategy: str) -> None:
    assert decide_calibration(strategy, brier_raw=0.08, brier_calibrated=0.09, pr_auc_raw=0.45, pr_auc_calibrated=0.40) is True


def test_unweighted_model_is_calibrated_when_brier_drops_and_pr_auc_barely_changes() -> None:
    assert decide_calibration(IMBALANCE_NONE, brier_raw=0.080, brier_calibrated=0.075, pr_auc_raw=0.450, pr_auc_calibrated=0.446) is True


def test_unweighted_model_is_not_calibrated_when_brier_does_not_improve() -> None:
    assert decide_calibration(IMBALANCE_NONE, brier_raw=0.075, brier_calibrated=0.080, pr_auc_raw=0.450, pr_auc_calibrated=0.450) is False


def test_unweighted_model_is_not_calibrated_when_pr_auc_drops_more_than_the_allowed_amount() -> None:
    assert decide_calibration(IMBALANCE_NONE, brier_raw=0.080, brier_calibrated=0.075, pr_auc_raw=0.450, pr_auc_calibrated=0.444) is False


@pytest.fixture(scope="module")
def labelled_customers() -> tuple[pd.DataFrame, pd.Series]:
    customers = make_customers(150)
    return customers, pd.Series(np.arange(150) % 4 == 0, index=customers.index).astype(int)


def test_configured_estimator_applies_the_searched_parameters() -> None:
    estimator = build_configured_estimator(RANDOM_FOREST, IMBALANCE_NONE, {"model__n_estimators": 7, "model__min_samples_leaf": 2}, calibrated=False)

    assert estimator.named_steps["model"].n_estimators == 7
    assert estimator.named_steps["model"].min_samples_leaf == 2


def test_calibrated_estimator_wraps_the_pipeline_with_isotonic_calibration(labelled_customers) -> None:
    customers, target = labelled_customers
    estimator = build_configured_estimator(RANDOM_FOREST, IMBALANCE_NONE, {"model__n_estimators": 10}, calibrated=True)

    proba = estimator.fit(customers, target).predict_proba(customers)[:, 1]

    assert isinstance(estimator, CalibratedClassifierCV)
    assert estimator.method == CALIBRATION_METHOD
    assert ((proba >= 0) & (proba <= 1)).all()


def test_final_pipeline_is_built_from_the_config() -> None:
    config = FinalModelConfig(XGBOOST, IMBALANCE_NONE, {"model__n_estimators": 9}, calibrated=False, threshold=0.2)

    assert build_final_pipeline(config).named_steps["model"].n_estimators == 9


def test_selected_config_is_read_from_the_summary_file(tmp_path) -> None:
    summary = {"selected": {"model": XGBOOST, "strategy": IMBALANCE_NONE, "params": {"model__max_depth": 4}, "calibrated": True, "threshold": 0.2151}}
    path = tmp_path / "hafta7_ozet.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    config = load_selected_config(path)

    assert config == FinalModelConfig(XGBOOST, IMBALANCE_NONE, {"model__max_depth": 4}, True, 0.2151)


def test_missing_summary_file_error_names_the_command_to_produce_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build_hiperparametre.py"):
        load_selected_config(tmp_path / "yok.json")
