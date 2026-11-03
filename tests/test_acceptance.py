import copy
import json

import pytest

from reports.final_rapor import ACCEPTANCE_CRITERIA
from src.acceptance import CRITERION_COUNT, NOT_VERIFIED, evaluate_acceptance
from src.config import PROJECT_ROOT

HAFTALIK_DIR = PROJECT_ROOT / "reports" / "haftalik"
VERIFICATION = {"commit": "abcdef1234567", "tests_passed": True, "test_count": 300, "numbers_match": True}
TIMING = {"single_prediction_seconds": 0.04}
CLEAN_CLONE_INDEXES = (2, 11)
FIRST_ANALYSIS_WEEK, LAST_ANALYSIS_WEEK = 2, 8


@pytest.fixture(scope="module")
def summaries() -> dict:
    return {week: json.loads((HAFTALIK_DIR / f"hafta{week}_ozet.json").read_text(encoding="utf-8")) for week in range(FIRST_ANALYSIS_WEEK, LAST_ANALYSIS_WEEK + 1)} | {9: {}}


def copy_of(value: dict) -> dict:
    return copy.deepcopy(value)


def test_there_is_one_result_per_prd_criterion(summaries) -> None:
    results = evaluate_acceptance(summaries, True, TIMING, VERIFICATION)

    assert len(results) == len(ACCEPTANCE_CRITERIA) == CRITERION_COUNT
    assert all(item.evidence for item in results)


def test_all_criteria_are_met_with_the_repository_results_and_a_clean_clone_check(summaries) -> None:
    results = evaluate_acceptance(summaries, True, TIMING, VERIFICATION)

    assert [name for name, item in zip(ACCEPTANCE_CRITERIA, results) if not item.met] == []


def test_clean_clone_criteria_are_not_met_without_a_verification_file(summaries) -> None:
    results = evaluate_acceptance(summaries, True, TIMING, None)

    for index in CLEAN_CLONE_INDEXES:
        assert not results[index].met and results[index].evidence == NOT_VERIFIED


def test_failed_clean_clone_check_is_reported_as_not_met(summaries) -> None:
    results = evaluate_acceptance(summaries, True, TIMING, {**VERIFICATION, "tests_passed": False, "numbers_match": False})

    assert not results[2].met and not results[11].met


def test_test_roc_auc_below_the_target_fails_criterion_five(summaries) -> None:
    changed = copy_of(summaries)
    changed[8]["test_metrics"]["selected"]["metrics"]["roc_auc"] = 0.70

    assert not evaluate_acceptance(changed, True, TIMING, VERIFICATION)[4].met


def test_pr_auc_below_the_target_fails_criterion_six(summaries) -> None:
    changed = copy_of(summaries)
    changed[8]["fog1"]["pr_auc_met"] = False

    assert not evaluate_acceptance(changed, True, TIMING, VERIFICATION)[5].met


def test_zero_recall_fails_criterion_seven(summaries) -> None:
    changed = copy_of(summaries)
    changed[8]["test_metrics"]["selected"]["metrics"]["recall"] = 0.0

    assert not evaluate_acceptance(changed, True, TIMING, VERIFICATION)[6].met


def test_reload_mismatch_fails_criterion_ten(summaries) -> None:
    assert not evaluate_acceptance(summaries, False, TIMING, VERIFICATION)[9].met


@pytest.mark.parametrize("timing", [{"single_prediction_seconds": 1.5}, None])
def test_slow_or_missing_timing_fails_criterion_eleven(summaries, timing) -> None:
    assert not evaluate_acceptance(summaries, True, timing, VERIFICATION)[10].met


def test_missing_weekly_summary_fails_criterion_thirteen(summaries) -> None:
    changed = {week: value for week, value in summaries.items() if week != 3}

    assert not evaluate_acceptance(changed, True, TIMING, VERIFICATION)[12].met
