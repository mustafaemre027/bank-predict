import inspect

import numpy as np
import pytest

from src.threshold import CRITERION_MAX_F1, CRITERION_TARGET_RECALL, select_threshold_for_recall, select_threshold_max_f1

# Kabul edenler 0,9 / 0,8 / 0,7 / 0,5 skorlu müşteriler. 0,5 eşiğinde 5 kişi aranır ve 4'ü gerçek kabuldür.
LABELS = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])


def test_max_f1_threshold_matches_hand_computed_example() -> None:
    choice = select_threshold_max_f1(LABELS, SCORES)

    assert choice.threshold == 0.5
    assert choice.precision == pytest.approx(4 / 5)
    assert choice.recall == pytest.approx(1.0)
    assert choice.f1 == pytest.approx(2 * 0.8 * 1.0 / 1.8)
    assert choice.criterion == CRITERION_MAX_F1


def test_max_f1_picks_the_higher_threshold_when_f1_is_tied() -> None:
    labels = np.array([0, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    choice = select_threshold_max_f1(labels, scores)

    assert choice.threshold == 0.8
    assert choice.f1 == pytest.approx(0.5)


def test_target_recall_threshold_is_the_highest_one_that_reaches_the_target() -> None:
    choice = select_threshold_for_recall(LABELS, SCORES, target_recall=0.6)

    assert choice.threshold == 0.7
    assert choice.recall == pytest.approx(0.75)
    assert choice.precision == pytest.approx(1.0)
    assert choice.criterion == CRITERION_TARGET_RECALL


def test_target_recall_of_one_needs_the_lowest_positive_score() -> None:
    choice = select_threshold_for_recall(LABELS, SCORES, target_recall=1.0)

    assert choice.threshold == 0.5
    assert choice.recall == pytest.approx(1.0)


@pytest.mark.parametrize("target_recall", [0.0, -0.1, 1.5])
def test_target_recall_outside_zero_to_one_is_rejected(target_recall: float) -> None:
    with pytest.raises(ValueError, match="Hedef Recall"):
        select_threshold_for_recall(LABELS, SCORES, target_recall=target_recall)


def test_reported_recall_matches_applying_the_chosen_threshold() -> None:
    choice = select_threshold_for_recall(LABELS, SCORES, target_recall=0.7)

    predicted_positive = SCORES >= choice.threshold

    assert (predicted_positive & (LABELS == 1)).sum() / LABELS.sum() == pytest.approx(choice.recall)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="farklı"):
        select_threshold_max_f1(LABELS, SCORES[:-1])


def test_single_class_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="iki sınıf"):
        select_threshold_max_f1(np.zeros(5, dtype=int), np.linspace(0.1, 0.9, 5))


@pytest.mark.parametrize("select", [select_threshold_max_f1, select_threshold_for_recall])
def test_selection_accepts_only_labels_and_oof_probabilities(select) -> None:
    parameters = [name for name in inspect.signature(select).parameters if name != "target_recall"]

    assert parameters == ["y_true", "oof_proba"]
