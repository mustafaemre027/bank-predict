import json

import pytest
from docx import Document

from reports.hafta5_rapor import (
    SUMMARY_PATH,
    build_report,
    check_ablation_keeps_defaults,
    describe_recall,
    describe_tree_gap,
)
from src.formatting import format_number
from src.report_text import describe_change
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260173"
WEEK5_HEADINGS = [
    "5. HAFTA İLERLEME RAPORU",
    "4.1. Ağaç Modelleri İçin Ön İşleme",
    "4.2. Rastgele Orman ve XGBoost",
    "4.3. Üç Modelin Karşılaştırılması",
    "4.4. Ablasyon Çalışması",
    "4.5. Sonuçların Tekrar Üretilebilirliği",
    "KAYNAKLAR",
]


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def week5_document(summary, tmp_path_factory):
    path = build_report(summary, CAPTAIN_ID).save(tmp_path_factory.mktemp("rapor") / "hafta5.docx")
    return Document(str(path))


def test_week5_report_contains_all_sections(week5_document) -> None:
    text = document_text(week5_document)

    assert [heading for heading in WEEK5_HEADINGS if heading not in text] == []


def test_week5_report_embeds_both_comparison_figures(week5_document) -> None:
    assert len(week5_document.inline_shapes) == 2


def test_week5_report_numbers_come_from_summary(week5_document, summary) -> None:
    best = summary["models"][summary["best_model"]]["summary"]["pr_auc"]

    text = document_text(week5_document)

    assert f"{format_number(best['mean'], 3)} ± {format_number(best['std'], 3)}" in text
    assert format_number(summary["split"]["train_rows"]) in text
    assert str(summary["feature_counts"]["tree"]) in text
    assert str(summary["feature_counts"]["logistic_regression"]) in text


def test_week5_report_names_the_selected_captain(week5_document) -> None:
    info_rows = {row.cells[0].text: row.cells[1].text for row in week5_document.tables[0].rows}

    assert info_rows["Takım Kaptanı:"] == "Mesut ASLAN (230260173)"


def test_week5_report_names_best_model_from_summary(summary, tmp_path) -> None:
    other_best = {**summary, "best_model": "random_forest"}

    text = document_text(Document(str(build_report(other_best, CAPTAIN_ID).save(tmp_path / "hafta5.docx"))))

    assert "en iyi ayırt eden Rastgele Orman oldu" in text


def test_report_refuses_to_describe_defaults_as_kept_when_an_ablation_is_a_candidate(summary) -> None:
    with_candidate = {**summary, "ablation_candidates": ["rf_without_day"]}

    with pytest.raises(ValueError, match="rf_without_day"):
        check_ablation_keeps_defaults(with_candidate)


def test_change_text_covers_no_change_and_no_improving_fold() -> None:
    unchanged = {"mean_difference": 0.0, "std_difference": 0.004, "folds_improved": 3, "fold_count": 5}
    worse = {"mean_difference": -0.0163, "std_difference": 0.0038, "folds_improved": 0, "fold_count": 5}

    assert describe_change(unchanged) == "PR-AUC ortalaması değişmedi ve 5 katmanın 3'ünde yükseldi."
    assert describe_change(worse) == "PR-AUC ortalamada 0,0163 azaldı ve hiçbir katmanda yükselmedi."


def test_tree_gap_is_not_ranked_when_difference_is_within_noise(summary) -> None:
    noisy = {**summary, "comparisons": {**summary["comparisons"], "xgboost_vs_random_forest": {"mean_difference": 0.006, "std_difference": 0.0084, "folds_improved": 4, "fold_count": 5}}}
    clear = {**summary, "comparisons": {**summary["comparisons"], "xgboost_vs_random_forest": {"mean_difference": 0.02, "std_difference": 0.004, "folds_improved": 5, "fold_count": 5}}}

    assert "üstün saymıyoruz" in describe_tree_gap(noisy)
    assert "üstün saymıyoruz" not in describe_tree_gap(clear)


def test_recall_text_only_calls_recall_low_when_every_model_is_below_half(summary) -> None:
    high_recall = json.loads(json.dumps(summary))
    for model in ("logistic_regression", "random_forest", "xgboost"):
        high_recall["models"][model]["summary"]["recall"]["mean"] = 0.8

    assert "küçük bir kısmını" in describe_recall(summary)
    assert "küçük bir kısmını" not in describe_recall(high_recall)
