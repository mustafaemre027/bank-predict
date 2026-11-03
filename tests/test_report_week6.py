import json

import pytest
from docx import Document

from reports.hafta6_rapor import (
    SUMMARY_PATH,
    build_report,
    check_class_weight_raises_recall,
    describe_best_strategies,
    describe_pr_auc_effects,
    describe_selected_threshold_f1,
)
from src.formatting import format_number
from src.report_text import lower_first, number_word
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260143"
WEEK6_HEADINGS = [
    "6. HAFTA İLERLEME RAPORU",
    "4.1. Sınıf Dengesizliği Stratejileri",
    "4.2. SMOTE'un Yalnızca Eğitim Verisine Uygulanması",
    "4.3. Strateji ve Model Karşılaştırması",
    "4.4. Karar Eşiğinin Seçilmesi",
    "4.5. Literatürle Karşılaştırma",
    "KAYNAKLAR",
]


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def week6_document(summary, tmp_path_factory):
    path = build_report(summary, CAPTAIN_ID).save(tmp_path_factory.mktemp("rapor") / "hafta6.docx")
    return Document(str(path))


def copy_of(summary: dict) -> dict:
    return json.loads(json.dumps(summary))


def test_week6_report_contains_all_sections(week6_document) -> None:
    text = document_text(week6_document)

    assert [heading for heading in WEEK6_HEADINGS if heading not in text] == []


def test_week6_report_embeds_both_figures(week6_document) -> None:
    assert len(week6_document.inline_shapes) == 2


def test_week6_report_numbers_come_from_summary(week6_document, summary) -> None:
    best = summary["experiments"][summary["ranking"][0]["key"]]

    text = document_text(week6_document)

    assert format_number(best["threshold_f1"]["threshold"], 2) in text
    assert format_number(best["threshold_recall"]["threshold"], 2) in text
    assert format_number(summary["split"]["train_rows"]) in text


def test_week6_report_names_the_selected_captain(week6_document) -> None:
    info_rows = {row.cells[0].text: row.cells[1].text for row in week6_document.tables[0].rows}

    assert info_rows["Takım Kaptanı:"] == "Mustafa Emre KAPLAN (230260143)"


def test_captain_row_keeps_the_smote_acronym_intact(week6_document) -> None:
    text = document_text(week6_document)

    assert "SMOTE'un yalnızca eğitim katmanlarına uygulandığını" in text
    assert "sMOTE" not in text


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Üç modelin", "üç modelin"), ("SMOTE'un", "SMOTE'un"), ("A", "a"), ("", ""), ("Işık", "ışık"), ("İlk", "ilk")],
)
def test_lower_first_leaves_acronyms_alone(text: str, expected: str) -> None:
    assert lower_first(text) == expected


def test_report_refuses_to_claim_higher_recall_when_class_weight_does_not_raise_it(summary) -> None:
    broken = copy_of(summary)
    broken["experiments"]["xgboost__class_weight"]["summary"]["recall"]["mean"] = 0.1

    with pytest.raises(ValueError, match="XGBoost"):
        check_class_weight_raises_recall(broken)


def test_pr_auc_text_names_the_strategies_that_improved_instead_of_claiming_no_gain(summary) -> None:
    improved = copy_of(summary)
    improved["strategy_effects"]["random_forest__smote"] = {"mean_difference": 0.02, "std_difference": 0.004, "folds_improved": 5, "fold_count": 5}

    assert "hiçbirinde PR-AUC'yi artırmadı" in describe_pr_auc_effects(summary)
    text = describe_pr_auc_effects(improved)
    assert "hiçbirinde" not in text
    assert "Rastgele Orman modelinde smote kullanıldığında" in text.replace("SMOTE", "smote")


def test_best_strategy_text_lists_strategies_when_not_every_model_prefers_the_unweighted_one(summary) -> None:
    mixed = copy_of(summary)
    mixed["best_by_model"]["random_forest"] = "smote"

    assert "eşik ayarlamasını kullanmaya karar verdik" in describe_best_strategies(summary)
    assert "Rastgele Orman için smote" in describe_best_strategies(mixed).replace("SMOTE", "smote")
    assert "karar verdik" not in describe_best_strategies(mixed)


def test_selected_threshold_text_claims_no_large_advantage_only_when_f1_values_are_close(summary) -> None:
    far_apart = copy_of(summary)
    far_apart["experiments"]["xgboost__smote"]["nested_summary_f1_threshold"]["f1"]["mean"] = 0.3

    assert "büyük bir üstünlük kalmadı" in describe_selected_threshold_f1(summary)
    assert "büyük bir üstünlük kalmadı" not in describe_selected_threshold_f1(far_apart)


@pytest.mark.parametrize(("number", "expected"), [(0, "sıfır"), (2, "iki"), (9, "dokuz"), (10, "on"), (11, "11")])
def test_number_word_spells_small_numbers_and_keeps_large_ones_as_digits(number: int, expected: str) -> None:
    assert number_word(number) == expected
