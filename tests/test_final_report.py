import json

import pytest
from docx import Document

from reports.final_rapor import (
    ACCEPTANCE_CRITERIA,
    HAFTALIK_DIR,
    SECTION_TITLES,
    build_final_report,
    final_file_name,
    load_summaries,
)
from src.formatting import format_number
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260054"
ACCEPTANCE_CRITERIA_COUNT = 13
FIGURE_COUNT = 5 + 1
PLACEHOLDER_MARKS = ["[TASLAK]", "yazılacak", "eklenecek", "doldurulacak"]
FORBIDDEN_PHRASES = ["Oysa", "belirgin şekilde", "ortaya konmuş", "dikkat çekici", "anlamına gelmez", "kadarı", ";"]


@pytest.fixture(scope="module")
def summaries() -> dict:
    return load_summaries()


@pytest.fixture(scope="module")
def security() -> dict:
    return json.loads((HAFTALIK_DIR / "hafta9_guvenlik.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def final_document(summaries, security, tmp_path_factory):
    path = build_final_report(summaries, CAPTAIN_ID, security).save(tmp_path_factory.mktemp("rapor") / "final.docx")
    return Document(str(path))


def test_final_report_contains_every_section_and_the_acceptance_table(final_document) -> None:
    text = document_text(final_document)

    assert [title for title in SECTION_TITLES if title not in text] == []
    assert "EK-A. Kabul Kriterleri Kanıt Tablosu" in text
    assert "FİNAL RAPORU" in text
    assert "HAFTA İLERLEME RAPORU" not in text
    assert "TAKIM KAPTANI GERİ BİLDİRİM FORMU" not in text


def test_acceptance_table_lists_all_thirteen_prd_criteria_with_evidence(final_document, summaries) -> None:
    acceptance_table = next(table for table in final_document.tables if [cell.text for cell in table.rows[0].cells] == ["No", "Kriter", "Durum", "Kanıt"])

    assert len(ACCEPTANCE_CRITERIA) == ACCEPTANCE_CRITERIA_COUNT
    assert len(acceptance_table.rows) == ACCEPTANCE_CRITERIA_COUNT + 1
    for row, item in zip(acceptance_table.rows[1:], summaries[9]["acceptance_criteria"]):
        assert row.cells[1].text == item["criterion"] and row.cells[3].text == item["evidence"]


def test_no_placeholder_text_is_left_in_the_report(final_document) -> None:
    text = document_text(final_document)

    assert [mark for mark in PLACEHOLDER_MARKS if mark in text] == []


def test_final_report_avoids_forbidden_phrases(final_document) -> None:
    text = document_text(final_document)

    assert [phrase for phrase in FORBIDDEN_PHRASES if phrase in text] == []


def test_final_report_embeds_the_evaluation_and_interface_figures(final_document) -> None:
    assert len(final_document.inline_shapes) >= FIGURE_COUNT


def test_final_report_numbers_come_from_the_weekly_summaries(final_document, summaries) -> None:
    text = document_text(final_document)

    assert format_number(summaries[2]["dataset_shape"]["rows"]) in text
    assert format_number(summaries[5]["feature_counts"]["tree"]) in text
    assert format_number(summaries[8]["test_metrics"]["selected"]["metrics"]["roc_auc"], 3) in text
    assert format_number(summaries[8]["time_based"]["roc_auc"], 3) in text
    assert summaries[9]["demo_prediction"]["band"] in text


def test_evaluation_and_service_sections_use_final_report_numbering(final_document) -> None:
    text = document_text(final_document)

    for heading in ("7.1. Final Modelin Test Sonuçları", "7.6. Hata Analizi", "8.1. Modelin Kaydedilmesi", "8.4. Güvenlik İncelemesi", "9.2. Sınırlılıklar"):
        assert heading in text
    assert "4.1. Final Modelin Test Sonuçları" not in text


def test_usability_trial_is_described_as_future_work(final_document) -> None:
    assert "kullanılabilirlik denemesi yapılacaktır" in document_text(final_document)


def test_final_file_name_uses_sorted_student_numbers() -> None:
    assert final_file_name() == "230260054_230260143_230260173_final.docx"
