import pytest
from docx import Document

from reports.final_rapor import (
    ACCEPTANCE_CRITERIA,
    PENDING_MARK,
    PENDING_SECTIONS,
    SECTION_TITLES,
    build_final_report,
    final_file_name,
    load_summaries,
)
from src.formatting import format_number
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260054"
ACCEPTANCE_CRITERIA_COUNT = 13


@pytest.fixture(scope="module")
def summaries() -> dict:
    return load_summaries()


@pytest.fixture(scope="module")
def draft_document(summaries, tmp_path_factory):
    path = build_final_report(summaries, CAPTAIN_ID, draft=True).save(tmp_path_factory.mktemp("rapor") / "final.docx")
    return Document(str(path))


def test_final_report_is_not_produced_without_the_draft_flag(summaries) -> None:
    with pytest.raises(ValueError, match="--taslak"):
        build_final_report(summaries, CAPTAIN_ID, draft=False)


def test_final_report_draft_contains_every_section_and_the_acceptance_table(draft_document) -> None:
    text = document_text(draft_document)

    assert [title for title in SECTION_TITLES if title not in text] == []
    assert "EK-A. Kabul Kriterleri Kanıt Tablosu" in text
    assert "FİNAL RAPORU" in text
    assert "HAFTA İLERLEME RAPORU" not in text
    assert "TAKIM KAPTANI GERİ BİLDİRİM FORMU" not in text


def test_acceptance_table_lists_all_thirteen_prd_criteria(draft_document) -> None:
    acceptance_table = next(table for table in draft_document.tables if table.rows[0].cells[0].text == "No")

    assert len(ACCEPTANCE_CRITERIA) == ACCEPTANCE_CRITERIA_COUNT
    assert len(acceptance_table.rows) == ACCEPTANCE_CRITERIA_COUNT + 1


def test_placeholders_appear_only_in_the_pending_sections_and_the_acceptance_table(draft_document) -> None:
    text = document_text(draft_document)

    assert text.count(PENDING_MARK) == len(PENDING_SECTIONS) + ACCEPTANCE_CRITERIA_COUNT


def test_final_report_numbers_come_from_the_weekly_summaries(draft_document, summaries) -> None:
    text = document_text(draft_document)

    assert format_number(summaries[2]["dataset_shape"]["rows"]) in text
    assert format_number(summaries[5]["feature_counts"]["tree"]) in text


def test_final_file_name_uses_sorted_student_numbers() -> None:
    assert final_file_name() == "230260054_230260143_230260173_final.docx"
