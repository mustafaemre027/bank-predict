import copy
import json

import pytest
from docx import Document

from reports.hafta9_rapor import (
    SECURITY_PATH,
    SUMMARY_PATH,
    WEEK8_SUMMARY_PATH,
    band_rule_text,
    build_report,
    check_summary,
    describe_acceptance,
    describe_security,
    describe_timing,
)
from src.formatting import format_number
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260054"
WEEK9_HEADINGS = [
    "9. HAFTA İLERLEME RAPORU",
    "3.4. Proje Sonrası",
    "4.1. Modelin Kaydedilmesi",
    "4.2. Tahmin Servisi ve Girdi Doğrulama",
    "4.3. Arayüz",
    "4.4. Güvenlik İncelemesi",
    "4.5. Projenin Genel Değerlendirmesi",
    "KAYNAKLAR",
]
FORBIDDEN_PHRASES = ["Oysa", "belirgin şekilde", "ortaya konmuş", "dikkat çekici", "anlamına gelmez", "kadarı", ";"]
USABILITY_CRITERION = "Arayüz ekip dışından bir kişiyle denendi"


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def week8() -> dict:
    return json.loads(WEEK8_SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def security() -> dict:
    return json.loads(SECURITY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def week9_document(summary, week8, security, tmp_path_factory):
    timing = {"single_prediction_seconds": 0.04, "repeats": 20}
    path = build_report(summary, CAPTAIN_ID, week8, security, timing).save(tmp_path_factory.mktemp("rapor") / "hafta9.docx")
    return Document(str(path))


def test_week9_report_contains_all_sections(week9_document) -> None:
    text = document_text(week9_document)

    assert [heading for heading in WEEK9_HEADINGS if heading not in text] == []


def test_week9_report_embeds_the_interface_screenshot(week9_document) -> None:
    assert len(week9_document.inline_shapes) == 1


def test_week9_report_numbers_come_from_the_summaries(week9_document, summary, week8) -> None:
    text = document_text(week9_document)

    assert format_number(week8["split"]["train_rows"]) in text
    assert format_number(summary["bundle"]["threshold"], 2) in text
    assert format_number(summary["demo_prediction"]["probability"] * 100, 1) in text
    for messages in summary["validation_examples"]["invalid"].values():
        assert messages[0] in text


def test_week9_report_names_the_selected_captain(week9_document) -> None:
    info_rows = {row.cells[0].text: row.cells[1].text for row in week9_document.tables[0].rows}

    assert info_rows["Takım Kaptanı:"] == "Ömer NARÇİÇEĞİ (230260054)"


def test_week9_report_avoids_forbidden_phrases(week9_document) -> None:
    text = document_text(week9_document)

    assert [phrase for phrase in FORBIDDEN_PHRASES if phrase in text] == []


def test_usability_trial_is_not_claimed_as_done(week9_document) -> None:
    text = document_text(week9_document)
    checklist = {row.cells[0].text: (row.cells[1].text, row.cells[2].text) for table in week9_document.tables[:3] for row in table.rows if len(row.cells) == 3}

    assert checklist[USABILITY_CRITERION] == ("☐", "√")
    assert "denemeyle sınanacaktır" in text and "yapılacaktır" in text


def test_acceptance_text_lists_unmet_criteria_or_confirms_all_are_met(summary) -> None:
    all_met = copy.deepcopy(summary)
    for item in all_met["acceptance_criteria"]:
        item["met"] = True
    one_unmet = copy.deepcopy(all_met)
    one_unmet["acceptance_criteria"][2]["met"] = False

    assert "tamamı karşılandı" in describe_acceptance(all_met)
    unmet_text = describe_acceptance(one_unmet)
    assert "12 tanesi karşılandı" in unmet_text and "test paketi temiz ortamda geçer" in unmet_text


def test_security_text_follows_the_findings(security) -> None:
    clean = {**security, "high": 0, "medium": 0}
    with_findings = {**security, "high": 1, "medium": 2}

    assert "bulgu çıkmadı" in describe_security(clean)
    assert "bir yüksek ve iki orta düzeyde bulgu çıktı" in describe_security(with_findings)


@pytest.mark.parametrize(("seconds", "verdict"), [(0.04, "altında kaldı"), (1.5, "üstüne çıktı")])
def test_timing_text_follows_the_one_second_limit(seconds, verdict) -> None:
    assert verdict in describe_timing({"single_prediction_seconds": seconds, "repeats": 20})
    assert describe_timing(None) == ""


def test_band_rule_text_uses_the_threshold_and_the_base_rate() -> None:
    text = band_rule_text({"threshold": 0.2435, "base_rate": 0.117})

    assert "en az 0,24" in text and "0,12 (eğitim verisindeki kabul oranı)" in text


def test_report_generation_stops_when_the_reloaded_model_does_not_match(summary) -> None:
    changed = copy.deepcopy(summary)
    changed["reload_matches_test"] = False

    with pytest.raises(ValueError, match="uyuşmuyor"):
        check_summary(changed)


def test_report_generation_stops_when_the_threshold_is_not_above_the_base_rate(summary) -> None:
    changed = copy.deepcopy(summary)
    changed["bundle"]["threshold"] = changed["bundle"]["base_rate"]

    with pytest.raises(ValueError, match="büyük olmalı"):
        check_summary(changed)
