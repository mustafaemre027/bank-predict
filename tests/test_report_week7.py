import json

import pytest
from docx import Document

from reports.hafta7_rapor import (
    SUMMARY_PATH,
    build_report,
    calibration_reason,
    describe_selection,
    describe_timing,
    format_duration,
    format_tuned_params,
)
from src.formatting import format_number
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260173"
WEEK7_HEADINGS = [
    "7. HAFTA İLERLEME RAPORU",
    "4.1. Hiperparametre Araması",
    "4.2. Ayarlanmış Modellerin Karşılaştırılması",
    "4.3. Olasılık Kalibrasyonu",
    "4.4. Final Aday Model ve Karar Eşiği",
    "4.5. Final Raporu İskeleti",
    "KAYNAKLAR",
]


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def week7_document(summary, tmp_path_factory):
    timing = {"search_seconds": {"a": 10.0}, "total_search_seconds": 10.0}
    path = build_report(summary, CAPTAIN_ID, timing).save(tmp_path_factory.mktemp("rapor") / "hafta7.docx")
    return Document(str(path))


def copy_of(summary: dict) -> dict:
    return json.loads(json.dumps(summary))


def test_week7_report_contains_all_sections(week7_document) -> None:
    text = document_text(week7_document)

    assert [heading for heading in WEEK7_HEADINGS if heading not in text] == []


def test_week7_report_embeds_search_and_calibration_figures(week7_document) -> None:
    assert len(week7_document.inline_shapes) == 2


def test_week7_report_numbers_come_from_summary(week7_document, summary) -> None:
    selected = summary["selected"]

    text = document_text(week7_document)

    assert format_number(summary["split"]["train_rows"]) in text
    assert format_number(selected["threshold_f1"]["threshold"], 2) in text
    assert format_number(selected["threshold_recall"]["threshold"], 2) in text
    for entry in summary["candidates"]:
        assert format_number(entry["tuned"]["summary"]["pr_auc"]["mean"], 3) in text


def test_week7_report_names_the_selected_captain(week7_document) -> None:
    info_rows = {row.cells[0].text: row.cells[1].text for row in week7_document.tables[0].rows}

    assert info_rows["Takım Kaptanı:"] == "Mesut ASLAN (230260173)"


def test_week7_report_states_calibration_decision_for_every_candidate(week7_document, summary) -> None:
    text = document_text(week7_document)

    for entry in summary["candidates"]:
        decision = "uygulandı" if entry["calibration"]["applied"] else "uygulanmadı"
        assert f"kalibrasyon {decision}, çünkü" in text


def test_selection_text_names_the_simpler_model_when_the_difference_is_within_noise(summary) -> None:
    best_model, other = summary["candidates"][0]["model"], summary["candidates"][1]["model"]
    simpler = {**summary["selected"], "model": other, "reason": "simpler_within_noise", "highest_pr_auc_model": best_model}
    highest = {**summary["selected"], "model": best_model, "reason": "highest_pr_auc"}

    assert "aynı başarıda daha basit" in describe_selection(simpler, summary["candidates"])
    assert "aynı başarıda daha basit" not in describe_selection(highest, summary["candidates"])
    assert "en yüksek aday" in describe_selection(highest, summary["candidates"])


@pytest.mark.parametrize(
    ("strategy", "applied", "brier_raw", "brier_calibrated", "expected"),
    [
        ("class_weight", True, 0.08, 0.09, "zorunludur"),
        ("none", True, 0.08, 0.07, "Brier skoru düştü"),
        ("none", False, 0.07, 0.08, "Brier skoru düşmedi"),
        ("none", False, 0.08, 0.07, "sınırından fazla azaldı"),
    ],
)
def test_calibration_reason_follows_the_numbers(strategy, applied, brier_raw, brier_calibrated, expected) -> None:
    entry = {"strategy": strategy, "calibration": {"applied": applied, "brier_raw": brier_raw, "brier_calibrated": brier_calibrated}}

    assert expected in calibration_reason(entry, max_drop=0.005)


def test_timing_text_reports_whether_the_search_stayed_within_the_time_limit() -> None:
    assert "altında kaldı" in describe_timing({"total_search_seconds": 600.0})
    assert "üstüne çıktı" in describe_timing({"total_search_seconds": 2400.0})
    assert describe_timing(None) == ""


@pytest.mark.parametrize(("seconds", "expected"), [(45.2, "45 saniye"), (125.0, "2 dakika 5 saniye")])
def test_duration_is_written_in_minutes_and_seconds(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_tuned_params_are_listed_without_the_pipeline_prefix_and_with_decimal_comma() -> None:
    assert format_tuned_params({"model__learning_rate": 0.04321, "model__max_depth": 5, "model__max_features": "sqrt"}) == (
        "learning_rate = 0,04321, max_depth = 5, max_features = sqrt"
    )
