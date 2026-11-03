import json

import pytest
from docx import Document

from reports.hafta8_rapor import (
    SUMMARY_PATH,
    build_report,
    check_confusion_adds_up,
    check_fog1_matches_metrics,
    check_gain_table_is_non_decreasing,
    check_recall_is_positive,
    check_threshold_matches_selection,
    describe_cv_versus_test,
    describe_fog1,
    describe_literature,
    describe_overlap,
    describe_time_based,
    describe_time_drop_hint,
)
from src.formatting import format_number
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260143"
WEEK8_HEADINGS = [
    "8. HAFTA İLERLEME RAPORU",
    "4.1. Final Modelin Test Sonuçları",
    "4.2. Karmaşıklık Matrisi",
    "4.3. Kazanç Tablosu",
    "4.4. Zamana Göre Ayrılmış Test",
    "4.5. Değişken Önemleri",
    "4.6. Hata Analizi",
    "4.7. Literatürle Karşılaştırma",
    "KAYNAKLAR",
]
# Raporun üslup kuralları: bu kalıplar metinde bulunmamalı.
FORBIDDEN_PHRASES = ["Oysa", "belirgin şekilde", "ortaya konmuş", "dikkat çekici", "anlamına gelmez", "kadarı", ";"]
EXPECTED_FIGURE_COUNT = 5


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def week8_document(summary, tmp_path_factory):
    path = build_report(summary, CAPTAIN_ID).save(tmp_path_factory.mktemp("rapor") / "hafta8.docx")
    return Document(str(path))


def copy_of(summary: dict) -> dict:
    return json.loads(json.dumps(summary))


def failing_fog1(summary: dict) -> dict:
    changed = copy_of(summary)
    changed["test_metrics"]["selected"]["metrics"]["roc_auc"] = 0.70
    changed["fog1"].update({"roc_auc_met": False, "met": False})
    return changed


def test_week8_report_contains_all_sections(week8_document) -> None:
    text = document_text(week8_document)

    assert [heading for heading in WEEK8_HEADINGS if heading not in text] == []


def test_week8_report_embeds_all_five_figures(week8_document) -> None:
    assert len(week8_document.inline_shapes) == EXPECTED_FIGURE_COUNT


def test_week8_report_numbers_come_from_summary(week8_document, summary) -> None:
    text = document_text(week8_document)
    counts = summary["confusion"]

    assert format_number(summary["split"]["test_rows"]) in text
    assert format_number(counts["tp"]) in text and format_number(counts["fn"]) in text
    assert format_number(summary["test_metrics"]["selected"]["metrics"]["roc_auc"], 3) in text
    assert format_number(summary["time_based"]["roc_auc"], 3) in text
    for row in summary["gain_table"]:
        assert format_number(row["captured_positives"]) in text
    for row in summary["permutation_importance"][:10]:
        assert row["feature"] in text


def test_week8_report_names_the_selected_captain(week8_document) -> None:
    info_rows = {row.cells[0].text: row.cells[1].text for row in week8_document.tables[0].rows}

    assert info_rows["Takım Kaptanı:"] == "Mustafa Emre KAPLAN (230260143)"


def test_week8_report_avoids_forbidden_phrases(week8_document) -> None:
    text = document_text(week8_document)

    assert [phrase for phrase in FORBIDDEN_PHRASES if phrase in text] == []


def test_fog1_text_reports_a_met_target(summary) -> None:
    text = describe_fog1(summary)

    assert "karşılandı" in text and "karşılanmadı" not in text


def test_fog1_text_switches_to_the_not_met_version_with_reason_and_next_step(summary) -> None:
    text = describe_fog1(failing_fog1(summary))

    assert "karşılanmadı" in text
    assert "Sağlanamayan ölçüt: ROC-AUC (0,700" in text
    assert "test sonucuna göre değiştirilmedi" in text
    assert "Sonraki adım" in text


def test_goal_sentence_follows_the_fog1_result(summary, tmp_path) -> None:
    passing = document_text(Document(str(build_report(summary, CAPTAIN_ID).save(tmp_path / "met.docx"))))
    failing = document_text(Document(str(build_report(failing_fog1(summary), CAPTAIN_ID).save(tmp_path / "missed.docx"))))

    assert "Bu hedef sekizinci haftada karşılanmıştır" in passing
    assert "Bu hedef sekizinci haftada karşılanamamıştır" in failing


def test_report_generation_stops_when_the_threshold_was_changed_after_week7(summary) -> None:
    changed = copy_of(summary)
    changed["test_metrics"]["selected"]["threshold"] = 0.3

    with pytest.raises(ValueError, match="eşikten"):
        check_threshold_matches_selection(changed)


def test_report_generation_stops_when_confusion_counts_do_not_add_up(summary) -> None:
    changed = copy_of(summary)
    changed["confusion"]["tp"] += 1

    with pytest.raises(ValueError, match="toplamı"):
        check_confusion_adds_up(changed)


def test_report_generation_stops_when_recall_is_zero(summary) -> None:
    changed = copy_of(summary)
    changed["recall_positive"] = False

    with pytest.raises(ValueError, match="Kabul kriteri 7"):
        check_recall_is_positive(changed)


def test_report_generation_stops_when_fog1_flag_contradicts_the_metrics(summary) -> None:
    changed = copy_of(summary)
    changed["fog1"]["met"] = not changed["fog1"]["met"]

    with pytest.raises(ValueError, match="tutarsız"):
        check_fog1_matches_metrics(changed)


def test_report_generation_stops_when_the_gain_table_decreases(summary) -> None:
    changed = copy_of(summary)
    changed["gain_table"][-1]["captured_positives"] = 0

    with pytest.raises(ValueError, match="Sıralama bozuk"):
        check_gain_table_is_non_decreasing(changed)


@pytest.mark.parametrize(
    ("test_value", "cv_std", "direction", "size"),
    [(0.50, 0.06, "üstünde", "küçük"), (0.30, 0.01, "altında", "büyük")],
)
def test_cv_versus_test_text_follows_direction_and_size(test_value, cv_std, direction, size) -> None:
    text = describe_cv_versus_test("PR-AUC", {"mean": 0.45, "std": cv_std}, test_value)

    assert f"ortalamasının (0,450) {direction}" in text
    assert f"standart sapmadan {size}" in text


def test_time_based_text_follows_the_direction_of_the_change(summary) -> None:
    lower, higher = copy_of(summary), copy_of(summary)
    lower["time_based"]["roc_auc"] = 0.60
    higher["time_based"]["roc_auc"] = 0.95

    assert "değerine göre düştü" in describe_time_based(lower)
    assert "değerine göre yükseldi" in describe_time_based(higher)


def test_calendar_hint_appears_only_when_time_based_result_dropped_and_month_or_day_lead(summary) -> None:
    dropped, not_dropped, no_calendar = copy_of(summary), copy_of(summary), copy_of(summary)
    dropped["time_based"]["roc_auc"] = 0.60
    dropped["permutation_importance"][0]["feature"] = "month"
    not_dropped["time_based"]["roc_auc"] = 0.95
    no_calendar["time_based"]["roc_auc"] = 0.60
    no_calendar["permutation_importance"] = [{"feature": name} for name in ["age", "job", "balance", "loan", "default"]]

    assert "month" in describe_time_drop_hint(dropped) and "sınanmamıştır" in describe_time_drop_hint(dropped)
    assert describe_time_drop_hint(not_dropped) == ""
    assert describe_time_drop_hint(no_calendar) == ""


def test_overlap_text_names_shared_variables_or_says_there_are_none(summary) -> None:
    none_shared = copy_of(summary)
    none_shared["importance_overlap_top5"] = {"count": 0, "variables": []}

    assert "ortak değişken bulunmuyor" in describe_overlap(none_shared)
    assert f"{summary['importance_overlap_top5']['variables'][0]}" in describe_overlap(summary)


def test_literature_text_follows_whether_the_model_beats_the_published_value(summary) -> None:
    above, below = copy_of(summary), copy_of(summary)
    above["test_metrics"]["selected"]["metrics"]["roc_auc"] = 0.9
    below["test_metrics"]["selected"]["metrics"]["roc_auc"] = 0.7

    assert "sayısal olarak yüksek kaldı" in describe_literature(above)
    assert "sayısal olarak düşük kaldı" in describe_literature(below)
