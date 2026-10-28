import json

import pytest
from docx import Document

from reports.hafta4_rapor import SUMMARY_PATH, build_report, check_decisions_match_data, describe_comparison, fold_count_text
from src.formatting import format_number, format_percent
from tests.test_report_docx import document_text

CAPTAIN_ID = "230260054"
WEEK4_HEADINGS = [
    "4. HAFTA İLERLEME RAPORU",
    "4.1. Ön İşleme Pipeline'ı",
    "4.2. Veri Sızıntısı Testleri",
    "4.3. Çapraz Doğrulama Düzeni",
    "4.4. İlk Lojistik Regresyon Modeli",
    "4.5. Üçüncü Haftadan Kalan Sorular",
    "KAYNAKLAR",
]


@pytest.fixture(scope="module")
def week4_document(tmp_path_factory):
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    path = build_report(summary, CAPTAIN_ID).save(tmp_path_factory.mktemp("rapor") / "hafta4.docx")
    return Document(str(path)), summary


def test_week4_report_contains_all_sections(week4_document) -> None:
    document, _ = week4_document

    text = document_text(document)

    assert [heading for heading in WEEK4_HEADINGS if heading not in text] == []


def test_week4_report_embeds_pr_roc_figure(week4_document) -> None:
    document, _ = week4_document

    assert len(document.inline_shapes) == 1


def test_week4_report_numbers_come_from_summary(week4_document) -> None:
    document, summary = week4_document
    logistic = summary["models"]["logistic_regression"]["summary"]

    text = document_text(document)

    assert f"{format_number(logistic['pr_auc']['mean'], 3)} ± {format_number(logistic['pr_auc']['std'], 3)}" in text
    assert format_number(summary["split"]["train_rows"]) in text
    assert format_percent(summary["oof_predicted_positive_rate"]) in text
    assert str(summary["model_feature_count"]) in text


@pytest.mark.parametrize(
    ("improved", "total", "expected"),
    [(4, 5, "5 katmanın 4'ünde"), (1, 5, "5 katmanın 1'inde"), (2, 5, "5 katmanın 2'sinde"), (0, 5, "5 katmanın 0'ında"), (5, 5, "katmanların hepsinde")],
)
def test_fold_count_text_uses_turkish_locative_suffix(improved: int, total: int, expected: str) -> None:
    assert fold_count_text(improved, total) == expected


def test_comparison_text_reports_direction_and_noise_level() -> None:
    worse = {"mean_difference": -0.002, "std_difference": 0.001, "folds_improved": 1, "fold_count": 5}

    text = describe_comparison(worse, "Bu durumda")

    assert "0,0020 azaldı" in text
    assert "standart sapmasından (0,0010) büyük" in text
    assert not text.startswith("Ancak")


def test_comparison_text_handles_no_improved_fold_and_zero_difference() -> None:
    no_fold = {"mean_difference": -0.001, "std_difference": 0.002, "folds_improved": 0, "fold_count": 5}
    unchanged = {"mean_difference": 0.0, "std_difference": 0.002, "folds_improved": 2, "fold_count": 5}

    assert "hiçbir katmanda yükselmedi" in describe_comparison(no_fold, "Bu durumda")
    assert "ortalaması değişmedi" in describe_comparison(unchanged, "Bu durumda")


def test_report_refuses_decisions_that_contradict_the_data() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["comparisons"]["age_group_added"]["mean_difference"] = -0.01

    with pytest.raises(ValueError, match="Yaş grubu"):
        check_decisions_match_data(summary)
