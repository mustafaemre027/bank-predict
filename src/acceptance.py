"""PRD'deki 13 kabul kriterinin özet JSON dosyalarından değerlendirilmesi (Ek-A kanıt tablosu).

Her kriter için `met` değeri sayılardan ve dosyaların varlığından türetilir, elle işaretlenmez. Temiz klon kriterleri (3 ve 12) yalnızca
doğrulama dosyası varsa karşılanmış sayılır. Bu dosyayı, temiz klon doğrulaması yapıldıktan sonra ekip üretir.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import FEATURE_COLUMNS, FOG1_MIN_ROC_AUC, PROJECT_ROOT
from src.modeling import LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST
from src.schemas import CustomerInput

EXPECTED_ROWS = 45211
EXPECTED_COLUMNS = 17
MAX_PREDICTION_SECONDS = 1.0
MIN_IMPORTANCE_ROWS = 10
GAIN_FRACTION_COUNT = 3
REPORT_WEEKS = range(3, 10)
SUMMARY_WEEKS = range(2, 10)
NOT_VERIFIED = "Temiz klon doğrulaması henüz yapılmadı (reports/haftalik/hafta9_dogrulama.json yok)."
CRITERION_COUNT = 13


@dataclass(frozen=True)
class Acceptance:
    met: bool
    evidence: str


def _clean_clone(verification: dict[str, Any] | None, key: str, message: str) -> Acceptance:
    if verification is None:
        return Acceptance(False, NOT_VERIFIED)
    return Acceptance(bool(verification[key]), f"{message} (commit {verification['commit'][:7]}).")


def evaluate_acceptance(
    summaries: dict[int, dict[str, Any]],
    reload_matches_test: bool,
    timing: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    root: Path = PROJECT_ROOT,
) -> list[Acceptance]:
    """Kabul kriterlerini, `reports/final_rapor.py` içindeki ACCEPTANCE_CRITERIA sırasıyla değerlendirir."""
    week2, week5, week6, week8 = summaries[2], summaries[5], summaries[6], summaries[8]
    metrics = week8["test_metrics"]["selected"]["metrics"]
    shape = week2["dataset_shape"]
    prediction_seconds = None if timing is None else timing["single_prediction_seconds"]
    report_files = [root / "reports" / f"hafta{week}_rapor.py" for week in REPORT_WEEKS]
    return [
        Acceptance(shape == {"rows": EXPECTED_ROWS, "cols": EXPECTED_COLUMNS}, f"hafta2_ozet.json dataset_shape: {shape['rows']} x {shape['cols']}. Test: tests/test_data.py."),
        Acceptance(
            "duration" not in FEATURE_COLUMNS and "duration" not in CustomerInput.model_fields,
            "duration, FEATURE_COLUMNS ve CustomerInput şemasında yok. Testler: tests/test_preprocessing.py, tests/test_schemas.py.",
        ),
        _clean_clone(verification, "tests_passed", f"Temiz klonda {verification['test_count'] if verification else 0} test geçti"),
        Acceptance(
            {LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST} <= set(week5["models"]) and week6["model_count"] == 3 and week6["strategy_count"] == 3,
            "hafta5_ozet.json models (üç model, 5 katlı CV) ve hafta6_ozet.json experiments (3 model x 3 strateji).",
        ),
        Acceptance(metrics["roc_auc"] >= FOG1_MIN_ROC_AUC, f"hafta8_ozet.json test_metrics.selected.metrics.roc_auc = {metrics['roc_auc']}."),
        Acceptance(week8["fog1"]["pr_auc_met"], f"hafta8_ozet.json test PR-AUC = {metrics['pr_auc']}, hedef {week8['fog1']['pr_auc_target']}."),
        Acceptance(
            metrics["recall"] > week2["baseline_model_metrics"]["recall"] and sum(week8["confusion"].values()) == week8["split"]["test_rows"],
            f"hafta8_ozet.json test Recall = {metrics['recall']}, referans modelde {week2['baseline_model_metrics']['recall']}. Karmaşıklık matrisi h8_karmasiklik.png.",
        ),
        Acceptance(len(week8["gain_table"]) >= GAIN_FRACTION_COUNT, "hafta8_ozet.json gain_table ve h8_kazanc.png."),
        Acceptance(
            len(week8["permutation_importance"]) >= MIN_IMPORTANCE_ROWS and (root / "reports" / "figures" / "h8_permutation_importance.png").exists(),
            "hafta8_ozet.json permutation_importance ve h8_permutation_importance.png.",
        ),
        Acceptance(reload_matches_test, "hafta9_ozet.json reload_matches_test: yüklenen modelin test metrikleri hafta8_ozet.json ile 4 ondalıkta aynı."),
        Acceptance(
            prediction_seconds is not None and prediction_seconds < MAX_PREDICTION_SECONDS,
            "hafta9_sure.json yok." if prediction_seconds is None else f"hafta9_sure.json single_prediction_seconds, {MAX_PREDICTION_SECONDS:g} saniyenin altında. Ölçüm tahmin servisi çağrısını kapsar, arayüz yalnızca bu çağrıyı gösterir.",
        ),
        _clean_clone(verification, "numbers_match", "Temiz klonda analiz betikleri depodaki özet JSON dosyalarıyla aynı sayıları üretti"),
        Acceptance(
            all(week in summaries for week in SUMMARY_WEEKS) and all(path.exists() for path in report_files),
            "hafta2-hafta9 özet JSON dosyaları ve reports/hafta3_rapor.py ... hafta9_rapor.py üreticileri mevcut.",
        ),
    ]


__all__ = ["CRITERION_COUNT", "Acceptance", "evaluate_acceptance"]
