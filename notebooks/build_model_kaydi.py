"""Hafta 9: final modelin kaydedilmesi, yeniden yükleme kontrolü, tahmin süresi ve kabul kriterleri özeti.

Kullanım: python notebooks/build_model_kaydi.py
Önce notebooks/build_hiperparametre.py ve notebooks/build_final_degerlendirme.py çalıştırılmış olmalıdır. Final model Hafta 7 özetindeki
yapılandırmayla tüm eğitim verisinde eğitilir, models/final_model.joblib olarak kaydedilir ve yeniden yüklenen modelin test metrikleri
Hafta 8 özetiyle karşılaştırılır (kabul kriteri 10). Tahmin süresi her çalıştırmada değiştiği için özet JSON'a değil hafta9_sure.json
dosyasına yazılır. Temiz klon kriterleri (3 ve 12) reports/haftalik/hafta9_dogrulama.json dosyası varsa oradan okunur.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import sklearn
import xgboost
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.final_rapor import ACCEPTANCE_CRITERIA
from src.acceptance import CRITERION_COUNT, evaluate_acceptance
from src.config import MODEL_PATH
from src.data_loader import load_raw, validate_raw
from src.evaluate import JSON_DECIMALS, compute_metrics, round_metrics
from src.final_model import build_final_pipeline, load_selected_config
from src.model_store import ModelBundle, load_bundle, save_bundle
from src.predict_service import REFERENCE_KEY, build_reference_values, predict_customer
from src.preprocessing import split_train_test
from src.schemas import CustomerInput, format_validation_errors

HAFTALIK_DIR = ROOT / "reports" / "haftalik"
SUMMARY_PATH = HAFTALIK_DIR / "hafta9_ozet.json"
TIMING_PATH = HAFTALIK_DIR / "hafta9_sure.json"
VERIFICATION_PATH = HAFTALIK_DIR / "hafta9_dogrulama.json"
TIMING_REPEATS = 20
FIRST_SUMMARY_WEEK = 2
LAST_ANALYSIS_WEEK = 8
# Sabit örnek müşteri: demo çıktısı ve doğrulama örnekleri bu kayıttan üretilir.
DEMO_CUSTOMER = {
    "age": 35, "job": "management", "marital": "married", "education": "tertiary", "default": "no", "balance": 2500,
    "housing": "no", "loan": "no", "contact": "cellular", "day": 12, "month": "mar", "campaign": 1, "pdays": 180,
    "previous": 2, "poutcome": "success",
}
INVALID_EXAMPLES = {
    "geçersiz kategori": {**DEMO_CUSTOMER, "job": "astronaut"},
    "aralık dışı sayı": {**DEMO_CUSTOMER, "age": 17},
    "eksik alan": {key: value for key, value in DEMO_CUSTOMER.items() if key != "balance"},
    "fazla alan (duration)": {**DEMO_CUSTOMER, "duration": 300},
}


def load_summaries() -> dict[int, dict[str, Any]]:
    summaries = {}
    for week in range(FIRST_SUMMARY_WEEK, LAST_ANALYSIS_WEEK + 1):
        path = HAFTALIK_DIR / f"hafta{week}_ozet.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadı. İlgili haftanın analiz betiğini çalıştırın (python run_all.py hepsini sırayla çalıştırır).")
        summaries[week] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def build_bundle(config, split, hyperparameter_summary: dict[str, Any]) -> ModelBundle:
    pipeline = build_final_pipeline(config).fit(split.X_train, split.y_train)
    return ModelBundle(
        pipeline=pipeline,
        threshold=config.threshold,
        base_rate=float(split.y_train.mean()),
        metadata={
            "model": config.model,
            "strategy": config.strategy,
            "calibrated": config.calibrated,
            "selected": hyperparameter_summary["selected"],
            "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
            REFERENCE_KEY: build_reference_values(split.X_train),
        },
    )


def reload_matches_test_metrics(bundle: ModelBundle, split, week8_summary: dict[str, Any]) -> bool:
    """Diskten yüklenen modelin test metrikleri Hafta 8'dekiyle 4 ondalıkta aynı mı."""
    reloaded = load_bundle(MODEL_PATH)
    original_proba = bundle.pipeline.predict_proba(split.X_test)[:, 1]
    reloaded_proba = reloaded.pipeline.predict_proba(split.X_test)[:, 1]
    if not np.array_equal(original_proba, reloaded_proba):
        return False
    metrics = round_metrics(compute_metrics(split.y_test, (reloaded_proba >= reloaded.threshold).astype(int), reloaded_proba))
    return metrics == week8_summary["test_metrics"]["selected"]["metrics"]


def measure_prediction_seconds(bundle: ModelBundle) -> float:
    predict_customer(DEMO_CUSTOMER, bundle)
    started = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        predict_customer(DEMO_CUSTOMER, bundle)
    return (time.perf_counter() - started) / TIMING_REPEATS


def demo_prediction(bundle: ModelBundle) -> dict[str, Any]:
    result = predict_customer(DEMO_CUSTOMER, bundle)
    return {
        "customer": DEMO_CUSTOMER,
        "probability": round(result.probability, JSON_DECIMALS),
        "will_accept": result.will_accept,
        "band": result.band,
        "top_factors": [{"feature": factor.feature, "direction": factor.direction, "effect": round(factor.effect, JSON_DECIMALS)} for factor in result.top_factors],
    }


def validation_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {"valid": {"customer": DEMO_CUSTOMER, "accepted": CustomerInput.model_validate(DEMO_CUSTOMER).model_dump() == DEMO_CUSTOMER}}
    invalid = {}
    for name, customer in INVALID_EXAMPLES.items():
        try:
            CustomerInput.model_validate(customer)
        except ValidationError as error:
            invalid[name] = format_validation_errors(error)
    examples["invalid"] = invalid
    return examples


def load_verification() -> dict[str, Any] | None:
    return json.loads(VERIFICATION_PATH.read_text(encoding="utf-8")) if VERIFICATION_PATH.exists() else None


def build_summary(bundle: ModelBundle, reload_matches: bool, timing: dict[str, Any], summaries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    acceptance = evaluate_acceptance({**summaries, 9: {}}, reload_matches, timing, load_verification())
    if len(acceptance) != CRITERION_COUNT or len(ACCEPTANCE_CRITERIA) != CRITERION_COUNT:
        raise ValueError("Kabul kriteri sayısı 13 olmalı.")
    metadata = {key: value for key, value in bundle.metadata.items() if key not in ("versions", REFERENCE_KEY)}
    return {
        "bundle": {**metadata, "threshold": round(bundle.threshold, JSON_DECIMALS), "base_rate": round(bundle.base_rate, JSON_DECIMALS)},
        "reload_matches_test": reload_matches,
        "validation_examples": validation_examples(),
        "demo_prediction": demo_prediction(bundle),
        "acceptance_criteria": [
            {"criterion": name, "met": item.met, "evidence": item.evidence} for name, item in zip(ACCEPTANCE_CRITERIA, acceptance)
        ],
    }


def main() -> None:
    df = load_raw()
    validate_raw(df)
    summaries = load_summaries()
    hyperparameter_summary = json.loads((HAFTALIK_DIR / "hafta7_ozet.json").read_text(encoding="utf-8"))
    split = split_train_test(df)
    bundle = build_bundle(load_selected_config(HAFTALIK_DIR / "hafta7_ozet.json"), split, hyperparameter_summary)
    save_bundle(bundle)
    reload_matches = reload_matches_test_metrics(bundle, split, summaries[8])
    if not reload_matches:
        raise ValueError("Yeniden yüklenen modelin test metrikleri Hafta 8 özetiyle uyuşmuyor. Kabul kriteri 10 sağlanamadı.")
    seconds = measure_prediction_seconds(load_bundle(MODEL_PATH))
    TIMING_PATH.write_text(json.dumps({"single_prediction_seconds": round(seconds, 4), "repeats": TIMING_REPEATS}, indent=2), encoding="utf-8")
    summary = build_summary(bundle, reload_matches, {"single_prediction_seconds": round(seconds, 4)}, summaries)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    unmet = [item["criterion"] for item in summary["acceptance_criteria"] if not item["met"]]
    print(f"Model: {MODEL_PATH}\nÖzet: {SUMMARY_PATH}\nTek tahmin süresi: {seconds * 1000:.0f} ms (ilk 3 etken dahil)")
    if unmet:
        print("Karşılanmayan kabul kriterleri:\n- " + "\n- ".join(unmet))


if __name__ == "__main__":
    main()
