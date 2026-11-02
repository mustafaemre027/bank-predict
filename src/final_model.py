"""Final modelin yapılandırması: aday seçimi, kalibrasyon kararı ve özet JSON'daki `selected` bloğundan modelin kurulması.

Kararlar kodda sabittir, böylece sonuç kişiye değil veriye bağlıdır ve raporda aynı kural anlatılır.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from src.config import CV_FOLDS
from src.modeling import IMBALANCE_NONE, LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, build_model

# Aynı başarıda daha basit model seçilir: daha hızlı çalışır ve daha kolay yorumlanır. Sıra basitten karmaşığa.
SIMPLICITY_ORDER = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST]
# Ağırlıksız modelde kalibrasyonun kabul edilmesi için PR-AUC ortalamasının en fazla bu kadar düşmesine izin verilir.
MAX_PR_AUC_DROP_FOR_CALIBRATION = 0.005
CALIBRATION_METHOD = "isotonic"


@dataclass(frozen=True)
class FinalModelConfig:
    model: str
    strategy: str
    params: dict[str, Any]
    calibrated: bool
    threshold: float


def select_final_candidate(candidates: list[dict[str, Any]], is_within_noise: Callable[[dict[str, Any], dict[str, Any]], bool]) -> dict[str, Any]:
    """PR-AUC ortalaması en yüksek aday seçilir. Ondan daha basit bir aday ona gürültü düzeyinde yakınsa en basit olanı seçilir.

    candidates: {"model", "pr_auc_mean", ...} sözlükleri. is_within_noise(en_iyi, diğer), iki adayın farkı katmanlardaki
    standart sapmanın altındaysa True döndürür.
    """
    best = max(candidates, key=lambda candidate: candidate["pr_auc_mean"])
    best_rank = SIMPLICITY_ORDER.index(best["model"])
    simpler_and_close = [
        candidate
        for candidate in candidates
        if SIMPLICITY_ORDER.index(candidate["model"]) < best_rank and is_within_noise(best, candidate)
    ]
    return min(simpler_and_close, key=lambda candidate: SIMPLICITY_ORDER.index(candidate["model"])) if simpler_and_close else best


def decide_calibration(strategy: str, brier_raw: float, brier_calibrated: float, pr_auc_raw: float, pr_auc_calibrated: float) -> bool:
    """Sınıf ağırlığı ve SMOTE olasılıkları tasarım gereği şişirdiği için kalibrasyon zorunludur.

    Ağırlıksız modelde kalibrasyon yalnızca Brier skorunu düşürürse ve PR-AUC ortalamasını en fazla
    MAX_PR_AUC_DROP_FOR_CALIBRATION kadar azaltırsa uygulanır.
    """
    if strategy != IMBALANCE_NONE:
        return True
    return brier_calibrated < brier_raw and pr_auc_raw - pr_auc_calibrated <= MAX_PR_AUC_DROP_FOR_CALIBRATION


def build_configured_estimator(model: str, strategy: str, params: dict[str, Any], calibrated: bool) -> BaseEstimator:
    """Fit edilmemiş final model: pipeline, aranan hiperparametreler ve gerekiyorsa isotonic kalibrasyon."""
    pipeline = build_model(model, strategy).set_params(**params)
    if not calibrated:
        return pipeline
    return CalibratedClassifierCV(estimator=pipeline, method=CALIBRATION_METHOD, cv=CV_FOLDS)


def build_final_pipeline(config: FinalModelConfig) -> BaseEstimator:
    return build_configured_estimator(config.model, config.strategy, config.params, config.calibrated)


def load_selected_config(summary_path: Path) -> FinalModelConfig:
    """Hafta 7 özet JSON'undaki `selected` bloğundan final model yapılandırmasını okur."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Özet dosyası bulunamadı: {summary_path}. Önce python notebooks/build_hiperparametre.py çalıştırın.")
    selected = json.loads(summary_path.read_text(encoding="utf-8"))["selected"]
    return FinalModelConfig(selected["model"], selected["strategy"], selected["params"], selected["calibrated"], selected["threshold"])


__all__ = [
    "CALIBRATION_METHOD",
    "MAX_PR_AUC_DROP_FOR_CALIBRATION",
    "SIMPLICITY_ORDER",
    "FinalModelConfig",
    "build_configured_estimator",
    "build_final_pipeline",
    "decide_calibration",
    "load_selected_config",
    "select_final_candidate",
]
