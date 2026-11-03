"""Final modelin diske kaydedilmesi ve yüklenmesi.

Model, eşik, eğitim kabul oranı ve tahmin servisinin ihtiyaç duyduğu referans değerler tek pakette (bundle) saklanır.
joblib dosyaları pickle tabanlıdır, bu yüzden yalnızca bu projenin kendi ürettiği dosya yüklenmelidir.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator

from src.config import MODEL_PATH


@dataclass(frozen=True)
class ModelBundle:
    pipeline: BaseEstimator
    threshold: float
    base_rate: float
    metadata: dict[str, Any]


def save_bundle(bundle: ModelBundle, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    if not path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {path}. Önce python notebooks/build_model_kaydi.py komutunu çalıştırın.")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError(f"{path} beklenen model paketini içermiyor. Dosyayı python notebooks/build_model_kaydi.py ile yeniden üretin.")
    return bundle


__all__ = ["ModelBundle", "load_bundle", "save_bundle"]
