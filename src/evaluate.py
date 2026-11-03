from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Özet JSON'a yazılan sayılar bu basamağa yuvarlanır; iki çalıştırmanın JSON'ları bu sayede birebir karşılaştırılabilir.
JSON_DECIMALS = 4
# Ablasyon varyantının varsayılan yapılmaya aday olması için iyileşmesi gereken katman payı (5 katmanın 4'ü).
MIN_IMPROVED_FOLD_SHARE = Fraction(4, 5)
# Kazanç tablosunda aranacak müşteri payları: olasılığa göre sıralı listenin ilk %10, %20 ve %30'u.
DEFAULT_GAIN_FRACTIONS = (0.1, 0.2, 0.3)


def round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(value, JSON_DECIMALS) for name, value in metrics.items()}


def compute_metrics(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray, y_proba: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Compute classification metrics for binary targets."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_proba_array = np.asarray(y_proba)

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, y_proba_array)),
        "pr_auc": float(average_precision_score(y_true_array, y_proba_array)),
    }


def confusion_counts(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> dict[str, int]:
    """İkili hedef için karmaşıklık matrisi sayıları: tn, fp, fn, tp."""
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1]).ravel()
    return {"tn": int(true_negative), "fp": int(false_positive), "fn": int(false_negative), "tp": int(true_positive)}


def gain_table(
    y_true: Sequence[int] | np.ndarray,
    proba: Sequence[float] | np.ndarray,
    fractions: Sequence[float] = DEFAULT_GAIN_FRACTIONS,
) -> list[dict[str, float]]:
    """Müşteriler kabul olasılığına göre azalan sıralanıp ilk `fraction` kadarı arandığında yakalanan kabul edenler.

    contacted = round(fraction · n). capture_rate yakalanan kabul edenlerin tüm kabul edenlere oranıdır.
    lift = (aranan müşterilerdeki kabul oranı) / (genel kabul oranı); rastgele aramada 1'dir.
    Eşit olasılıkta sıra, veri sırasına göre kararlı (stable) tutulur.
    """
    y_array, proba_array = np.asarray(y_true), np.asarray(proba)
    if len(y_array) != len(proba_array):
        raise ValueError(f"Etiket ve olasılık sayıları farklı: {len(y_array)} ve {len(proba_array)}.")
    total_positives = int(y_array.sum())
    if total_positives == 0:
        raise ValueError("Kazanç tablosu için etiketlerde en az bir pozitif örnek bulunmalı.")
    sorted_labels = y_array[np.argsort(-proba_array, kind="stable")]
    base_rate = total_positives / len(y_array)
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"Oran 0 ile 1 arasında olmalı, {fraction} verildi.")
        contacted = max(1, round(fraction * len(y_array)))
        captured = int(sorted_labels[:contacted].sum())
        rows.append(
            {
                "fraction": float(fraction),
                "contacted": contacted,
                "captured_positives": captured,
                "capture_rate": captured / total_positives,
                "lift": (captured / contacted) / base_rate,
            }
        )
    return rows


def _require_multiple_folds(fold_count: int) -> None:
    # Örneklem standart sapması (ddof=1) tek katmanda tanımsızdır; NaN sessizce JSON'a ve rapora geçmesin.
    if fold_count < 2:
        raise ValueError(f"En az 2 katman gerekli, {fold_count} katman verildi.")


def summarize_cv(fold_metrics: list[dict[str, float]]) -> pd.DataFrame:
    """Katman metriklerini metrik başına ortalama ve standart sapma (örneklem, ddof=1) tablosuna toplar."""
    _require_multiple_folds(len(fold_metrics))
    table = pd.DataFrame(fold_metrics)
    return pd.DataFrame({"mean": table.mean(), "std": table.std(ddof=1)})


def compare_folds(base: list[dict[str, float]], variant: list[dict[str, float]], metric: str) -> dict[str, float]:
    """Aynı katmanlarda eğitilmiş iki modeli bir metrikte katman katman karşılaştırır (variant - base)."""
    if len(base) != len(variant):
        raise ValueError(f"Katman sayıları farklı: {len(base)} ve {len(variant)}. Modeller aynı CV düzeniyle eğitilmeli.")
    _require_multiple_folds(len(base))
    differences = np.array([after[metric] - before[metric] for before, after in zip(base, variant)])
    return {
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std(ddof=1)),
        "folds_improved": int((differences > 0).sum()),
        "fold_count": len(differences),
    }


def build_comparison_table(summaries: dict[str, pd.DataFrame], metrics: list[str]) -> pd.DataFrame:
    """Modelleri satır, (metrik, mean/std) çiftlerini sütun yapan karşılaştırma tablosu. Girdi summarize_cv çıktısıdır."""
    rows = {
        model: {(metric, statistic): float(summary.loc[metric, statistic]) for metric in metrics for statistic in ("mean", "std")}
        for model, summary in summaries.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize_for_json(fold_metrics: list[dict[str, float]]) -> dict[str, object]:
    """Bir modelin katman metriklerini ve ortalama ± std özetini özet JSON'a yazılacak biçimde yuvarlar."""
    return {
        "folds": [round_metrics(metrics) for metrics in fold_metrics],
        "summary": {metric: round_metrics(values) for metric, values in summarize_cv(fold_metrics).to_dict(orient="index").items()},
    }


def is_within_noise(comparison: dict[str, float]) -> bool:
    """Ortalama fark, katmanlardaki farkların standart sapmasının altındaysa fark gürültü düzeyindedir."""
    return abs(comparison["mean_difference"]) < comparison["std_difference"]


def min_folds_improved_for_candidate(fold_count: int) -> int:
    return math.ceil(MIN_IMPROVED_FOLD_SHARE * fold_count)


def is_ablation_candidate(comparison: dict[str, float]) -> bool:
    """Bir varyantın varsayılanı değiştirmeye aday olması için ortalama artışı katmanlar arası std'yi aşmalı ve katmanların çoğunda iyileşme olmalı."""
    return (
        comparison["mean_difference"] > 0
        and comparison["mean_difference"] > comparison["std_difference"]
        and comparison["folds_improved"] >= min_folds_improved_for_candidate(comparison["fold_count"])
    )


def fold_metrics_at_nested_threshold(
    y_true: Sequence[int] | np.ndarray,
    oof_proba: np.ndarray,
    fold_ids: np.ndarray,
    select_threshold: Callable[[np.ndarray, np.ndarray], float],
) -> list[dict[str, float]]:
    """Her katmanın metriklerini, eşiği o katman dışındaki katmanların OOF tahminlerinden seçerek hesaplar.

    Eşik, üzerinde ölçüldüğü tahminlere bakmadan seçildiği için sonuç, eşiği tüm OOF tahminlerinden seçip aynı tahminlerde ölçmekten
    daha az iyimserdir.
    """
    y_array = np.asarray(y_true)
    metrics = []
    for fold in np.unique(fold_ids):
        held_out = fold_ids == fold
        threshold = select_threshold(y_array[~held_out], oof_proba[~held_out])
        proba = oof_proba[held_out]
        metrics.append(compute_metrics(y_array[held_out], (proba >= threshold).astype(int), proba))
    return metrics


def calibration_summary(y_true: Sequence[int] | np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Brier skoru ve olasılığa göre sıralı, eşit sayıda kayıt içeren dilimler için ortalama tahmin ve gerçek kabul oranı.

    Eşit sayılı dilimler kullanılır, çünkü olasılıklar küçük değerlere yığıldığından eşit genişlikli dilimlerin çoğu boş kalır.
    """
    y_array = np.asarray(y_true)
    order = np.argsort(proba, kind="stable")
    bins = [
        {
            "mean_predicted": float(proba[indices].mean()),
            "observed_rate": float(y_array[indices].mean()),
            "count": len(indices),
        }
        for indices in np.array_split(order, n_bins)
    ]
    return {"brier": float(brier_score_loss(y_array, proba)), "bins": bins}


def fold_composition_for_json(X: pd.DataFrame, y: pd.Series, cv: Any) -> list[dict[str, Any]]:
    """Katmanlamanın her katmanda hedef oranını koruduğunu göstermek için katman boyutları ve pozitif oranları."""
    return [
        {
            "train_rows": len(train_index),
            "validation_rows": len(validation_index),
            "validation_positive_rate": round(float(y.iloc[validation_index].mean()), JSON_DECIMALS),
        }
        for train_index, validation_index in cv.split(X, y)
    ]
