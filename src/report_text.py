"""Haftalık raporlarda ortak kullanılan metin yardımcıları: Türkçe ekler, sayı biçimleri ve karşılaştırma cümleleri.

Rapordaki sayılar ve sayıya dayalı hükümler (arttı/azaldı, kaç katmanda yükseldi) özet JSON'dan gelir.
"""
from __future__ import annotations

from typing import Any

from src.formatting import format_number

Summary = dict[str, Any]

# Rakamın okunuşuna göre bulunma eki: 1'inde, 2'sinde, 3'ünde, 4'ünde, 5'inde, 6'sında, 7'sinde, 8'inde, 9'unda, 0'ında.
LOCATIVE_SUFFIXES = {0: "ında", 1: "inde", 2: "sinde", 3: "ünde", 4: "ünde", 5: "inde", 6: "sında", 7: "sinde", 8: "inde", 9: "unda"}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

MONTH_NAMES_TR = {
    "jan": "ocak", "feb": "şubat", "mar": "mart", "apr": "nisan", "may": "mayıs", "jun": "haziran",
    "jul": "temmuz", "aug": "ağustos", "sep": "eylül", "oct": "ekim", "nov": "kasım", "dec": "aralık",
}


def decimal(value: float, digits: int = 3) -> str:
    return format_number(value, digits)


def mean_std(summary: Summary, model: str, metric: str) -> str:
    values = summary["models"][model]["summary"][metric]
    return f"{decimal(values['mean'])} ± {decimal(values['std'])}"


def mean_of(summary: Summary, model: str, metric: str) -> float:
    return summary["models"][model]["summary"][metric]["mean"]


def fold_count_text(improved: int, total: int) -> str:
    """"5 katmanın 4'ünde" gibi, sayının okunuşuna uygun bulunma ekiyle katman ifadesi."""
    if improved == total:
        return "katmanların hepsinde"
    return f"{total} katmanın {improved}'{LOCATIVE_SUFFIXES[improved % 10]}"


def signed(value: float) -> str:
    return f"+{decimal(value, 4)}" if value > 0 else decimal(value, 4)


def is_within_noise(comparison: dict[str, float]) -> bool:
    return abs(comparison["mean_difference"]) < comparison["std_difference"]


def describe_change(comparison: dict[str, float]) -> str:
    """"PR-AUC ortalamada 0,0384 arttı ve katmanların hepsinde yükseldi" gibi yön ve katman sayısı ifadesi."""
    improved = comparison["folds_improved"]
    folds = "hiçbir katmanda yükselmedi" if improved == 0 else f"{fold_count_text(improved, comparison['fold_count'])} yükseldi"
    difference = comparison["mean_difference"]
    if difference == 0:
        return f"PR-AUC ortalaması değişmedi ve {folds}."
    direction = "arttı" if difference > 0 else "azaldı"
    return f"PR-AUC ortalamada {decimal(abs(difference), 4)} {direction} ve {folds}."


def describe_comparison(comparison: dict[str, float], change: str) -> str:
    """Bir karşılaştırmanın sonucunu yön, katman sayısı ve gürültüye göre büyüklüğüyle birlikte cümleye döker."""
    difference = comparison["mean_difference"]
    if difference == 0:
        return f"{change} {describe_change(comparison)}"
    spread = decimal(comparison["std_difference"], 4)
    is_small = is_within_noise(comparison)
    # "Ancak" yalnızca küçük bir iyileşmeyi sınırlarken anlamlı; düşüşten sonra gereksiz bir karşıtlık kurar.
    lead = "Ancak bu fark" if is_small and difference > 0 else "Bu fark"
    size = f"{lead}, katmanlardaki farkların standart sapmasından ({spread}) {'küçük' if is_small else 'büyük'}."
    return f"{change} {describe_change(comparison)} {size}"


__all__ = [
    "LOCATIVE_SUFFIXES",
    "METRIC_LABELS",
    "MONTH_NAMES_TR",
    "Summary",
    "decimal",
    "describe_change",
    "describe_comparison",
    "fold_count_text",
    "is_within_noise",
    "mean_of",
    "mean_std",
    "signed",
]
