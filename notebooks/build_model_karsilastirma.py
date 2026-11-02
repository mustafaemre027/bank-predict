"""Hafta 5: Lojistik Regresyon, Rastgele Orman ve XGBoost'u aynı 5 katlı katmanlı CV'de karşılaştırır, ön işleme ablasyonu yapar.

Kullanım: python notebooks/build_model_karsilastirma.py
Rapora giren her sayı bu betiğin ürettiği reports/haftalik/hafta5_ozet.json dosyasından gelir. Test seti yalnızca ayrılır,
hiçbir modelde kullanılmaz (Hafta 8'e kadar kilitli).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import nbformat as nbf
import pandas as pd
from sklearn.base import BaseEstimator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw, validate_raw
from src.eda_plots import save_figure
from src.evaluate import JSON_DECIMALS, compare_folds, is_ablation_candidate, round_metrics, summarize_for_json
from src.model_plots import plot_model_comparison, plot_pr_curves_overlay
from src.modeling import (
    DEFAULT_THRESHOLD,
    LOGISTIC_REGRESSION,
    MODEL_LABELS,
    RANDOM_FOREST,
    RF_PARAMS,
    XGB_PARAMS,
    XGBOOST,
    CrossValidationResult,
    build_baseline,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    cross_validate_model,
    fold_composition,
    make_cv,
)
from src.notebook_build import write_and_execute_notebook
from src.preprocessing import build_linear_feature_pipeline, build_tree_feature_pipeline, split_train_test

NOTEBOOK_PATH = ROOT / "notebooks" / "04_model_karsilastirma.ipynb"
FIGURES_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta5_ozet.json"
COMPARISON_METRIC = "pr_auc"
BASELINE = "baseline"

# Her karşılaştırma (eski, yeni) çiftidir; fark yeni - eski olarak hesaplanır, pozitif fark yeni modelin iyi olduğunu gösterir.
MODEL_COMPARISONS = {
    "random_forest_vs_lr": (LOGISTIC_REGRESSION, RANDOM_FOREST),
    "xgboost_vs_lr": (LOGISTIC_REGRESSION, XGBOOST),
    "xgboost_vs_random_forest": (RANDOM_FOREST, XGBOOST),
}

# Ablasyon varyantı: ad -> (karşılaştırılacak ana model, değişikliğin kısa tanımı, varyantı kuran fonksiyon).
# Log dönüşümü ağaçlarda denenmez, çünkü monoton dönüşüm ağaçların bölme noktalarını değiştirmez.
ABLATIONS: dict[str, tuple[str, str, Callable[[], BaseEstimator]]] = {
    "lr_without_day": (LOGISTIC_REGRESSION, "day çıkarıldı", lambda: build_logistic_regression(include_day=False)),
    "lr_without_age_group": (LOGISTIC_REGRESSION, "yaş grubu çıkarıldı", lambda: build_logistic_regression(include_age_group=False)),
    "lr_without_log": (LOGISTIC_REGRESSION, "log dönüşümü kapatıldı", lambda: build_logistic_regression(use_log=False)),
    "rf_without_day": (RANDOM_FOREST, "day çıkarıldı", lambda: build_random_forest(include_day=False)),
    "rf_with_age_group": (RANDOM_FOREST, "yaş grubu eklendi", lambda: build_random_forest(include_age_group=True)),
    "xgb_without_day": (XGBOOST, "day çıkarıldı", lambda: build_xgboost(include_day=False)),
    "xgb_with_age_group": (XGBOOST, "yaş grubu eklendi", lambda: build_xgboost(include_age_group=True)),
}


def build_main_models() -> dict[str, BaseEstimator]:
    return {
        BASELINE: build_baseline(),
        LOGISTIC_REGRESSION: build_logistic_regression(),
        RANDOM_FOREST: build_random_forest(),
        XGBOOST: build_xgboost(),
    }


def count_model_features(X_train: pd.DataFrame) -> dict[str, int]:
    """Ön işleme çıktısındaki sütun sayısını bulmak için pipeline'lar tüm eğitim verisine fit edilir.

    Bu fit yalnızca betimleme içindir; CV skorları her katmanda ayrı fit edilen modellerden gelir, test verisi kullanılmaz.
    """
    return {
        LOGISTIC_REGRESSION: len(build_linear_feature_pipeline().fit(X_train).named_steps["preprocess"].get_feature_names_out()),
        "tree": len(build_tree_feature_pipeline().fit(X_train).named_steps["preprocess"].get_feature_names_out()),
    }


def build_summary(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, CrossValidationResult], pd.Series]:
    split = split_train_test(df)
    cv = make_cv()
    results = {name: cross_validate_model(model, split.X_train, split.y_train, cv) for name, model in build_main_models().items()}
    ablation_results = {name: cross_validate_model(build(), split.X_train, split.y_train, cv) for name, (_, _, build) in ABLATIONS.items()}
    ablation = {}
    for name, (main_model, change, _) in ABLATIONS.items():
        comparison = round_metrics(compare_folds(results[main_model].fold_metrics, ablation_results[name].fold_metrics, COMPARISON_METRIC))
        ablation[name] = {"model": main_model, "change": change, **summarize_for_json(ablation_results[name].fold_metrics), "comparison": comparison}
    model_names = [name for name in results if name != BASELINE]
    pr_auc_by_model = {name: summarize_for_json(results[name].fold_metrics)["summary"][COMPARISON_METRIC]["mean"] for name in model_names}
    summary = {
        "split": {
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "train_positive_rate": round(float(split.y_train.mean()), JSON_DECIMALS),
            "test_positive_rate": round(float(split.y_test.mean()), JSON_DECIMALS),
            "raw_feature_count": split.X_train.shape[1],
        },
        "cv_folds": fold_composition(split.X_train, split.y_train),
        "threshold": DEFAULT_THRESHOLD,
        "feature_counts": count_model_features(split.X_train),
        "hyperparameters": {RANDOM_FOREST: RF_PARAMS, XGBOOST: XGB_PARAMS},
        "models": {name: summarize_for_json(result.fold_metrics) for name, result in results.items()},
        "comparisons": {
            name: round_metrics(compare_folds(results[base].fold_metrics, results[variant].fold_metrics, COMPARISON_METRIC))
            for name, (base, variant) in MODEL_COMPARISONS.items()
        },
        "ablation": ablation,
        "ablation_candidates": [name for name, entry in ablation.items() if is_ablation_candidate(entry["comparison"])],
        "best_model": max(pr_auc_by_model, key=pr_auc_by_model.__getitem__),
        "comparison_metric": COMPARISON_METRIC,
    }
    return summary, results, split.y_train


def save_figures(summary: dict[str, Any], results: dict[str, CrossValidationResult], y_train: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_names = [name for name in results if name != BASELINE]
    summary_by_model = {name: summary["models"][name]["summary"] for name in model_names}
    reference_summary = summary["models"][BASELINE]["summary"]
    save_figure(plot_model_comparison(summary_by_model, MODEL_LABELS, reference_summary), FIGURES_DIR / "h5_model_karsilastirma.png")
    oof_by_model = {name: results[name].oof_proba for name in model_names}
    save_figure(plot_pr_curves_overlay(y_train, oof_by_model, MODEL_LABELS), FIGURES_DIR / "h5_pr_egrileri.png")


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Hafta 5 — Rastgele Orman, XGBoost ve Ablasyon\n\n"
            "Ağaç modelleri için ön işleme `src/preprocessing.py`, modeller `src/modeling.py`, karşılaştırma tablosu `src/evaluate.py` "
            "içindedir. Rapordaki sayıların kaynağı `notebooks/build_model_karsilastirma.py` ile üretilen `hafta5_ozet.json` dosyasıdır. "
            "Test seti ayrılır ama hiçbir hücrede kullanılmaz."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            "from src.data_loader import load_raw\n"
            "from src.evaluate import build_comparison_table, summarize_cv\n"
            "from src.model_plots import plot_model_comparison, plot_pr_curves_overlay\n"
            "from src.modeling import MODEL_LABELS, build_logistic_regression, build_random_forest, build_xgboost, cross_validate_model, make_cv\n"
            "from src.preprocessing import build_tree_feature_pipeline, split_train_test\n"
            "\n"
            "split = split_train_test(load_raw())\n"
            "print('eğitim:', split.X_train.shape, 'test:', split.X_test.shape)"
        ),
        nbf.v4.new_markdown_cell("## 1. Ağaç modelleri için ön işleme çıktısı"),
        nbf.v4.new_code_cell(
            "trees = build_tree_feature_pipeline().fit(split.X_train)\n"
            "names = trees.named_steps['preprocess'].get_feature_names_out()\n"
            "print(len(names), 'özellik')\n"
            "list(names)"
        ),
        nbf.v4.new_markdown_cell("## 2. Üç model, aynı 5 katlı katmanlı CV"),
        nbf.v4.new_code_cell(
            "cv = make_cv()\n"
            "builders = {'logistic_regression': build_logistic_regression, 'random_forest': build_random_forest, 'xgboost': build_xgboost}\n"
            "results = {name: cross_validate_model(build(), split.X_train, split.y_train, cv) for name, build in builders.items()}\n"
            "summaries = {MODEL_LABELS[name]: summarize_cv(result.fold_metrics) for name, result in results.items()}\n"
            "build_comparison_table(summaries, ['pr_auc', 'roc_auc', 'recall', 'precision', 'f1']).round(4)"
        ),
        nbf.v4.new_code_cell(
            "oof = {name: result.oof_proba for name, result in results.items()}\n"
            "plot_pr_curves_overlay(split.y_train, oof, MODEL_LABELS);"
        ),
    ]
    write_and_execute_notebook(nb, NOTEBOOK_PATH)


def main() -> None:
    df = load_raw()
    validate_raw(df)
    summary, results, y_train = build_summary(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_figures(summary, results, y_train)
    create_notebook()
    print(f"Özet: {SUMMARY_PATH}\nGrafikler: {FIGURES_DIR}\nNotebook: {NOTEBOOK_PATH}")
    if summary["ablation_candidates"]:
        print(f"UYARI: Varsayılanı değiştirmeye aday ablasyon varyantları var: {summary['ablation_candidates']}. Kullanıcıya sorulmalı.")


if __name__ == "__main__":
    main()
