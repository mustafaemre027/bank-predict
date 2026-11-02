"""Hafta 6: üç modeli üç sınıf dengesizliği stratejisiyle aynı 5 katlı katmanlı CV'de karşılaştırır ve karar eşiğini OOF tahminlerinden seçer.

Kullanım: python notebooks/build_dengesizlik.py
Rapora giren her sayı bu betiğin ürettiği reports/haftalik/hafta6_ozet.json dosyasından gelir. Test seti yalnızca ayrılır,
hiçbir modelde ve hiçbir eşik seçiminde kullanılmaz (Hafta 8'e kadar kilitli).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from itertools import product
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_RECALL
from src.data_loader import load_raw, validate_raw
from src.eda_plots import save_figure
from src.evaluate import (
    JSON_DECIMALS,
    compare_folds,
    fold_composition_for_json,
    fold_metrics_at_nested_threshold,
    round_metrics,
    summarize_for_json,
)
from src.model_plots import plot_strategy_comparison, plot_threshold_curves
from src.modeling import (
    DEFAULT_THRESHOLD,
    IMBALANCE_NONE,
    IMBALANCE_STRATEGIES,
    LOGISTIC_REGRESSION,
    MODEL_LABELS,
    RANDOM_FOREST,
    STRATEGY_LABELS,
    XGBOOST,
    CrossValidationResult,
    build_baseline,
    build_model,
    cross_validate_model,
    make_cv,
)
from src.notebook_build import write_and_execute_notebook
from src.preprocessing import split_train_test
from src.threshold import ThresholdChoice, select_threshold_for_recall, select_threshold_max_f1

NOTEBOOK_PATH = ROOT / "notebooks" / "05_dengesizlik_esik.ipynb"
FIGURES_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta6_ozet.json"
COMPARISON_METRIC = "pr_auc"
MODELS = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST]
# Hiperparametre aramasına giren aday sayısı (Hafta 7).
TUNING_CANDIDATE_COUNT = 2


MODEL_STRATEGY_PAIRS = list(product(MODELS, IMBALANCE_STRATEGIES))


def experiment_key(model: str, strategy: str) -> str:
    return f"{model}__{strategy}"


def choice_to_json(choice: ThresholdChoice) -> dict[str, Any]:
    values = asdict(choice)
    criterion = values.pop("criterion")
    return {**round_metrics(values), "criterion": criterion}


def nested_summary(y_train: pd.Series, result: CrossValidationResult, select: Callable[[np.ndarray, np.ndarray], ThresholdChoice]) -> dict[str, dict[str, float]]:
    """Eşik her katman için diğer katmanların OOF tahminlerinden seçilir ve o katmanda ölçülür."""
    folds = fold_metrics_at_nested_threshold(y_train, result.oof_proba, result.fold_ids, lambda labels, scores: select(labels, scores).threshold)
    return summarize_for_json(folds)["summary"]


def describe_experiment(model: str, strategy: str, result: CrossValidationResult, y_train: pd.Series, choices: tuple[ThresholdChoice, ThresholdChoice]) -> dict[str, Any]:
    f1_choice, recall_choice = choices
    return {
        "model": model,
        "strategy": strategy,
        **summarize_for_json(result.fold_metrics),
        "threshold_f1": choice_to_json(f1_choice),
        "threshold_recall": choice_to_json(recall_choice),
        "nested_summary_f1_threshold": nested_summary(y_train, result, select_threshold_max_f1),
        "nested_summary_recall_threshold": nested_summary(y_train, result, lambda labels, scores: select_threshold_for_recall(labels, scores, TARGET_RECALL)),
        "predicted_positive_rate_at_f1_threshold": round(float((result.oof_proba >= f1_choice.threshold).mean()), JSON_DECIMALS),
        "predicted_positive_rate_at_recall_threshold": round(float((result.oof_proba >= recall_choice.threshold).mean()), JSON_DECIMALS),
    }


def pr_auc_mean(experiment: dict[str, Any]) -> float:
    return experiment["summary"][COMPARISON_METRIC]["mean"]


def rank_experiments(experiments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(experiments.items(), key=lambda item: pr_auc_mean(item[1]), reverse=True)
    return [{"key": key, "model": entry["model"], "strategy": entry["strategy"], "pr_auc_mean": pr_auc_mean(entry)} for key, entry in ordered]


def pick_tuning_candidates(ranking: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Sıralamada birbirinden farklı ilk iki model, her biri kendi en iyi stratejisiyle."""
    candidates: dict[str, str] = {}
    for entry in ranking:
        candidates.setdefault(entry["model"], entry["strategy"])
        if len(candidates) == TUNING_CANDIDATE_COUNT:
            break
    return [{"model": model, "strategy": strategy} for model, strategy in candidates.items()]


def build_strategy_effects(results: dict[str, CrossValidationResult]) -> dict[str, dict[str, float]]:
    """Her model için, ağırlıksız hâle göre diğer iki stratejinin PR-AUC farkı (strateji - ağırlıksız)."""
    effects = {}
    for model, strategy in MODEL_STRATEGY_PAIRS:
        if strategy == IMBALANCE_NONE:
            continue
        unweighted = results[experiment_key(model, IMBALANCE_NONE)].fold_metrics
        effects[experiment_key(model, strategy)] = round_metrics(compare_folds(unweighted, results[experiment_key(model, strategy)].fold_metrics, COMPARISON_METRIC))
    return effects


def build_summary(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, CrossValidationResult], dict[str, tuple[ThresholdChoice, ThresholdChoice]], pd.Series]:
    split = split_train_test(df)
    cv = make_cv()
    baseline = cross_validate_model(build_baseline(), split.X_train, split.y_train, cv)
    results = {
        experiment_key(model, strategy): cross_validate_model(build_model(model, strategy), split.X_train, split.y_train, cv)
        for model, strategy in MODEL_STRATEGY_PAIRS
    }
    choices = {
        key: (select_threshold_max_f1(split.y_train, result.oof_proba), select_threshold_for_recall(split.y_train, result.oof_proba, TARGET_RECALL))
        for key, result in results.items()
    }
    experiments = {
        experiment_key(model, strategy): describe_experiment(model, strategy, results[experiment_key(model, strategy)], split.y_train, choices[experiment_key(model, strategy)])
        for model, strategy in MODEL_STRATEGY_PAIRS
    }
    ranking = rank_experiments(experiments)
    summary = {
        "split": {
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "train_positive_rate": round(float(split.y_train.mean()), JSON_DECIMALS),
        },
        "cv_folds": fold_composition_for_json(split.X_train, split.y_train, cv),
        "default_threshold": DEFAULT_THRESHOLD,
        "target_recall": TARGET_RECALL,
        "model_count": len(MODELS),
        "strategy_count": len(IMBALANCE_STRATEGIES),
        "baseline": summarize_for_json(baseline.fold_metrics),
        "experiments": experiments,
        "strategy_effects": build_strategy_effects(results),
        "best_by_model": {model: next(entry["strategy"] for entry in ranking if entry["model"] == model) for model in MODELS},
        "ranking": ranking,
        "candidates_for_tuning": pick_tuning_candidates(ranking),
        "comparison_metric": COMPARISON_METRIC,
    }
    return summary, results, choices, split.y_train


def strategy_panel(summary: dict[str, Any], title: str, metric: str, summary_key: str) -> tuple[str, dict[str, dict[str, dict[str, float]]]]:
    values = {
        model: {strategy: summary["experiments"][experiment_key(model, strategy)][summary_key][metric] for strategy in IMBALANCE_STRATEGIES}
        for model in MODELS
    }
    return title, values


def save_figures(
    summary: dict[str, Any],
    results: dict[str, CrossValidationResult],
    choices: dict[str, tuple[ThresholdChoice, ThresholdChoice]],
    y_train: pd.Series,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        strategy_panel(summary, "PR-AUC", COMPARISON_METRIC, "summary"),
        strategy_panel(summary, "F1-Score (0,5 eşiğinde)", "f1", "summary"),
        strategy_panel(summary, "F1-Score (OOF'tan seçilen eşikte)", "f1", "nested_summary_f1_threshold"),
    ]
    save_figure(plot_strategy_comparison(panels, MODEL_LABELS, STRATEGY_LABELS), FIGURES_DIR / "h6_strateji_karsilastirma.png")
    best = summary["ranking"][0]
    f1_choice, recall_choice = choices[best["key"]]
    title = f"{MODEL_LABELS[best['model']]}, {STRATEGY_LABELS[best['strategy']]}"
    save_figure(plot_threshold_curves(y_train, results[best["key"]].oof_proba, f1_choice, recall_choice, title), FIGURES_DIR / "h6_esik_egrisi.png")


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Hafta 6 — Sınıf Dengesizliği ve Eşik Seçimi\n\n"
            "Stratejiler `src/modeling.py` (`build_model`), eşik seçimi `src/threshold.py` içindedir. Rapordaki sayıların kaynağı "
            "`notebooks/build_dengesizlik.py` ile üretilen `hafta6_ozet.json` dosyasıdır (3 model × 3 strateji). Bu notebook yalnızca "
            "XGBoost ile yöntemi gösterir. Test seti ayrılır ama hiçbir hücrede kullanılmaz."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            "import pandas as pd\n"
            "from src.data_loader import load_raw\n"
            "from src.evaluate import summarize_cv\n"
            "from src.model_plots import plot_threshold_curves\n"
            "from src.modeling import IMBALANCE_STRATEGIES, STRATEGY_LABELS, build_model, cross_validate_model, make_cv\n"
            "from src.preprocessing import split_train_test\n"
            "from src.threshold import select_threshold_for_recall, select_threshold_max_f1\n"
            "\n"
            "split = split_train_test(load_raw())\n"
            "print('eğitim:', split.X_train.shape, 'test:', split.X_test.shape)"
        ),
        nbf.v4.new_markdown_cell("## 1. XGBoost, üç strateji, aynı 5 katlı CV"),
        nbf.v4.new_code_cell(
            "cv = make_cv()\n"
            "results = {s: cross_validate_model(build_model('xgboost', s), split.X_train, split.y_train, cv) for s in IMBALANCE_STRATEGIES}\n"
            "pd.concat({STRATEGY_LABELS[s]: summarize_cv(r.fold_metrics) for s, r in results.items()}, axis=1).round(4)"
        ),
        nbf.v4.new_markdown_cell("## 2. Eşik, yalnızca OOF tahminlerinden seçilir"),
        nbf.v4.new_code_cell(
            "oof = results['none'].oof_proba\n"
            "f1_choice = select_threshold_max_f1(split.y_train, oof)\n"
            "recall_choice = select_threshold_for_recall(split.y_train, oof)\n"
            "print(f1_choice)\n"
            "print(recall_choice)\n"
            "plot_threshold_curves(split.y_train, oof, f1_choice, recall_choice, 'XGBoost, ağırlıksız');"
        ),
    ]
    write_and_execute_notebook(nb, NOTEBOOK_PATH)


def main() -> None:
    df = load_raw()
    validate_raw(df)
    summary, results, choices, y_train = build_summary(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_figures(summary, results, choices, y_train)
    create_notebook()
    print(f"Özet: {SUMMARY_PATH}\nGrafikler: {FIGURES_DIR}\nNotebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
