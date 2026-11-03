"""Hafta 8: final modelin test setinde tek seferlik değerlendirmesi, açıklanabilirlik ve hata analizi.

Kullanım: python notebooks/build_final_degerlendirme.py
Önce notebooks/build_hiperparametre.py çalıştırılmış olmalıdır. Final yapılandırma (model, parametreler, kalibrasyon, eşik)
reports/haftalik/hafta7_ozet.json dosyasındaki `selected` bloğundan okunur. Test seti bu betikte bir kez açılır; hiçbir parametre
test sonucuna göre değiştirilmez. Rapora giren her sayı reports/haftalik/hafta8_ozet.json dosyasından gelir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FOG1_MIN_PR_AUC, FOG1_MIN_ROC_AUC, LITERATURE_ROC_AUC
from src.data_loader import load_raw, validate_raw
from src.eda_plots import save_figure
from src.evaluate import JSON_DECIMALS, compute_metrics, confusion_counts, gain_table, round_metrics
from src.explain import (
    PERIOD_FEATURE,
    error_profile,
    logistic_coefficients,
    metrics_by_group,
    permutation_importance_table,
    rank_variables_by_coefficient,
    top_overlap,
    with_campaign_period,
)
from src.final_model import FinalModelConfig, build_final_pipeline, load_selected_config
from src.model_plots import plot_confusion_matrix, plot_gain_curve, plot_lr_coefficients, plot_permutation_importance, plot_pr_roc_curves
from src.modeling import IMBALANCE_NONE, LOGISTIC_REGRESSION, MODEL_LABELS, build_model
from src.notebook_build import write_and_execute_notebook
from src.preprocessing import TrainTestSplit, add_engineered_features, split_by_time, split_train_test

NOTEBOOK_PATH = ROOT / "notebooks" / "07_final_degerlendirme.ipynb"
FIGURES_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta8_ozet.json"
HYPERPARAMETER_SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta7_ozet.json"
DEFAULT_THRESHOLD = 0.5
OVERLAP_TOP_N = 5
COEFFICIENT_COUNT = 15
ERROR_PROFILE_COLUMNS = ["poutcome", "contact", "month", "age_group", "was_contacted_before"]
GROUP_COLUMNS = ["contact", PERIOD_FEATURE]


def evaluate_at_threshold(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, JSON_DECIMALS),
        "metrics": round_metrics(compute_metrics(y_true, predictions, proba)),
        "confusion": confusion_counts(y_true, predictions),
        "predicted_positive_share": round(float(predictions.mean()), JSON_DECIMALS),
    }


def fit_final_model(config: FinalModelConfig, split: TrainTestSplit):
    return build_final_pipeline(config).fit(split.X_train, split.y_train)


def time_based_evaluation(df: pd.DataFrame, config: FinalModelConfig, random_test_index: pd.Index) -> dict[str, Any]:
    """Aynı yapılandırma ilk %80 satırda eğitilir, son %20 satırda ölçülür. Eşik kullanılmaz, yalnızca sıralama metrikleri verilir."""
    split = split_by_time(df)
    proba = fit_final_model(config, split).predict_proba(split.X_test)[:, 1]
    metrics = compute_metrics(split.y_test, (proba >= config.threshold).astype(int), proba)
    return {
        "train_rows": len(split.X_train),
        "test_rows": len(split.X_test),
        "train_positive_rate": round(float(split.y_train.mean()), JSON_DECIMALS),
        "test_positive_rate": round(float(split.y_test.mean()), JSON_DECIMALS),
        "roc_auc": round(metrics["roc_auc"], JSON_DECIMALS),
        "pr_auc": round(metrics["pr_auc"], JSON_DECIMALS),
        # Son %20 satırların rastgele test setiyle ortak kayıt sayısı: iki değerlendirme birbirinden bağımsız değildir.
        "overlap_with_random_test": int(split.X_test.index.isin(random_test_index).sum()),
    }


def logistic_reference(split: TrainTestSplit, raw_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hafta 7'de Lojistik Regresyon aranmadığı için Hafta 4-6'daki varsayılan yapılandırmayla eğitim verisinde fit edilir."""
    fitted = build_model(LOGISTIC_REGRESSION, IMBALANCE_NONE).fit(split.X_train, split.y_train)
    coefficients = logistic_coefficients(fitted)
    return coefficients, rank_variables_by_coefficient(coefficients, raw_columns)


def frame_to_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame'i JSON'a yazılacak kayıtlara çevirir; NaN (tek sınıflı grupta PR-AUC) null olur."""
    rounded = table.round(JSON_DECIMALS).astype(object).where(table.notna(), None)
    return rounded.to_dict(orient="records")


def round_profile(profile: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        column: [{key: round(value, JSON_DECIMALS) if isinstance(value, float) else value for key, value in row.items()} for row in rows]
        for column, rows in profile.items()
    }


def build_summary(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_selected_config(HYPERPARAMETER_SUMMARY_PATH)
    hyperparameter_summary = json.loads(HYPERPARAMETER_SUMMARY_PATH.read_text(encoding="utf-8"))
    split = split_train_test(df)
    raw_columns = list(split.X_train.columns)

    model = fit_final_model(config, split)
    proba = model.predict_proba(split.X_test)[:, 1]
    y_test = split.y_test

    evaluations = {
        "selected": evaluate_at_threshold(y_test, proba, config.threshold),
        "default_0_5": evaluate_at_threshold(y_test, proba, DEFAULT_THRESHOLD),
        # Recall eşiği özet JSON'da 4 basamağa yuvarlanmış tutulur; tam değer yalnızca F1 eşiği için saklanmıştı.
        "target_recall": evaluate_at_threshold(y_test, proba, hyperparameter_summary["selected"]["threshold_recall"]["threshold"]),
    }
    selected_predictions = (proba >= config.threshold).astype(int)
    test_metrics = evaluations["selected"]["metrics"]
    roc_met, pr_met = test_metrics["roc_auc"] >= FOG1_MIN_ROC_AUC, test_metrics["pr_auc"] >= FOG1_MIN_PR_AUC

    importance = permutation_importance_table(model, split.X_test, y_test)
    coefficients, variable_ranking = logistic_reference(split, raw_columns)
    # LR değişken sıralaması ham değişkene toplandığı için permutation importance listesiyle aynı adlarla karşılaştırılır.
    overlap = top_overlap(list(importance["feature"]), list(variable_ranking["variable"]), OVERLAP_TOP_N)

    engineered_test = with_campaign_period(add_engineered_features(split.X_test))
    summary = {
        "selected": hyperparameter_summary["selected"],
        "split": {"train_rows": len(split.X_train), "test_rows": len(split.X_test), "test_positive_rate": round(float(y_test.mean()), JSON_DECIMALS)},
        "test_metrics": evaluations,
        "confusion": evaluations["selected"]["confusion"],
        "gain_table": [{key: round(value, JSON_DECIMALS) if isinstance(value, float) else value for key, value in row.items()} for row in gain_table(y_test, proba)],
        "fog1": {
            "roc_auc_target": FOG1_MIN_ROC_AUC,
            "pr_auc_target": FOG1_MIN_PR_AUC,
            "roc_auc_met": roc_met,
            "pr_auc_met": pr_met,
            "met": roc_met and pr_met,
        },
        "recall_positive": test_metrics["recall"] > 0,
        "time_based": time_based_evaluation(df, config, split.X_test.index),
        "permutation_importance": frame_to_records(importance),
        "lr_coefficients": frame_to_records(coefficients.head(COEFFICIENT_COUNT)),
        "lr_variable_ranking": frame_to_records(variable_ranking),
        "importance_overlap_top5": {"count": len(overlap), "variables": overlap},
        "error_profile": round_profile(error_profile(engineered_test, y_test, selected_predictions, ERROR_PROFILE_COLUMNS)),
        "group_metrics": {column: frame_to_records(metrics_by_group(engineered_test, y_test, proba, column)) for column in GROUP_COLUMNS},
        "literature": {"moro_2014_roc_auc": LITERATURE_ROC_AUC, "model_label": MODEL_LABELS[config.model]},
    }
    artifacts = {"y_test": y_test, "proba": proba, "importance": importance, "coefficients": coefficients, "confusion": summary["confusion"]}
    return summary, artifacts


def save_figures(artifacts: dict[str, Any], model_label: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(plot_pr_roc_curves(artifacts["y_test"], artifacts["proba"], None, model_label), FIGURES_DIR / "h8_test_pr_roc.png")
    save_figure(plot_confusion_matrix(artifacts["confusion"]), FIGURES_DIR / "h8_karmasiklik.png")
    save_figure(plot_gain_curve(artifacts["y_test"], artifacts["proba"]), FIGURES_DIR / "h8_kazanc.png")
    save_figure(plot_permutation_importance(artifacts["importance"]), FIGURES_DIR / "h8_permutation_importance.png")
    save_figure(plot_lr_coefficients(artifacts["coefficients"]), FIGURES_DIR / "h8_lr_katsayilari.png")


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Hafta 8 — Final Değerlendirme ve Açıklanabilirlik\n\n"
            "Rapordaki sayıların kaynağı `notebooks/build_final_degerlendirme.py` ile üretilen `hafta8_ozet.json` dosyasıdır. "
            "Bu notebook o dosyayı okuyup özetler; modeli yeniden eğitmez, test setini yeniden açmaz."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "import pandas as pd\n"
            "\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "summary = json.loads((PROJECT_ROOT / 'reports' / 'haftalik' / 'hafta8_ozet.json').read_text(encoding='utf-8'))\n"
            "summary['fog1']"
        ),
        nbf.v4.new_markdown_cell("## 1. Test metrikleri (üç eşik)"),
        nbf.v4.new_code_cell("pd.DataFrame({name: item['metrics'] | {'threshold': item['threshold']} for name, item in summary['test_metrics'].items()}).T"),
        nbf.v4.new_markdown_cell("## 2. Kazanç tablosu"),
        nbf.v4.new_code_cell("pd.DataFrame(summary['gain_table'])"),
        nbf.v4.new_markdown_cell("## 3. Zamana göre ayrılmış test"),
        nbf.v4.new_code_cell("pd.Series(summary['time_based'])"),
        nbf.v4.new_markdown_cell("## 4. Permutation importance (ham değişkenler, test seti)"),
        nbf.v4.new_code_cell("pd.DataFrame(summary['permutation_importance'])"),
        nbf.v4.new_markdown_cell("## 5. Lojistik Regresyon katsayıları"),
        nbf.v4.new_code_cell("pd.DataFrame(summary['lr_coefficients'])"),
    ]
    write_and_execute_notebook(nb, NOTEBOOK_PATH)


def main() -> None:
    df = load_raw()
    validate_raw(df)
    summary, artifacts = build_summary(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_figures(artifacts, summary["literature"]["model_label"])
    create_notebook()
    print(f"Özet: {SUMMARY_PATH}\nGrafikler: {FIGURES_DIR}\nNotebook: {NOTEBOOK_PATH}")
    if not summary["fog1"]["met"]:
        print("UYARI: FOG-1 test setinde karşılanmadı. Rapor dürüstçe yazılmalı ve kullanıcıya haber verilmeli.")


if __name__ == "__main__":
    main()
