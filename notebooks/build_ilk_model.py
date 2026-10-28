"""Hafta 4: ön işleme pipeline'ı ile ilk Lojistik Regresyon modelini 5 katlı katmanlı CV'de eğitir ve referans modelle karşılaştırır.

Kullanım: python notebooks/build_ilk_model.py
Rapora giren her sayı bu betiğin ürettiği reports/haftalik/hafta4_ozet.json dosyasından gelir. Test seti yalnızca ayrılır,
hiçbir modelde kullanılmaz (Hafta 8'e kadar kilitli).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
from src.evaluate import compare_folds, summarize_cv
from src.model_plots import plot_pr_roc_curves
from src.modeling import DEFAULT_THRESHOLD, CrossValidationResult, build_baseline, build_logistic_regression, cross_validate_model, make_cv
from src.config import MONTH_ORDER
from src.preprocessing import MONTH_FEATURE, build_linear_feature_pipeline, split_train_test

NOTEBOOK_PATH = ROOT / "notebooks" / "03_ilk_model.ipynb"
FIGURES_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta4_ozet.json"
DECIMALS = 4
# Eğitim verisindeki payı %2'nin altında kalan aylar birleştirilir. Bu eşik Hafta 3'te az gözlemli bulunan dört ayı
# (aralık, mart, eylül, ekim) ayırır, sonraki en küçük ay olan ocak yaklaşık %3 paya sahiptir.
RARE_MONTH_MIN_FREQUENCY = 0.02
COMPARISON_METRIC = "pr_auc"

MAIN_MODEL = "logistic_regression"
BASELINE = "baseline"
WITHOUT_AGE_GROUP = "lr_without_age_group"
RARE_MONTHS_MERGED = "lr_rare_months_merged"
# Her karşılaştırma (eski, yeni) çiftidir; fark yeni - eski olarak hesaplanır, pozitif fark iyileşme demektir.
COMPARISONS = {
    "logistic_vs_baseline": (BASELINE, MAIN_MODEL),
    "age_group_added": (WITHOUT_AGE_GROUP, MAIN_MODEL),
    "rare_months_merged": (MAIN_MODEL, RARE_MONTHS_MERGED),
}


def build_models() -> dict[str, BaseEstimator]:
    return {
        BASELINE: build_baseline(),
        MAIN_MODEL: build_logistic_regression(include_age_group=True),
        WITHOUT_AGE_GROUP: build_logistic_regression(include_age_group=False),
        RARE_MONTHS_MERGED: build_logistic_regression(include_age_group=True, rare_month_min_frequency=RARE_MONTH_MIN_FREQUENCY),
    }


def rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(value, DECIMALS) for name, value in metrics.items()}


def fold_composition(X: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    """Katmanlamanın her katmanda hedef oranını koruduğunu göstermek için katman boyutları ve pozitif oranları."""
    return [
        {
            "train_rows": len(train_index),
            "validation_rows": len(validation_index),
            "validation_positive_rate": round(float(y.iloc[validation_index].mean()), DECIMALS),
        }
        for train_index, validation_index in make_cv().split(X, y)
    ]


def build_summary(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, CrossValidationResult], pd.Series]:
    split = split_train_test(df)
    cv = make_cv()
    results = {name: cross_validate_model(model, split.X_train, split.y_train, cv) for name, model in build_models().items()}
    feature_names, merged_months = fitted_feature_details(split.X_train)
    summary = {
        "split": {
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "train_positive_rate": round(float(split.y_train.mean()), DECIMALS),
            "test_positive_rate": round(float(split.y_test.mean()), DECIMALS),
            "raw_feature_count": split.X_train.shape[1],
        },
        "cv_folds": fold_composition(split.X_train, split.y_train),
        "threshold": DEFAULT_THRESHOLD,
        # Varsayılan eşikte "kabul eder" denen müşterilerin oranı; düşük Recall'un nedenini rapora sayıyla taşır.
        "oof_predicted_positive_rate": round(float((results[MAIN_MODEL].oof_proba >= DEFAULT_THRESHOLD).mean()), DECIMALS),
        "model_feature_count": len(feature_names),
        "rare_month_min_frequency": RARE_MONTH_MIN_FREQUENCY,
        "merged_months": merged_months,
        "models": {
            name: {
                "folds": [rounded(metrics) for metrics in result.fold_metrics],
                "summary": {metric: rounded(values) for metric, values in summarize_cv(result.fold_metrics).to_dict(orient="index").items()},
            }
            for name, result in results.items()
        },
        "comparisons": {
            name: rounded(compare_folds(results[base].fold_metrics, results[variant].fold_metrics, COMPARISON_METRIC))
            for name, (base, variant) in COMPARISONS.items()
        },
        "comparison_metric": COMPARISON_METRIC,
    }
    return summary, results, split.y_train


def fitted_feature_details(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Modele giren özellik adlarını ve birleştirilen ayları bulmak için ön işleme tüm eğitim verisine fit edilir.

    Bu fit yalnızca betimleme içindir; CV skorları her katmanda ayrı fit edilen modellerden gelir, test verisi kullanılmaz.
    """
    main_features = build_linear_feature_pipeline().fit(X_train)
    merged_features = build_linear_feature_pipeline(rare_month_min_frequency=RARE_MONTH_MIN_FREQUENCY).fit(X_train)
    # Eşiğin altında kalan ay yoksa scikit-learn bu alanı None bırakır.
    infrequent = merged_features.named_steps["preprocess"].named_transformers_[MONTH_FEATURE].infrequent_categories_[0]
    merged_months = sorted(map(str, infrequent if infrequent is not None else []), key=MONTH_ORDER.index)
    return list(main_features.named_steps["preprocess"].get_feature_names_out()), merged_months


def save_figures(results: dict[str, CrossValidationResult], y_train: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    main = results[MAIN_MODEL]
    save_figure(plot_pr_roc_curves(y_train, main.oof_proba, main.fold_ids, "Lojistik Regresyon"), FIGURES_DIR / "h4_pr_roc.png")


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Hafta 4 — Ön İşleme Pipeline'ı ve İlk Model\n\n"
            "Ön işleme `src/preprocessing.py`, CV düzeni ve modeller `src/modeling.py`, özet tablo `src/evaluate.py` içindedir. "
            "Rapordaki sayıların kaynağı `notebooks/build_ilk_model.py` ile üretilen `hafta4_ozet.json` dosyasıdır. "
            "Test seti ayrılır ama hiçbir hücrede kullanılmaz."
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
            "from src.model_plots import plot_pr_roc_curves\n"
            "from src.modeling import build_baseline, build_logistic_regression, cross_validate_model, make_cv\n"
            "from src.preprocessing import build_linear_feature_pipeline, split_train_test\n"
            "\n"
            "split = split_train_test(load_raw())\n"
            "print('eğitim:', split.X_train.shape, 'test:', split.X_test.shape)"
        ),
        nbf.v4.new_markdown_cell("## 1. Ön işleme çıktısı"),
        nbf.v4.new_code_cell(
            "features = build_linear_feature_pipeline().fit(split.X_train)\n"
            "names = features.named_steps['preprocess'].get_feature_names_out()\n"
            "print(len(names), 'özellik')\n"
            "list(names)"
        ),
        nbf.v4.new_markdown_cell("## 2. 5 katlı katmanlı CV: referans model ve Lojistik Regresyon"),
        nbf.v4.new_code_cell(
            "cv = make_cv()\n"
            "baseline = cross_validate_model(build_baseline(), split.X_train, split.y_train, cv)\n"
            "logistic = cross_validate_model(build_logistic_regression(), split.X_train, split.y_train, cv)\n"
            "pd.concat({'Referans': summarize_cv(baseline.fold_metrics), 'Lojistik Regresyon': summarize_cv(logistic.fold_metrics)}, axis=1).round(4)"
        ),
        nbf.v4.new_code_cell("plot_pr_roc_curves(split.y_train, logistic.oof_proba, logistic.fold_ids, 'Lojistik Regresyon');"),
    ]
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as file:
        nbf.write(nb, file)


def execute_notebook() -> None:
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NOTEBOOK_PATH)],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    df = load_raw()
    validate_raw(df)
    summary, results, y_train = build_summary(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_figures(results, y_train)
    create_notebook()
    execute_notebook()
    print(f"Özet: {SUMMARY_PATH}\nGrafik: {FIGURES_DIR / 'h4_pr_roc.png'}\nNotebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
