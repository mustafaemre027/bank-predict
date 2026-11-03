"""Hafta 7: en iyi iki model ve stratejisi için hiperparametre araması, kalibrasyon kontrolü, final aday ve karar eşiği.

Kullanım: python notebooks/build_hiperparametre.py
Adaylar reports/haftalik/hafta6_ozet.json içindeki candidates_for_tuning listesinden okunur. Rapora giren her sayı bu betiğin
ürettiği reports/haftalik/hafta7_ozet.json dosyasından gelir. Test seti yalnızca ayrılır, aramada, kalibrasyonda ve eşik seçiminde
kullanılmaz (Hafta 8'e kadar kilitli). Arama süreleri her çalıştırmada değiştiği için özet JSON'a değil hafta7_sure.json dosyasına yazılır.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_RECALL
from src.data_loader import load_raw, validate_raw
from src.eda_plots import save_figure
from src.evaluate import (
    JSON_DECIMALS,
    calibration_summary,
    compare_folds,
    fold_composition_for_json,
    is_within_noise,
    round_metrics,
    summarize_for_json,
)
from src.final_model import (
    MAX_PR_AUC_DROP_FOR_CALIBRATION,
    SIMPLICITY_ORDER,
    build_configured_estimator,
    decide_calibration,
    select_final_candidate,
)
from src.model_plots import plot_calibration_curves, plot_search_results
from src.modeling import (
    MODEL_LABELS,
    STRATEGY_LABELS,
    CrossValidationResult,
    build_model,
    cross_validate_model,
    make_cv,
    nested_threshold_summary,
)
from src.notebook_build import write_and_execute_notebook
from src.preprocessing import split_train_test
from src.threshold import choice_to_json, select_threshold_for_recall, select_threshold_max_f1
from src.tuning import N_ITER, PARAM_SPECS, describe_spec, most_influential_params, tune_model

NOTEBOOK_PATH = ROOT / "notebooks" / "06_hiperparametre.ipynb"
FIGURES_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta7_ozet.json"
TIMING_PATH = ROOT / "reports" / "haftalik" / "hafta7_sure.json"
CANDIDATES_PATH = ROOT / "reports" / "haftalik" / "hafta6_ozet.json"
COMPARISON_METRIC = "pr_auc"
CALIBRATION_BINS = 10
INFLUENTIAL_PARAM_COUNT = 2
# FOG-5: tam hiperparametre araması standart bir dizüstü bilgisayarda bu süreyi aşmamalı.
SEARCH_TIME_LIMIT_SECONDS = 30 * 60


def load_candidates() -> list[dict[str, str]]:
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Aday listesi bulunamadı: {CANDIDATES_PATH}. Önce python notebooks/build_dengesizlik.py çalıştırın.")
    return json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))["candidates_for_tuning"]


def pr_auc_mean(result: CrossValidationResult) -> float:
    return summarize_for_json(result.fold_metrics)["summary"][COMPARISON_METRIC]["mean"]


def evaluate_candidate(model: str, strategy: str, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Bir aday için arama, varsayılanla karşılaştırma ve kalibrasyon kontrolü. Ağır nesneler (sonuçlar, arama) ayrıca döner."""
    cv = make_cv()
    default = cross_validate_model(build_model(model, strategy), X, y, cv)
    search = tune_model(build_model(model, strategy), PARAM_SPECS[model], X, y)
    tuned = cross_validate_model(build_model(model, strategy).set_params(**search.best_params), X, y, cv)
    calibrated = cross_validate_model(build_configured_estimator(model, strategy, search.best_params, calibrated=True), X, y, cv)
    brier_raw = calibration_summary(y, tuned.oof_proba, CALIBRATION_BINS)
    brier_calibrated = calibration_summary(y, calibrated.oof_proba, CALIBRATION_BINS)
    apply_calibration = decide_calibration(strategy, brier_raw["brier"], brier_calibrated["brier"], pr_auc_mean(tuned), pr_auc_mean(calibrated))
    entry = {
        "model": model,
        "strategy": strategy,
        "default": summarize_for_json(default.fold_metrics),
        "best_search_score": round(search.best_score, JSON_DECIMALS),
        "tuned_params": search.best_params,
        "tuned": summarize_for_json(tuned.fold_metrics),
        "comparison_vs_default": round_metrics(compare_folds(default.fold_metrics, tuned.fold_metrics, COMPARISON_METRIC)),
        "calibration": {
            "brier_raw": round(brier_raw["brier"], JSON_DECIMALS + 2),
            "brier_calibrated": round(brier_calibrated["brier"], JSON_DECIMALS + 2),
            "pr_auc_raw": round(pr_auc_mean(tuned), JSON_DECIMALS),
            "pr_auc_calibrated": round(pr_auc_mean(calibrated), JSON_DECIMALS),
            "applied": apply_calibration,
        },
    }
    return {"entry": entry, "tuned": tuned, "calibrated": calibrated, "search": search, "bins": (brier_raw["bins"], brier_calibrated["bins"])}


def select_final(evaluated: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ayarlanmış adaylar arasından final adayı ve seçim gerekçesini döndürür."""
    candidates = [{"key": key, "model": item["entry"]["model"], "pr_auc_mean": item["entry"]["tuned"]["summary"][COMPARISON_METRIC]["mean"]} for key, item in evaluated.items()]

    def within_noise(best: dict[str, Any], other: dict[str, Any]) -> bool:
        comparison = compare_folds(evaluated[other["key"]]["tuned"].fold_metrics, evaluated[best["key"]]["tuned"].fold_metrics, COMPARISON_METRIC)
        return is_within_noise(comparison)

    chosen = select_final_candidate(candidates, within_noise)
    best = max(candidates, key=lambda candidate: candidate["pr_auc_mean"])
    reason = "highest_pr_auc" if chosen["key"] == best["key"] else "simpler_within_noise"
    return chosen, {"reason": reason, "highest_pr_auc_model": best["model"]}


def build_summary(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, dict[str, Any]], pd.Series, dict[str, float]]:
    split = split_train_test(df)
    candidates = load_candidates()
    evaluated = {f"{item['model']}__{item['strategy']}": evaluate_candidate(item["model"], item["strategy"], split.X_train, split.y_train) for item in candidates}
    chosen, selection = select_final(evaluated)
    final = evaluated[chosen["key"]]
    final_result = final["calibrated"] if final["entry"]["calibration"]["applied"] else final["tuned"]
    base_rate = float(split.y_train.mean())
    f1_choice = select_threshold_max_f1(split.y_train, final_result.oof_proba)
    recall_choice = select_threshold_for_recall(split.y_train, final_result.oof_proba, TARGET_RECALL)
    if f1_choice.threshold <= base_rate:
        raise ValueError(f"Seçilen eşik ({f1_choice.threshold:.4f}) eğitim verisindeki kabul oranından ({base_rate:.4f}) büyük olmalı. Olasılık bandı tanımlanamaz.")
    selected = {
        **selection,
        "model": chosen["model"],
        "strategy": final["entry"]["strategy"],
        "params": final["entry"]["tuned_params"],
        "calibrated": final["entry"]["calibration"]["applied"],
        # Tam hassasiyetli eşik: Hafta 8 ve 9 bu değeri doğrudan kullanır. Yuvarlanmış hâli threshold_f1 içindedir.
        "threshold": f1_choice.threshold,
        "threshold_f1": choice_to_json(f1_choice),
        "threshold_recall": choice_to_json(recall_choice),
        "oof_summary": summarize_for_json(final_result.fold_metrics)["summary"],
        "nested_summary_f1_threshold": nested_threshold_summary(split.y_train, final_result, lambda labels, scores: select_threshold_max_f1(labels, scores).threshold),
        "nested_summary_recall_threshold": nested_threshold_summary(
            split.y_train, final_result, lambda labels, scores: select_threshold_for_recall(labels, scores, TARGET_RECALL).threshold
        ),
        "base_rate": round(base_rate, JSON_DECIMALS),
    }
    summary = {
        "split": {"train_rows": len(split.X_train), "test_rows": len(split.X_test), "train_positive_rate": round(base_rate, JSON_DECIMALS)},
        "cv_folds": fold_composition_for_json(split.X_train, split.y_train, make_cv()),
        "n_iter": N_ITER,
        "target_recall": TARGET_RECALL,
        "max_pr_auc_drop_for_calibration": MAX_PR_AUC_DROP_FOR_CALIBRATION,
        "simplicity_order": SIMPLICITY_ORDER,
        "search_space": {
            model: {param: describe_spec(spec) for param, spec in PARAM_SPECS[model].items()} for model in dict.fromkeys(item["model"] for item in candidates)
        },
        "candidates": [evaluated[f"{item['model']}__{item['strategy']}"]["entry"] for item in candidates],
        "selected": selected,
    }
    timings = {key: round(item["search"].seconds, 1) for key, item in evaluated.items()}
    return summary, evaluated, split.y_train, timings


def save_figures(evaluated: dict[str, dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    searches = []
    calibration_panels = []
    for item in evaluated.values():
        entry, search = item["entry"], item["search"]
        label = f"{MODEL_LABELS[entry['model']]} ({STRATEGY_LABELS[entry['strategy']].lower()})"
        searches.append((entry["model"], label, search.cv_results, most_influential_params(search.cv_results, INFLUENTIAL_PARAM_COUNT)))
        raw_bins, calibrated_bins = item["bins"]
        calibration_panels.append((label, entry["model"], {"Kalibrasyonsuz": raw_bins, "Kalibre edilmiş (isotonic)": calibrated_bins}))
    save_figure(plot_search_results(searches), FIGURES_DIR / "h7_arama_sonuclari.png")
    save_figure(plot_calibration_curves(calibration_panels), FIGURES_DIR / "h7_kalibrasyon.png")


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Hafta 7 — Hiperparametre Optimizasyonu ve Kalibrasyon\n\n"
            "Arama `src/tuning.py`, kalibrasyon ve aday seçimi kuralları `src/final_model.py` içindedir. Rapordaki sayıların kaynağı "
            "`notebooks/build_hiperparametre.py` ile üretilen `hafta7_ozet.json` dosyasıdır (30 yapılandırma, iki aday). Bu notebook "
            "yöntemi XGBoost ile 5 yapılandırmalık küçük bir aramayla gösterir. Test seti ayrılır ama hiçbir hücrede kullanılmaz."
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
            "from src.evaluate import calibration_summary\n"
            "from src.modeling import build_model, cross_validate_model, make_cv\n"
            "from src.preprocessing import split_train_test\n"
            "from src.tuning import PARAM_SPECS, describe_spec, most_influential_params, tune_model\n"
            "\n"
            "split = split_train_test(load_raw())\n"
            "print('eğitim:', split.X_train.shape, 'test:', split.X_test.shape)"
        ),
        nbf.v4.new_markdown_cell("## 1. Arama uzayı"),
        nbf.v4.new_code_cell("pd.Series({param: describe_spec(spec) for param, spec in PARAM_SPECS['xgboost'].items()})"),
        nbf.v4.new_markdown_cell("## 2. XGBoost için 5 yapılandırmalık küçük arama (PR-AUC'ye göre)"),
        nbf.v4.new_code_cell(
            "search = tune_model(build_model('xgboost'), PARAM_SPECS['xgboost'], split.X_train, split.y_train, n_iter=5)\n"
            "print('en iyi parametreler:', search.best_params)\n"
            "print('en iyi CV PR-AUC:', round(search.best_score, 4))\n"
            "print('en etkili parametreler:', most_influential_params(search.cv_results))"
        ),
        nbf.v4.new_markdown_cell("## 3. Kalibrasyon özeti (ayarlanmış model, OOF tahminleri)"),
        nbf.v4.new_code_cell(
            "tuned = cross_validate_model(build_model('xgboost').set_params(**search.best_params), split.X_train, split.y_train, make_cv())\n"
            "summary = calibration_summary(split.y_train, tuned.oof_proba)\n"
            "print('Brier:', round(summary['brier'], 5))\n"
            "pd.DataFrame(summary['bins']).round(4)"
        ),
    ]
    write_and_execute_notebook(nb, NOTEBOOK_PATH)


def main() -> None:
    df = load_raw()
    validate_raw(df)
    summary, evaluated, _, timings = build_summary(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    TIMING_PATH.write_text(json.dumps({"search_seconds": timings, "total_search_seconds": round(sum(timings.values()), 1)}, indent=2), encoding="utf-8")
    save_figures(evaluated)
    create_notebook()
    total_seconds = sum(timings.values())
    print(f"Özet: {SUMMARY_PATH}\nGrafikler: {FIGURES_DIR}\nNotebook: {NOTEBOOK_PATH}\nToplam arama süresi: {total_seconds:.0f} sn")
    if total_seconds > SEARCH_TIME_LIMIT_SECONDS:
        print(f"UYARI: Toplam arama süresi {SEARCH_TIME_LIMIT_SECONDS // 60} dakikayı aştı (FOG-5). Kullanıcıya sorulmalı.")


if __name__ == "__main__":
    main()
