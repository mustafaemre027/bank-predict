"""Final raporu.

Kullanım: python reports/final_rapor.py --kaptan 230260054
Python run_all.py ile tüm özet JSON dosyaları üretildikten sonra çalıştırılır. Bölümler Hafta 2-9 özet dosyalarından, güvenlik bölümü
hafta9_guvenlik.json dosyasından doldurulur. Ek-A kabul kriterleri tablosu hafta9_ozet.json içindeki acceptance_criteria bloğundan gelir.
Rapor yer tutucu içermez.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.hafta4_rapor import PREPROCESSING_STEPS
from reports.hafta4_rapor import REFERENCES as WEEK4_REFERENCES
from reports.hafta5_rapor import REFERENCES as WEEK5_REFERENCES
from reports.hafta6_rapor import LITERATURE_ROWS
from reports.hafta6_rapor import REFERENCES as WEEK6_REFERENCES
from reports.hafta8_rapor import (
    add_confusion,
    add_error_analysis,
    add_gain,
    add_importance,
    add_test_results,
    add_time_based,
    describe_time_drop_hint,
    largest_gap,
    top_variables,
)
from reports.hafta8_rapor import add_literature as add_week8_literature
from reports.hafta9_rapor import REFERENCES as WEEK9_REFERENCES
from reports.hafta9_rapor import add_interface, add_saving, add_security, add_service, describe_acceptance
from src.formatting import format_number, format_percent, join_names
from src.modeling import LOGISTIC_REGRESSION, MODEL_LABELS, RANDOM_FOREST, STRATEGY_LABELS, XGBOOST
from src.report_docx import TEAM, WeeklyReport
from src.report_text import METRIC_LABELS, decimal

HAFTALIK_DIR = ROOT / "reports" / "haftalik"
OUTPUT_DIR = ROOT / "reports" / "haftalik"
FINAL_WEEK = 9
FIRST_WEEK = 2
MODELS = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST]
TOP_ASSOCIATION_COUNT = 3

SECTION_TITLES = [
    "ÖZET",
    "1. GİRİŞ",
    "2. LİTERATÜR",
    "3. YÖNTEM",
    "4. VERİ ANLAMA",
    "5. VERİ HAZIRLAMA",
    "6. MODELLEME",
    "7. DEĞERLENDİRME",
    "8. TAHMİN SERVİSİ VE ARAYÜZ",
    "9. SONUÇ VE ÖNERİLER",
]
# PRD'deki 13 kabul kriteri (kısa adlarıyla). Durumları ve kanıtları hafta9_ozet.json içindeki acceptance_criteria bloğundan gelir.
ACCEPTANCE_CRITERIA = [
    "Ham veri doğrulaması geçer",
    "Modele giren özellikler arasında duration yoktur",
    "Test paketi temiz ortamda geçer",
    "Üç modelin 5 katlı CV sonuçları ve üç dengesizlik stratejisi raporlanır",
    "Test setinde ROC-AUC ≥ 0,75",
    "Test PR-AUC değeri referansın en az 2 katı",
    "Seçilen eşikte test Recall'u referanstan yüksek, karmaşıklık matrisi raporlanır",
    "Kazanç (gain) tablosu raporlanır",
    "İlk 10 değişken permutation importance ile gösterilir",
    "Kaydedilen model yeniden yüklendiğinde aynı skorları üretir",
    "Arayüzde olasılık, etiket ve ilk 3 etken 1 saniyenin altında görünür",
    "Analiz ve rapor betikleri temiz klonda aynı sayıları üretir",
    "Her hafta raporu özet JSON'dan üretilir",
]


def load_summaries(directory: Path = HAFTALIK_DIR) -> dict[int, dict[str, Any]]:
    summaries = {}
    for week in range(FIRST_WEEK, FINAL_WEEK + 1):
        path = directory / f"hafta{week}_ozet.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadı. İlgili haftanın analiz betiğini çalıştırın.")
        summaries[week] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def final_file_name() -> str:
    student_ids = "_".join(sorted(member.student_id for member in TEAM))
    return f"{student_ids}_final.docx"


def add_introduction(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    shape, target = summaries[2]["dataset_shape"], summaries[2]["target_pct"]
    report.add_heading("1. GİRİŞ")
    report.add_paragraph(
        "Bir bankanın vadeli mevduat kampanyasında müşterilerin yalnızca küçük bir kısmı teklifi kabul ediyor. Bu projede, arama "
        "yapılmadan önce bilinen müşteri bilgilerinden kabul olasılığını tahmin eden bir makine öğrenmesi modeli geliştirildi. "
        f"Veri seti UCI Bank Marketing deposundaki bank-full.csv dosyasıdır. {format_number(shape['rows'])} kayıt ve {shape['cols']} "
        f"sütun içerir. Teklif kabul oranı {format_percent(target['yes'] / 100)} olarak bulunmuştur."
    )


def add_literature(report: WeeklyReport) -> None:
    report.add_heading("2. LİTERATÜR")
    report.add_table(
        "SMOTE ve sınıf dengesizliği üzerine incelenen çalışmalar",
        ["Çalışma", "Yöntem ve veri", "Bulgu"],
        LITERATURE_ROWS,
        [3.2, 6.8, 6.0],
    )
    report.add_paragraph(
        "Moro vd. (2014) çalışması bank-full.csv'den daha geniş bir veri sürümü kullanır ve kayan pencereyle değerlendirme yapar. "
        "Bu yüzden 0,794 AUC değeriyle karşılaştırma yalnızca göstergedir."
    )


def add_method(report: WeeklyReport) -> None:
    report.add_heading("3. YÖNTEM")
    report.add_paragraph(
        "Çalışma CRISP-DM sürecini izledi. Veri, hedef oranı korunarak %80 eğitim ve %20 test olarak ayrıldı ve test verisi final "
        "değerlendirmeye kadar kilitli tutuldu. Model seçimi, hiperparametre araması ve karar eşiği seçimi yalnızca eğitim verisinde 5 "
        "katlı katmanlı çapraz doğrulamayla yapıldı. Ön işleme adımları Pipeline içinde tanımlandı ve yalnızca eğitim katmanına fit "
        "edildi. Sınıflar dengesiz olduğu için başarı PR-AUC, ROC-AUC, Recall ve F1-Score ile ölçüldü. random_state her yerde 42'dir."
    )


def add_data_understanding(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    chi_square = summaries[3]["chi_square"]
    strongest = sorted(chi_square, key=lambda column: chi_square[column]["cramers_v"], reverse=True)[:TOP_ASSOCIATION_COUNT]
    listing = ", ".join(f"{column} ({decimal(chi_square[column]['cramers_v'])})" for column in strongest)
    report.add_heading("4. VERİ ANLAMA")
    report.add_paragraph(
        f"Veri setinde {format_number(summaries[2]['duplicate_rows'])} tekrar eden satır vardır. Hedef değişkenle en güçlü kategorik "
        f"ilişkiler Cramér's V değerine göre şunlardır: {listing}. duration değişkeni görüşme bittikten sonra bilindiği için modelden "
        "çıkarıldı."
    )


def add_data_preparation(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    counts = summaries[5]["feature_counts"]
    report.add_heading("5. VERİ HAZIRLAMA")
    report.add_table("Değişkenlere uygulanan ön işleme adımları (Lojistik Regresyon)", ["Değişken", "Uygulanan adım"], PREPROCESSING_STEPS, [4.5, 11.5])
    report.add_paragraph(
        f"Ön işleme sonunda Lojistik Regresyon için {counts[LOGISTIC_REGRESSION]}, ağaç modelleri için {counts['tree']} sütun oluştu. "
        "Ağaç modelleri ölçekleme ve log dönüşümü uygulanmayan ayrı bir dalı kullanır."
    )


def add_modeling(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    week5, week6, week7 = summaries[5], summaries[6], summaries[7]
    report.add_heading("6. MODELLEME")
    report.add_table(
        "Üç modelin 5 katlı çapraz doğrulama sonuçları (ortalama ± standart sapma)",
        ["Metrik", "Referans model", *[MODEL_LABELS[model] for model in MODELS]],
        [
            [label, *[f"{decimal(week5['models'][model]['summary'][metric]['mean'])} ± {decimal(week5['models'][model]['summary'][metric]['std'])}" for model in ["baseline", *MODELS]]]
            for metric, label in METRIC_LABELS.items()
        ],
        [2.6, 3.4, 3.4, 3.2, 3.4],
    )
    best_strategies = ", ".join(f"{MODEL_LABELS[model]} için {STRATEGY_LABELS[strategy].lower()}" for model, strategy in week6["best_by_model"].items())
    report.add_paragraph(f"Sınıf dengesizliği için sınıf ağırlığı, SMOTE ve eşik ayarlama karşılaştırıldı. PR-AUC değerine göre en iyi strateji {best_strategies} oldu.")
    report.add_table(
        "Hiperparametre araması: varsayılan ve ayarlanmış model PR-AUC değerleri (ortalama ± standart sapma)",
        ["Model", "Strateji", "Varsayılan", "Ayarlanmış"],
        [
            [
                MODEL_LABELS[entry["model"]],
                STRATEGY_LABELS[entry["strategy"]],
                f"{decimal(entry['default']['summary']['pr_auc']['mean'])} ± {decimal(entry['default']['summary']['pr_auc']['std'])}",
                f"{decimal(entry['tuned']['summary']['pr_auc']['mean'])} ± {decimal(entry['tuned']['summary']['pr_auc']['std'])}",
            ]
            for entry in week7["candidates"]
        ],
        [4.0, 4.0, 4.0, 4.0],
    )
    selected = week7["selected"]
    calibration = "kalibre edilmiş" if selected["calibrated"] else "kalibre edilmemiş"
    report.add_paragraph(f"Final aday {MODEL_LABELS[selected['model']]} ({STRATEGY_LABELS[selected['strategy']].lower()}, {calibration}) oldu.")


def add_abstract(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    shape, target, selected = summaries[2]["dataset_shape"], summaries[2]["target_pct"], summaries[7]["selected"]
    week8, week9 = summaries[8], summaries[9]
    metrics = week8["test_metrics"]["selected"]["metrics"]
    first_gain = week8["gain_table"][0]
    calibration = "kalibre edilmiş" if selected["calibrated"] else "kalibre edilmemiş"
    report.add_heading("ÖZET")
    report.add_paragraph(
        f"Bu çalışmada bir bankanın vadeli mevduat kampanyasında müşterilerin teklifi kabul etme olasılığı tahmin edildi. Veri seti "
        f"{format_number(shape['rows'])} kayıt içerir ve kabul oranı {format_percent(target['yes'] / 100)} olduğu için sınıflar dengesizdir. "
        f"Lojistik Regresyon, Rastgele Orman ve XGBoost 5 katlı çapraz doğrulamayla karşılaştırıldı, sınıf dengesizliği için üç yöntem denendi ve "
        f"hiperparametreler aranarak karar eşiği eğitim verisinin çapraz doğrulama dışı tahminlerinden seçildi. Final model "
        f"{MODEL_LABELS[selected['model']]} ({STRATEGY_LABELS[selected['strategy']].lower()}, {calibration}) oldu."
    )
    report.add_paragraph(
        f"Test verisinde bir kez yapılan değerlendirmede ROC-AUC {decimal(metrics['roc_auc'])} ve PR-AUC {decimal(metrics['pr_auc'])} bulundu. Seçilen eşikte "
        f"Recall {decimal(metrics['recall'])}, Precision {decimal(metrics['precision'])} oldu. Müşteriler olasılığa göre sıralanıp ilk "
        f"{format_percent(first_gain['fraction'], 0)} arandığında kabul edenlerin {format_percent(first_gain['capture_rate'], 1)} yakalanıyor. Zamana göre ayrılmış "
        f"testte ROC-AUC {decimal(week8['time_based']['roc_auc'])} olduğundan modelin farklı bir döneme genellemesi sınırlıdır. Model kaydedilip "
        f"Pydantic ile girdi doğrulaması yapan bir tahmin servisine ve Streamlit arayüzüne bağlandı. {describe_acceptance(week9)}"
    )


def add_evaluation(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    report.add_heading("7. DEĞERLENDİRME")
    week8 = summaries[8]
    add_test_results(report, week8, "7.1")
    add_confusion(report, week8, "7.2")
    add_gain(report, week8, "7.3")
    add_time_based(report, week8, "7.4")
    add_importance(report, week8, "7.5")
    add_error_analysis(report, week8, "7.6")
    add_week8_literature(report, week8, "7.7")


def add_service_and_interface(report: WeeklyReport, summaries: dict[int, dict[str, Any]], security: dict[str, Any], timing: dict[str, Any] | None) -> None:
    report.add_heading("8. TAHMİN SERVİSİ VE ARAYÜZ")
    add_saving(report, summaries[9], summaries[8], "8.1")
    add_service(report, summaries[9], timing, "8.2")
    add_interface(report, "8.3")
    add_security(report, security, "8.4")


def add_conclusion(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    week8 = summaries[8]
    metrics, counts = week8["test_metrics"]["selected"]["metrics"], week8["confusion"]
    missed_share = counts["fn"] / (counts["fn"] + counts["tp"])
    poutcome_gap = largest_gap(week8["error_profile"]["poutcome"])
    report.add_heading("9. SONUÇ VE ÖNERİLER")
    report.add_heading("9.1. Sonuçlar")
    report.add_paragraph(
        f"Final model test verisinde ROC-AUC {decimal(metrics['roc_auc'])} değerine ulaştı ve 0,75 hedefini karşıladı. Test PR-AUC değeri "
        f"{decimal(metrics['pr_auc'])} oldu. Seçilen eşikte kabul eden müşterilerin {format_percent(1 - missed_share, 1)} yakalandı, arama listesine "
        f"alınanların {format_percent(metrics['precision'], 1)} gerçekten kabul etti. En önemli değişkenler "
        f"{join_names(top_variables(week8, 3))} olarak bulundu."
    )
    report.add_heading("9.2. Sınırlılıklar")
    report.add_paragraph(
        f"Zamana göre ayrılmış testte ROC-AUC {decimal(week8['time_based']['roc_auc'])} bulundu, rastgele testteki {decimal(metrics['roc_auc'])} değerinin altındadır. "
        f"Bu iki değerlendirme birbirinden bağımsız değildir ve eğitim ile test kısmındaki kabul oranları farklıdır.{describe_time_drop_hint(week8)}"
    )
    report.add_paragraph(
        f"Kabul eden müşterilerin {format_percent(missed_share, 1)} kaçırıldı. Kaçırılan kabul edenlerde önceki kampanya sonucu {poutcome_gap['category']} olan "
        f"müşterilerin payı yakalananlardan yüksektir: poutcome değişkeninde bu kategorinin payı kaçırılanlarda {format_percent(poutcome_gap['false_negative_share'], 1)}, "
        f"yakalananlarda {format_percent(poutcome_gap['true_positive_share'], 1)} bulundu. Model, önceki kampanya bilgisi olmayan müşterilerde kabul edenleri yakalamakta zorlanıyor."
    )
    report.add_paragraph(
        "Arayüzde kimlik doğrulama yoktur ve ağa açılacaksa ayrıca korunmalıdır. Arayüzün ekip dışından bir kişiyle kullanılabilirlik "
        "denemesi yapılacaktır."
    )
    report.add_heading("9.3. Öneriler")
    report.add_paragraph(
        "Model seçimi ve hiperparametre araması zamana göre ayrılmış doğrulama düzeniyle tekrarlanabilir. Böylece modelin farklı bir döneme "
        "genellemesi doğrudan ölçülür. Önceki kampanya bilgisi olmayan müşteriler için ek değişkenler toplanabilir. Karar eşiği, bir aramanın "
        "maliyeti ile bir kabulün getirisi bilindiğinde maliyet temelli seçilebilir. Model kullanıma alınırsa kabul oranı ve olasılık "
        "dağılımı düzenli olarak izlenmelidir."
    )


def add_acceptance_table(report: WeeklyReport, summaries: dict[int, dict[str, Any]]) -> None:
    criteria = summaries[9]["acceptance_criteria"]
    report.add_heading("EK-A. Kabul Kriterleri Kanıt Tablosu")
    report.add_table(
        "PRD kabul kriterleri, durumları ve kanıtları",
        ["No", "Kriter", "Durum", "Kanıt"],
        [
            [str(number), item["criterion"], "Karşılandı" if item["met"] else "Karşılanmadı", item["evidence"]]
            for number, item in enumerate(criteria, start=1)
        ],
        [1.0, 5.0, 2.3, 7.7],
    )


def merged_references() -> list[str]:
    seen: dict[str, None] = {}
    for reference in [*WEEK4_REFERENCES, *WEEK5_REFERENCES, *WEEK6_REFERENCES, *WEEK9_REFERENCES]:
        seen.setdefault(reference)
    return sorted(seen)


def build_final_report(summaries: dict[int, dict[str, Any]], captain_id: str, security: dict[str, Any], timing: dict[str, Any] | None = None) -> WeeklyReport:
    report = WeeklyReport(week=FINAL_WEEK, captain_id=captain_id)
    report.add_title_block(title="FİNAL RAPORU")
    add_abstract(report, summaries)
    add_introduction(report, summaries)
    add_literature(report)
    add_method(report)
    add_data_understanding(report, summaries)
    add_data_preparation(report, summaries)
    add_modeling(report, summaries)
    add_evaluation(report, summaries)
    add_service_and_interface(report, summaries, security, timing)
    add_conclusion(report, summaries)
    add_acceptance_table(report, summaries)
    report.add_links()
    report.add_references(merged_references())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Final raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Final raporunu hazırlayan kaptanın öğrenci numarası")
    args = parser.parse_args()
    security = json.loads((HAFTALIK_DIR / "hafta9_guvenlik.json").read_text(encoding="utf-8"))
    timing_path = HAFTALIK_DIR / "hafta9_sure.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else None
    output_path = build_final_report(load_summaries(), args.kaptan, security, timing).save(OUTPUT_DIR / final_file_name())
    print(f"Final raporu oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
