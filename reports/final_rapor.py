"""Final raporu iskeleti.

Kullanım: python reports/final_rapor.py --kaptan 230260054 --taslak
H2-H7 bölümleri haftalık özet JSON dosyalarından üretilir. 7, 8, 9. bölümler, Özet ve Ek-A kabul kriterleri tablosu Hafta 8 ve 9
bulguları gelene kadar [TASLAK] işaretli yer tutucudur. --taslak bayrağı olmadan rapor üretilmez, böylece yer tutucu metin teslim
edilen bir rapora sızmaz.
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
from src.formatting import format_number, format_percent
from src.modeling import LOGISTIC_REGRESSION, MODEL_LABELS, RANDOM_FOREST, STRATEGY_LABELS, XGBOOST
from src.report_docx import TEAM, WeeklyReport
from src.report_text import METRIC_LABELS, decimal

HAFTALIK_DIR = ROOT / "reports" / "haftalik"
OUTPUT_DIR = ROOT / "reports" / "haftalik"
FINAL_WEEK = 9
PENDING_MARK = "[TASLAK]"
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
# Hafta 8 ve 9 sonuçları gelene kadar yer tutucu olan bölümler.
PENDING_SECTIONS = ["ÖZET", "7. DEĞERLENDİRME", "8. TAHMİN SERVİSİ VE ARAYÜZ", "9. SONUÇ VE ÖNERİLER"]
PENDING_NOTES = {
    "ÖZET": "Projenin amacı, yöntemi ve ana bulguları (test sonuçları dahil) Hafta 9'da yazılacak.",
    "7. DEĞERLENDİRME": "Final modelin test seti sonuçları, karmaşıklık matrisi, kazanç tablosu, zaman bazlı test, değişken önemleri ve hata analizi Hafta 8'de eklenecek.",
    "8. TAHMİN SERVİSİ VE ARAYÜZ": "Model kaydı, tahmin servisi, girdi doğrulama ve Streamlit arayüzü Hafta 9'da eklenecek.",
    "9. SONUÇ VE ÖNERİLER": "Sonuçlar, sınırlılıklar ve öneriler Hafta 9'da yazılacak.",
}

# PRD'deki 13 kabul kriteri (kısa adlarıyla). Durumları Hafta 9'da özet JSON'dan doldurulacak.
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
    for week in range(2, 8):
        path = directory / f"hafta{week}_ozet.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadı. İlgili haftanın analiz betiğini çalıştırın.")
        summaries[week] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def final_file_name() -> str:
    student_ids = "_".join(sorted(member.student_id for member in TEAM))
    return f"{student_ids}_final.docx"


def pending_paragraph(report: WeeklyReport, section: str) -> None:
    report.add_paragraph(f"{PENDING_MARK} {PENDING_NOTES[section]}")


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


def add_acceptance_table(report: WeeklyReport) -> None:
    report.add_heading("EK-A. Kabul Kriterleri Kanıt Tablosu")
    report.add_table(
        "PRD kabul kriterleri ve kanıtları",
        ["No", "Kriter", "Kanıt"],
        [[str(number), criterion, f"{PENDING_MARK} Hafta 9'da doldurulacak"] for number, criterion in enumerate(ACCEPTANCE_CRITERIA, start=1)],
        [1.2, 9.8, 5.0],
    )


def merged_references() -> list[str]:
    seen: dict[str, None] = {}
    for reference in [*WEEK4_REFERENCES, *WEEK5_REFERENCES, *WEEK6_REFERENCES]:
        seen.setdefault(reference)
    return sorted(seen)


def build_final_report(summaries: dict[int, dict[str, Any]], captain_id: str, draft: bool) -> WeeklyReport:
    if not draft:
        raise ValueError(f"Final raporunda {PENDING_MARK} işaretli bölümler var ({', '.join(PENDING_SECTIONS)}). Taslak için --taslak bayrağını verin.")
    report = WeeklyReport(week=FINAL_WEEK, captain_id=captain_id)
    report.add_title_block(title="FİNAL RAPORU")
    for section in SECTION_TITLES:
        if section in PENDING_SECTIONS:
            report.add_heading(section)
            pending_paragraph(report, section)
        elif section == "1. GİRİŞ":
            add_introduction(report, summaries)
        elif section == "2. LİTERATÜR":
            add_literature(report)
        elif section == "3. YÖNTEM":
            add_method(report)
        elif section == "4. VERİ ANLAMA":
            add_data_understanding(report, summaries)
        elif section == "5. VERİ HAZIRLAMA":
            add_data_preparation(report, summaries)
        elif section == "6. MODELLEME":
            add_modeling(report, summaries)
    add_acceptance_table(report)
    report.add_links()
    report.add_references(merged_references())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Final raporu iskeletini .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Final raporunu hazırlayan kaptanın öğrenci numarası")
    parser.add_argument("--taslak", action="store_true", help="[TASLAK] yer tutucuları içeren raporun üretilmesine izin verir")
    args = parser.parse_args()
    output_path = build_final_report(load_summaries(), args.kaptan, args.taslak).save(OUTPUT_DIR / final_file_name())
    print(f"Taslak rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
