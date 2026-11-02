"""7. hafta ilerleme raporunu üretir.

Kullanım: python reports/hafta7_rapor.py --kaptan 230260173
Önce notebooks/build_hiperparametre.py çalıştırılmalıdır. Metindeki tüm sayılar ve sayıya dayalı hükümler (iyileşme/kötüleşme,
kalibrasyon kararı, final aday gerekçesi) hafta7_ozet.json dosyasından türetilir; yalnızca yorum cümleleri bu haftanın bulgularına göre
yazılmıştır. Arama süresi hafta7_sure.json dosyasından okunur, dosya yoksa süre cümlesi yazılmaz. Bölüm 2 (ekip içi çalışma süreci)
işaretleri taslaktır; teslimden önce takım kaptanı tarafından gözden geçirilmelidir.
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

from reports.final_rapor import PENDING_SECTIONS, SECTION_TITLES
from reports.hafta6_rapor import NEXT_WEEK as WEEK6_PLAN
from src.formatting import format_number, format_percent, join_names
from src.modeling import IMBALANCE_NONE, MODEL_LABELS, STRATEGY_LABELS
from src.report_docx import CAPTAIN_ROLE, TEAM, TEAM_PROCESS_CRITERIA, WeeklyReport, report_file_name
from src.report_text import Summary, decimal, describe_comparison, lower_first, number_word, signed

WEEK = 7
CAPTAIN_DUTY = "Takım kaptanlığı ve haftalık raporun hazırlanması"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta7_ozet.json"
TIMING_PATH = ROOT / "reports" / "haftalik" / "hafta7_sure.json"
FIGURES_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR = ROOT / "reports" / "haftalik"
DONE_THIS_WEEK = WEEK6_PLAN
# FOG-5: tam hiperparametre araması bu süreyi aşmamalı.
SEARCH_TIME_LIMIT_SECONDS = 30 * 60
SECONDS_PER_MINUTE = 60
REASON_HIGHEST = "highest_pr_auc"
REASON_SIMPLER = "simpler_within_noise"

# Kriterler 6. hafta raporunun "Gelecek Hafta Planı" tablosunda bu hafta için verilen görevlerden türetilmiştir.
PROGRESS_CRITERIA = [
    "En iyi iki model için hiperparametre araması yapıldı",
    "Arama süresi ölçüldü",
    "Olasılık kalibrasyonu kontrol edildi",
    "Final aday model ve karar eşiği belirlendi",
    "Final raporunun iskeleti hazırlandı",
]

WORK_PACKAGES = [
    ("1", "Literatür Taraması ve Veri Seti Temini", "Tüm Ekip", "Tamamlandı"),
    ("2", "Veri Ön İşleme ve Temizleme", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("3", "Keşifsel Veri Analizi", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("4", "Makine Öğrenmesi Modellerinin Kurulması", "Ömer NARÇİÇEĞİ", "Tamamlandı"),
    ("5", "Model Eğitimi ve Optimizasyon", "Ömer NARÇİÇEĞİ", "Tamamlandı"),
    ("6", "Modellerin Test Edilmesi", "Mesut ASLAN", "Devam ediyor"),
    ("7", "Sonuçların Değerlendirilmesi ve Proje Raporunun Hazırlanması", "Tüm Ekip", "Devam ediyor"),
]

NEXT_WEEK = [
    ("Ömer NARÇİÇEĞİ", "Final modelin tüm eğitim verisiyle eğitilmesi ve test verisinde bir kez değerlendirilmesi"),
    ("Mesut ASLAN", "Karmaşıklık matrisi, kazanç tablosu ve zamana göre ayrılmış son %20 kayıtta ek değerlendirme"),
    ("Mustafa Emre KAPLAN", "Permutation importance ve Lojistik Regresyon katsayılarıyla değişken önemlerinin incelenmesi, hatalı tahminlerin analizi"),
]

REFERENCES = [
    "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. "
    "Decision Support Systems, 62, 22-31.",
    "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing veri seti. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. Proceedings of the 4th "
    "International Conference on the Practical Applications of Knowledge Discovery and Data Mining, 29-39.",
    "Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281-305.",
    "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. Proceedings of the 22nd "
    "International Conference on Machine Learning, 625-632.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary "
    "classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
    "Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. Proceedings of "
    "the 14th International Joint Conference on Artificial Intelligence, 1137-1143.",
]


def candidate_label(entry: dict[str, Any]) -> str:
    return f"{MODEL_LABELS[entry['model']]} ({STRATEGY_LABELS[entry['strategy']].lower()})"


def format_tuned_value(value: object) -> str:
    return f"{value:.4g}".replace(".", ",") if isinstance(value, float) else str(value)


def format_tuned_params(params: dict[str, object]) -> str:
    return ", ".join(f"{name.removeprefix('model__')} = {format_tuned_value(value)}" for name, value in params.items())


def format_duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    return f"{minutes} dakika {rest} saniye" if minutes else f"{rest} saniye"


def load_timing() -> dict[str, Any] | None:
    """Arama süreleri her çalıştırmada değiştiği için özet JSON'da değil ayrı dosyadadır; dosya yoksa süre anlatılmaz."""
    if not TIMING_PATH.exists():
        return None
    return json.loads(TIMING_PATH.read_text(encoding="utf-8"))


def describe_timing(timing: dict[str, Any] | None) -> str:
    if timing is None:
        return ""
    total = timing["total_search_seconds"]
    limit = f"{SEARCH_TIME_LIMIT_SECONDS // SECONDS_PER_MINUTE} dakikalık sınırın "
    verdict = f"{limit}altında kaldı" if total <= SEARCH_TIME_LIMIT_SECONDS else f"{limit}üstüne çıktı"
    return f" İki aramanın toplam süresi {format_duration(total)} oldu ve {verdict}."


def calibration_reason(entry: dict[str, Any], max_drop: float) -> str:
    """Kalibrasyonun uygulanıp uygulanmadığını sayılara dayanarak gerekçelendirir."""
    calibration = entry["calibration"]
    if entry["strategy"] != IMBALANCE_NONE:
        return "bu strateji olasılıkları tasarım gereği şişirdiği için kalibrasyon zorunludur"
    if calibration["applied"]:
        return "Brier skoru düştü ve PR-AUC ortalaması izin verilen sınırdan fazla azalmadı"
    if calibration["brier_calibrated"] >= calibration["brier_raw"]:
        return "Brier skoru düşmedi"
    return f"PR-AUC ortalaması {decimal(max_drop, 3)} sınırından fazla azaldı"


def describe_selection(selected: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    final = next(entry for entry in candidates if entry["model"] == selected["model"])
    if selected["reason"] == REASON_HIGHEST:
        return f"PR-AUC ortalaması en yüksek aday {candidate_label(final)} olduğu için final model olarak seçildi."
    return (f"PR-AUC ortalaması en yüksek model {MODEL_LABELS[selected['highest_pr_auc_model']]} idi. Ancak aradaki fark katmanlar "
            f"arasındaki standart sapmanın altında kaldığı için aynı başarıda daha basit ve hızlı olan {candidate_label(final)} seçildi.")


def add_introduction(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "Altıncı haftada sınıf dengesizliği için üç yöntem karşılaştırılmış ve karar eşiği çapraz doğrulama dışı tahminlerden "
        "seçilmişti. Hiperparametre araması için PR-AUC sıralamasındaki ilk iki farklı model belirlenmişti.",
        f"Yedinci haftada bu iki model için {summary['n_iter']} yapılandırmalık rastgele arama yapıldı ve ayarlanmış modeller varsayılan "
        "hâlleriyle aynı katmanlarda karşılaştırıldı. Olasılık kalibrasyonu kontrol edildi, final aday model ve karar eşiği belirlendi. "
        "Ayrıca final raporunun iskeleti hazırlandı. Test verisi bu hafta da kullanılmadı.",
    ])


def add_goals(report: WeeklyReport) -> None:
    report.add_heading("1.1. Amaç ve Hedefler")
    report.add_labeled_box([
        ("Amaç:", "Bankanın hedef kitlesini doğru analiz ederek pazarlama kampanyalarının başarı oranını artıracak ve maliyetleri "
                  "düşürecek ölçülebilir bir makine öğrenmesi tahmin modeli ortaya koymaktır."),
        ("Hedefler:", "Bank Marketing veri setini temin etmek ve doğrulamak. Bu hedef ikinci haftada tamamlanmıştır. Değişkenlerin "
                      "hedefle ilişkisini keşifsel analiz ve istatistiksel testlerle incelemek. Bu hedef üçüncü haftada "
                      "tamamlanmıştır. Bulgulara göre ön işleme adımlarını kurmak. Bu hedef dördüncü ve beşinci haftalarda "
                      "tamamlanmıştır. Lojistik Regresyon, Rastgele Orman ve XGBoost modellerini çapraz doğrulama ile eğitmek, "
                      "sınıf dengesizliği için farklı yöntemleri karşılaştırmak ve hiperparametreleri çapraz doğrulama ile "
                      "ayarlamak. Bu hedef yedinci haftada tamamlanmıştır. Veri setindeki sınıflar dengesiz olduğu için "
                      "modelleri yalnızca Accuracy ile değil, Precision, Recall, F1-Score, ROC-AUC ve PR-AUC ile de "
                      "değerlendirmek. Test verisinde ROC-AUC değeri en az 0,75 olan bir model elde etmek ve bu modeli basit "
                      "bir arayüzle kullanılabilir hâle getirmek."),
    ])


def add_method(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin modelleme aşamasında çalışıldı (Wirth ve Hipp, 2000). Veri bölmesi ve çapraz doğrulama katmanları "
        f"önceki haftalarla aynı kaldı. Eğitim verisinde {format_number(summary['split']['train_rows'])}, test verisinde "
        f"{format_number(summary['split']['test_rows'])} kayıt bulunmaktadır. Test verisi aramada, kalibrasyonda ve eşik seçiminde "
        "kullanılmadı.",
        f"Hiperparametre araması rastgele arama ile yapıldı (Bergstra ve Bengio, 2012). Her model için {summary['n_iter']} yapılandırma "
        "örneklendi ve her biri 5 katlı çapraz doğrulamada PR-AUC değerine göre puanlandı (Saito ve Rehmsmeier, 2015). Arama "
        "paralel çalıştığı için modellerin kendi iş parçacıkları kapatıldı, böylece sonuçlar tekrar üretilebilir kaldı.",
        "Olasılık kalibrasyonu isotonic yöntemle, çapraz doğrulamanın her eğitim katmanında ayrı yapıldı (Niculescu-Mizil ve Caruana, 2005). "
        "Kalibrasyonun uygulanıp uygulanmayacağı önceden kurala bağlandı. Sınıf ağırlığı ve SMOTE olasılıkları tasarım gereği "
        "şişirdiği için bu stratejilerde kalibrasyon zorunludur. Ağırlıksız modelde kalibrasyon yalnızca Brier skorunu düşürürse ve "
        f"PR-AUC ortalamasını en fazla {decimal(summary['max_pr_auc_drop_for_calibration'], 3)} azaltırsa uygulanır.",
        "Final aday, ayarlanmış modeller arasında PR-AUC ortalaması en yüksek olandır. Daha basit bir model ona katmanlar arası "
        "standart sapma içinde yakınsa o seçilir. Basitlik sırası Lojistik Regresyon, Rastgele Orman, XGBoost şeklindedir. Karar "
        "eşiği final modelin out-of-fold tahminlerinden seçildi.",
    ])


def add_project_management(report: WeeklyReport) -> None:
    report.add_heading("3. PROJE YÖNETİMİ")
    report.add_heading("3.1. Takım Üyeleri Tanımı")
    member_rows = []
    for number, member in enumerate(TEAM, start=1):
        role = f"{CAPTAIN_ROLE} / {member.role}" if member == report.captain else member.role
        member_rows.append([str(number), member.student_id, member.full_name, role])
    report.add_table("Takım üyeleri", ["Üye No", "Numara", "Adı Soyadı", "Görevi"], member_rows, [1.5, 2.5, 4.5, 7.5])
    report.add_heading("3.2. Görev Dağılımı")
    report.add_table(
        "İş paketleri ve durumları",
        ["İP No", "İş Paketinin Adı ve Hedefleri", "Görevlendirilen Takım Üyesi", "Durum"],
        [list(row) for row in WORK_PACKAGES],
        [1.3, 7.2, 4.3, 3.2],
    )
    report.add_heading("3.3. Bu Hafta Yapılan Çalışmalar")
    done_rows = [
        [name, f"{CAPTAIN_DUTY}, {lower_first(work)}" if name == report.captain.full_name else work]
        for name, work in DONE_THIS_WEEK
    ]
    report.add_table("Bu hafta yapılan çalışmalar", ["Takım Üyesi", "Yapılan Çalışma"], done_rows, [4.5, 11.5])
    report.add_heading("3.4. Gelecek Hafta Planı")
    report.add_paragraph("Sekizinci haftada final model tüm eğitim verisiyle eğitilecek ve test verisinde bir kez değerlendirilecektir.")
    report.add_table("Gelecek hafta planı", ["Takım Üyesi", "Planlanan Çalışma"], [list(row) for row in NEXT_WEEK], [4.5, 11.5])


def add_search(report: WeeklyReport, summary: Summary, timing: dict[str, Any] | None) -> None:
    candidates = summary["candidates"]
    listing = join_names([candidate_label(entry) for entry in candidates])
    rows = [
        [MODEL_LABELS[model], param.removeprefix("model__"), description]
        for model, space in summary["search_space"].items()
        for param, description in space.items()
    ]
    report.add_heading("4.1. Hiperparametre Araması")
    report.add_paragraph(
        f"Altıncı haftadaki sıralamaya göre arama {listing} için yapıldı. Her model için {summary['n_iter']} yapılandırma örneklendi. "
        f"Aranan parametreler ve dağılımları aşağıdaki tabloda verilmiştir.{describe_timing(timing)}"
    )
    report.add_table("Hiperparametre arama uzayı", ["Model", "Parametre", "Dağılım"], rows, [4.0, 5.0, 7.0])


def add_tuned_models(report: WeeklyReport, summary: Summary) -> None:
    candidates = summary["candidates"]
    report.add_heading("4.2. Ayarlanmış Modellerin Karşılaştırılması")
    report.add_table(
        "Varsayılan ve ayarlanmış modellerin PR-AUC değerleri (ortalama ± standart sapma)",
        ["Model", "Varsayılan", "Ayarlanmış", "Fark"],
        [
            [
                candidate_label(entry),
                f"{decimal(entry['default']['summary']['pr_auc']['mean'])} ± {decimal(entry['default']['summary']['pr_auc']['std'])}",
                f"{decimal(entry['tuned']['summary']['pr_auc']['mean'])} ± {decimal(entry['tuned']['summary']['pr_auc']['std'])}",
                signed(entry["comparison_vs_default"]["mean_difference"]),
            ]
            for entry in candidates
        ],
        [5.6, 3.8, 3.8, 2.8],
    )
    report.add_paragraph(" ".join(
        describe_comparison(entry["comparison_vs_default"], f"{MODEL_LABELS[entry['model']]} için arama sonrası parametreler kullanıldığında")
        for entry in candidates
    ))
    report.add_paragraph(
        "En iyi yapılandırma, karşılaştırmada kullanılan aynı katmanlardaki skora göre seçildiği için bu karşılaştırma biraz "
        "iyimserdir. Ayarlamanın gerçek etkisini sekizinci haftada test verisinde göreceğiz. Seçilen parametreler aşağıdaki tabloda verilmiştir."
    )
    report.add_table(
        "Aramada bulunan en iyi parametreler",
        ["Model", "Parametreler"],
        [[candidate_label(entry), format_tuned_params(entry["tuned_params"])] for entry in candidates],
        [5.0, 11.0],
    )
    report.add_figure(
        FIGURES_DIR / "h7_arama_sonuclari.png",
        "Her adayda skoru en çok etkileyen iki parametrenin değeri ve CV PR-AUC skoru (her nokta bir yapılandırma, kenarlıklı nokta en iyisi)",
    )


def add_calibration(report: WeeklyReport, summary: Summary) -> None:
    candidates = summary["candidates"]
    max_drop = summary["max_pr_auc_drop_for_calibration"]
    report.add_heading("4.3. Olasılık Kalibrasyonu")
    report.add_table(
        "Ayarlanmış modellerde kalibrasyon: Brier skoru ve PR-AUC",
        ["Model", "Brier (kalibrasyonsuz)", "Brier (isotonic)", "PR-AUC (kalibrasyonsuz)", "PR-AUC (isotonic)", "Karar"],
        [
            [
                candidate_label(entry),
                decimal(entry["calibration"]["brier_raw"], 4),
                decimal(entry["calibration"]["brier_calibrated"], 4),
                decimal(entry["calibration"]["pr_auc_raw"]),
                decimal(entry["calibration"]["pr_auc_calibrated"]),
                "Uygulandı" if entry["calibration"]["applied"] else "Uygulanmadı",
            ]
            for entry in candidates
        ],
        [4.2, 2.5, 2.3, 2.7, 2.3, 2.0],
    )
    report.add_paragraph(" ".join(
        f"{candidate_label(entry)} için kalibrasyon {'uygulandı' if entry['calibration']['applied'] else 'uygulanmadı'}, çünkü "
        f"{calibration_reason(entry, max_drop)}."
        for entry in candidates
    ))
    report.add_figure(
        FIGURES_DIR / "h7_kalibrasyon.png",
        "Kalibrasyon eğrileri: eşit sayıda müşteri içeren dilimlerde ortalama tahmin edilen olasılık ve gerçek kabul oranı (OOF tahminleri)",
    )


def add_final_candidate(report: WeeklyReport, summary: Summary) -> None:
    selected = summary["selected"]
    final = next(entry for entry in summary["candidates"] if entry["model"] == selected["model"])
    f1_choice, recall_choice = selected["threshold_f1"], selected["threshold_recall"]
    nested_f1, nested_recall = selected["nested_summary_f1_threshold"], selected["nested_summary_recall_threshold"]

    def row(label: str, choice: dict[str, Any], nested: dict[str, dict[str, float]]) -> list[str]:
        return [
            label,
            decimal(choice["threshold"], 2),
            *[f"{decimal(nested[metric]['mean'])} ± {decimal(nested[metric]['std'])}" for metric in ("f1", "recall", "precision")],
        ]

    report.add_heading("4.4. Final Aday Model ve Karar Eşiği")
    calibration = "kalibre edilmiş" if selected["calibrated"] else "kalibre edilmemiş"
    report.add_paragraph(
        f"{describe_selection(selected, summary['candidates'])} Final aday {candidate_label(final)}, {calibration} olasılıklarla "
        f"kullanılacak. Ayarlanmış modelin PR-AUC değeri {decimal(final['tuned']['summary']['pr_auc']['mean'])} ± "
        f"{decimal(final['tuned']['summary']['pr_auc']['std'])}, ROC-AUC değeri {decimal(final['tuned']['summary']['roc_auc']['mean'])} ± "
        f"{decimal(final['tuned']['summary']['roc_auc']['std'])} bulundu."
    )
    report.add_table(
        "Final adayın out-of-fold tahminlerinden seçilen eşikler (katman sonuçları, ortalama ± standart sapma)",
        ["Ölçüt", "Eşik", "F1", "Recall", "Precision"],
        [
            row("F1 en yüksek", f1_choice, nested_f1),
            row(f"Recall en az {decimal(summary['target_recall'], 2)}", recall_choice, nested_recall),
        ],
        [4.6, 2.0, 3.2, 3.2, 3.0],
    )
    report.add_paragraph(
        "Eşik, final modelin out-of-fold tahminlerinden seçildi ve test verisine bakılmadı. Tablodaki katman sonuçları için eşik her "
        "katmanda o katman dışındaki tahminlerden seçilip yalnızca o katmanda ölçüldü. Olasılık bandı için alt sınır eğitim verisindeki "
        f"kabul oranıdır ({format_percent(selected['base_rate'])}), üst sınır ise F1'i en yüksek yapan eşiktir "
        f"({decimal(f1_choice['threshold'], 2)}). İş senaryosunda daha çok kabul eden müşteri yakalamak istenirse Recall ölçütüne göre "
        f"seçilen daha düşük eşik ({decimal(recall_choice['threshold'], 2)}) kullanılabilir. Bu karar sekizinci haftada test sonuçlarıyla birlikte değerlendirilecektir."
    )


def add_final_report_skeleton(report: WeeklyReport) -> None:
    report.add_heading("4.5. Final Raporu İskeleti")
    report.add_paragraph(
        f"Final raporunun iskeleti reports/final_rapor.py dosyasında hazırlandı. Rapor {number_word(len(SECTION_TITLES))} bölümden ve "
        "kabul kriterleri ek tablosundan oluşuyor. Giriş, literatür, yöntem, veri anlama, veri hazırlama ve modelleme bölümleri haftalık "
        f"özet dosyalarından otomatik doldurulmaktadır. Bölümlerden {number_word(len(PENDING_SECTIONS))} tanesi ile kabul kriterleri tablosu sekizinci ve "
        "dokuzuncu hafta sonuçları gelene kadar yer tutucu olarak bırakıldı. Literatür tablosu, Moro ve arkadaşlarının çalışmasının "
        "daha geniş bir veri sürümü kullandığı notunu içermektedir."
    )


def build_report(summary: Summary, captain_id: str, timing: dict[str, Any] | None = None) -> WeeklyReport:
    report = WeeklyReport(week=WEEK, captain_id=captain_id)
    # Ekip süreci işaretleri taslaktır; kaptan teslimden önce gerçek duruma göre değiştirmelidir.
    report.add_captain_form(
        team_process=[(criterion, True) for criterion in TEAM_PROCESS_CRITERIA],
        progress=[(criterion, True) for criterion in PROGRESS_CRITERIA],
    )
    report.add_title_block()
    add_introduction(report, summary)
    add_goals(report)
    add_method(report, summary)
    add_project_management(report)
    report.add_heading("4. TEORİ VE SONUÇLAR")
    report.add_box([
        "Müşterilerin durumları, sosyoekonomik özellikleri ve geçmiş bankacılık alışkanlıkları, pazarlama kampanyalarına verecekleri "
        "yanıtı doğrudan etkilemektedir. Bu hafta bu ilişkileri en iyi yakalayan modelin ayarlanması ve tahmin olasılıklarının "
        "güvenilirliği incelendi. Sonuçlar aşağıdaki alt başlıklarda verilmiştir.",
    ])
    add_search(report, summary, timing)
    add_tuned_models(report, summary)
    add_calibration(report, summary)
    add_final_candidate(report, summary)
    add_final_report_skeleton(report)
    report.add_links()
    report.add_references(REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="7. hafta ilerleme raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    output_path = build_report(summary, args.kaptan, load_timing()).save(OUTPUT_DIR / report_file_name(WEEK))
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
