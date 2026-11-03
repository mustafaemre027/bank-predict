"""8. hafta ilerleme raporunu üretir.

Kullanım: python reports/hafta8_rapor.py --kaptan 230260143
Önce notebooks/build_final_degerlendirme.py çalıştırılmalıdır. Metindeki tüm sayılar ve sayıya dayalı hükümler (FOG-1 sonucu, CV ile test
farkı, zamana göre testin yönü, en önemli değişkenler, en çok kaçırılan kategoriler) hafta8_ozet.json dosyasından türetilir. Veri metnin
dayandığı karara aykırıysa rapor üretimi hata vererek durur. Bölüm 2 (ekip içi çalışma süreci) işaretleri taslaktır, teslimden önce
takım kaptanı tarafından gözden geçirilmelidir.
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

from reports.hafta7_rapor import NEXT_WEEK as WEEK7_PLAN
from reports.hafta7_rapor import REFERENCES as WEEK7_REFERENCES
from src.explain import DEFAULT_REPEATS
from src.formatting import format_number, format_percent, join_names
from src.report_docx import CAPTAIN_ROLE, TEAM, TEAM_PROCESS_CRITERIA, WeeklyReport, report_file_name
from src.report_text import Summary, decimal, lower_first, number_word

WEEK = 8
CAPTAIN_DUTY = "Takım kaptanlığı ve haftalık raporun hazırlanması"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta8_ozet.json"
FIGURES_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR = ROOT / "reports" / "haftalik"
DONE_THIS_WEEK = WEEK7_PLAN
THRESHOLD_LABELS = {
    "selected": "Seçilen eşik (F1 en yüksek)",
    "default_0_5": "Varsayılan eşik",
    "target_recall": "Recall hedefli eşik",
}
IMPORTANCE_ROWS = 10
COEFFICIENT_ROWS = 10
TOP_VARIABLE_COUNT = 5
ERROR_TABLE_COLUMNS = ["poutcome", "contact", "was_contacted_before"]
ERROR_TEXT_COLUMNS = ["poutcome", "contact", "month"]
TIME_DROP_CANDIDATES = ("month", "day")
TIME_DROP_RANK_LIMIT = 4
NO_VALUE = "-"

# Kriterler 7. hafta raporunun "Gelecek Hafta Planı" tablosunda bu hafta için verilen görevlerden türetilmiştir.
PROGRESS_CRITERIA = [
    "Final model tüm eğitim verisiyle eğitildi",
    "Test verisinde bir kez değerlendirildi",
    "Karmaşıklık matrisi ve kazanç tablosu hazırlandı",
    "Zamana göre ayrılmış veride ek değerlendirme yapıldı",
    "Değişken önemleri incelendi",
    "Hatalı tahminler analiz edildi",
]

WORK_PACKAGES = [
    ("1", "Literatür Taraması ve Veri Seti Temini", "Tüm Ekip", "Tamamlandı"),
    ("2", "Veri Ön İşleme ve Temizleme", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("3", "Keşifsel Veri Analizi", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("4", "Makine Öğrenmesi Modellerinin Kurulması", "Ömer NARÇİÇEĞİ", "Tamamlandı"),
    ("5", "Model Eğitimi ve Optimizasyon", "Ömer NARÇİÇEĞİ", "Tamamlandı"),
    ("6", "Modellerin Test Edilmesi", "Mesut ASLAN", "Tamamlandı"),
    ("7", "Sonuçların Değerlendirilmesi ve Proje Raporunun Hazırlanması", "Tüm Ekip", "Devam ediyor"),
]

NEXT_WEEK = [
    ("Ömer NARÇİÇEĞİ", "Final modelin kaydedilmesi, tahmin servisi ve final raporunun tamamlanması"),
    ("Mesut ASLAN", "Tahmin servisinde girdi doğrulaması, testleri ve süre ölçümü, sunumun hazırlanması"),
    ("Mustafa Emre KAPLAN", "Streamlit arayüzünün geliştirilmesi ve ekip dışından bir kişiyle denenmesi"),
]

REFERENCES = [
    *WEEK7_REFERENCES,
    "Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by "
    "studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20(177), 1-81.",
]


def percent(ratio: float) -> str:
    return format_percent(ratio, 1)


def threshold_text(value: float) -> str:
    return decimal(value, 2)


def check_threshold_matches_selection(summary: Summary) -> None:
    """Testte kullanılan eşik, Hafta 7'de OOF tahminlerinden seçilen eşikle aynı olmalı. Aksi halde test sonucuna göre eşik ayarlanmış olabilir."""
    used, chosen = summary["test_metrics"]["selected"]["threshold"], round(summary["selected"]["threshold"], 4)
    if used != chosen:
        raise ValueError(f"Testte kullanılan eşik ({used}) Hafta 7'de seçilen eşikten ({chosen}) farklı. Eşik test sonucuna göre değiştirilemez.")


def check_confusion_adds_up(summary: Summary) -> None:
    counts = summary["confusion"]
    if sum(counts.values()) != summary["split"]["test_rows"]:
        raise ValueError("Karmaşıklık matrisindeki sayıların toplamı test kayıt sayısına eşit değil.")


def check_recall_is_positive(summary: Summary) -> None:
    """Kabul kriteri 7: seçilen eşikte Recall sıfırdan büyük olmalı. Sıfırsa model kimseyi kabul eden olarak işaretlememiştir."""
    if not summary["recall_positive"] or summary["test_metrics"]["selected"]["metrics"]["recall"] <= 0:
        raise ValueError("Seçilen eşikte test Recall değeri 0. Kabul kriteri 7 sağlanmadı, kullanıcıya sorulmalı.")


def check_fog1_matches_metrics(summary: Summary) -> None:
    fog1, metrics = summary["fog1"], summary["test_metrics"]["selected"]["metrics"]
    expected = metrics["roc_auc"] >= fog1["roc_auc_target"] and metrics["pr_auc"] >= fog1["pr_auc_target"]
    if fog1["met"] != expected:
        raise ValueError("fog1.met değeri test metrikleriyle tutarsız. Özet dosyasını yeniden üretin.")


def check_gain_table_is_non_decreasing(summary: Summary) -> None:
    captured = [row["captured_positives"] for row in summary["gain_table"]]
    if captured != sorted(captured):
        raise ValueError("Kazanç tablosunda daha çok müşteri aranırken yakalanan kabul eden sayısı azalıyor. Sıralama bozuk.")


def check_summary(summary: Summary) -> None:
    check_threshold_matches_selection(summary)
    check_confusion_adds_up(summary)
    check_recall_is_positive(summary)
    check_fog1_matches_metrics(summary)
    check_gain_table_is_non_decreasing(summary)


def describe_fog1(summary: Summary) -> str:
    """FOG-1 sonucu. Hedef karşılanmadıysa hangi ölçütün sağlanmadığı, CV ile fark ve sonraki adım yazılır."""
    fog1 = summary["fog1"]
    metrics = summary["test_metrics"]["selected"]["metrics"]
    cv = summary["selected"]["oof_summary"]
    roc, pr = decimal(metrics["roc_auc"]), decimal(metrics["pr_auc"])
    if fog1["met"]:
        return (
            f"FOG-1 hedefi karşılandı. Test ROC-AUC değeri {roc} ile {decimal(fog1['roc_auc_target'], 2)} hedefinin, PR-AUC değeri {pr} ile "
            f"{decimal(fog1['pr_auc_target'])} hedefinin üzerinde kalmıştır. PR-AUC hedefi, referans modelin kabul oranının iki katı olarak belirlenmişti."
        )
    missed = []
    if not fog1["roc_auc_met"]:
        missed.append(f"ROC-AUC ({roc}, hedef {decimal(fog1['roc_auc_target'], 2)})")
    if not fog1["pr_auc_met"]:
        missed.append(f"PR-AUC ({pr}, hedef {decimal(fog1['pr_auc_target'])})")
    return (
        f"FOG-1 hedefi karşılanmadı. Sağlanamayan ölçüt: {join_names(missed)}. Çapraz doğrulamada ROC-AUC {decimal(cv['roc_auc']['mean'])}, "
        f"PR-AUC {decimal(cv['pr_auc']['mean'])} bulunmuştu. Model, eşik ve hiperparametreler test sonucuna göre değiştirilmedi. "
        "Sonraki adım olarak yeni değişkenlerin ve farklı model ailelerinin denenmesi önerilir ve bu sonuç final raporunda olduğu gibi verilecektir."
    )


def describe_cv_versus_test(label: str, cv: dict[str, float], test: float) -> str:
    """Test değerinin CV ortalamasına göre yönünü ve farkın katmanlar arası standart sapmaya oranını verir."""
    difference = test - cv["mean"]
    direction = "üstünde" if difference > 0 else "altında"
    size = "küçük" if abs(difference) <= cv["std"] else "büyük"
    return (
        f"{label} için test değeri ({decimal(test)}) çapraz doğrulama ortalamasının ({decimal(cv['mean'])}) {direction} kaldı. Aradaki fark "
        f"{decimal(abs(difference), 4)}, katmanlar arası standart sapma ise {decimal(cv['std'], 4)} olduğundan fark standart sapmadan {size}."
    )


def describe_time_based(summary: Summary) -> str:
    """Zamana göre ayrılmış testin rastgele testle yönünü ve eğitim-test kabul oranı farkını verir."""
    time_based, random_roc = summary["time_based"], summary["test_metrics"]["selected"]["metrics"]["roc_auc"]
    direction = "düştü" if time_based["roc_auc"] < random_roc else "yükseldi"
    return (
        f"Zamana göre ayrılmış testte ROC-AUC değeri {decimal(time_based['roc_auc'])} bulundu ve rastgele testteki {decimal(random_roc)} "
        f"değerine göre {direction}. Eğitim kısmındaki kabul oranı {percent(time_based['train_positive_rate'])}, test kısmındaki kabul oranı "
        f"{percent(time_based['test_positive_rate'])} olduğundan iki dönemin yapısı birbirinden farklıdır. PR-AUC değeri kabul oranına "
        "bağlı olduğu için iki ayrımdaki PR-AUC değerleri doğrudan kıyaslanamaz, ROC-AUC kıyaslanabilir."
    )


def top_variables(summary: Summary, count: int = TOP_VARIABLE_COUNT) -> list[str]:
    return [row["feature"] for row in summary["permutation_importance"][:count]]


def describe_time_drop_hint(summary: Summary) -> str:
    """Ay ve gün, önem sıralamasının başındaysa modelin takvime bağlı olabileceği bir olasılık olarak not edilir. Sınanmadığı açıkça yazılır."""
    leading = top_variables(summary, TIME_DROP_RANK_LIMIT)
    present = [name for name in TIME_DROP_CANDIDATES if name in leading]
    if not present or summary["time_based"]["roc_auc"] >= summary["test_metrics"]["selected"]["metrics"]["roc_auc"]:
        return ""
    return (
        f" Modelin en önemli değişkenleri arasında {join_names(present)} bulunduğundan bu düşüşün bir nedeni modelin takvime bağlı örüntülere "
        "güvenmesi olabilir. Bu varsayım bu çalışmada ayrıca sınanmamıştır."
    )


def describe_overlap(summary: Summary) -> str:
    overlap = summary["importance_overlap_top5"]
    permutation_top = join_names(top_variables(summary))
    lr_top = join_names([row["variable"] for row in summary["lr_variable_ranking"][:TOP_VARIABLE_COUNT]])
    if overlap["count"] == 0:
        shared = "İki listenin ilk beş değişkeninde ortak değişken bulunmuyor."
    else:
        shared = f"İki listenin ilk beş değişkeninde {number_word(overlap['count'])} ortak değişken bulunuyor: {join_names(overlap['variables'])}."
    return (
        f"Permutation importance sıralamasında ilk beş değişken {permutation_top}, Lojistik Regresyon'da ise {lr_top} olarak bulundu. {shared}"
    )


def largest_gap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Kaçırılan kabul edenlerdeki payı yakalananlardakinden en çok büyük olan kategori."""
    return max(rows, key=lambda row: row["false_negative_share"] - row["true_positive_share"])


def describe_error_gaps(summary: Summary) -> str:
    sentences = []
    for column in ERROR_TEXT_COLUMNS:
        row = largest_gap(summary["error_profile"][column])
        sentences.append(
            f"{column} değişkeninde en büyük fark {row['category']} kategorisinde görüldü: bu kategorinin payı kaçırılan kabul edenlerde "
            f"{percent(row['false_negative_share'])}, yakalananlarda {percent(row['true_positive_share'])} bulundu."
        )
    return " ".join(sentences)


def group_rows(summary: Summary) -> list[tuple[str, dict[str, Any]]]:
    labels = {"contact": "contact", "period": "dönem"}
    return [(f"{labels[column]} = {row['group']}", row) for column, rows in summary["group_metrics"].items() for row in rows]


def describe_groups(summary: Summary) -> str:
    sentences = []
    for column, rows in summary["group_metrics"].items():
        scored = [row for row in rows if row["pr_auc"] is not None]
        lowest, highest = min(scored, key=lambda row: row["pr_auc"]), max(scored, key=lambda row: row["pr_auc"])
        name = "contact" if column == "contact" else "dönem"
        sentences.append(
            f"{name} bazında PR-AUC en düşük {lowest['group']} grubunda ({decimal(lowest['pr_auc'])}, kabul oranı {percent(lowest['positive_rate'])}), "
            f"en yüksek {highest['group']} grubunda ({decimal(highest['pr_auc'])}, kabul oranı {percent(highest['positive_rate'])}) bulundu."
        )
    return " ".join(sentences)


def describe_literature(summary: Summary) -> str:
    literature, own = summary["literature"], summary["test_metrics"]["selected"]["metrics"]["roc_auc"]
    relation = "yüksek" if own >= literature["moro_2014_roc_auc"] else "düşük"
    return (
        f"Moro ve arkadaşları (2014) en iyi modellerinde {decimal(literature['moro_2014_roc_auc'])} AUC bildirmiştir. Bu çalışmada "
        f"{literature['model_label']} için rastgele ayrılmış test verisinde ROC-AUC {decimal(own)} bulundu, yani sayısal olarak {relation} kaldı. "
        "Ancak Moro ve arkadaşları bank-full.csv'den daha geniş bir veri sürümü kullanmış, özellik seçiminden önce çok daha fazla değişkenle "
        "çalışmış ve değerlendirmeyi zamana göre kayan pencereyle yapmıştır. Bu yüzden iki değer yalnızca gösterge olarak okunmalıdır. "
        f"Kayan pencere zamana göre ayrılmış teste daha yakın bir düzen olduğundan bu çalışmanın o ayrımdaki ROC-AUC değeri ({decimal(summary['time_based']['roc_auc'])}) "
        "de birlikte düşünülmelidir."
    )


def add_introduction(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "Yedinci haftada final aday model, olasılık kalibrasyonu ve karar eşiği belirlenmiş, test verisi o zamana kadar hiçbir "
        "aşamada kullanılmamıştı.",
        f"Sekizinci haftada final model tüm eğitim verisiyle eğitildi ve test verisinde ilk kez, tek seferlik değerlendirildi. "
        "Karmaşıklık matrisi ve kazanç tablosu hazırlandı, zamana göre ayrılmış veride ek bir değerlendirme yapıldı. "
        "Değişken önemleri ve hatalı tahminler incelendi. Model, eşik ve hiperparametrelerde test sonucuna göre değişiklik yapılmadı.",
    ])


def add_goals(report: WeeklyReport, summary: Summary) -> None:
    met = summary["fog1"]["met"]
    roc_goal = (
        "Test verisinde ROC-AUC değeri en az 0,75 olan bir model elde etmek. Bu hedef sekizinci haftada "
        f"{'karşılanmıştır' if met else 'karşılanamamıştır'}."
    )
    interface_goal = "Bu modeli basit bir arayüzle kullanılabilir hâle getirmek. Bu hedef dokuzuncu haftada ele alınacaktır."
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
                      f"değerlendirmek. {roc_goal} {interface_goal}"),
    ])


def add_method(report: WeeklyReport, summary: Summary) -> None:
    split = summary["split"]
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin değerlendirme aşamasında çalışıldı (Wirth ve Hipp, 2000). Final model, yedinci hafta özet dosyasındaki "
        f"yapılandırmayla (model, hiperparametreler, kalibrasyon ve karar eşiği) {format_number(split['train_rows'])} kayıttan oluşan tüm eğitim "
        f"verisinde eğitildi ve {format_number(split['test_rows'])} kayıtlık test verisinde bir kez değerlendirildi. Test sonucuna bakılarak "
        "hiçbir ayar değiştirilmedi.",
        "Test metrikleri seçilen eşikte hesaplandı. Karşılaştırma için 0,5 varsayılan eşiği ve eğitim verisinin out-of-fold tahminlerinde "
        "Recall değerini 0,70'e ulaştıran eşik de verildi. Kazanç tablosunda müşteriler kabul olasılığına göre sıralanıp ilk %10, %20 ve %30'u "
        "arandığında yakalanan kabul eden sayısı ve lift değeri hesaplandı. Lift, aranan müşterilerdeki kabul oranının genel kabul oranına bölümüdür.",
        "Zamana göre ayrılmış değerlendirmede veri dosyasının tarih sırası korundu. İlk %80 satırda eğitim, son %20 satırda test yapıldı. "
        "Bu son %20 kısım rastgele ayrılmış test setiyle bazı kayıtları paylaştığı için sonuç bilgi amaçlıdır.",
        f"Değişken önemi için ham değişkenler üzerinde permutation importance kullanıldı (Fisher vd., 2019). Her değişken test verisinde "
        f"{DEFAULT_REPEATS} kez karıştırıldı ve PR-AUC değerindeki düşüş ölçüldü. Ayrıca Lojistik Regresyon, eğitim verisinde dördüncü hafta "
        "yapılandırmasıyla eğitilip katsayıları incelendi, çünkü yedinci haftada bu model için arama yapılmamıştı. Hatalı tahminler, "
        "kabul ettiği hâlde aranmayacak olarak işaretlenen müşteriler (yanlış negatifler) ile doğru yakalanan müşterilerin karşılaştırılmasıyla incelendi.",
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
    report.add_paragraph(
        "Dokuzuncu haftada final model kaydedilecek, tahmin servisi ve Streamlit arayüzü hazırlanacak ve final raporu tamamlanacaktır."
    )
    report.add_table("Gelecek hafta planı", ["Takım Üyesi", "Planlanan Çalışma"], [list(row) for row in NEXT_WEEK], [4.5, 11.5])


def add_test_results(report: WeeklyReport, summary: Summary) -> None:
    evaluations, cv = summary["test_metrics"], summary["selected"]["oof_summary"]
    metrics = evaluations["selected"]["metrics"]
    report.add_heading("4.1. Final Modelin Test Sonuçları")
    report.add_paragraph(
        f"Final model {summary['literature']['model_label']} olup {'kalibre edilmiş' if summary['selected']['calibrated'] else 'kalibre edilmemiş'} olasılıklarla çalışıyor. Test verisinde "
        f"{format_number(summary['split']['test_rows'])} kayıt bulunuyor ve kabul oranı {percent(summary['split']['test_positive_rate'])}. "
        f"ROC-AUC ve PR-AUC eşikten bağımsızdır, diğer metrikler üç farklı eşikte verilmiştir."
    )
    report.add_table(
        "Test verisinde üç eşikte metrikler",
        ["Eşik türü", "Eşik", "Accuracy", "Precision", "Recall", "F1-Score"],
        [
            [
                THRESHOLD_LABELS[key],
                threshold_text(item["threshold"]),
                *[decimal(item["metrics"][name]) for name in ("accuracy", "precision", "recall", "f1")],
            ]
            for key, item in evaluations.items()
        ],
        [4.8, 1.8, 2.3, 2.4, 2.3, 2.4],
    )
    report.add_table(
        "Çapraz doğrulama ve test sonuçlarının karşılaştırması",
        ["Metrik", "Çapraz doğrulama (ortalama ± std)", "Test"],
        [
            [label, f"{decimal(cv[name]['mean'])} ± {decimal(cv[name]['std'])}", decimal(metrics[name])]
            for label, name in (("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"))
        ],
        [4.0, 7.0, 5.0],
    )
    report.add_paragraph(
        f"{describe_cv_versus_test('ROC-AUC', cv['roc_auc'], metrics['roc_auc'])} "
        f"{describe_cv_versus_test('PR-AUC', cv['pr_auc'], metrics['pr_auc'])}"
    )
    report.add_paragraph(
        f"{describe_fog1(summary)} Seçilen eşikte test Recall değeri {decimal(metrics['recall'])} olup sıfırdan büyüktür. "
        f"Recall hedefli eşikte ({threshold_text(evaluations['target_recall']['threshold'])}) Recall {decimal(evaluations['target_recall']['metrics']['recall'])}, "
        f"Precision {decimal(evaluations['target_recall']['metrics']['precision'])} bulundu."
    )
    report.add_figure(FIGURES_DIR / "h8_test_pr_roc.png", "Test verisinde Precision-Recall ve ROC eğrileri")


def add_confusion(report: WeeklyReport, summary: Summary) -> None:
    counts, selected = summary["confusion"], summary["test_metrics"]["selected"]
    positives = counts["tp"] + counts["fn"]
    report.add_heading("4.2. Karmaşıklık Matrisi")
    report.add_table(
        f"Seçilen eşikte ({threshold_text(selected['threshold'])}) test verisi karmaşıklık matrisi",
        ["Gerçek sınıf", "Kabul etmez olarak tahmin", "Kabul eder olarak tahmin"],
        [
            ["Kabul etmez", format_number(counts["tn"]), format_number(counts["fp"])],
            ["Kabul eder", format_number(counts["fn"]), format_number(counts["tp"])],
        ],
        [4.0, 6.0, 6.0],
    )
    report.add_paragraph(
        f"Testteki {format_number(positives)} kabul eden müşterinin {format_number(counts['tp'])} tanesi yakalandı, {format_number(counts['fn'])} tanesi "
        f"kaçırıldı. Arama listesine alınan müşteri oranı {percent(selected['predicted_positive_share'])} oldu. Bu listedeki "
        f"{format_number(counts['tp'] + counts['fp'])} müşterinin {format_number(counts['tp'])} tanesi gerçekten kabul etti "
        f"(Precision {decimal(selected['metrics']['precision'])}). Kabul etmeyecek {format_number(counts['fp'])} müşteri de gereksiz yere aranmış oldu."
    )
    report.add_figure(FIGURES_DIR / "h8_karmasiklik.png", "Seçilen eşikte test verisi karmaşıklık matrisi (hücrelerde sayı ve gerçek sınıf içindeki pay)")


def add_gain(report: WeeklyReport, summary: Summary) -> None:
    rows = summary["gain_table"]
    first, last = rows[0], rows[-1]
    report.add_heading("4.3. Kazanç Tablosu")
    report.add_table(
        "Müşteriler kabul olasılığına göre sıralanıp ilk dilimler arandığında yakalanan kabul edenler",
        ["Aranan pay", "Aranan müşteri", "Yakalanan kabul eden", "Yakalama oranı", "Lift"],
        [
            [
                format_percent(row["fraction"], 0),
                format_number(row["contacted"]),
                format_number(row["captured_positives"]),
                percent(row["capture_rate"]),
                decimal(row["lift"], 2),
            ]
            for row in rows
        ],
        [2.6, 3.4, 3.8, 3.4, 2.8],
    )
    report.add_paragraph(
        f"Olasılığı en yüksek {format_percent(first['fraction'], 0)} müşteri aranırsa yakalama oranı {percent(first['capture_rate'])} oluyor ve "
        f"bu, rastgele aramaya göre {decimal(first['lift'], 2)} kat verimlidir. Aranan pay {format_percent(last['fraction'], 0)} değerine çıkınca yakalama oranı "
        f"{percent(last['capture_rate'])}, lift ise {decimal(last['lift'], 2)} oluyor. Daha çok müşteri arandıkça yakalama oranı artıyor ama arama başına verim düşüyor."
    )
    report.add_figure(FIGURES_DIR / "h8_kazanc.png", "Kümülatif kazanç eğrisi: aranan müşteri payına göre yakalanan kabul eden payı")


def add_time_based(report: WeeklyReport, summary: Summary) -> None:
    time_based, random_metrics = summary["time_based"], summary["test_metrics"]["selected"]["metrics"]
    report.add_heading("4.4. Zamana Göre Ayrılmış Test")
    report.add_paragraph(
        f"Veri dosyası tarih sırasındadır. Final yapılandırma ilk {format_number(time_based['train_rows'])} satırda eğitildi ve son "
        f"{format_number(time_based['test_rows'])} satırda ölçüldü. Bu son kısımdaki kayıtların {format_number(time_based['overlap_with_random_test'])} tanesi "
        "rastgele ayrılmış test setinde de bulunuyor. Bu yüzden sonuç bilgi amaçlıdır ve bağımsız ikinci bir test sayılmamalıdır."
    )
    report.add_table(
        "Rastgele ve zamana göre ayrılmış testin karşılaştırması",
        ["Ayrım", "ROC-AUC", "PR-AUC", "Test kabul oranı"],
        [
            ["Rastgele ayrım", decimal(random_metrics["roc_auc"]), decimal(random_metrics["pr_auc"]), percent(summary["split"]["test_positive_rate"])],
            ["Zamana göre ayrım", decimal(time_based["roc_auc"]), decimal(time_based["pr_auc"]), percent(time_based["test_positive_rate"])],
        ],
        [5.0, 3.4, 3.4, 4.2],
    )
    report.add_paragraph(f"{describe_time_based(summary)}{describe_time_drop_hint(summary)}")


def add_importance(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("4.5. Değişken Önemleri")
    report.add_paragraph(
        "Permutation importance değerleri final modelin test verisindeki PR-AUC düşüşü olarak ölçüldü. Değer ne kadar büyükse model o değişkene "
        "o kadar çok dayanıyor. Lojistik Regresyon katsayıları ayrı bir modele aittir ve ölçeklenmiş sayısal değişkenler ile One-Hot sütunları için log-odds cinsindendir."
    )
    report.add_table(
        f"Permutation importance, ilk {IMPORTANCE_ROWS} değişken ({DEFAULT_REPEATS} tekrar, ortalama ± standart sapma)",
        ["Sıra", "Değişken", "PR-AUC düşüşü"],
        [
            [str(rank), row["feature"], f"{decimal(row['importance_mean'], 4)} ± {decimal(row['importance_std'], 4)}"]
            for rank, row in enumerate(summary["permutation_importance"][:IMPORTANCE_ROWS], start=1)
        ],
        [1.8, 6.0, 8.2],
    )
    report.add_figure(FIGURES_DIR / "h8_permutation_importance.png", "Permutation importance, ilk 10 değişken")
    report.add_table(
        f"Lojistik Regresyon, mutlak değere göre ilk {COEFFICIENT_ROWS} katsayı",
        ["Sıra", "Özellik", "Katsayı"],
        [
            [str(rank), row["feature"].split("__", 1)[-1], decimal(row["coefficient"], 3)]
            for rank, row in enumerate(summary["lr_coefficients"][:COEFFICIENT_ROWS], start=1)
        ],
        [1.8, 8.0, 6.2],
    )
    report.add_figure(FIGURES_DIR / "h8_lr_katsayilari.png", "Lojistik Regresyon katsayıları (turuncu: kabul olasılığını artırır, mavi: azaltır)")
    report.add_paragraph(
        f"{describe_overlap(summary)} Lojistik Regresyon sıralaması, özelliklerin ham değişkene göre en büyük mutlak katsayısı alınarak "
        "permutation importance ile aynı değişken adlarına çevrilmiştir."
    )


def add_error_analysis(report: WeeklyReport, summary: Summary) -> None:
    counts = summary["confusion"]
    rows = [
        [column, row["category"], percent(row["false_negative_share"]), percent(row["true_positive_share"])]
        for column in ERROR_TABLE_COLUMNS
        for row in summary["error_profile"][column]
    ]
    report.add_heading("4.6. Hata Analizi")
    report.add_paragraph(
        f"Seçilen eşikte {format_number(counts['fn'])} kabul eden müşteri kaçırıldı ve {format_number(counts['tp'])} kabul eden yakalandı. "
        "Aşağıdaki tabloda bu iki grubun seçili değişkenlerdeki dağılımı verilmiştir. Paylar her grubun kendi toplamına göre hesaplandı."
    )
    report.add_table(
        "Kaçırılan ve yakalanan kabul edenlerin değişken kategorilerine göre dağılımı",
        ["Değişken", "Kategori", "Kaçırılanlarda pay", "Yakalananlarda pay"],
        rows,
        [4.2, 3.6, 4.1, 4.1],
    )
    report.add_paragraph(describe_error_gaps(summary))
    report.add_table(
        "Test verisinde grup bazlı kabul oranı ve PR-AUC",
        ["Grup", "Kayıt sayısı", "Kabul oranı", "PR-AUC"],
        [
            [name, format_number(row["count"]), percent(row["positive_rate"]), NO_VALUE if row["pr_auc"] is None else decimal(row["pr_auc"])]
            for name, row in group_rows(summary)
        ],
        [6.0, 3.4, 3.4, 3.2],
    )
    report.add_paragraph(describe_groups(summary))


def add_literature(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("4.7. Literatürle Karşılaştırma")
    report.add_paragraph(describe_literature(summary))


def build_report(summary: Summary, captain_id: str) -> WeeklyReport:
    check_summary(summary)
    report = WeeklyReport(week=WEEK, captain_id=captain_id)
    # Ekip süreci işaretleri taslaktır, kaptan teslimden önce gerçek duruma göre değiştirmelidir.
    report.add_captain_form(
        team_process=[(criterion, True) for criterion in TEAM_PROCESS_CRITERIA],
        progress=[(criterion, True) for criterion in PROGRESS_CRITERIA],
    )
    report.add_title_block()
    add_introduction(report, summary)
    add_goals(report, summary)
    add_method(report, summary)
    add_project_management(report)
    report.add_heading("4. TEORİ VE SONUÇLAR")
    report.add_box([
        "Müşterilerin durumları, sosyoekonomik özellikleri ve geçmiş bankacılık alışkanlıkları, pazarlama kampanyalarına verecekleri "
        "yanıtı doğrudan etkilemektedir. Bu hafta bu ilişkiyi yakalayan final modelin hiç görmediği veride ne kadar başarılı olduğu ve "
        "hangi değişkenlere dayandığı incelendi. Sonuçlar aşağıdaki alt başlıklarda verilmiştir.",
    ])
    add_test_results(report, summary)
    add_confusion(report, summary)
    add_gain(report, summary)
    add_time_based(report, summary)
    add_importance(report, summary)
    add_error_analysis(report, summary)
    add_literature(report, summary)
    report.add_links()
    report.add_references(REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="8. hafta ilerleme raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    output_path = build_report(summary, args.kaptan).save(OUTPUT_DIR / report_file_name(WEEK))
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
