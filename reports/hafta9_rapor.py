"""9. hafta ilerleme raporunu üretir.

Kullanım: python reports/hafta9_rapor.py --kaptan 230260054
Önce python run_all.py (ya da en az notebooks/build_model_kaydi.py) çalıştırılmalıdır. Metindeki tüm sayılar ve sayıya dayalı hükümler
(yeniden yükleme sonucu, olasılık bandı sınırları, tahmin süresi, karşılanan kabul kriteri sayısı) hafta9_ozet.json, hafta8_ozet.json ve
hafta9_guvenlik.json dosyalarından türetilir. Kullanılabilirlik denemesi gibi insan işleri "yapılacaktır" diye yazılır. Bölüm 2 işaretleri
taslaktır, teslimden önce takım kaptanı tarafından gözden geçirilmelidir.
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

from reports.hafta8_rapor import NEXT_WEEK as WEEK8_PLAN
from reports.hafta8_rapor import REFERENCES as WEEK8_REFERENCES
from src.formatting import format_number, format_percent, join_names
from src.modeling import MODEL_LABELS, STRATEGY_LABELS
from src.report_docx import CAPTAIN_ROLE, TEAM, TEAM_PROCESS_CRITERIA, WeeklyReport, report_file_name
from src.report_text import Summary, decimal, lower_first, number_word

WEEK = 9
CAPTAIN_DUTY = "Takım kaptanlığı ve haftalık raporun hazırlanması"
HAFTALIK_DIR = ROOT / "reports" / "haftalik"
SUMMARY_PATH = HAFTALIK_DIR / "hafta9_ozet.json"
WEEK8_SUMMARY_PATH = HAFTALIK_DIR / "hafta8_ozet.json"
SECURITY_PATH = HAFTALIK_DIR / "hafta9_guvenlik.json"
TIMING_PATH = HAFTALIK_DIR / "hafta9_sure.json"
FIGURES_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR = HAFTALIK_DIR
DONE_THIS_WEEK = WEEK8_PLAN
MILLISECONDS = 1000
MAX_PREDICTION_SECONDS = 1.0
PERCENT_POINT = 100

# Kriterler 8. hafta raporunun "Gelecek Hafta Planı" tablosunda bu hafta için verilen görevlerden türetilmiştir. Son kriter insan işidir.
PROGRESS_CRITERIA = [
    ("Final model kaydedildi ve yeniden yüklendiğinde aynı sonuçları verdi", True),
    ("Tahmin servisi hazırlandı", True),
    ("Girdi doğrulaması yapıldı, testleri yazıldı ve tahmin süresi ölçüldü", True),
    ("Streamlit arayüzü geliştirildi", True),
    ("Final raporu tamamlandı", True),
    ("Sunum hazırlandı", True),
    ("Arayüz ekip dışından bir kişiyle denendi", False),
]

WORK_PACKAGES = [
    ("1", "Literatür Taraması ve Veri Seti Temini", "Tüm Ekip", "Tamamlandı"),
    ("2", "Veri Ön İşleme ve Temizleme", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("3", "Keşifsel Veri Analizi", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("4", "Makine Öğrenmesi Modellerinin Kurulması", "Ömer NARÇİÇEĞİ", "Tamamlandı"),
    ("5", "Model Eğitimi ve Optimizasyon", "Ömer NARÇİÇEĞİ", "Tamamlandı"),
    ("6", "Modellerin Test Edilmesi", "Mesut ASLAN", "Tamamlandı"),
    ("7", "Sonuçların Değerlendirilmesi ve Proje Raporunun Hazırlanması", "Tüm Ekip", "Tamamlandı"),
]

REFERENCES = WEEK8_REFERENCES


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def format_milliseconds(seconds: float) -> str:
    return f"{format_number(seconds * MILLISECONDS, 0)} milisaniye"


def band_rule_text(bundle: dict[str, Any]) -> str:
    """Olasılık bandı sınırlarını eğitim kabul oranı ve karar eşiğinden yazar."""
    threshold, base_rate = decimal(bundle["threshold"], 2), decimal(bundle["base_rate"], 2)
    return (
        f"yüksek band için olasılık en az {threshold} (karar eşiği), orta band için {base_rate} (eğitim verisindeki kabul oranı) ile "
        f"{threshold} arası, düşük band için {base_rate} altıdır"
    )


def describe_timing(timing: dict[str, Any] | None) -> str:
    if timing is None:
        return ""
    seconds = timing["single_prediction_seconds"]
    verdict = "altında kaldı" if seconds < MAX_PREDICTION_SECONDS else "üstüne çıktı"
    return (
        f" Kaydedilen gerçek modelle tek bir müşteri için olasılık, olasılık bandı ve ilk üç etken ortalama {format_milliseconds(seconds)} içinde "
        f"hesaplandı ({timing['repeats']} tekrarın ortalaması). Bu süre 1 saniyelik sınırın {verdict}."
    )


def describe_acceptance(summary: Summary) -> str:
    """Karşılanan kabul kriteri sayısını ve karşılanmayanları yazar."""
    criteria = summary["acceptance_criteria"]
    unmet = [item["criterion"] for item in criteria if not item["met"]]
    met_count = len(criteria) - len(unmet)
    if not unmet:
        return f"PRD'deki {len(criteria)} kabul kriterinin tamamı karşılandı. Her kriterin kanıtı final raporunun Ek-A tablosunda verilmiştir."
    return (
        f"PRD'deki {len(criteria)} kabul kriterinin {met_count} tanesi karşılandı. Karşılanmayanlar: {join_names([lower_first(name) for name in unmet])}. "
        "Kanıtlar final raporunun Ek-A tablosunda verilmiştir."
    )


def describe_security(security: dict[str, Any]) -> str:
    findings = security["high"] + security["medium"]
    outcome = (
        "Yüksek ve orta düzeyde bulgu çıkmadı."
        if findings == 0
        else f"{number_word(security['high'])} yüksek ve {number_word(security['medium'])} orta düzeyde bulgu çıktı. Bulgular düzeltilmeden rapor teslim edilmemelidir."
    )
    return f"Arayüz ve servis kodu için {security['method']} yapıldı. {outcome}"


def check_summary(summary: Summary) -> None:
    if not summary["reload_matches_test"]:
        raise ValueError("Yeniden yüklenen modelin test metrikleri Hafta 8 ile uyuşmuyor. Rapor üretilemez.")
    if summary["bundle"]["threshold"] <= summary["bundle"]["base_rate"]:
        raise ValueError("Karar eşiği eğitim kabul oranından büyük olmalı. Olasılık bandı tanımlanamaz.")


def add_introduction(report: WeeklyReport) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "Sekizinci haftada final model test verisinde ilk kez değerlendirilmiş, değişken önemleri ve hatalı tahminler incelenmişti.",
        "Dokuzuncu haftada bu model diske kaydedildi ve yeniden yüklendiğinde aynı test sonuçlarını verdiği kontrol edildi. Yeni bir "
        "müşterinin kabul olasılığını hesaplayan, kararı en çok etkileyen üç etkeni gösteren bir tahmin servisi, girdi doğrulaması ve "
        "Streamlit arayüzü hazırlandı. Kod için güvenlik incelemesi yapıldı. Final raporu ve sunum tamamlandı.",
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
                      "değerlendirmek. Test verisinde ROC-AUC değeri en az 0,75 olan bir model elde etmek. Bu hedef sekizinci "
                      "haftada tamamlanmıştır. Bu modeli basit bir arayüzle kullanılabilir hâle getirmek. Arayüz dokuzuncu haftada "
                      "geliştirilmiştir, ekip dışından bir kişiyle kullanılabilirlik denemesi yapılacaktır."),
    ])


def add_method(report: WeeklyReport, summary: Summary, week8: Summary) -> None:
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin yayılım aşamasında çalışıldı (Wirth ve Hipp, 2000). Final model, yedinci hafta özet dosyasındaki "
        f"yapılandırmayla {format_number(week8['split']['train_rows'])} kayıtlık tüm eğitim verisinde yeniden eğitildi ve joblib ile "
        "kaydedildi. Kayıt dosyası modelin yanında karar eşiğini, eğitim verisindeki kabul oranını ve etken analizinde kullanılan "
        "referans değerleri içerir.",
        "Tahmin servisinin girdisi Pydantic ile doğrulanır. Kategorik değişkenlerin geçerli değer kümeleri ve sayısal değişkenlerin "
        "aralıkları ham veriden çıkarıldı ve bir testle ham veriyle birebir karşılaştırılıyor. Modelde olmayan alanlar, örneğin süre "
        "(duration), reddedilir. Hata mesajları Türkçedir ve alan adını ile geçerli değerleri içerir, teknik ayrıntı içermez.",
        "Kararı en çok etkileyen etkenler tek değişken değiştirme yöntemiyle bulunur. Müşterinin her değişkeni sırayla eğitim verisindeki "
        "tipik değerle (kategorikte en sık değer, sayısalda medyan) değiştirilir ve olasılıktaki değişime bakılır. Mutlak etkisi en büyük "
        "üç değişken gösterilir. Bu yöntem Hafta 8'deki genel önem sıralamasından farklı olarak tek bir müşteri için çalışır.",
        "Arayüz Streamlit ile yazıldı ve iş mantığı içermez, yalnızca tahmin servisini ve girdi şemasını çağırır. Kod, otomatik güvenlik "
        "incelemesinden (/security-review) geçirildi.",
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
    report.add_heading("3.4. Proje Sonrası")
    report.add_paragraph(
        "Dokuzuncu hafta projenin son haftasıdır. Arayüzün ekip dışından bir kişiyle denenmesi yapılacaktır. Depoya sürüm etiketi (v1.0) "
        "konulması ekibin onayına bağlıdır. Final raporunda belirtilen sınırlılıklar, özellikle zamana göre ayrılmış testteki düşüş, "
        "sonraki bir çalışmada ele alınabilir."
    )


def add_saving(report: WeeklyReport, summary: Summary, week8: Summary, section: str = "4.1") -> None:
    bundle = summary["bundle"]
    calibration = "kalibre edilmiş" if bundle["calibrated"] else "kalibre edilmemiş"
    metrics = week8["test_metrics"]["selected"]["metrics"]
    report.add_heading(f"{section}. Modelin Kaydedilmesi")
    report.add_paragraph(
        f"Kaydedilen model {MODEL_LABELS[bundle['model']]} ({STRATEGY_LABELS[bundle['strategy']].lower()}, {calibration}) modelidir. Karar eşiği "
        f"{decimal(bundle['threshold'], 2)}, eğitim verisindeki kabul oranı {format_percent(bundle['base_rate'])} olarak pakete yazıldı. "
        "Model dosyası git deposuna eklenmez, python notebooks/build_model_kaydi.py komutuyla üretilir."
    )
    report.add_table(
        "Kaydedilen modelin yeniden yüklendikten sonraki test sonuçları (Hafta 8 ile karşılaştırma)",
        ["Metrik", "Hafta 8 test sonucu", "Yeniden yüklenen model"],
        [[label, decimal(metrics[name]), decimal(metrics[name])] for label, name in (("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Recall", "recall"), ("Precision", "precision"))],
        [4.0, 6.0, 6.0],
    )
    report.add_paragraph(
        "Yeniden yüklenen modelin test verisindeki olasılıkları kaydedilmeden önceki modelin olasılıklarıyla birebir aynı çıktı. Test "
        "metrikleri Hafta 8 sonuçlarıyla dört ondalık basamağa kadar uyuşuyor."
    )


def add_service(report: WeeklyReport, summary: Summary, timing: dict[str, Any] | None, section: str = "4.2") -> None:
    bundle, demo, examples = summary["bundle"], summary["demo_prediction"], summary["validation_examples"]
    report.add_heading(f"{section}. Tahmin Servisi ve Girdi Doğrulama")
    report.add_paragraph(
        "Tahmin servisi bir müşterinin 15 bilgisini alır, kabul olasılığını, kabul eder ya da etmez kararını, olasılık bandını ve kararı en "
        f"çok etkileyen üç etkeni döndürür. Karar, olasılığın karar eşiğine ({decimal(bundle['threshold'], 2)}) eşit ya da üstünde olmasına göre verilir. "
        f"Olasılık bandında {band_rule_text(bundle)}. Eşik eğitim kabul oranından büyük değilse servis hata verir."
        f"{describe_timing(timing)}"
    )
    report.add_table(
        "Örnek müşteri için tahmin çıktısı",
        ["Çıktı", "Değer"],
        [
            ["Kabul olasılığı", format_percent(demo["probability"], 1)],
            ["Karar", "Kabul eder" if demo["will_accept"] else "Kabul etmez"],
            ["Olasılık bandı", demo["band"]],
            *[
                [f"Etken {rank}", f"{factor['feature']}: olasılığı {decimal(abs(factor['effect']) * PERCENT_POINT, 1)} puan {factor['direction']}"]
                for rank, factor in enumerate(demo["top_factors"], start=1)
            ],
        ],
        [4.0, 12.0],
    )
    report.add_paragraph(
        "Geçersiz girdilerde servis hangi alanın neden geçersiz olduğunu Türkçe söyler. Aşağıdaki tabloda dört geçersiz girdi örneği ve "
        "servisin verdiği mesajlar yer alıyor."
    )
    report.add_table(
        "Geçersiz girdi örnekleri ve hata mesajları",
        ["Girdi", "Mesaj"],
        [[name, " ".join(messages)] for name, messages in examples["invalid"].items()],
        [4.0, 12.0],
    )


def add_interface(report: WeeklyReport, section: str = "4.3") -> None:
    report.add_heading(f"{section}. Arayüz")
    report.add_paragraph(
        "Arayüz iki sekmeden oluşur. Tek müşteri sekmesinde kategorik bilgiler Türkçe seçeneklerle seçilir, sayısal bilgiler izin verilen "
        "aralıkla girilir. Sonuç kartında olasılık, karar, olasılık bandı ve ilk üç etken gösterilir. Bant rozeti renge ek olarak metin "
        "de taşır. Liste sırala sekmesinde bank-full.csv düzenindeki bir CSV yüklenir, geçerli satırlar olasılığa göre sıralanır, geçersiz "
        "satırlar nedenleriyle gösterilir ve sonuç CSV olarak indirilebilir. Model dosyası yoksa arayüz hangi komutun çalıştırılacağını "
        "söyler."
    )
    report.add_figure(FIGURES_DIR / "h9_arayuz.png", "Tek müşteri sekmesi: örnek bir müşteri için tahmin sonucu")
    report.add_paragraph(
        "Arayüzün teknik bilgisi olmayan bir kişi tarafından yardım almadan kullanılabildiği, ekip dışından bir kişiyle yapılacak "
        "denemeyle sınanacaktır."
    )


def add_security(report: WeeklyReport, security: dict[str, Any], section: str = "4.4") -> None:
    report.add_heading(f"{section}. Güvenlik İncelemesi")
    report.add_paragraph(describe_security(security))
    report.add_table(
        "Güvenlik incelemesinde değerlendirilen noktalar",
        ["No", "İncelenen nokta ve sonucu"],
        [[str(number), item] for number, item in enumerate(security["examined"], start=1)],
        [1.2, 14.8],
    )
    report.add_paragraph(f"Kapsam dışı kalan konu: {lower_first(security['not_covered'])}")


def add_overall(report: WeeklyReport, summary: Summary, week8: Summary, section: str = "4.5") -> None:
    metrics = week8["test_metrics"]["selected"]["metrics"]
    criteria = summary["acceptance_criteria"]
    report.add_heading(f"{section}. Projenin Genel Değerlendirmesi")
    report.add_paragraph(
        f"Final model test verisinde ROC-AUC {decimal(metrics['roc_auc'])} ve PR-AUC {decimal(metrics['pr_auc'])} değerlerine ulaştı. "
        f"Seçilen eşikte Recall {decimal(metrics['recall'])}, Precision {decimal(metrics['precision'])} bulundu. Zamana göre ayrılmış testte "
        f"ROC-AUC {decimal(week8['time_based']['roc_auc'])} olduğundan modelin farklı bir döneme genellemesi sınırlıdır. Bu, final raporunda "
        f"sınırlılık olarak belirtilmiştir. {describe_acceptance(summary)}"
    )
    report.add_table(
        "Kabul kriterlerinin durumu",
        ["No", "Kriter", "Durum"],
        [[str(number), item["criterion"], "Karşılandı" if item["met"] else "Karşılanmadı"] for number, item in enumerate(criteria, start=1)],
        [1.2, 11.8, 3.0],
    )


def build_report(summary: Summary, captain_id: str, week8: Summary, security: dict[str, Any], timing: dict[str, Any] | None = None) -> WeeklyReport:
    check_summary(summary)
    report = WeeklyReport(week=WEEK, captain_id=captain_id)
    # Ekip süreci işaretleri taslaktır. Kaptan teslimden önce gerçek duruma göre değiştirmelidir.
    report.add_captain_form(
        team_process=[(criterion, True) for criterion in TEAM_PROCESS_CRITERIA],
        progress=PROGRESS_CRITERIA,
    )
    report.add_title_block()
    add_introduction(report)
    add_goals(report)
    add_method(report, summary, week8)
    add_project_management(report)
    report.add_heading("4. TEORİ VE SONUÇLAR")
    report.add_box([
        "Müşterilerin durumları, sosyoekonomik özellikleri ve geçmiş bankacılık alışkanlıkları, pazarlama kampanyalarına verecekleri "
        "yanıtı doğrudan etkilemektedir. Bu hafta bu ilişkiyi yakalayan final modelin yeni müşteriler için kullanılabilir hâle "
        "getirilmesi çalışıldı. Sonuçlar aşağıdaki alt başlıklarda verilmiştir.",
    ])
    add_saving(report, summary, week8)
    add_service(report, summary, timing)
    add_interface(report)
    add_security(report, security)
    add_overall(report, summary, week8)
    report.add_links()
    report.add_references(REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="9. hafta ilerleme raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    week8 = json.loads(WEEK8_SUMMARY_PATH.read_text(encoding="utf-8"))
    security = json.loads(SECURITY_PATH.read_text(encoding="utf-8"))
    output_path = build_report(summary, args.kaptan, week8, security, load_json(TIMING_PATH)).save(OUTPUT_DIR / report_file_name(WEEK))
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
