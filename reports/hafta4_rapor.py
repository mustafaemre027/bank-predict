"""4. hafta ilerleme raporunu üretir.

Kullanım: python reports/hafta4_rapor.py --kaptan 230260054
Önce notebooks/build_ilk_model.py çalıştırılmalıdır. Metindeki tüm sayılar ve sayıya dayalı hükümler (iyileşme/kötüleşme,
katman sayıları) hafta4_ozet.json dosyasından türetilir; yalnızca yorum cümleleri bu haftanın bulgularına göre yazılmıştır.
Bölüm 2 (ekip içi çalışma süreci) işaretleri taslaktır; teslimden önce takım kaptanı tarafından gözden geçirilmelidir.
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

from src.formatting import format_number, format_percent, join_names
from src.report_docx import (
    CAPTAIN_ROLE,
    TEAM,
    TEAM_PROCESS_CRITERIA,
    WeeklyReport,
    report_file_name,
)

Summary = dict[str, Any]

WEEK = 4
CAPTAIN_DUTY = "Takım kaptanlığı ve haftalık raporun hazırlanması"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta4_ozet.json"
FIGURES_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR = ROOT / "reports" / "haftalik"
# Hedef (FOG-1) test setinde ölçülecek; bu hafta yalnızca CV ortalamasının hedefin üstünde olup olmadığı belirtilir.
TARGET_ROC_AUC = 0.75
# Standart sapma ortalamanın %5'inden azsa sonuçların katmandan katmana az değiştiği söylenir.
SMALL_RELATIVE_SPREAD = 0.05

# Rakamın okunuşuna göre bulunma eki: 1'inde, 2'sinde, 3'ünde, 4'ünde, 5'inde, 6'sında, 7'sinde, 8'inde, 9'unda, 0'ında.
LOCATIVE_SUFFIXES = {0: "ında", 1: "inde", 2: "sinde", 3: "ünde", 4: "ünde", 5: "inde", 6: "sında", 7: "sinde", 8: "inde", 9: "unda"}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

# Kriterler 3. hafta raporunun "Gelecek Hafta Planı" tablosunda bu hafta için verilen görevlerden türetilmiştir.
PROGRESS_CRITERIA = [
    "Ön işleme pipeline'ı kuruldu",
    "Veri sızıntısını kontrol eden testler yazıldı",
    "5 katlı katmanlı çapraz doğrulama düzeni kuruldu",
    "İlk Lojistik Regresyon modeli eğitildi",
    "Model referans modelle karşılaştırıldı",
    "Çapraz doğrulama sonuç tablosu ile PR ve ROC eğrileri hazırlandı",
]

WORK_PACKAGES = [
    ("1", "Literatür Taraması ve Veri Seti Temini", "Tüm Ekip", "Tamamlandı"),
    ("2", "Veri Ön İşleme ve Temizleme", "Mustafa Emre KAPLAN", "Lojistik Regresyon için tamamlandı"),
    ("3", "Keşifsel Veri Analizi", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("4", "Makine Öğrenmesi Modellerinin Kurulması", "Ömer NARÇİÇEĞİ", "Devam ediyor"),
    ("5", "Model Eğitimi ve Optimizasyon", "Ömer NARÇİÇEĞİ", "Başlamadı"),
    ("6", "Modellerin Test Edilmesi", "Mesut ASLAN", "Başladı"),
    ("7", "Sonuçların Değerlendirilmesi ve Proje Raporunun Hazırlanması", "Tüm Ekip", "Devam ediyor"),
]

DONE_THIS_WEEK = [
    ("Ömer NARÇİÇEĞİ", "5 katlı katmanlı çapraz doğrulama düzeninin kurulması, ilk Lojistik Regresyon modelinin eğitilip referans "
     "modelle karşılaştırılması, yaş grubu özelliği ve az gözlemli ayların birleştirilmesi için karşılaştırma deneyleri"),
    ("Mustafa Emre KAPLAN", "Ön işleme adımlarının kodlanması (was_contacted_before, yaş grubu, pdays dönüşümü, One-Hot kodlama, log "
     "dönüşümü ve ölçekleme), katmanlı eğitim/test bölme fonksiyonu ve veri sızıntısını kontrol eden testler"),
    ("Mesut ASLAN", "Çapraz doğrulama sonuçlarını ortalama ± standart sapma tablosuna dönüştüren ve iki modeli katman katman "
     "karşılaştıran değerlendirme fonksiyonları, PR ve ROC eğrisi grafikleri ve bunların testleri"),
]

NEXT_WEEK = [
    ("Ömer NARÇİÇEĞİ", "Rastgele Orman ve XGBoost modellerinin ağaç modelleri için ölçeklemesiz bir ön işleme ile kurulması ve aynı "
     "çapraz doğrulama düzeninde eğitilmesi"),
    ("Mesut ASLAN", "Üç modelin karşılaştırma tablosu ve grafikleri, aynı tohumla sonuçların tekrar üretilebildiğini kontrol eden test"),
    ("Mustafa Emre KAPLAN", "Ön işleme ablasyonu: day, yaş grubu ve log dönüşümü çıkarıldığında skorların nasıl değiştiğinin incelenmesi"),
]

PREPROCESSING_STEPS = [
    ["duration", "Görüşme bitmeden bilinemediği için bölme sırasında veriden çıkarıldı."],
    ["pdays", "Daha önce aranmamış müşterilerde -1 olan değer 0 yapıldı. Müşterinin daha önce aranıp aranmadığı was_contacted_before "
     "(0/1) adlı yeni bir değişkenle gösterildi."],
    ["age", "Sayısal değer standartlaştırıldı. Ayrıca 6 aralıklı yaş grubu özelliği One-Hot ile eklendi."],
    ["balance", "İşareti koruyan log dönüşümü (sign(x)·log(1+|x|)) ve standartlaştırma."],
    ["campaign, previous", "log(1+x) dönüşümü ve standartlaştırma."],
    ["day, pdays", "Standartlaştırma."],
    ["default, housing, loan", "yes = 1, no = 0."],
    ["job, marital, education, contact, month, poutcome", "One-Hot Encoding. unknown ayrı bir kategori olarak bırakıldı. Eğitim verisinde "
     "olmayan bir kategoriyle karşılaşıldığında hata verilmez, o değişkenin bütün sütunlarına 0 yazılır."],
]

LEAKAGE_TESTS = [
    ["Bölmeden sonra duration ve y değişkenleri girdiler arasında yok", "Görüşme süresinin ve hedefin modele girmediğini"],
    ["Eğitim ve test verisinde ortak kayıt yok", "Aynı müşterinin iki tarafta birden bulunmadığını"],
    ["Ölçekleme değerleri yalnızca eğitim verisinden hesaplanıyor", "Başka bir veri dönüştürüldüğünde ortalamanın değişmediğini"],
    ["Bir kaydın dönüşümü yanında gelen kayıtlara bağlı değil", "Ön işleme adımlarının yeni veriyle yeniden hesaplanmadığını"],
    ["Çapraz doğrulamada model yalnızca o katmanın eğitim kayıtlarıyla eğitiliyor", "Doğrulama katmanındaki kayıtların eğitimde kullanılmadığını"],
    ["Bölme ve çapraz doğrulama her çalıştırmada aynı sonucu veriyor", "random_state = 42 ile sonuçların tekrar elde edilebildiğini"],
]

REFERENCES = [
    "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. "
    "Decision Support Systems, 62, 22-31.",
    "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing veri seti. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. Proceedings of the 4th "
    "International Conference on the Practical Applications of Knowledge Discovery and Data Mining, 29-39.",
    "Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.",
    "Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. Proceedings of "
    "the 14th International Joint Conference on Artificial Intelligence, 1137-1143.",
    "Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. "
    "ACM Transactions on Knowledge Discovery from Data, 6(4), 1-21.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary "
    "classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
]


def decimal(value: float, digits: int = 3) -> str:
    return format_number(value, digits)


def mean_std(summary: Summary, model: str, metric: str) -> str:
    values = summary["models"][model]["summary"][metric]
    return f"{decimal(values['mean'])} ± {decimal(values['std'])}"


def mean_of(summary: Summary, model: str, metric: str) -> float:
    return summary["models"][model]["summary"][metric]["mean"]


def fold_count_text(improved: int, total: int) -> str:
    """"5 katmanın 4'ünde" gibi, sayının okunuşuna uygun bulunma ekiyle katman ifadesi."""
    if improved == total:
        return "katmanların hepsinde"
    return f"{total} katmanın {improved}'{LOCATIVE_SUFFIXES[improved % 10]}"


def signed(value: float) -> str:
    return f"+{decimal(value, 4)}" if value > 0 else decimal(value, 4)


def is_within_noise(comparison: dict[str, float]) -> bool:
    return abs(comparison["mean_difference"]) < comparison["std_difference"]


def describe_comparison(comparison: dict[str, float], change: str) -> str:
    """Bir karşılaştırmanın sonucunu yön, katman sayısı ve gürültüye göre büyüklüğüyle birlikte cümleye döker."""
    difference = comparison["mean_difference"]
    improved = comparison["folds_improved"]
    folds = "hiçbir katmanda yükselmedi" if improved == 0 else f"{fold_count_text(improved, comparison['fold_count'])} yükseldi"
    if difference == 0:
        return f"{change} PR-AUC ortalaması değişmedi ve {folds}."
    direction = "arttı" if difference > 0 else "azaldı"
    spread = decimal(comparison["std_difference"], 4)
    is_small = is_within_noise(comparison)
    # "Ancak" yalnızca küçük bir iyileşmeyi sınırlarken anlamlı; düşüşten sonra gereksiz bir karşıtlık kurar.
    lead = "Ancak bu fark" if is_small and difference > 0 else "Bu fark"
    size = f"{lead}, katmanlardaki farkların standart sapmasından ({spread}) {'küçük' if is_small else 'büyük'}."
    return f"{change} PR-AUC ortalamada {decimal(abs(difference), 4)} {direction} ve {folds}. {size}"


def check_decisions_match_data(summary: Summary) -> None:
    """Rapordaki kararlar model varsayılanlarına göre yazılıdır; veri tersini gösterirse rapor sessizce yanlış karar anlatmasın."""
    comparisons = summary["comparisons"]
    if comparisons["age_group_added"]["mean_difference"] <= 0:
        raise ValueError("Yaş grubu PR-AUC'yi iyileştirmiyor ama build_logistic_regression varsayılan olarak onu kullanıyor. "
                         "Kararı, model varsayılanını ve rapor metnini birlikte gözden geçirin.")
    if comparisons["rare_months_merged"]["mean_difference"] > 0:
        raise ValueError("Ay birleştirme PR-AUC'yi iyileştiriyor ama model varsayılanı ayları birleştirmiyor. "
                         "Kararı, model varsayılanını ve rapor metnini birlikte gözden geçirin.")


def describe_fold_spread(summary: Summary, model: str) -> str:
    values = summary["models"][model]["summary"]
    is_stable = all(values[metric]["std"] < SMALL_RELATIVE_SPREAD * values[metric]["mean"] for metric in ("pr_auc", "roc_auc"))
    if is_stable:
        return "Standart sapmaların küçük olması, sonuçların katmandan katmana fazla değişmediğini göstermektedir."
    return "Standart sapmalar ise sonuçların katmandan katmana belirli ölçüde değiştiğini göstermektedir."


def add_introduction(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "Üçüncü haftada keşifsel veri analizi yapılmış, değişkenlerle hedef değişken arasındaki ilişkiler istatistiksel testlerle "
        "incelenmiş ve ön işleme adımları için bir karar taslağı hazırlanmıştı.",
        "Dördüncü haftada bu taslaktaki kararlar koda döküldü. Veri seti eğitim ve test verisi olarak ayrıldı. Test verisi son "
        "değerlendirmeye kadar kullanılmayacağı için bu haftaki bütün çalışmalar eğitim verisi üzerinde yapıldı. Eğitim verisinde "
        "5 katlı çapraz doğrulama kuruldu ve ilk model olarak Lojistik Regresyon eğitildi. Bu model ikinci haftada oluşturduğumuz "
        "referans modelle karşılaştırıldı. Ayrıca geçen haftadan açık kalan iki soruya, yaş grubu özelliğinin modele bir katkısı "
        "olup olmadığına ve az arama yapılan ayların birleştirilmesinin gerekip gerekmediğine bakıldı.",
    ])


def add_goals(report: WeeklyReport) -> None:
    report.add_heading("1.1. Amaç ve Hedefler")
    report.add_labeled_box([
        ("Amaç:", "Bankanın hedef kitlesini doğru analiz ederek pazarlama kampanyalarının başarı oranını artıracak ve maliyetleri "
                  "düşürecek ölçülebilir bir makine öğrenmesi tahmin modeli ortaya koymaktır."),
        ("Hedefler:", "Bank Marketing veri setini temin etmek ve doğrulamak. Bu hedef ikinci haftada tamamlanmıştır. Değişkenlerin "
                      "hedefle ilişkisini keşifsel analiz ve istatistiksel testlerle incelemek. Bu hedef üçüncü haftada "
                      "tamamlanmıştır. Bulgulara göre ön işleme adımlarını kurmak. Bu hedefin Lojistik Regresyon için gereken kısmı dördüncü "
                      "haftada tamamlanmıştır. "
                      "Lojistik Regresyon, Rastgele Orman ve XGBoost modellerini çapraz doğrulama ile eğitmek ve sınıf dengesizliği "
                      "için farklı yöntemleri karşılaştırmak. Veri setindeki sınıflar dengesiz olduğu için modelleri yalnızca "
                      "Accuracy ile değil; Precision, Recall, F1-Score, ROC-AUC ve PR-AUC ile değerlendirmek. Test verisinde ROC-AUC "
                      "değeri en az 0,75 olan bir model elde etmek ve bu modeli basit bir arayüzle kullanılabilir hâle getirmek."),
    ])


def add_method(report: WeeklyReport, summary: Summary) -> None:
    split = summary["split"]
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin veri hazırlama aşaması üzerinde çalışıldı ve modelleme aşamasına geçildi (Wirth ve Hipp, 2000). "
        "Veri seti, kabul oranı iki tarafta da aynı kalacak şekilde %80 eğitim ve %20 test verisi olarak ayrıldı. Eğitim verisinde "
        f"{format_number(split['train_rows'])}, test verisinde {format_number(split['test_rows'])} kayıt bulunmaktadır. Test verisi "
        "sekizinci haftadaki son değerlendirmeye kadar hiçbir modelde kullanılmayacaktır.",
        "Ön işleme adımları scikit-learn kütüphanesinin Pipeline ve ColumnTransformer yapılarıyla tanımlandı (Pedregosa vd., 2011). "
        "Bu yapıda standartlaştırma için gereken ortalama ve standart sapma gibi değerler yalnızca eğitim verisinden hesaplanır. Bu "
        "değerler bütün veri üzerinden hesaplansaydı doğrulama verisine ait bilgi modele karışacak ve sonuçlar olduğundan iyi "
        "görünecekti. Bu duruma veri sızıntısı denir (Kaufman vd., 2012). Kodda böyle bir sızıntı olmadığını kontrol etmek için "
        "ayrıca testler yazıldı.",
        "Modeller 5 katlı katmanlı çapraz doğrulama ile değerlendirildi (Kohavi, 1995). Bu yöntemde eğitim verisi, her katmanda kabul "
        "oranı aynı kalacak şekilde beş katmana bölünür. Model her seferinde dört katmanla eğitilir ve kalan katmanda değerlendirilir. "
        "Böylece her kayıt bir kez doğrulama verisinde yer almış olur. Veri bölünmeden önce karıştırıldı ve sonuçların tekrar elde "
        "edilebilmesi için random_state = 42 kullanıldı. Bütün modeller aynı katmanlarla eğitildiği için sonuçlar katman katman "
        "karşılaştırılabilmektedir.",
        "İlk model olarak Lojistik Regresyon seçildi (Hosmer vd., 2013). Bu model hızlı çalışır ve katsayıları yorumlanabilir. Bu "
        "yüzden sonraki modeller için iyi bir başlangıç noktasıdır. Düzenlileştirme olarak scikit-learn'ün varsayılan L2 cezası "
        "kullanıldı. Sınıflar dengesiz olduğu için modelleri karşılaştırırken esas olarak PR-AUC değerine baktık (Saito ve "
        f"Rehmsmeier, 2015). Precision, Recall ve F1-Score değerleri {decimal(summary['threshold'], 1)} karar eşiğiyle hesaplandı. "
        "Daha uygun bir eşik altıncı haftada seçilecektir.",
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
        [name, f"{CAPTAIN_DUTY}, {work[0].lower()}{work[1:]}" if name == report.captain.full_name else work]
        for name, work in DONE_THIS_WEEK
    ]
    report.add_table("Bu hafta yapılan çalışmalar", ["Takım Üyesi", "Yapılan Çalışma"], done_rows, [4.5, 11.5])
    report.add_heading("3.4. Gelecek Hafta Planı")
    report.add_paragraph("Beşinci haftada Rastgele Orman ve XGBoost modelleri kurulacak ve üç model aynı çapraz doğrulama düzeninde karşılaştırılacaktır.")
    report.add_table("Gelecek hafta planı", ["Takım Üyesi", "Planlanan Çalışma"], [list(row) for row in NEXT_WEEK], [4.5, 11.5])


def add_preprocessing(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("4.1. Ön İşleme Pipeline'ı")
    report.add_paragraph(
        "Karar taslağında Lojistik Regresyon için belirlenen adımların hepsi src/preprocessing.py dosyasında kodlandı. Hiçbir kayıt "
        "silinmedi. Aykırı değerlerin etkisi log dönüşümüyle azaltıldı, unknown değerler de ayrı bir kategori olarak bırakıldı. "
        "Ağaç tabanlı modeller için ölçekleme yapılmayan ön işleme beşinci haftada eklenecektir. Değişkenlere uygulanan adımlar "
        "aşağıdaki tabloda verilmiştir."
    )
    report.add_table("Değişkenlere uygulanan ön işleme adımları", ["Değişken", "Uygulanan adım"], PREPROCESSING_STEPS, [4.5, 11.5])
    report.add_paragraph(
        f"duration çıkarıldıktan sonra kalan {summary['split']['raw_feature_count']} değişken bu adımlardan sonra "
        f"{summary['model_feature_count']} sütuna çıktı. Sütun sayısının artmasının nedeni One-Hot Encoding'de her kategori için "
        "ayrı bir sütun oluşturulmasıdır."
    )


def add_leakage_tests(report: WeeklyReport) -> None:
    report.add_heading("4.2. Veri Sızıntısı Testleri")
    report.add_paragraph(
        "Ön işleme ve çapraz doğrulama kodunda veri sızıntısı olmadığını göstermek için aşağıdaki testler yazıldı. Bu testlerin bir "
        "kısmı gerçek veriyle, bir kısmı ise sonucu önceden bilinen küçük örnek tablolarla çalışmaktadır. Bu hafta yazılan diğer "
        "testlerle birlikte projedeki bütün testler başarıyla geçmektedir."
    )
    report.add_table("Veri sızıntısını kontrol eden testler", ["Test", "Neyi kontrol ediyor"], LEAKAGE_TESTS, [8.0, 8.0])


def add_cross_validation(report: WeeklyReport, summary: Summary) -> None:
    folds = summary["cv_folds"]
    rates = [fold["validation_positive_rate"] for fold in folds]
    report.add_heading("4.3. Çapraz Doğrulama Düzeni")
    report.add_table(
        "Çapraz doğrulama katmanları",
        ["Katman", "Eğitim kaydı", "Doğrulama kaydı", "Doğrulamada kabul oranı"],
        [
            [str(number), format_number(fold["train_rows"]), format_number(fold["validation_rows"]), format_percent(fold["validation_positive_rate"])]
            for number, fold in enumerate(folds, start=1)
        ],
        [2.5, 4.5, 4.5, 4.5],
    )
    report.add_paragraph(
        f"Doğrulama katmanlarındaki kabul oranları {format_percent(min(rates))} ile {format_percent(max(rates))} arasında "
        f"değişmektedir. Eğitim verisindeki genel oran {format_percent(summary['split']['train_positive_rate'])} olduğu için "
        "katmanlara bölerken sınıf oranının korunduğu görülmektedir. Sonraki haftalarda kurulacak modeller de aynı katmanlarla "
        "değerlendirilecektir."
    )


def add_first_model(report: WeeklyReport, summary: Summary) -> None:
    main, baseline = "logistic_regression", "baseline"
    comparison = summary["comparisons"]["logistic_vs_baseline"]
    pr_ratio = mean_of(summary, main, "pr_auc") / mean_of(summary, baseline, "pr_auc")
    roc_auc = mean_of(summary, main, "roc_auc")
    target_text = (
        f"ROC-AUC değeri {mean_std(summary, main, 'roc_auc')} çıktı ve hedeflediğimiz {decimal(TARGET_ROC_AUC, 2)} değerinin üzerinde "
        "kaldı. Bu hedefin asıl olarak test verisinde sağlanması gerekiyor, bunu sekizinci haftada göreceğiz."
        if roc_auc >= TARGET_ROC_AUC
        else f"ROC-AUC değeri {mean_std(summary, main, 'roc_auc')} çıktı ve hedeflediğimiz {decimal(TARGET_ROC_AUC, 2)} değerinin "
        "altında kaldı."
    )
    report.add_heading("4.4. İlk Lojistik Regresyon Modeli")
    report.add_table(
        "Referans model ve Lojistik Regresyon: 5 katlı çapraz doğrulama sonuçları (ortalama ± standart sapma)",
        ["Metrik", "Referans model", "Lojistik Regresyon"],
        [[label, mean_std(summary, baseline, metric), mean_std(summary, main, metric)] for metric, label in METRIC_LABELS.items()],
        [4.0, 6.0, 6.0],
    )
    report.add_paragraph(
        f"Lojistik Regresyon modelinin PR-AUC değeri {mean_std(summary, main, 'pr_auc')} olarak bulundu. Referans modelde bu değer "
        f"{decimal(mean_of(summary, baseline, 'pr_auc'))} olarak kalmıştı. Yani model yaklaşık {decimal(pr_ratio, 1)} kat daha "
        f"yüksek bir değere ulaştı ve {fold_count_text(comparison['folds_improved'], comparison['fold_count'])} referans modeli "
        f"geçti. {target_text} {describe_fold_spread(summary, main)}"
    )
    report.add_paragraph(
        f"Accuracy değeri Lojistik Regresyon için {decimal(mean_of(summary, main, 'accuracy'))}, herkese \"kabul etmez\" diyen "
        f"referans model için {decimal(mean_of(summary, baseline, 'accuracy'))} çıktı. İki değerin bu kadar yakın olması, "
        "Accuracy'yi tek başına kullanmama kararını neden aldığımızı göstermektedir. Recall değeri ise "
        f"{mean_std(summary, main, 'recall')} ile düşük kaldı. Model {decimal(summary['threshold'], 1)} eşiğinde kampanyayı kabul eden "
        "müşterilerin ancak küçük bir kısmını bulabiliyor. Bunun nedeni, sınıflar dengesiz olduğu için modelin çok az müşteriye "
        f"{decimal(summary['threshold'], 1)} ve üzeri olasılık vermesidir. Bu müşterilerin oranı "
        f"{format_percent(summary['oof_predicted_positive_rate'])} iken veride kabul edenlerin oranı "
        f"{format_percent(summary['split']['train_positive_rate'])}. Bu sorun için altıncı haftada sınıf ağırlıklandırma, SMOTE ve "
        "eşik seçimi denenecektir."
    )
    report.add_figure(
        FIGURES_DIR / "h4_pr_roc.png",
        "Lojistik Regresyon'un çapraz doğrulama tahminlerinden çizilen PR ve ROC eğrileri (kesikli çizgi: referans model)",
    )


def add_open_questions(report: WeeklyReport, summary: Summary) -> None:
    comparisons = summary["comparisons"]
    age = comparisons["age_group_added"]
    months = comparisons["rare_months_merged"]
    month_names = {"jan": "ocak", "feb": "şubat", "mar": "mart", "apr": "nisan", "may": "mayıs", "jun": "haziran",
                   "jul": "temmuz", "aug": "ağustos", "sep": "eylül", "oct": "ekim", "nov": "kasım", "dec": "aralık"}
    merged = join_names([month_names[month] for month in summary["merged_months"]])
    check_decisions_match_data(summary)
    report.add_heading("4.5. Üçüncü Haftadan Kalan Sorular")
    report.add_table(
        "Yaş grubu ve ay birleştirme denemeleri (PR-AUC, ortalama ± standart sapma)",
        ["Model", "PR-AUC", "Ana modele göre fark"],
        [
            ["Lojistik Regresyon (ana model, yaş grubu var)", mean_std(summary, "logistic_regression", "pr_auc"), "-"],
            ["Yaş grubu olmadan", mean_std(summary, "lr_without_age_group", "pr_auc"), signed(-age["mean_difference"])],
            ["Az gözlemli aylar birleştirilmiş", mean_std(summary, "lr_rare_months_merged", "pr_auc"), signed(months["mean_difference"])],
        ],
        [7.5, 4.25, 4.25],
    )
    report.add_paragraph(
        f"{describe_comparison(age, 'Yaş grubu özelliği eklendiğinde')} "
        + ("İyileşme küçük olduğu hâlde yaş grubunu modelde tutmaya karar verdik." if is_within_noise(age)
           else "Bu yüzden yaş grubunu modelde tutmaya karar verdik.")
        + " Bu özelliğe beşinci haftadaki ablasyon çalışmasında ağaç modelleriyle birlikte tekrar bakılacaktır."
    )
    report.add_paragraph(
        f"Az gözlemli aylar için eğitim verisindeki payı %{decimal(summary['rare_month_min_frequency'] * 100, 0)} sınırının altında "
        f"kalan {merged} ayları tek bir kategoride birleştirilerek model yeniden eğitildi. {describe_comparison(months, 'Bu durumda')} "
        "Birleştirme bir iyileşme sağlamadığı için ayları ayrı kategoriler olarak bırakmaya karar verdik."
    )


def build_report(summary: Summary, captain_id: str) -> WeeklyReport:
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
        "yanıtı doğrudan etkilemektedir. Üçüncü haftada bu ilişkilerin veride bulunduğu görülmüştü. Bu hafta kurulan ilk model de "
        "kampanyayı kabul edecek müşterileri rastgele tahminden çok daha iyi ayırt edebildi. Bu haftanın sonuçları aşağıdaki alt "
        "başlıklarda verilmiştir.",
    ])
    add_preprocessing(report, summary)
    add_leakage_tests(report)
    add_cross_validation(report, summary)
    add_first_model(report, summary)
    add_open_questions(report, summary)
    report.add_links()
    report.add_references(REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="4. hafta ilerleme raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    output_path = build_report(summary, args.kaptan).save(OUTPUT_DIR / report_file_name(WEEK))
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
