"""5. hafta ilerleme raporunu üretir.

Kullanım: python reports/hafta5_rapor.py --kaptan 230260173
Önce notebooks/build_model_karsilastirma.py çalıştırılmalıdır. Metindeki tüm sayılar ve sayıya dayalı hükümler (en iyi model,
iyileşme/kötüleşme, katman sayıları) hafta5_ozet.json dosyasından türetilir; yalnızca yorum cümleleri bu haftanın bulgularına göre
yazılmıştır. Bölüm 2 (ekip içi çalışma süreci) işaretleri taslaktır; teslimden önce takım kaptanı tarafından gözden geçirilmelidir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AGE_LABELS, CV_FOLDS
from src.evaluate import min_folds_improved_for_candidate
from src.formatting import format_number, join_names
from src.modeling import LOGISTIC_REGRESSION, MODEL_LABELS, RANDOM_FOREST, XGBOOST
from src.report_docx import (
    CAPTAIN_ROLE,
    TEAM,
    TEAM_PROCESS_CRITERIA,
    WeeklyReport,
    report_file_name,
)
from src.report_text import (
    METRIC_LABELS,
    Summary,
    decimal,
    describe_change,
    describe_comparison,
    is_within_noise,
    mean_of,
    mean_std,
    signed,
)

WEEK = 5
CAPTAIN_DUTY = "Takım kaptanlığı ve haftalık raporun hazırlanması"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta5_ozet.json"
FIGURES_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR = ROOT / "reports" / "haftalik"
BASELINE = "baseline"
MODELS = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST]
# Hedef (FOG-1) test setinde ölçülecek; bu hafta yalnızca CV ortalamasının hedefin üstünde olup olmadığı belirtilir.
TARGET_ROC_AUC = 0.75
# Varsayılan 0,5 eşiğinde Recall ortalaması bu değerin altındaysa modelin kabul edenleri az yakaladığı söylenir.
LOW_RECALL_LIMIT = 0.5
TREE_MODELS_WITHOUT_DAY = ["rf_without_day", "xgb_without_day"]

# Kriterler 4. hafta raporunun "Gelecek Hafta Planı" tablosunda bu hafta için verilen görevlerden türetilmiştir.
PROGRESS_CRITERIA = [
    "Ağaç modelleri için ölçeklemesiz ön işleme kuruldu",
    "Rastgele Orman ve XGBoost modelleri aynı çapraz doğrulama düzeninde eğitildi",
    "Üç modelin karşılaştırma tablosu ve grafikleri hazırlandı",
    "Sonuçların aynı tohumla tekrar üretilebildiği test edildi",
    "Ön işleme ablasyonu yapıldı",
]

WORK_PACKAGES = [
    ("1", "Literatür Taraması ve Veri Seti Temini", "Tüm Ekip", "Tamamlandı"),
    ("2", "Veri Ön İşleme ve Temizleme", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("3", "Keşifsel Veri Analizi", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("4", "Makine Öğrenmesi Modellerinin Kurulması", "Ömer NARÇİÇEĞİ", "Tamamlandı"),
    ("5", "Model Eğitimi ve Optimizasyon", "Ömer NARÇİÇEĞİ", "Devam ediyor"),
    ("6", "Modellerin Test Edilmesi", "Mesut ASLAN", "Devam ediyor"),
    ("7", "Sonuçların Değerlendirilmesi ve Proje Raporunun Hazırlanması", "Tüm Ekip", "Devam ediyor"),
]

DONE_THIS_WEEK = [
    ("Ömer NARÇİÇEĞİ", "Rastgele Orman ve XGBoost modellerinin ağaç modelleri için ölçeklemesiz bir ön işleme ile kurulması ve aynı "
     "çapraz doğrulama düzeninde eğitilmesi"),
    ("Mesut ASLAN", "Üç modelin karşılaştırma tablosu ve grafikleri, aynı tohumla sonuçların tekrar üretilebildiğini kontrol eden test"),
    ("Mustafa Emre KAPLAN", "Ön işleme ablasyonu: day, yaş grubu ve log dönüşümü çıkarıldığında skorların nasıl değiştiğinin incelenmesi"),
]

NEXT_WEEK = [
    ("Ömer NARÇİÇEĞİ", "Sınıf dengesizliği için sınıf ağırlıklandırma, SMOTE ve ağırlıksız model ile eşik ayarlama stratejilerinin "
     "üç modelde aynı çapraz doğrulama düzeninde karşılaştırılması"),
    ("Mesut ASLAN", "Çapraz doğrulama dışı tahminlerden karar eşiği seçen fonksiyonun (F1 en yüksek ve hedef Recall) yazılması ve testleri"),
    ("Mustafa Emre KAPLAN", "SMOTE'un yalnızca eğitim katmanlarına uygulandığını kontrol eden testin yazılması ve sonuçların SMOTE "
     "kullanan çalışmalarla karşılaştırılması"),
]

PREPROCESSING_BRANCHES = [
    ["balance, campaign, previous", "İşareti koruyan log dönüşümü (balance) ya da log(1+x) (campaign, previous), ardından standartlaştırma", "Ham değer"],
    ["age, day, pdays", "Standartlaştırma", "Ham değer"],
    ["Yaş grubu", f"{len(AGE_LABELS)} aralıklı yaş grubu One-Hot ile eklendi", "Varsayılan olarak yok, ablasyonda denendi"],
    ["default, housing, loan", "yes = 1, no = 0", "yes = 1, no = 0"],
    ["job, marital, education, contact, month, poutcome", "One-Hot Encoding, unknown ayrı kategori", "One-Hot Encoding, unknown ayrı kategori"],
    ["was_contacted_before", "pdays ≠ -1 ise 1", "pdays ≠ -1 ise 1"],
]

# Ablasyon varyantlarının metindeki cümle başlangıçları (sayılar karşılaştırma sonucundan gelir, burada yalnızca değişiklik anlatılır).
ABLATION_LEADS = {
    "lr_without_day": "Lojistik Regresyon'dan day çıkarıldığında",
    "lr_without_age_group": "Lojistik Regresyon'dan yaş grubu çıkarıldığında",
    "lr_without_log": "Lojistik Regresyon'da log dönüşümü kapatıldığında",
    "rf_without_day": "Rastgele Orman'dan day çıkarıldığında",
    "rf_with_age_group": "Rastgele Orman'a yaş grubu eklendiğinde",
    "xgb_without_day": "XGBoost'tan day çıkarıldığında",
    "xgb_with_age_group": "XGBoost'a yaş grubu eklendiğinde",
}
# Gürültü içinde kalan farkları listelerken kullanılan kısa adlar.
ABLATION_SHORT_NAMES = {
    "lr_without_day": "Lojistik Regresyon'dan day'in çıkarılması",
    "lr_without_age_group": "Lojistik Regresyon'dan yaş grubunun çıkarılması",
    "lr_without_log": "Lojistik Regresyon'da log dönüşümünün kapatılması",
    "rf_without_day": "Rastgele Orman'dan day'in çıkarılması",
    "rf_with_age_group": "Rastgele Orman'a yaş grubunun eklenmesi",
    "xgb_without_day": "XGBoost'tan day'in çıkarılması",
    "xgb_with_age_group": "XGBoost'a yaş grubunun eklenmesi",
}
LINEAR_ABLATIONS = ["lr_without_day", "lr_without_age_group", "lr_without_log"]
TREE_ABLATIONS = ["rf_without_day", "rf_with_age_group", "xgb_without_day", "xgb_with_age_group"]

REFERENCES = [
    "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. "
    "Decision Support Systems, 62, 22-31.",
    "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing veri seti. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. Proceedings of the 4th "
    "International Conference on the Practical Applications of Knowledge Discovery and Data Mining, 29-39.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International "
    "Conference on Knowledge Discovery and Data Mining, 785-794.",
    "Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. Proceedings of "
    "the 14th International Joint Conference on Artificial Intelligence, 1137-1143.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary "
    "classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
]


def check_ablation_keeps_defaults(summary: Summary) -> None:
    """Metin ön işleme varsayılanlarının korunduğunu anlatır; veri bir varyantı aday gösteriyorsa rapor sessizce yanlış karar anlatmasın."""
    if summary["ablation_candidates"]:
        raise ValueError(
            f"Şu ablasyon varyantları varsayılanı değiştirmeye aday: {summary['ablation_candidates']}. "
            "Kararı, model varsayılanını ve rapor metnini birlikte gözden geçirin."
        )


def format_parameter(value: object) -> str:
    return format_number(value, 2) if isinstance(value, float) else str(value)


def format_parameters(parameters: dict[str, object]) -> str:
    return ", ".join(f"{name} = {format_parameter(value)}" for name, value in parameters.items())


def ablation_sentences(summary: Summary, names: list[str]) -> str:
    changes = " ".join(f"{ABLATION_LEADS[name]} {describe_change(summary['ablation'][name]['comparison'])}" for name in names)
    within_noise = [ABLATION_SHORT_NAMES[name] for name in names if is_within_noise(summary["ablation"][name]["comparison"])]
    if not within_noise:
        return changes
    return f"{changes} {join_names(within_noise)} için fark, katmanlardaki farkların standart sapmasının altında kaldı."


def describe_day_importance(summary: Summary) -> str:
    """Day'i çıkarmak iki ağaç modelinde de gürültünün üstünde bir düşüş yaratıyorsa bunu gerekçe olarak yazar."""
    comparisons = [summary["ablation"][name]["comparison"] for name in TREE_MODELS_WITHOUT_DAY]
    if all(comparison["mean_difference"] < 0 and not is_within_noise(comparison) for comparison in comparisons):
        return " Ağaç modellerinde day değişkenini çıkarmak skoru düşürdüğü için bu değişken modelde tutuldu."
    return ""


def describe_roc_auc_target(summary: Summary) -> str:
    below = [MODEL_LABELS[model] for model in MODELS if mean_of(summary, model, "roc_auc") < TARGET_ROC_AUC]
    if not below:
        return (f"Üç modelin ROC-AUC ortalaması da hedeflediğimiz {decimal(TARGET_ROC_AUC, 2)} değerinin üzerinde kaldı. "
                "Bu hedefin asıl olarak test verisinde sağlanması gerekiyor, bunu sekizinci haftada göreceğiz.")
    return f"ROC-AUC ortalaması hedeflediğimiz {decimal(TARGET_ROC_AUC, 2)} değerinin altında kalan model: {', '.join(below)}."


def describe_recall(summary: Summary) -> str:
    recalls = ", ".join(f"{MODEL_LABELS[model]} için {mean_std(summary, model, 'recall')}" for model in MODELS)
    lead = f"Recall değerleri {decimal(summary['threshold'], 1)} eşiğinde {recalls} çıktı."
    if max(mean_of(summary, model, "recall") for model in MODELS) < LOW_RECALL_LIMIT:
        return (f"{lead} Üç model de kampanyayı kabul edecek müşterilerin ancak küçük bir kısmını bulabiliyor. Sınıflar dengesiz olduğu "
                "için modeller çok az müşteriye yüksek olasılık veriyor. Altıncı haftada sınıf ağırlıklandırma, SMOTE ve eşik "
                "seçimi bu sorun için denenecektir.")
    return f"{lead} Karar eşiği altıncı haftada seçilecektir."


def describe_tree_gap(summary: Summary) -> str:
    comparison = summary["comparisons"]["xgboost_vs_random_forest"]
    if is_within_noise(comparison):
        return ("İki ağaç modeli arasındaki fark bu veriyle gürültü düzeyinde kaldığı için birini diğerinden üstün saymıyoruz. "
                "Ayarlanmış hâlleri yedinci haftada tekrar karşılaştırılacaktır.")
    return f"{MODEL_LABELS[summary['best_model']]} bu hafta en yüksek PR-AUC değerini veren model oldu."


def add_introduction(report: WeeklyReport) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "Dördüncü haftada ön işleme adımları koda dökülmüş ve ilk model olarak Lojistik Regresyon 5 katlı çapraz doğrulamada "
        "eğitilmişti. Bu model referans modelin üzerine çıkmıştı.",
        "Beşinci haftada Rastgele Orman ve XGBoost modelleri kuruldu. Bu iki model ağaç tabanlı olduğu için ölçeklemeye ihtiyaç "
        "duymaz. Bu yüzden onlar için ölçeklemesiz ayrı bir ön işleme hazırlandı. Üç model aynı katmanlarla ve aynı metriklerle "
        "karşılaştırıldı. Ayrıca day, yaş grubu ve log dönüşümünün sonuca katkısı, bu adımlar tek tek çıkarılarak ölçüldü. Test "
        "verisi bu hafta da kullanılmadı.",
    ])


def add_goals(report: WeeklyReport) -> None:
    report.add_heading("1.1. Amaç ve Hedefler")
    report.add_labeled_box([
        ("Amaç:", "Bankanın hedef kitlesini doğru analiz ederek pazarlama kampanyalarının başarı oranını artıracak ve maliyetleri "
                  "düşürecek ölçülebilir bir makine öğrenmesi tahmin modeli ortaya koymaktır."),
        ("Hedefler:", "Bank Marketing veri setini temin etmek ve doğrulamak. Bu hedef ikinci haftada tamamlanmıştır. Değişkenlerin "
                      "hedefle ilişkisini keşifsel analiz ve istatistiksel testlerle incelemek. Bu hedef üçüncü haftada "
                      "tamamlanmıştır. Bulgulara göre ön işleme adımlarını kurmak. Bu hedef dördüncü ve beşinci haftalarda "
                      "tamamlanmıştır. Lojistik Regresyon, Rastgele Orman ve XGBoost modellerini çapraz doğrulama ile eğitmek ve "
                      "sınıf dengesizliği için farklı yöntemleri karşılaştırmak. Bu hedefin modelleri eğitme kısmı beşinci haftada "
                      "tamamlanmıştır. Veri setindeki sınıflar dengesiz olduğu için modelleri yalnızca Accuracy ile değil, "
                      "Precision, Recall, F1-Score, ROC-AUC ve PR-AUC ile de değerlendirmek. Test verisinde ROC-AUC değeri en az "
                      "0,75 olan bir model elde etmek ve bu modeli basit bir arayüzle kullanılabilir hâle getirmek."),
    ])


def add_method(report: WeeklyReport, summary: Summary) -> None:
    split = summary["split"]
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin modelleme aşamasında çalışıldı (Wirth ve Hipp, 2000). Veri seti geçen haftaki bölmeyle aynı "
        f"kaldı. Eğitim verisinde {format_number(split['train_rows'])}, test verisinde {format_number(split['test_rows'])} kayıt "
        "bulunmaktadır. Test verisi sekizinci haftadaki son değerlendirmeye kadar hiçbir modelde kullanılmayacaktır.",
        "Lojistik Regresyon'a ek olarak iki ağaç tabanlı model kuruldu. Rastgele Orman, bootstrap örneklemleriyle eğitilmiş çok "
        "sayıda karar ağacının oylamasına dayanır (Breiman, 2001). XGBoost ise ağaçları art arda ekleyen ve her yeni ağacı "
        "öncekilerin hatasını azaltacak biçimde eğiten bir gradyan artırma yöntemidir (Chen ve Guestrin, 2016). Ağaçlar bölme "
        "noktasını değişkenin yalnızca sıralamasına göre seçtiği için ölçekleme ve log dönüşümü onların sonucunu değiştirmez. Bu "
        "yüzden sayısal değişkenler ham değerleriyle kullanıldı. One-Hot Encoding, was_contacted_before değişkeni ve pdays "
        "düzenlemesi Lojistik Regresyon ile aynı bırakıldı.",
        f"Üç model de dördüncü haftadaki {CV_FOLDS} katlı katmanlı çapraz doğrulama düzeniyle ve aynı katmanlarla eğitildi "
        "(Kohavi, 1995). Böylece sonuçlar katman katman karşılaştırılabilmektedir. Karşılaştırmada esas olarak PR-AUC değerine "
        f"bakıldı (Saito ve Rehmsmeier, 2015). Precision, Recall ve F1-Score değerleri {decimal(summary['threshold'], 1)} karar "
        "eşiğiyle hesaplandı. Hiperparametreler bu hafta aranmadı, sabit tutuldu. Ayarlanmaları yedinci haftada yapılacaktır.",
        "Ön işleme adımlarının katkısını görmek için ablasyon çalışması yapıldı. Bir adım çıkarıldı (ağaç modellerinde yaş grubu ise "
        "eklendi) ve model aynı katmanlarla yeniden eğitilerek PR-AUC farkı katman katman hesaplandı. Bir değişikliğin varsayılan "
        "ön işlemeyi değiştirmesi için iki koşul belirlendi: ortalama artış katmanlar arasındaki standart sapmayı aşmalı ve "
        f"{CV_FOLDS} katmanın en az {min_folds_improved_for_candidate(CV_FOLDS)}'ünde iyileşme görülmeli.",
        "Rastgele Orman birden çok iş parçacığında çalıştırıldığında ağaçların olasılıkları her seferinde farklı sırayla "
        "toplandığı için sonuçlar ondalık sayının son basamaklarında değişebiliyordu. Sonuçların birebir tekrar edilebilmesi için "
        "bu model tek iş parçacığında çalıştırıldı.",
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
    report.add_paragraph("Altıncı haftada sınıf dengesizliği için üç yöntem karşılaştırılacak ve karar eşiği seçilecektir.")
    report.add_table("Gelecek hafta planı", ["Takım Üyesi", "Planlanan Çalışma"], [list(row) for row in NEXT_WEEK], [4.5, 11.5])


def add_tree_preprocessing(report: WeeklyReport, summary: Summary) -> None:
    counts = summary["feature_counts"]
    report.add_heading("4.1. Ağaç Modelleri İçin Ön İşleme")
    report.add_paragraph(
        "Ağaç modelleri için src/preprocessing.py dosyasına ayrı bir ön işleme dalı eklendi. Bu dalda sayısal değişkenlere log "
        "dönüşümü ve standartlaştırma uygulanmadı. Kategorik değişkenler Lojistik Regresyon'daki gibi kodlandı ve unknown değerler "
        "ayrı bir kategori olarak kaldı. İki dalın farkı aşağıdaki tabloda verilmiştir."
    )
    report.add_table(
        "Lojistik Regresyon ve ağaç modelleri için ön işleme adımları",
        ["Değişken", "Lojistik Regresyon", "Rastgele Orman ve XGBoost"],
        PREPROCESSING_BRANCHES,
        [4.5, 6.0, 5.5],
    )
    gap = counts[LOGISTIC_REGRESSION] - counts["tree"]
    gap_text = (
        " Aradaki fark, yalnızca Lojistik Regresyon'da bulunan yaş grubu sütunlarından gelmektedir."
        if gap == len(AGE_LABELS)
        else ""
    )
    report.add_paragraph(
        f"duration çıkarıldıktan sonra kalan {summary['split']['raw_feature_count']} değişken, Lojistik Regresyon için "
        f"{counts[LOGISTIC_REGRESSION]}, ağaç modelleri için {counts['tree']} sütuna çıktı.{gap_text} Yaş grubunu ağaçlarda "
        "varsayılan olarak kullanmadık, çünkü ağaç yaşın U şeklindeki ilişkisini ham yaş değerinden de öğrenebilir. Bu varsayımı "
        "ablasyon çalışmasında denedik."
    )


def add_tree_models(report: WeeklyReport, summary: Summary) -> None:
    parameters = summary["hyperparameters"]
    report.add_heading("4.2. Rastgele Orman ve XGBoost")
    forest_trees, boosting_trees = parameters[RANDOM_FOREST]["n_estimators"], parameters[XGBOOST]["n_estimators"]
    tree_counts = (
        f"İki model de {forest_trees} ağaçla kuruldu."
        if forest_trees == boosting_trees
        else f"Rastgele Orman {forest_trees} ağaçla, XGBoost ise {boosting_trees} ağaçla kuruldu."
    )
    report.add_paragraph(
        f"{tree_counts} Hiperparametreler bu hafta aranmadı ve aşağıdaki sabit değerler kullanıldı. Bu yüzden ağaç modellerinin "
        "bu haftaki skorları, ayarlanmış hâllerinin skorundan düşük olabilir."
    )
    report.add_table(
        "Ağaç modellerinin bu hafta kullanılan sabit hiperparametreleri",
        ["Model", "Hiperparametreler"],
        [[MODEL_LABELS[model], format_parameters(parameters[model])] for model in (RANDOM_FOREST, XGBOOST)],
        [4.0, 12.0],
    )


def add_model_comparison(report: WeeklyReport, summary: Summary) -> None:
    comparisons = summary["comparisons"]
    report.add_heading("4.3. Üç Modelin Karşılaştırılması")
    report.add_table(
        "Referans model ve üç modelin 5 katlı çapraz doğrulama sonuçları (ortalama ± standart sapma)",
        ["Metrik", "Referans model", *[MODEL_LABELS[model] for model in MODELS]],
        [[label, mean_std(summary, BASELINE, metric), *[mean_std(summary, model, metric) for model in MODELS]] for metric, label in METRIC_LABELS.items()],
        [2.6, 3.4, 3.4, 3.2, 3.4],
    )
    report.add_paragraph(
        f"PR-AUC değeri Lojistik Regresyon için {mean_std(summary, LOGISTIC_REGRESSION, 'pr_auc')}, Rastgele Orman için "
        f"{mean_std(summary, RANDOM_FOREST, 'pr_auc')}, XGBoost için {mean_std(summary, XGBOOST, 'pr_auc')} olarak bulundu. "
        f"Referans modelde bu değer {decimal(mean_of(summary, BASELINE, 'pr_auc'))} idi. "
        f"{describe_comparison(comparisons['random_forest_vs_lr'], 'Lojistik Regresyon yerine Rastgele Orman kullanıldığında')} "
        f"{describe_comparison(comparisons['xgboost_vs_lr'], 'Lojistik Regresyon yerine XGBoost kullanıldığında')} "
        f"{describe_comparison(comparisons['xgboost_vs_random_forest'], 'Rastgele Orman yerine XGBoost kullanıldığında')} "
        f"{describe_tree_gap(summary)}"
    )
    report.add_paragraph(
        f"ROC-AUC değerleri sırasıyla {mean_std(summary, LOGISTIC_REGRESSION, 'roc_auc')}, "
        f"{mean_std(summary, RANDOM_FOREST, 'roc_auc')} ve {mean_std(summary, XGBOOST, 'roc_auc')} çıktı. "
        f"{describe_roc_auc_target(summary)} {describe_recall(summary)}"
    )
    report.add_figure(FIGURES_DIR / "h5_model_karsilastirma.png", "Üç modelin PR-AUC ve ROC-AUC ortalamaları (çubuklar: standart sapma, kesikli çizgi: referans model)")
    report.add_figure(FIGURES_DIR / "h5_pr_egrileri.png", "Üç modelin çapraz doğrulama tahminlerinden çizilen PR eğrileri (kesikli çizgi: referans model)")


def add_ablation(report: WeeklyReport, summary: Summary) -> None:
    check_ablation_keeps_defaults(summary)
    report.add_heading("4.4. Ablasyon Çalışması")
    report.add_table(
        "Ön işleme ablasyonu: PR-AUC (ortalama ± standart sapma) ve ana modele göre fark",
        ["Model", "Değişiklik", "PR-AUC", "Ana modele göre fark"],
        [
            [
                MODEL_LABELS[entry["model"]],
                entry["change"],
                f"{decimal(entry['summary']['pr_auc']['mean'])} ± {decimal(entry['summary']['pr_auc']['std'])}",
                signed(entry["comparison"]["mean_difference"]),
            ]
            for entry in summary["ablation"].values()
        ],
        [3.6, 5.0, 3.8, 3.6],
    )
    report.add_paragraph(
        "Tablodaki fark, varyantın PR-AUC ortalaması ile ana modelin ortalaması arasındaki farktır. Negatif değer, değişikliğin "
        "skoru düşürdüğünü gösterir. Log dönüşümü ağaç modellerinde denenmedi, çünkü monoton bir dönüşüm ağaçların bölme "
        "noktalarını değiştirmez."
    )
    report.add_paragraph(ablation_sentences(summary, LINEAR_ABLATIONS))
    report.add_paragraph(ablation_sentences(summary, TREE_ABLATIONS))
    report.add_paragraph(
        "Hiçbir değişiklik, ortalamada standart sapmayı aşan ve katmanların çoğunda görülen bir iyileşme sağlamadığı için ön işleme "
        f"adımlarının hepsini olduğu gibi bıraktık.{describe_day_importance(summary)} Üçüncü haftadan kalan yaş grubu "
        "sorusuna ağaç modelleri için de cevap bulundu: yaş grubu ağaçlarda bir iyileşme getirmedi."
    )


def add_reproducibility(report: WeeklyReport) -> None:
    report.add_heading("4.5. Sonuçların Tekrar Üretilebilirliği")
    report.add_paragraph(
        "Sonuçların aynı tohumla tekrar elde edilebildiğini iki yolla kontrol ettik. Birincisi, her model için aynı veriyle ve "
        "aynı tohumla çapraz doğrulama iki kez çalıştırıldı ve out-of-fold tahminlerinin birebir aynı olduğunu kontrol eden "
        "testler yazıldı. İkincisi, analiz betiği baştan sona iki kez çalıştırıldı ve ürettiği özet dosyasının iki seferde de "
        "birebir aynı çıktığı görüldü. Bu kontrolü yaparken Rastgele Orman'ın çok iş parçacığındaki küçük sapmasını fark ettik "
        "ve modeli tek iş parçacığına aldık."
    )


def build_report(summary: Summary, captain_id: str) -> WeeklyReport:
    report = WeeklyReport(week=WEEK, captain_id=captain_id)
    # Ekip süreci işaretleri taslaktır; kaptan teslimden önce gerçek duruma göre değiştirmelidir.
    report.add_captain_form(
        team_process=[(criterion, True) for criterion in TEAM_PROCESS_CRITERIA],
        progress=[(criterion, True) for criterion in PROGRESS_CRITERIA],
    )
    report.add_title_block()
    add_introduction(report)
    add_goals(report)
    add_method(report, summary)
    add_project_management(report)
    report.add_heading("4. TEORİ VE SONUÇLAR")
    report.add_box([
        "Müşterilerin durumları, sosyoekonomik özellikleri ve geçmiş bankacılık alışkanlıkları, pazarlama kampanyalarına verecekleri "
        "yanıtı doğrudan etkilemektedir. Üçüncü haftada bu ilişkilerin veride bulunduğu görülmüştü. Bu hafta kurulan modeller "
        f"arasında kampanyayı kabul edecek müşterileri en iyi ayırt eden {MODEL_LABELS[summary['best_model']]} oldu. Bu haftanın "
        "sonuçları aşağıdaki alt başlıklarda verilmiştir.",
    ])
    add_tree_preprocessing(report, summary)
    add_tree_models(report, summary)
    add_model_comparison(report, summary)
    add_ablation(report, summary)
    add_reproducibility(report)
    report.add_links()
    report.add_references(REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="5. hafta ilerleme raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    output_path = build_report(summary, args.kaptan).save(OUTPUT_DIR / report_file_name(WEEK))
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
