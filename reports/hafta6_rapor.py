"""6. hafta ilerleme raporunu üretir.

Kullanım: python reports/hafta6_rapor.py --kaptan 230260143
Önce notebooks/build_dengesizlik.py çalıştırılmalıdır. Metindeki tüm sayılar ve sayıya dayalı hükümler (artış/azalış, en iyi strateji,
aday modeller) hafta6_ozet.json dosyasından türetilir; yalnızca yorum cümleleri bu haftanın bulgularına göre yazılmıştır.
Bölüm 2 (ekip içi çalışma süreci) işaretleri taslaktır; teslimden önce takım kaptanı tarafından gözden geçirilmelidir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.hafta5_rapor import NEXT_WEEK as DONE_THIS_WEEK
from src.config import CV_FOLDS
from src.formatting import format_number, format_percent, join_names
from src.modeling import (
    IMBALANCE_CLASS_WEIGHT,
    IMBALANCE_NONE,
    IMBALANCE_SMOTE,
    IMBALANCE_STRATEGIES,
    LOGISTIC_REGRESSION,
    MODEL_LABELS,
    RANDOM_FOREST,
    STRATEGY_LABELS,
    XGBOOST,
)
from src.report_docx import CAPTAIN_ROLE, TEAM, TEAM_PROCESS_CRITERIA, WeeklyReport, report_file_name
from src.report_text import Summary, decimal, describe_comparison, lower_first, number_word

WEEK = 6
CAPTAIN_DUTY = "Takım kaptanlığı ve haftalık raporun hazırlanması"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta6_ozet.json"
FIGURES_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR = ROOT / "reports" / "haftalik"
MODELS = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST]
# Aynı modelin stratejileri arasında F1 farkı bu değerin altındaysa stratejiler arasında F1 bakımından büyük üstünlük olmadığı söylenir.
SMALL_F1_SPREAD = 0.02

MODEL_LOCATIVE = {LOGISTIC_REGRESSION: "Lojistik Regresyon'da", RANDOM_FOREST: "Rastgele Orman'da", XGBOOST: "XGBoost'ta"}

# Kriterler 5. hafta raporunun "Gelecek Hafta Planı" tablosunda bu hafta için verilen görevlerden türetilmiştir.
PROGRESS_CRITERIA = [
    "Sınıf ağırlıklandırma, SMOTE ve eşik ayarlama stratejileri üç modelde karşılaştırıldı",
    "Karar eşiğini çapraz doğrulama dışı tahminlerden seçen fonksiyon yazıldı ve test edildi",
    "SMOTE'un yalnızca eğitim katmanlarına uygulandığı test edildi",
    "Sonuçlar SMOTE kullanan çalışmalarla karşılaştırıldı",
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

NEXT_WEEK = [
    ("Ömer NARÇİÇEĞİ", "En iyi iki model ve strateji için PR-AUC'ye göre rastgele hiperparametre araması ve arama süresinin ölçülmesi"),
    ("Mesut ASLAN", "Olasılık kalibrasyonunun kontrol edilmesi, kalibrasyon eğrisi ve arama sonuçlarının görselleştirilmesi"),
    ("Mustafa Emre KAPLAN", "Final raporunun iskeletinin hazırlanması, ilk altı haftanın bulgularının ve literatür karşılaştırma tablosunun derlenmesi"),
]

STRATEGY_DESCRIPTIONS = [
    [STRATEGY_LABELS[IMBALANCE_NONE], "Model sınıfları eşit ağırlıkla öğrenir. Dengesizlik, model eğitildikten sonra karar eşiği değiştirilerek giderilir."],
    [STRATEGY_LABELS[IMBALANCE_CLASS_WEIGHT], "Azınlık sınıfındaki hatalar daha ağır cezalandırılır. Lojistik Regresyon'da class_weight = balanced, Rastgele Orman'da "
     "balanced_subsample kullanıldı. XGBoost'ta scale_pos_weight, her eğitimde o eğitim verisindeki negatif/pozitif oranından hesaplandı."],
    [STRATEGY_LABELS[IMBALANCE_SMOTE], "Azınlık sınıfı için ön işlemeden sonra sentetik örnekler üretilir. Yalnızca eğitim katmanında çalışır."],
]

LITERATURE_ROWS = [
    ["Arifah vd., 2023", "Aynı 45.211 kayıtlık veri, XGBoost, SMOTE ile ve SMOTE olmadan",
     "SMOTE uygulandığında Recall yükselmiş, ROC-AUC 0,94 civarında neredeyse aynı kalmıştır."],
    ["Dong vd., 2026", "Topluluk yöntemleri, görüşme süresi modelde, SMOTE yalnızca eğitim verisine ve sınıf ağırlıklandırmasıyla birlikte",
     "En iyi model 0,908 ROC-AUC değerine ulaşmıştır."],
]

REFERENCES = [
    "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. "
    "Decision Support Systems, 62, 22-31.",
    "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing veri seti. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. Proceedings of the 4th "
    "International Conference on the Practical Applications of Knowledge Discovery and Data Mining, 29-39.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International "
    "Conference on Knowledge Discovery and Data Mining, 785-794.",
    "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. "
    "Journal of Artificial Intelligence Research, 16, 321-357.",
    "Arifah, D., Saragih, T. H., Kartini, D., Muliadi, M., & Mazdadi, M. I. (2023). Application of SMOTE to handle imbalance class in "
    "deposit classification using the extreme gradient boosting algorithm. Jurnal Ilmiah Teknik Elektro Komputer dan Informatika, "
    "9(2), 396-410.",
    "Dong, B., Zhang, X., Liu, Y., Zhang, T., Liu, X., Hou, L., Meng, L., Guo, Z., & Mulati, A. (2026). Risk-sensitive machine "
    "learning for financial decision modeling under imbalanced data: Evidence from bank telemarketing. Entropy, 28(3), 354.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary "
    "classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
    "Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. Proceedings of "
    "the 14th International Joint Conference on Artificial Intelligence, 1137-1143.",
]


def experiment(summary: Summary, model: str, strategy: str) -> dict:
    return summary["experiments"][f"{model}__{strategy}"]


def metric_mean(summary: Summary, model: str, strategy: str, metric: str, section: str = "summary") -> float:
    return experiment(summary, model, strategy)[section][metric]["mean"]


def mean_std_of(summary: Summary, model: str, strategy: str, metric: str, section: str = "summary") -> str:
    values = experiment(summary, model, strategy)[section][metric]
    return f"{decimal(values['mean'])} ± {decimal(values['std'])}"


def check_class_weight_raises_recall(summary: Summary) -> None:
    """Metin sınıf ağırlığının 0,5 eşiğinde Recall'u yükselttiğini anlatır; veri tersini gösterirse rapor sessizce yanlış anlatmasın."""
    for model in MODELS:
        if metric_mean(summary, model, IMBALANCE_CLASS_WEIGHT, "recall") <= metric_mean(summary, model, IMBALANCE_NONE, "recall"):
            raise ValueError(f"{MODEL_LABELS[model]} için sınıf ağırlığı 0,5 eşiğinde Recall'u yükseltmiyor. Rapor metnini gözden geçirin.")


def describe_recall_changes(summary: Summary, strategy: str) -> str:
    """"Lojistik Regresyon'da 0,186 iken 0,627" biçiminde, yön belirtmeden önce/sonra değerleri sıralar."""
    parts = [
        f"{MODEL_LOCATIVE[model]} {decimal(metric_mean(summary, model, IMBALANCE_NONE, 'recall'))} iken "
        f"{decimal(metric_mean(summary, model, strategy, 'recall'))} oldu"
        for model in MODELS
    ]
    return join_names(parts)


def describe_pr_auc_effects(summary: Summary) -> str:
    effects = summary["strategy_effects"]
    improved = {key: effect for key, effect in effects.items() if effect["mean_difference"] > 0}
    if improved:
        sentences = []
        for key, effect in improved.items():
            model, strategy = key.split("__")
            sentences.append(describe_comparison(effect, f"{MODEL_LABELS[model]} modelinde {STRATEGY_LABELS[strategy].lower()} kullanıldığında"))
        remaining = len(effects) - len(improved)
        rest = f" Diğer {number_word(remaining)} karşılaştırmada PR-AUC artmadı." if remaining else ""
        return f"{' '.join(sentences)}{rest}"
    drops = [-effect["mean_difference"] for effect in effects.values()]
    never_improved = sum(effect["folds_improved"] for effect in effects.values()) == 0
    detail = " Hiçbir katmanda da iyileşme görülmedi." if never_improved else ""
    return (f"Sınıf ağırlığı ve SMOTE, ağırlıksız hâle göre {number_word(len(effects))} karşılaştırmanın hiçbirinde PR-AUC'yi artırmadı. "
            f"Ortalama düşüş {decimal(min(drops), 4)} ile {decimal(max(drops), 4)} arasında kaldı.{detail} "
            "Bu yöntemler modelin müşterileri sıralama kalitesini artırmadı.")


def f1_spread_within_models(summary: Summary) -> float:
    spreads = []
    for model in MODELS:
        values = [metric_mean(summary, model, strategy, "f1", "nested_summary_f1_threshold") for strategy in IMBALANCE_STRATEGIES]
        spreads.append(max(values) - min(values))
    return max(spreads)


def describe_selected_threshold_f1(summary: Summary) -> str:
    spread = f1_spread_within_models(summary)
    text = (f"Eşik, ölçüldüğü katmanın dışındaki katmanların tahminlerinden seçildiğinde F1 değerleri aynı modelin "
            f"{number_word(len(IMBALANCE_STRATEGIES))} stratejisi arasında en fazla {decimal(spread)} fark etti.")
    if spread < SMALL_F1_SPREAD:
        return f"{text} Eşik doğru seçilince stratejiler arasında F1 bakımından büyük bir üstünlük kalmadı."
    return text


def describe_best_strategies(summary: Summary) -> str:
    best = summary["best_by_model"]
    if all(strategy == IMBALANCE_NONE for strategy in best.values()):
        return ("Üç modelde de en yüksek PR-AUC değeri ağırlıksız hâlde görüldü. Bu yüzden dengesizlik için eşik ayarlamasını kullanmaya "
                "karar verdik.")
    listing = join_names([f"{MODEL_LABELS[model]} için {STRATEGY_LABELS[strategy].lower()}" for model, strategy in best.items()])
    return f"PR-AUC değerine göre en iyi strateji {listing} oldu."


def describe_tuning_candidates(summary: Summary) -> str:
    candidates = summary["candidates_for_tuning"]
    listing = join_names([f"{MODEL_LABELS[entry['model']]} ({STRATEGY_LABELS[entry['strategy']].lower()})" for entry in candidates])
    return f"Yedinci haftada hiperparametre araması için PR-AUC sıralamasındaki ilk {number_word(len(candidates))} farklı model seçildi: {listing}."


def describe_literature_comparison(summary: Summary) -> str:
    roc_spread = max(
        max(metric_mean(summary, model, strategy, "roc_auc") for strategy in IMBALANCE_STRATEGIES)
        - min(metric_mean(summary, model, strategy, "roc_auc") for strategy in IMBALANCE_STRATEGIES)
        for model in MODELS
    )
    return (
        "Arifah ve arkadaşları XGBoost ile SMOTE uygulandığında Recall'un yükseldiğini, ROC-AUC'nin ise neredeyse aynı kaldığını "
        f"bildirmişti. Bizim çalışmamızda SMOTE kullanıldığında 0,5 eşiğinde Recall {describe_recall_changes(summary, IMBALANCE_SMOTE)}. "
        f"ROC-AUC ise aynı modelin üç stratejisi arasında en fazla {decimal(roc_spread)} değişti. Dong ve arkadaşları SMOTE'u yalnızca "
        "eğitim verisine uygulamış ve sınıf ağırlıklandırmasıyla birlikte kullanmıştır. Biz de SMOTE'u yalnızca eğitim katmanlarında "
        "kullandık, ancak iki yöntemi birlikte denemedik. Dong ve arkadaşlarının modeline görüşme süresi girmiştir, Arifah ve "
        "arkadaşları ise bunu belirtmemiştir. Bu yüzden ROC-AUC değerleri bizim sonuçlarımızla doğrudan karşılaştırılamaz."
    )


def add_introduction(report: WeeklyReport) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "Beşinci haftada Lojistik Regresyon, Rastgele Orman ve XGBoost aynı çapraz doğrulama düzeninde karşılaştırılmıştı. "
        "Üç modelin de 0,5 karar eşiğinde kabul edecek müşterilerin ancak küçük bir kısmını bulabildiği görülmüştü.",
        "Altıncı haftada bu sorun için üç yöntem denendi: sınıf ağırlıklandırma, SMOTE ve karar eşiğinin ayarlanması. Üç yöntem üç "
        "modelde aynı katmanlarla karşılaştırıldı ve karar eşiği, test verisine bakılmadan çapraz doğrulama dışı tahminlerden "
        "seçildi. Sonuçlar SMOTE kullanan çalışmalarla karşılaştırıldı.",
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
                      "sınıf dengesizliği için farklı yöntemleri karşılaştırmak. Bu hedef altıncı haftada tamamlanmıştır, "
                      "hiperparametre ayarı yedinci haftada yapılacaktır. Veri setindeki sınıflar dengesiz olduğu için modelleri "
                      "yalnızca Accuracy ile değil, Precision, Recall, F1-Score, ROC-AUC ve PR-AUC ile de değerlendirmek. Test "
                      "verisinde ROC-AUC değeri en az 0,75 olan bir model elde etmek ve bu modeli basit bir arayüzle kullanılabilir "
                      "hâle getirmek."),
    ])


def add_method(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin modelleme aşamasında çalışıldı (Wirth ve Hipp, 2000). Veri bölmesi ve çapraz doğrulama katmanları "
        f"geçen haftayla aynı kaldı. Eğitim verisinde {format_number(summary['split']['train_rows'])}, test verisinde "
        f"{format_number(summary['split']['test_rows'])} kayıt bulunmaktadır. Test verisi sekizinci haftaya kadar ne modellerde ne de "
        "eşik seçiminde kullanıldı.",
        "Sınıf dengesizliği için üç yöntem denendi. Sınıf ağırlıklandırmada azınlık sınıfındaki hatalar daha ağır cezalandırılır. "
        "SMOTE azınlık sınıfı için sentetik örnekler üretir (Chawla vd., 2002). SMOTE ön işlemeden sonra çalıştığı için One-Hot "
        "sütunlarında 0 ile 1 arasında kesirli değerler üretebilir, bunu bu yöntemin bilinen bir sınırı olarak kabul ettik. Üçüncü "
        "yöntemde model ağırlıksız eğitilir ve kabul kararı için kullanılan olasılık eşiği ayarlanır. {number_word(summary['model_count'] * summary['strategy_count'])} kombinasyonun "
        f"({number_word(summary['model_count'])} model, {number_word(summary['strategy_count'])} yöntem) hepsi aynı {CV_FOLDS} katlı katmanlı çapraz doğrulamayla "
        "eğitildi (Kohavi, 1995).",
        "Yöntemleri karşılaştırırken esas olarak PR-AUC ve F1-Score değerine baktık (Saito ve Rehmsmeier, 2015). PR-AUC eşikten "
        "bağımsızdır, yani modelin müşterileri kabul olasılığına göre ne kadar iyi sıraladığını gösterir. F1-Score ise bir karar eşiği "
        "gerektirir ve iki eşikte hesaplandı: varsayılan 0,5 ve çapraz doğrulama dışı (out-of-fold) tahminlerden seçilen eşik. Seçilen "
        "eşikle F1 hesaplanırken eşik, ölçülen katmanın dışındaki katmanlardan seçildi.",
        "Eşik iki ölçütle seçildi. Birincisi F1-Score'u en yüksek yapan eşiktir. Eşit F1 değerlerinde daha yüksek eşik seçildi, çünkü "
        f"bu daha az müşteri aranması demektir. İkincisi, Recall değerini en az {decimal(summary['target_recall'], 2)} yapan "
        "en yüksek eşiktir. Seçim fonksiyonları yalnızca gerçek etiketleri ve out-of-fold olasılıkları alır, test "
        "verisi bu fonksiyonlara verilemez.",
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
    report.add_paragraph("Yedinci haftada seçilen modeller için hiperparametre araması yapılacak ve olasılık kalibrasyonu kontrol edilecektir.")
    report.add_table("Gelecek hafta planı", ["Takım Üyesi", "Planlanan Çalışma"], [list(row) for row in NEXT_WEEK], [4.5, 11.5])


def add_strategies(report: WeeklyReport) -> None:
    report.add_heading("4.1. Sınıf Dengesizliği Stratejileri")
    report.add_paragraph(
        "Bu hafta denenen üç stratejinin nasıl uygulandığı aşağıdaki tabloda verilmiştir. Üçüncü strateji olan eşik ayarlama için ayrı bir "
        "model kurulmadı. Ağırlıksız model kullanıldı ve eşik 4.4 bölümünde anlatıldığı gibi seçildi."
    )
    report.add_table("Sınıf dengesizliği stratejileri", ["Strateji", "Uygulama"], STRATEGY_DESCRIPTIONS, [4.0, 12.0])


def add_smote_leakage(report: WeeklyReport) -> None:
    report.add_heading("4.2. SMOTE'un Yalnızca Eğitim Verisine Uygulanması")
    report.add_paragraph(
        "SMOTE doğrulama katmanına da uygulansaydı sentetik örnekler doğrulama sonuçlarına karışır ve skorlar olduğundan iyi görünürdü. "
        "Bunu önlemek için SMOTE, imbalanced-learn kütüphanesinin Pipeline yapısına ön işlemeden sonra ve modelden önce eklendi. Bu "
        "yapı örneklemeyi yalnızca eğitim sırasında uygular, tahmin sırasında hiçbir örnekleme yapmaz. Bunu kontrol etmek için bir test "
        "yazıldı. Testte, kendisine verilen satır sayısını ve sınıf oranını kaydeden sahte bir model SMOTE'lu pipeline'la çapraz "
        "doğrulamadan geçirildi. Eğitim katmanlarının SMOTE sonrası büyüdüğü ve sınıf oranının 0,5 olduğu, doğrulama katmanlarının ise "
        "satır sayısının hiç değişmediği görüldü."
    )


def add_strategy_comparison(report: WeeklyReport, summary: Summary) -> None:
    check_class_weight_raises_recall(summary)
    rows = [
        [
            MODEL_LABELS[model],
            STRATEGY_LABELS[strategy],
            mean_std_of(summary, model, strategy, "pr_auc"),
            mean_std_of(summary, model, strategy, "recall"),
            mean_std_of(summary, model, strategy, "f1"),
            mean_std_of(summary, model, strategy, "f1", "nested_summary_f1_threshold"),
        ]
        for model in MODELS
        for strategy in IMBALANCE_STRATEGIES
    ]
    report.add_heading("4.3. Strateji ve Model Karşılaştırması")
    report.add_table(
        "Model ve strateji kombinasyonlarının 5 katlı çapraz doğrulama sonuçları (ortalama ± standart sapma)",
        ["Model", "Strateji", "PR-AUC", "Recall (0,5 eşiği)", "F1 (0,5 eşiği)", "F1 (seçilen eşik)"],
        rows,
        [3.0, 2.8, 2.6, 2.8, 2.4, 2.4],
    )
    report.add_paragraph(
        f"{describe_pr_auc_effects(summary)} Sınıf ağırlığı kullanıldığında 0,5 eşiğindeki Recall {describe_recall_changes(summary, IMBALANCE_CLASS_WEIGHT)}. "
        f"{describe_selected_threshold_f1(summary)}"
    )
    report.add_paragraph(f"{describe_best_strategies(summary)} {describe_tuning_candidates(summary)}")
    report.add_figure(
        FIGURES_DIR / "h6_strateji_karsilastirma.png",
        "Model ve stratejilere göre PR-AUC, 0,5 eşiğinde F1-Score ve OOF'tan seçilen eşikte F1-Score (çubuklar: standart sapma)",
    )


def add_threshold_selection(report: WeeklyReport, summary: Summary) -> None:
    best = summary["ranking"][0]
    best_model, best_strategy = best["model"], best["strategy"]
    best_experiment = experiment(summary, best_model, best_strategy)
    f1_choice, recall_choice = best_experiment["threshold_f1"], best_experiment["threshold_recall"]
    rows = []
    for model in MODELS:
        strategy = summary["best_by_model"][model]
        entry = experiment(summary, model, strategy)
        rows.append([
            f"{MODEL_LABELS[model]} ({STRATEGY_LABELS[strategy].lower()})",
            decimal(entry["threshold_f1"]["threshold"], 2),
            mean_std_of(summary, model, strategy, "f1", "nested_summary_f1_threshold"),
            mean_std_of(summary, model, strategy, "recall", "nested_summary_f1_threshold"),
            decimal(entry["threshold_recall"]["threshold"], 2),
            mean_std_of(summary, model, strategy, "precision", "nested_summary_recall_threshold"),
        ])
    report.add_heading("4.4. Karar Eşiğinin Seçilmesi")
    report.add_table(
        "Her modelin en iyi stratejisinde out-of-fold tahminlerden seçilen eşikler ve katman sonuçları (ortalama ± standart sapma)",
        ["Model", "F1 en yüksek eşik", "F1", "Recall", f"Recall {decimal(summary['target_recall'], 2)} eşiği", "Precision (bu eşikte)"],
        rows,
        [4.6, 2.2, 2.4, 2.4, 2.2, 2.2],
    )
    report.add_paragraph(
        "Tablodaki F1, Recall ve Precision değerleri için eşik her katmanda, o katman dışındaki katmanların out-of-fold tahminlerinden "
        "seçildi ve yalnızca o katmanda ölçüldü. Böylece eşik, değerlendirildiği tahminlere bakılmadan seçilmiş oldu. Metindeki tek "
        "eşik değerleri ise bütün out-of-fold tahminlerden seçildi. Eşiğin gerçek etkisini sekizinci haftada test verisinde ölçeceğiz."
    )
    report.add_paragraph(
        f"En yüksek PR-AUC değerine sahip kombinasyonda ({MODEL_LABELS[best_model]}, {STRATEGY_LABELS[best_strategy].lower()}) F1'i en "
        f"yüksek yapan eşik {decimal(f1_choice['threshold'], 2)} oldu. Bu eşikte kabul edecek olarak işaretlenen müşteri oranı "
        f"{format_percent(best_experiment['predicted_positive_rate_at_f1_threshold'], 1)} çıktı. Recall değerini en az "
        f"{decimal(summary['target_recall'], 2)} yapan en yüksek eşik {decimal(recall_choice['threshold'], 2)} bulundu. Bu eşikte aranan "
        f"müşteri oranı {format_percent(best_experiment['predicted_positive_rate_at_recall_threshold'], 1)}, aranan müşterilerin "
        f"gerçekten kabul etme oranı (Precision) {format_percent(best_experiment['nested_summary_recall_threshold']['precision']['mean'], 1)} "
        "oldu. İki eşik arasındaki seçim, daha az arama yapmakla daha çok kabul eden müşteri yakalamak arasındaki iş kararıdır."
    )
    report.add_figure(
        FIGURES_DIR / "h6_esik_egrisi.png",
        "En yüksek PR-AUC değerine sahip kombinasyonda eşiğe göre Precision, Recall ve F1-Score (OOF tahminleri)",
    )


def add_literature(report: WeeklyReport, summary: Summary) -> None:
    ours = [
        "Bu çalışma (altıncı hafta)",
        f"{summary['model_count']} model × {summary['strategy_count']} strateji, görüşme süresi modelde yok, {CV_FOLDS} katlı CV",
        f"{decimal(summary['experiments']['xgboost__none']['summary']['roc_auc']['mean'])} ROC-AUC (XGBoost, ağırlıksız, CV ortalaması)",
    ]
    report.add_heading("4.5. Literatürle Karşılaştırma")
    report.add_table("SMOTE kullanan çalışmalar ve bu çalışmanın karşılaştırması", ["Çalışma", "Yöntem ve veri", "Bulgu"], [*LITERATURE_ROWS, ours], [3.2, 6.8, 6.0])
    report.add_paragraph(describe_literature_comparison(summary))


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
        "yanıtı doğrudan etkilemektedir. Bu hafta, modellerin kabul edecek müşterileri yakalama oranının karar eşiğiyle nasıl "
        "değiştiği incelendi. Sonuçlar aşağıdaki alt başlıklarda verilmiştir.",
    ])
    add_strategies(report)
    add_smote_leakage(report)
    add_strategy_comparison(report, summary)
    add_threshold_selection(report, summary)
    add_literature(report, summary)
    report.add_links()
    report.add_references(REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="6. hafta ilerleme raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    output_path = build_report(summary, args.kaptan).save(OUTPUT_DIR / report_file_name(WEEK))
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
