"""3. hafta ilerleme raporunu üretir.

Kullanım: python reports/hafta3_rapor.py --kaptan 230260054
Önce notebooks/build_kesifsel_analiz.py çalıştırılmalıdır. Metindeki tüm sayılar ve sayıya dayalı hükümler (anlamlılık, yön,
sıralama) hafta3_ozet.json dosyasından türetilir; yalnızca yorum cümleleri bu haftanın bulgularına göre yazılmıştır.
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

from docx.shared import Cm

from src.formatting import format_number, format_p_value, format_percent
from src.report_docx import (
    CAPTAIN_ROLE,
    TEAM,
    TEAM_PROCESS_CRITERIA,
    WeeklyReport,
    report_file_name,
)

Summary = dict[str, Any]

WEEK = 3
SIGNIFICANCE_LEVEL = 0.05
NEGLIGIBLE_CORRELATION = 0.1
CAPTAIN_DUTY = "Takım kaptanlığı ve haftalık raporun hazırlanması"
SUMMARY_PATH = ROOT / "reports" / "haftalik" / "hafta3_ozet.json"
FIGURES_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR = ROOT / "reports" / "haftalik"

# Kriterler 2. hafta raporunun "Gelecek Hafta Planı" tablosunda bu hafta için verilen görevlerden türetilmiştir.
PROGRESS_CRITERIA = [
    "Kategorik değişkenlere göre kabul oranları incelendi ve grafikleri hazırlandı",
    "Sayısal değişkenlerin dağılımları ve aykırı değerleri incelendi",
    "Korelasyon ısı haritası hazırlandı",
    "Ki-kare testi ve Cramér's V ile kategorik değişkenler incelendi",
    "Ön işleme adımları için karar taslağı hazırlandı",
    "Proje belgeleri (PRD, gereksinim analizi, yol haritası) güncellendi",
]

WORK_PACKAGES = [
    ("1", "Literatür Taraması ve Veri Seti Temini", "Tüm Ekip", "Tamamlandı"),
    ("2", "Veri Ön İşleme ve Temizleme", "Mustafa Emre KAPLAN", "Karar taslağı hazırlandı"),
    ("3", "Keşifsel Veri Analizi", "Mustafa Emre KAPLAN", "Tamamlandı"),
    ("4", "Makine Öğrenmesi Modellerinin Kurulması", "Ömer NARÇİÇEĞİ", "Başladı"),
    ("5", "Model Eğitimi ve Optimizasyon", "Ömer NARÇİÇEĞİ", "Başlamadı"),
    ("6", "Modellerin Test Edilmesi", "Mesut ASLAN", "Başlamadı"),
    ("7", "Sonuçların Değerlendirilmesi ve Proje Raporunun Hazırlanması", "Tüm Ekip", "Devam ediyor"),
]

DONE_THIS_WEEK = [
    ("Ömer NARÇİÇEĞİ", "Kategorik değişkenlerle hedef değişken arasındaki ilişkinin ki-kare testi ve Cramér's V ile incelenmesi, "
     "sayısal değişkenler için Mann-Whitney U testi, ön işleme karar taslağının hazırlanması, PRD, gereksinim analizinin ikinci "
     "sürümü ve yol haritasının yazılması"),
    ("Mustafa Emre KAPLAN", "Meslek, ay, eğitim, iletişim türü ve önceki kampanya sonucuna göre kabul oranlarının incelenmesi ve "
     "grafiklerin hazırlanması"),
    ("Mesut ASLAN", "Sayısal değişkenlerin dağılımları, balance, campaign ve previous değişkenlerinde aykırı değer analizi ve "
     "korelasyon ısı haritası"),
]

NEXT_WEEK = [
    ("Mustafa Emre KAPLAN", "Ön işleme karar taslağına göre ön işleme pipeline'ının yazılması (was_contacted_before ve yaş grubu "
     "özellikleri dahil) ve veri sızıntısını kontrol eden testler"),
    ("Ömer NARÇİÇEĞİ", "Eğitim verisi üzerinde 5 katlı çapraz doğrulamanın kurulması, ilk Lojistik Regresyon modelinin eğitilmesi "
     "ve referans modelle karşılaştırılması"),
    ("Mesut ASLAN", "Çapraz doğrulama sonuçlarını tabloya döken değerlendirme fonksiyonları, PR ve ROC eğrisi grafikleri ve "
     "bunların testleri"),
]

REFERENCES = [
    "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. "
    "Decision Support Systems, 62, 22-31.",
    "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing veri seti. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International "
    "Conference on Knowledge Discovery and Data Mining, 785-794.",
    "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. "
    "Journal of Artificial Intelligence Research, 16, 321-357.",
    "Elsalamony, H. A. (2014). Bank direct marketing analysis of data mining techniques. International Journal of Computer "
    "Applications, 85(7), 12-22.",
    "Arifah, D., Saragih, T. H., Kartini, D., Muliadi, M., & Mazdadi, M. I. (2023). Application of SMOTE to handle imbalance class "
    "in deposit classification using the extreme gradient boosting algorithm. Jurnal Ilmiah Teknik Elektro Komputer dan "
    "Informatika, 9(2), 396-410.",
    "Dong, B., Zhang, X., Liu, Y., Zhang, T., Liu, X., Hou, L., Meng, L., Guo, Z., & Mulati, A. (2026). Risk-sensitive machine "
    "learning for financial decision modeling under imbalanced data: Evidence from bank telemarketing. Entropy, 28(3), 354.",
    "Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. Proceedings of the 4th "
    "International Conference on the Practical Applications of Knowledge Discovery and Data Mining, 29-39.",
    "Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.",
    "Cochran, W. G. (1954). Some methods for strengthening the common χ² tests. Biometrics, 10(4), 417-451.",
    "Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the "
    "other. The Annals of Mathematical Statistics, 18(1), 50-60.",
    "Kerby, D. S. (2014). The simple difference formula: An approach to teaching nonparametric correlation. Comprehensive "
    "Psychology, 3, 11.IT.3.1.",
    "Spearman, C. (1904). The proof and measurement of association between two things. The American Journal of Psychology, "
    "15(1), 72-101.",
]


def rate(summary: Summary, column: str, category: str) -> str:
    return format_percent(summary["acceptance_rates"][column][category]["acceptance_rate"])


def count(summary: Summary, column: str, category: str) -> str:
    return format_number(summary["acceptance_rates"][column][category]["count"])


def decimal(value: float, digits: int = 3) -> str:
    return format_number(value, digits)


def add_introduction(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "İkinci haftada proje altyapısı kurulmuş, veri seti indirilip doğrulanmış ve her müşteriye \"kabul etmez\" diyen bir "
        "referans model oluşturulmuştu. duration değişkeninin görüşme bittikten sonra öğrenildiği için modelden çıkarılmasına da "
        "o hafta karar verilmişti.",
        "Üçüncü haftada keşifsel veri analizi yapıldı. Önceki raporda planlandığı gibi kategorik değişkenlere göre kabul oranları, "
        "sayısal değişkenlerin dağılımları ve aykırı değerleri, değişkenler arasındaki korelasyon ve kategorik değişkenlerle hedef "
        f"arasındaki ilişki incelendi. Bütün analizlerde {format_number(summary['row_count'])} kaydın tamamı kullanıldı. Çıkan "
        "sonuçlara göre dördüncü haftada yapılacak ön işleme adımları için bir karar taslağı hazırlandı. Proje belgeleri de bu "
        "hafta genişletildi.",
    ])


def add_goals(report: WeeklyReport) -> None:
    report.add_heading("1.1. Amaç ve Hedefler")
    report.add_labeled_box([
        ("Amaç:", "Bankanın hedef kitlesini doğru analiz ederek pazarlama kampanyalarının başarı oranını artıracak ve maliyetleri "
                  "düşürecek ölçülebilir bir makine öğrenmesi tahmin modeli ortaya koymaktır."),
        ("Hedefler:", "Bank Marketing veri setini temin etmek ve doğrulamak. Bu hedef ikinci haftada tamamlanmıştır. Değişkenlerin "
                      "hedefle ilişkisini keşifsel analiz ve istatistiksel testlerle incelemek. Bu hedef üçüncü haftada "
                      "tamamlanmıştır. Bulgulara göre ön işleme adımlarını kurmak. Lojistik Regresyon, Rastgele Orman ve XGBoost "
                      "modellerini çapraz doğrulama ile eğitmek ve sınıf dengesizliği için farklı yöntemleri karşılaştırmak. Veri "
                      "setindeki sınıflar dengesiz olduğu için modelleri yalnızca Accuracy ile değil; Precision, Recall, F1-Score, "
                      "ROC-AUC ve PR-AUC ile değerlendirmek. Test verisinde ROC-AUC değeri en az 0,75 olan bir model elde etmek ve "
                      "bu modeli basit bir arayüzle kullanılabilir hâle getirmek."),
    ])


def add_method(report: WeeklyReport, summary: Summary) -> None:
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin veriyi anlama aşaması üzerinde çalışıldı (Wirth ve Hipp, 2000). Önce her kategorik değişken "
        "için kategorilere göre müşteri sayısı ve kampanyayı kabul edenlerin oranı hesaplandı. Bir kategorinin oranı genel orana "
        "bölünerek lift değeri bulundu. Lift değeri 1'den büyükse o gruptaki müşteriler ortalamadan daha fazla kabul etmiş demektir.",
        "Sayısal değişkenler için ortalama, medyan, standart sapma ve çarpıklık hesaplandı. Aykırı değerleri bulmak için IQR "
        "yöntemi kullanıldı. Bu yöntemde birinci ve üçüncü çeyrek arasındaki farkın 1,5 katından daha uzakta kalan değerler aykırı "
        "sayılır. Değişkenlerin çoğu çarpık dağıldığı için korelasyon hesabında Pearson yerine sıralamaya dayanan Spearman "
        "korelasyonu tercih edildi (Spearman, 1904).",
        "Kategorik değişkenlerle hedef arasında ilişki olup olmadığına ki-kare bağımsızlık testiyle bakıldı. Ki-kare testi bir "
        "ilişkinin var olup olmadığını gösterir ama ne kadar güçlü olduğunu göstermez. Bu yüzden ilişkinin gücü Cramér's V ile "
        "ölçüldü (Cramér, 1946). V değeri 0 ile 1 arasında değişir ve 0'a yaklaştıkça ilişki zayıflar. Testin geçerli olması için "
        "beklenen frekansların 5'ten küçük olmaması gerekir. Bu koşul her tablo için ayrıca kontrol edildi (Cochran, 1954).",
        "Sayısal değişkenler normal dağılmadığı için kabul eden ve etmeyen müşteriler Mann-Whitney U testiyle karşılaştırıldı (Mann "
        "ve Whitney, 1947). Farkın büyüklüğü için rank-biserial korelasyon hesaplandı (Kerby, 2014). Veride "
        f"{format_number(summary['row_count'])} kayıt olduğu için çok küçük farklar bile istatistiksel olarak anlamlı çıkabiliyor. "
        "Bu nedenle değişkenleri karşılaştırırken p-değerinden çok etki büyüklüğüne baktık.",
        "Hesaplamalar src/eda.py dosyasında fonksiyonlar olarak yazıldı ve sonucu elle hesaplanabilen küçük örnek verilerle test "
        "edildi. Rapordaki sayılar ve grafikler notebooks/build_kesifsel_analiz.py betiğiyle üretildi.",
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
    report.add_paragraph("Dördüncü haftada ön işleme adımları kodlanacak ve ilk model eğitilecektir.")
    report.add_table("Gelecek hafta planı", ["Takım Üyesi", "Planlanan Çalışma"], [list(row) for row in NEXT_WEEK], [4.5, 11.5])


def add_categorical_findings(report: WeeklyReport, summary: Summary) -> None:
    rates = summary["acceptance_rates"]
    unknown_share = summary["contact_unknown_share_by_month"]
    other_months_max = max(share for month, share in unknown_share.items() if month not in ("may", "jun"))
    report.add_heading("4.1. Kategorik Değişkenlere Göre Kabul Oranları")
    report.add_paragraph(
        f"Veri setinde kampanyayı kabul edenlerin oranı {format_percent(summary['overall_acceptance_rate'])}. Grafiklerdeki kesikli "
        "çizgi bu oranı gösteriyor. En büyük fark önceki kampanyanın sonucunda görüldü. Önceki kampanyada teklifi kabul etmiş "
        f"{count(summary, 'poutcome', 'success')} müşteride bu kampanyayı kabul etme oranı {rate(summary, 'poutcome', 'success')}. "
        f"Bu oran genel oranın yaklaşık {decimal(rates['poutcome']['success']['lift'], 1)} katı. Daha önce hiç aranmamış "
        f"müşterilerde (poutcome = unknown) oran {rate(summary, 'poutcome', 'unknown')} olarak kaldı."
    )
    report.add_paragraph(
        f"İletişim türü unknown olan {count(summary, 'contact', 'unknown')} müşteride kabul oranı {rate(summary, 'contact', 'unknown')}, "
        f"cep telefonuyla aranan müşterilerde ise {rate(summary, 'contact', 'cellular')}. Bu unknown değerlerin aylara göre "
        "dağılımına da baktık. İletişim türü kaydedilmemiş aramaların oranı mayısta "
        f"{format_percent(unknown_share['may'])}, haziranda {format_percent(unknown_share['jun'])}, diğer aylarda ise en fazla "
        f"{format_percent(other_months_max)}. Yani bu değerler rastgele dağılmamış, iki ayda toplanmış. Bu yüzden unknown değerleri silmeyip ayrı bir kategori olarak tutmaya karar verdik. Eğitim "
        f"düzeyine bakıldığında yükseköğretim mezunlarının kabul oranı ({rate(summary, 'education', 'tertiary')}) ilköğretim "
        f"mezunlarından ({rate(summary, 'education', 'primary')}) daha yüksek."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_poutcome_contact_education.png", "Önceki kampanya sonucu, iletişim türü ve eğitim düzeyine göre kabul oranları", width=Cm(13))
    report.add_paragraph(
        "Aylara göre kabul oranları arasında büyük farklar var. Mart "
        f"({rate(summary, 'month', 'mar')}), eylül ({rate(summary, 'month', 'sep')}), ekim ({rate(summary, 'month', 'oct')}) ve "
        f"aralıkta ({rate(summary, 'month', 'dec')}) oranlar yüksek, fakat bu aylarda çok az arama yapılmış. Örneğin aralıkta "
        f"yalnızca {count(summary, 'month', 'dec')} arama var. En çok aramanın yapıldığı mayısta ({count(summary, 'month', 'may')} "
        f"arama) ise kabul oranı en düşük seviyede ({rate(summary, 'month', 'may')}). Az arama yapılan aylardaki yüksek oranlara "
        "güvenmek zor. Modelin bu aylara fazla bağlanıp bağlanmadığını çapraz doğrulamada kontrol edeceğiz."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_ay.png", "Son iletişim ayına göre kabul oranları", width=Cm(14))
    report.add_paragraph(
        f"Meslekler arasında en yüksek kabul oranı öğrencilerde ({rate(summary, 'job', 'student')}) ve emeklilerde "
        f"({rate(summary, 'job', 'retired')}), en düşük oran ise mavi yakalılarda ({rate(summary, 'job', 'blue-collar')}) görüldü. "
        f"Konut kredisi olmayan müşterilerin kabul oranı {rate(summary, 'housing', 'no')}, konut kredisi olanların "
        f"{rate(summary, 'housing', 'yes')}. Bireysel kredisi olanlarda oran {rate(summary, 'loan', 'yes')} ile genel oranın altında "
        f"kalırken bekârlarda {rate(summary, 'marital', 'single')} ile genel oranın üstünde."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_meslek.png", "Mesleğe göre kabul oranları", width=Cm(14))


def add_age_campaign_findings(report: WeeklyReport, summary: Summary) -> None:
    ages = summary["age_bin_rates"]
    campaigns = summary["campaign_bin_rates"]
    contacted = summary["previously_contacted_rates"]
    report.add_heading("4.2. Yaş, Arama Sayısı ve Önceki İletişim")
    report.add_paragraph(
        "Yaş ile kabul arasındaki ilişki doğrusal değil. 18-25 yaş grubunda kabul oranı "
        f"{format_percent(ages['18-25']['acceptance_rate'])}, 61 yaş ve üstünde {format_percent(ages['61+']['acceptance_rate'])}, "
        f"orta yaş gruplarında ise {format_percent(ages['41-50']['acceptance_rate'])} ile "
        f"{format_percent(ages['31-40']['acceptance_rate'])} arasında. Grafikte U harfine benzeyen bu durum meslek sonuçlarıyla da "
        "uyumlu, çünkü öğrenciler ve emekliler bu iki uçta yer alıyor. Lojistik Regresyon yaşı tek bir sayı olarak kullandığında "
        "bu şekli yakalayamaz. Bu yüzden dördüncü haftada yaşı gruplara ayıran bir özellik de deneyeceğiz."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_yas.png", "Yaş grubuna göre kabul oranları", width=Cm(14))
    report.add_paragraph(
        "Bu kampanyada bir müşteri ne kadar çok arandıysa kabul oranı genel olarak o kadar düştü. Bir kez aranan müşterilerde oran "
        f"{format_percent(campaigns['1']['acceptance_rate'])} iken 11 ve daha fazla aranan müşterilerde "
        f"{format_percent(campaigns['11+']['acceptance_rate'])}. Önceki bir kampanyada aranmış "
        f"{format_number(contacted['contacted_count'])} müşterinin kabul oranı {format_percent(contacted['contacted'])}, hiç "
        f"aranmamış müşterilerin ise {format_percent(contacted['not_contacted'])}."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_campaign.png", "Bu kampanyadaki arama sayısına göre kabul oranları", width=Cm(14))


def add_numeric_findings(report: WeeklyReport, summary: Summary) -> None:
    stats = summary["numeric_summary"]
    outliers = summary["iqr_outliers"]
    contacted_outliers = summary["iqr_outliers_contacted_only"]
    outlier_rates = summary["outlier_acceptance"]
    report.add_heading("4.3. Sayısal Değişkenlerin Dağılımı ve Aykırı Değerler")
    report.add_table(
        "Sayısal değişkenlerin betimsel istatistikleri",
        ["Değişken", "Ortalama", "Medyan", "Std. sapma", "En küçük", "En büyük", "Çarpıklık"],
        [
            [column, decimal(values["mean"], 2), decimal(values["50%"], 0), decimal(values["std"], 2),
             decimal(values["min"], 0), decimal(values["max"], 0), decimal(values["skewness"], 2)]
            for column, values in stats.items()
        ],
    )
    report.add_paragraph(
        f"balance (çarpıklık {decimal(stats['balance']['skewness'], 2)}), campaign ({decimal(stats['campaign']['skewness'], 2)}) ve "
        f"previous ({decimal(stats['previous']['skewness'], 2)}) değişkenleri sağa çok çarpık ve ortalamaları medyanlarının "
        "üstünde. Lojistik Regresyon için bu değişkenlere logaritmik dönüşüm uygulayacağız. Ağaç tabanlı modeller bu tür "
        "çarpıklıktan etkilenmediği için orada ham değerler kullanılabilir."
    )
    report.add_table(
        "IQR yöntemiyle aykırı değerler",
        ["Değişken", "Alt sınır", "Üst sınır", "Aykırı sayısı", "Aykırı oranı"],
        [
            [column, decimal(values["lower_bound"], 1), decimal(values["upper_bound"], 1), format_number(values["outlier_count"]),
             format_percent(values["outlier_pct"] / 100)]
            for column, values in outliers.items()
        ] + [
            [f"{column} (yalnızca aranmışlar)", decimal(values["lower_bound"], 1), decimal(values["upper_bound"], 1),
             format_number(values["outlier_count"]), format_percent(values["outlier_pct"] / 100)]
            for column, values in contacted_outliers.items()
        ],
    )
    report.add_paragraph(
        "Aykırı değerleri silip silmeyeceğimize karar vermek için önce bu kayıtların kabul oranlarına baktık. balance değişkeninde "
        f"aykırı kayıtların kabul oranı {format_percent(outlier_rates['balance']['outlier_rate'])}, diğer kayıtların "
        f"{format_percent(outlier_rates['balance']['inlier_rate'])}. campaign değişkeninde ise aykırı kayıtlarda oran "
        f"{format_percent(outlier_rates['campaign']['outlier_rate'])}, diğerlerinde {format_percent(outlier_rates['campaign']['inlier_rate'])}. "
        "İki değişkende de aykırı kayıtlar hedef hakkında bilgi taşıdığı için bunları silmemeye karar verdik. previous değişkeninde "
        "müşterilerin çoğunun değeri 0 olduğu için IQR sıfır çıkıyor ve daha önce aranmış her müşteri aykırı görünüyor "
        f"({format_percent(outliers['previous']['outlier_pct'] / 100)}). Bu yüzden bu değişkende IQR yöntemini yalnızca daha önce "
        f"aranmış müşteriler arasında uyguladık ve aykırı oranı {format_percent(contacted_outliers['previous']['outlier_pct'] / 100)} çıktı."
    )
    report.add_figure(
        FIGURES_DIR / "h3_sayisal_dagilimlar.png",
        "Kabul eden ve etmeyen müşterilerde sayısal değişkenlerin normalize dağılımları (balance ve campaign için en uç %1'lik kuyruklar görselden çıkarılmıştır)",
    )


def join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} ve {names[-1]}"


def describe_strong_pairs(pairs: list[dict[str, Any]]) -> str:
    if not pairs:
        return "Sayısal değişkenler arasında |r| ≥ 0,5 olan güçlü bir ilişki bulunmamıştır."
    described = join_names([f"{pair['first']} ile {pair['second']} arasında (r = {decimal(pair['spearman'], 3)})" for pair in pairs])
    if len(pairs) == 1:
        lead = f"Sayısal değişkenler arasındaki tek güçlü ilişki (|r| ≥ 0,5) {described}. Bu iki değişken"
    else:
        lead = f"Sayısal değişkenler arasındaki güçlü ilişkiler (|r| ≥ 0,5) {described}. Bu değişkenler"
    return (
        f"{lead} neredeyse aynı bilgiyi taşıyor. Birlikte Lojistik Regresyon'a girdiklerinde katsayıları yorumlamak "
        "zorlaşacağı için L2 düzenlileştirme kullanacağız."
    )


def add_correlation_findings(report: WeeklyReport, summary: Summary) -> None:
    ranking = summary["target_correlation_ranking"]
    top = ranking[:3]
    age = next(item for item in ranking if item["column"] == "age")
    age_text = (
        f"Yaşın hedefle korelasyonu ise {decimal(age['spearman'], 3)} ile neredeyse sıfır. Bölüm 4.2'de yaşa göre kabul oranının "
        "çok değiştiğini görmüştük. Korelasyonun bunu göstermemesinin nedeni ilişkinin U şeklinde olması. Kabul oranı hem gençlerde "
        "hem yaşlılarda arttığı için sıralamaya dayalı korelasyonda bu iki etki birbirini götürüyor."
        if abs(age["spearman"]) < NEGLIGIBLE_CORRELATION
        else f"Yaşın hedefle korelasyonu {decimal(age['spearman'], 3)}."
    )
    top_text = join_names([f"{item['column']} ({decimal(item['spearman'], 3)})" for item in top])
    report.add_heading("4.4. Korelasyon Analizi")
    report.add_paragraph(
        f"{describe_strong_pairs(summary['strong_correlations'])} Hedef değişkenle mutlak değerce en yüksek korelasyonu "
        f"gösteren değişkenler {top_text}. {age_text}"
    )
    report.add_figure(FIGURES_DIR / "h3_korelasyon.png", "Sayısal değişkenler ve hedef değişken arasındaki Spearman korelasyon matrisi", width=Cm(12))


def describe_chi_square_significance(chi_square: dict[str, Any]) -> str:
    not_significant = [column for column, values in chi_square.items() if values["p_value"] >= SIGNIFICANCE_LEVEL]
    if not not_significant:
        threshold = "p < 0,001" if all(values["p_value"] < 0.001 for values in chi_square.values()) else "p < 0,05"
        return f"Ki-kare testine göre bütün kategorik değişkenler hedefle anlamlı bir ilişki gösteriyor ({threshold})."
    return f"{join_names(not_significant)} dışındaki kategorik değişkenler hedefle anlamlı bir ilişki gösteriyor (p < 0,05)."


def describe_mann_whitney(mann_whitney: dict[str, Any]) -> str:
    significant = {column: values for column, values in mann_whitney.items() if values["p_value"] < SIGNIFICANCE_LEVEL}
    higher = [column for column, values in significant.items() if values["rank_biserial"] > 0]
    lower = [column for column, values in significant.items() if values["rank_biserial"] < 0]
    not_significant = [column for column in mann_whitney if column not in significant]
    parts = []
    if higher:
        parts.append(f"{join_names(higher)} kabul edenlerde daha yüksek")
    if lower:
        parts.append(f"{join_names(lower)} kabul edenlerde daha düşük")
    sentence = f"{', '.join(parts)} değerler alıyor." if parts else "Hiçbir sayısal değişkende anlamlı fark bulunmadı."
    if not_significant:
        details = join_names([
            f"{column} (p = {format_p_value(mann_whitney[column]['p_value'])}, r = {decimal(mann_whitney[column]['rank_biserial'], 3)})"
            for column in not_significant
        ])
        sentence += f" {details} için iki grup arasındaki fark anlamlı çıkmadı."
    return sentence


def describe_age_test(mann_whitney: dict[str, Any]) -> str:
    age = mann_whitney["age"]
    if age["p_value"] < SIGNIFICANCE_LEVEL:
        return ""
    return (
        " Yaşta fark çıkmaması Bölüm 4.2'deki U şekliyle açıklanabilir. Kabul edenler hem gençlerde hem yaşlılarda yoğunlaştığı "
        f"için iki grubun medyanları birbirine çok yakın kalıyor ({decimal(age['median_yes'], 0)} ve {decimal(age['median_no'], 0)})."
    )


def add_statistical_tests(report: WeeklyReport, summary: Summary) -> None:
    chi_square = summary["chi_square"]
    mann_whitney = summary["mann_whitney"]
    report.add_heading("4.5. İstatistiksel Testler")
    report.add_table(
        "Kategorik değişkenler için ki-kare bağımsızlık testi (Cramér's V'ye göre sıralı)",
        ["Değişken", "χ²", "Serbestlik derecesi", "p-değeri", "Cramér's V"],
        [
            [column, decimal(values["chi2"], 1), format_number(values["dof"]), format_p_value(values["p_value"]), decimal(values["cramers_v"], 3)]
            for column, values in chi_square.items()
        ],
    )
    low_expected = sum(values["low_expected_cells"] for values in chi_square.values())
    low_expected_text = (
        "Beklenen frekansı 5'in altında kalan hücre olmadığı için testin koşulu da sağlanıyor."
        if low_expected == 0
        else f"Toplam {low_expected} hücrede beklenen frekans 5'in altında kaldığı için bu değişkenlerin sonuçlarına dikkatle bakılmalı."
    )
    ordered = list(chi_square)
    strongest, second, weakest = ordered[0], ordered[1], ordered[-1]
    report.add_paragraph(
        f"{describe_chi_square_significance(chi_square)} {low_expected_text} İlişkinin ne kadar güçlü olduğunu görmek için "
        f"Cramér's V değerlerine de baktık. En güçlü ilişki {strongest} (V = {decimal(chi_square[strongest]['cramers_v'], 3)}) "
        f"ve {second} (V = {decimal(chi_square[second]['cramers_v'], 3)}) değişkenlerinde, en zayıf ilişki ise {weakest} "
        f"(V = {decimal(chi_square[weakest]['cramers_v'], 3)}) değişkeninde görüldü."
    )
    report.add_table(
        "Sayısal değişkenler için Mann-Whitney U testi (etki büyüklüğüne göre sıralı)",
        ["Değişken", "Medyan (kabul eden)", "Medyan (etmeyen)", "p-değeri", "Rank-biserial r"],
        [
            [column, decimal(values["median_yes"], 0), decimal(values["median_no"], 0), format_p_value(values["p_value"]), decimal(values["rank_biserial"], 3)]
            for column, values in mann_whitney.items()
        ],
    )
    report.add_paragraph(
        "Rank-biserial r pozitifse kabul eden müşterilerde o değişkenin değerleri daha yüksek, negatifse daha düşük demektir. "
        f"{describe_mann_whitney(mann_whitney)}{describe_age_test(mann_whitney)} Bu aşamada hiçbir değişkeni elemedik. Hangi "
        "değişkenin modele ne kadar katkı verdiğine çapraz doğrulama ve permutation importance ile bakacağız."
    )
    report.add_figure(FIGURES_DIR / "h3_etki_buyuklukleri.png", "Değişkenlerin hedefle ilişkisinin etki büyüklükleri")


def add_preprocessing_decisions(report: WeeklyReport) -> None:
    report.add_heading("4.6. Ön İşleme Karar Taslağı")
    report.add_paragraph(
        "Bu bulgulara göre dördüncü haftada uygulanacak ön işleme adımlarını belirledik. Kararların gerekçeleriyle birlikte ayrıntılı "
        "hâli proje deposundaki docs/on_isleme_kararlari.md dosyasında bulunuyor. Bütün dönüşümler yalnızca eğitim verisi üzerinde "
        "öğrenilecek, test verisine son değerlendirmeye kadar dokunulmayacak."
    )
    report.add_table(
        "Ön işleme kararlarının özeti",
        ["Konu", "Karar"],
        [
            ["unknown değerler", "Silinmez. Ayrı bir kategori olarak One-Hot Encoding ile kodlanır."],
            ["Aykırı değerler", "Silinmez. Hedefle ilgili bilgi taşıdıkları için etkileri logaritmik dönüşümle azaltılır."],
            ["pdays = -1", "was_contacted_before adında yeni bir ikili özellik eklenir. Daha önce aranmamış müşterilerde pdays 0 yapılır."],
            ["age", "Sayısal değerin yanında yaş grubu özelliği de denenir."],
            ["balance, campaign, previous", "Lojistik Regresyon için log dönüşümü ve standartlaştırma uygulanır. Ağaç modellerinde ham değer kullanılır."],
            ["İkili değişkenler", "default, housing ve loan 0/1 olarak kodlanır."],
            ["Değişken seçimi", "Bu aşamada değişken çıkarılmaz. Karar etki büyüklüklerine ve çapraz doğrulama sonuçlarına göre verilecek."],
        ],
        [4.5, 11.5],
    )


def add_documents(report: WeeklyReport) -> None:
    report.add_heading("4.7. Proje Belgelerinin Güncellenmesi")
    report.add_paragraph(
        "Bu hafta proje belgeleri de genişletildi. Ürün gereksinim belgesine (PRD) kullanıcı hikâyeleri ve ölçülebilir kabul "
        "kriterleri eklendi. Gereksinim analizinin ikinci sürümünde her gereksinim için öncelik, kabul kriteri, doğrulama yöntemi "
        "ve hedef hafta belirlendi. Bu sürüme veri sızıntısı analizi, risk listesi ve gereksinimleri kod ve testlerle eşleştiren "
        "bir izlenebilirlik tablosu da eklendi. Ayrıca her hafta için hedefleri, üyelerin görevlerini ve takım kaptanı sırasını "
        "gösteren 10 haftalık bir yol haritası hazırlandı. Belgelerin hepsi deponun docs klasöründe bulunuyor."
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
        "yanıtı doğrudan etkilemektedir. Bu hafta yapılan analizler bu ilişkilerin veride gerçekten bulunduğunu gösterdi. Kabul "
        "oranını en çok değiştiren değişkenler önceki kampanyanın sonucu, iletişim ayı, iletişim türü, yaş ve bu kampanyadaki arama "
        "sayısı oldu. Bu ilişkilerin bir kısmı doğrusal olmadığı için ağaç tabanlı modellerin bu veride daha başarılı olabileceğini "
        "düşünüyoruz. Bu haftanın sonuçları aşağıdaki alt başlıklarda verilmiştir.",
    ])
    add_categorical_findings(report, summary)
    add_age_campaign_findings(report, summary)
    add_numeric_findings(report, summary)
    add_correlation_findings(report, summary)
    add_statistical_tests(report, summary)
    add_preprocessing_decisions(report)
    add_documents(report)
    report.add_links()
    report.add_references(REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="3. hafta ilerleme raporunu .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    output_path = build_report(summary, args.kaptan).save(OUTPUT_DIR / report_file_name(WEEK))
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
