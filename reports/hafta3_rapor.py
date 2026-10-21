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

PROGRESS_CRITERIA = [
    "Kategorik değişkenlere göre kabul oranları incelendi",
    "Sayısal değişkenlerin dağılımları ve aykırı değerleri incelendi",
    "Korelasyon analizi yapıldı",
    "Ki-kare ve Mann-Whitney U testleri uygulandı",
    "Ön işleme karar taslağı hazırlandı",
    "PRD, gereksinim analizi ve yol haritası güncellendi",
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
    ("Ömer NARÇİÇEĞİ", "Kategorik değişkenler için ki-kare/Cramér's V ve sayısal değişkenler için Mann-Whitney U testleri; ön işleme "
     "karar taslağı; PRD, gereksinim analizi (2. sürüm) ve 10 haftalık yol haritası"),
    ("Mustafa Emre KAPLAN", "Meslek, ay, eğitim, iletişim türü ve önceki kampanya sonucuna göre kabul oranlarının incelenmesi, "
     "grafiklerin yorumlanması"),
    ("Mesut ASLAN", "Sayısal değişkenlerin dağılımları, balance, campaign ve previous değişkenlerinde IQR ile aykırı değer analizi, "
     "Spearman korelasyon ısı haritasının yorumlanması"),
]

NEXT_WEEK = [
    ("Mustafa Emre KAPLAN", "Ön işleme karar taslağına göre ColumnTransformer tabanlı ön işleme pipeline'ının yazılması "
     "(was_contacted_before ve yaş grubu özellikleri dahil) ve veri sızıntısını kontrol eden birim testleri"),
    ("Ömer NARÇİÇEĞİ", "Eğitim verisi üzerinde 5 katlı stratified çapraz doğrulama düzeninin kurulması ve ilk Lojistik Regresyon "
     "modelinin eğitilip referans modelle karşılaştırılması"),
    ("Mesut ASLAN", "Çapraz doğrulama sonuçlarını tablo hâlinde toplayan değerlendirme fonksiyonları ile PR ve ROC eğrisi "
     "grafiklerinin hazırlanması ve testlerinin yazılması"),
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
        "İkinci haftada proje altyapısı kurulmuş, Bank Marketing veri seti doğrulanmış, unknown değerlerin ve pdays değişkeninin "
        "anlamı ortaya konmuş, duration değişkeni veri sızıntısı nedeniyle modelden çıkarılmış ve her müşteriye \"kabul etmez\" "
        "diyen bir referans model kurulmuştu.",
        "Üçüncü haftada CRISP-DM sürecinin \"veriyi anlama\" aşaması tamamlandı. Her değişkenin kampanyayı kabul etme oranıyla "
        "ilişkisi incelendi, sayısal değişkenlerin dağılımları ve aykırı değerleri analiz edildi, değişkenler arası korelasyonlara "
        f"bakıldı ve bu ilişkilerin anlamlılığı ile büyüklüğü istatistiksel testlerle ölçüldü. Analizler {format_number(summary['row_count'])} "
        "kaydın tamamı üzerinde yapıldı. Bulgulara dayanarak dördüncü haftada uygulanacak ön işleme adımlarının karar taslağı "
        "hazırlandı. Ayrıca proje belgeleri genişletilerek ürün gereksinim belgesi (PRD), gereksinim analizinin ikinci sürümü ve "
        "10 haftalık yol haritası oluşturuldu.",
    ])


def add_goals(report: WeeklyReport) -> None:
    report.add_heading("1.1. Amaç ve Hedefler")
    report.add_labeled_box([
        ("Amaç:", "Bankanın hedef kitlesini doğru analiz ederek pazarlama kampanyalarının başarı oranını artıracak ve maliyetleri "
                  "düşürecek ölçülebilir bir makine öğrenmesi tahmin modeli ortaya koymaktır."),
        ("Hedefler:", "Bank Marketing veri setini temin etmek ve doğrulamak (2. hafta, tamamlandı). Değişkenlerin hedefle ilişkisini "
                      "keşifsel analiz ve istatistiksel testlerle ortaya koymak (3. hafta, tamamlandı). Bulgulara dayalı ön işleme "
                      "pipeline'ını kurmak (4. hafta). Lojistik Regresyon, Rastgele Orman ve XGBoost modellerini çapraz doğrulama "
                      "ile eğitip sınıf dengesizliği yöntemlerini karşılaştırmak (5-7. haftalar). Seçilen modelin test verisinde "
                      "ROC-AUC değerinin en az 0,75 olması ve PR-AUC değerinin referans modelin belirgin şekilde üzerine çıkması "
                      "(8. hafta). Modeli kaydedip basit bir tahmin arayüzüyle kullanılabilir hâle getirmek (9. hafta)."),
    ])


def add_method(report: WeeklyReport) -> None:
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Bu hafta CRISP-DM sürecinin veriyi anlama aşaması yürütülmüştür (Wirth ve Hipp, 2000). Her kategorik değişken için "
        "kategori bazında kayıt sayısı, kampanyayı kabul eden müşteri sayısı ve kabul oranı hesaplanmıştır. Ayrıca kategorinin kabul "
        "oranının genel kabul oranına bölünmesiyle bulunan lift değeri hesaplanmıştır. Lift değerinin 1'den büyük olması, o "
        "kategorideki müşterilerin ortalamadan daha istekli olduğunu gösterir.",
        "Sayısal değişkenler için betimsel istatistikler ve çarpıklık katsayısı hesaplanmıştır. Aykırı değerler, birinci ve üçüncü "
        "çeyreklik arasındaki farkın (IQR) 1,5 katı dışında kalan değerler olarak belirlenmiştir. Değişkenlerin önemli bir kısmı "
        "çarpık dağıldığı için doğrusal ilişki varsayan Pearson yerine sıralamaya dayalı Spearman korelasyonu kullanılmıştır "
        "(Spearman, 1904).",
        "Kategorik değişkenlerle hedef arasındaki ilişki ki-kare bağımsızlık testiyle sınanmış ve ilişkinin gücü Cramér's V ile "
        "ölçülmüştür (Cramér, 1946). Testin güvenilirliği için beklenen frekansların 5'in altına düşüp düşmediği kontrol edilmiştir "
        "(Cochran, 1954). Sayısal değişkenler normal dağılmadığı için kabul eden ve etmeyen gruplar parametrik olmayan Mann-Whitney U "
        "testiyle karşılaştırılmıştır (Mann ve Whitney, 1947). Etki büyüklüğü olarak rank-biserial korelasyon kullanılmıştır "
        "(Kerby, 2014). Örneklem çok büyük olduğunda çok küçük farklar bile anlamlı çıkar. Bu nedenle değişkenler p-değerine göre "
        "değil, etki büyüklüğüne göre değerlendirilmiştir.",
        "Hesaplamalar src/eda.py modülünde fonksiyonlar hâlinde yazılmış ve elle doğrulanabilen küçük örnek verilerle birim "
        "testleriyle kontrol edilmiştir. Rapordaki tüm sayılar notebooks/build_kesifsel_analiz.py betiğinin ürettiği hafta3_ozet.json "
        "dosyasından alınmıştır. Betik tekrar çalıştırıldığında aynı sonuçlar elde edilir.",
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
        [name, f"{CAPTAIN_DUTY}; {work[0].lower()}{work[1:]}" if name == report.captain.full_name else work]
        for name, work in DONE_THIS_WEEK
    ]
    report.add_table("Bu hafta yapılan çalışmalar", ["Takım Üyesi", "Yapılan Çalışma"], done_rows, [4.5, 11.5])
    report.add_heading("3.4. Gelecek Hafta Planı")
    report.add_paragraph("Dördüncü haftada ön işleme pipeline'ı kurulacak ve ilk makine öğrenmesi modeli çapraz doğrulama ile eğitilecektir.")
    report.add_table("Gelecek hafta planı", ["Takım Üyesi", "Planlanan Çalışma"], [list(row) for row in NEXT_WEEK], [4.5, 11.5])


def add_categorical_findings(report: WeeklyReport, summary: Summary) -> None:
    rates = summary["acceptance_rates"]
    report.add_heading("4.1. Kategorik Değişkenlere Göre Kabul Oranları")
    report.add_paragraph(
        f"Veri setindeki genel kabul oranı {format_percent(summary['overall_acceptance_rate'])} olup grafiklerde kesikli çizgiyle "
        f"gösterilmiştir. En belirgin fark önceki kampanyanın sonucunda görülmektedir. Önceki kampanyayı kabul etmiş "
        f"({count(summary, 'poutcome', 'success')} müşteri) müşterilerde bu kampanyayı kabul etme oranı "
        f"{rate(summary, 'poutcome', 'success')} olmuştur. Bu oran genel oranın yaklaşık {decimal(rates['poutcome']['success']['lift'], 1)} katıdır. Daha önce hiç "
        f"aranmamış müşterilerde (unknown) oran {rate(summary, 'poutcome', 'unknown')} düzeyindedir."
    )
    report.add_paragraph(
        f"İletişim türü unknown olan {count(summary, 'contact', 'unknown')} müşteride kabul oranı yalnızca "
        f"{rate(summary, 'contact', 'unknown')} olarak ölçülmüştür. Cep telefonuyla arananlarda bu oran {rate(summary, 'contact', 'cellular')} düzeyindedir. "
        f"unknown değerlerin aylara dağılımına bakıldığında iletişim türü kaydedilmemiş aramaların payı mayıs ayında "
        f"{format_percent(summary['contact_unknown_share_by_month']['may'])}, haziran ayında "
        f"{format_percent(summary['contact_unknown_share_by_month']['jun'])} olarak bulunmuştur. "
        "Bu durum, unknown değerinin rastgele bir eksiklik olmadığını, veri toplamanın belirli bir dönemine ait olduğunu "
        "göstermektedir. Bu nedenle unknown değerler silinmeyecek, ayrı bir kategori olarak modelde tutulacaktır. Eğitim düzeyinde "
        f"yükseköğretim mezunlarının kabul oranı ({rate(summary, 'education', 'tertiary')}) ilköğretim mezunlarınınkinden "
        f"({rate(summary, 'education', 'primary')}) belirgin şekilde yüksektir."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_poutcome_contact_education.png", "Önceki kampanya sonucu, iletişim türü ve eğitim düzeyine göre kabul oranları", width=Cm(13))
    report.add_paragraph(
        f"Aylara göre kabul oranları çok farklıdır. Mart ({rate(summary, 'month', 'mar')}), eylül ({rate(summary, 'month', 'sep')}), "
        f"ekim ({rate(summary, 'month', 'oct')}) ve aralıkta ({rate(summary, 'month', 'dec')}) kabul oranı yüksektir. Ancak bu aylarda "
        f"yapılan arama sayısı azdır (aralıkta yalnızca {count(summary, 'month', 'dec')} arama). En çok aramanın yapıldığı mayıs ayında "
        f"({count(summary, 'month', 'may')} arama) kabul oranı en düşük seviyededir ({rate(summary, 'month', 'may')}). Az gözlemli "
        "ayların yüksek oranları modelin bu ayları ezberlemesine yol açabileceği için etkileri çapraz doğrulamada izlenecektir."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_ay.png", "Son iletişim ayına göre kabul oranları", width=Cm(14))
    report.add_paragraph(
        f"Meslekler arasında öğrenciler ({rate(summary, 'job', 'student')}) ve emekliler ({rate(summary, 'job', 'retired')}) en "
        f"yüksek kabul oranına sahiptir. Mavi yakalı çalışanlarda ise oran {rate(summary, 'job', 'blue-collar')} düzeyindedir. Konut kredisi "
        f"olmayanların kabul oranı {rate(summary, 'housing', 'no')}, olanların {rate(summary, 'housing', 'yes')} olarak ölçülmüştür. Bireysel kredisi "
        f"olanlarda ({rate(summary, 'loan', 'yes')}) ve bekârlarda ({rate(summary, 'marital', 'single')}) da genel orandan "
        "belirgin farklar görülmektedir."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_meslek.png", "Mesleğe göre kabul oranları", width=Cm(14))


def add_age_campaign_findings(report: WeeklyReport, summary: Summary) -> None:
    ages = summary["age_bin_rates"]
    campaigns = summary["campaign_bin_rates"]
    contacted = summary["previously_contacted_rates"]
    report.add_heading("4.2. Yaş, Arama Sayısı ve Önceki İletişim")
    report.add_paragraph(
        f"Yaş ile kabul arasındaki ilişki doğrusal değildir. 18-25 yaş grubunda kabul oranı {format_percent(ages['18-25']['acceptance_rate'])}, "
        f"61 yaş ve üzerinde {format_percent(ages['61+']['acceptance_rate'])} iken orta yaş gruplarında oran "
        f"{format_percent(ages['41-50']['acceptance_rate'])} ile {format_percent(ages['31-40']['acceptance_rate'])} arasında kalmaktadır. "
        "U şeklindeki bu ilişki meslek bulgusuyla da tutarlıdır (öğrenciler ve emekliler). Doğrusal bir model yaşı tek bir sayısal "
        "değişken olarak kullandığında bu ilişkiyi yakalayamaz. Bu nedenle dördüncü haftada yaş grubu özelliği de denenecektir."
    )
    report.add_figure(FIGURES_DIR / "h3_kabul_yas.png", "Yaş grubuna göre kabul oranları", width=Cm(14))
    report.add_paragraph(
        f"Bu kampanyada müşteriyle yapılan arama sayısı arttıkça kabul oranı düzenli olarak düşmektedir. Tek aramada oran "
        f"{format_percent(campaigns['1']['acceptance_rate'])} iken 11 ve üzeri aramada "
        f"{format_percent(campaigns['11+']['acceptance_rate'])} seviyesine düşmektedir. Önceki bir kampanyada aranmış {format_number(contacted['contacted_count'])} müşterinin kabul oranı "
        f"{format_percent(contacted['contacted'])}, hiç aranmamış müşterilerin ise {format_percent(contacted['not_contacted'])} olarak bulunmuştur."
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
        f"previous ({decimal(stats['previous']['skewness'], 2)}) değişkenleri çok sağa çarpıktır. Ortalamaları medyanlarının çok üzerindedir. "
        "Bu değişkenler Lojistik Regresyon için logaritmik dönüşümle yumuşatılacaktır. Ağaç tabanlı modeller bu dönüşüme ihtiyaç "
        "duymaz."
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
        "Aykırı değerlerin silinip silinmeyeceğine karar vermek için aykırı kayıtların kabul oranlarına bakılmıştır. balance "
        f"değişkeninde aykırı kayıtların kabul oranı {format_percent(outlier_rates['balance']['outlier_rate'])}, diğer kayıtlarınki "
        f"{format_percent(outlier_rates['balance']['inlier_rate'])} olarak bulunmuştur. campaign değişkeninde ise aykırı kayıtlarda "
        f"{format_percent(outlier_rates['campaign']['outlier_rate'])}, diğerlerinde {format_percent(outlier_rates['campaign']['inlier_rate'])} düzeyindedir. "
        "Aykırı değerler hedefle ilgili bilgi taşıdığı için silinmeyecektir. previous değişkeninde müşterilerin çoğunluğu 0 değerini "
        f"aldığından IQR sıfırdır ve önceden aranmış her müşteri aykırı görünmektedir ({format_percent(outliers['previous']['outlier_pct'] / 100)}). "
        "Bu yüzden IQR yöntemi bu değişkene yalnızca aranmış müşteriler içinde uygulanmış, oran "
        f"{format_percent(contacted_outliers['previous']['outlier_pct'] / 100)} olarak bulunmuştur."
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
        return "Sayısal değişkenler arasında |r| ≥ 0,5 düzeyinde güçlü bir ilişki bulunmamıştır."
    described = join_names([f"{pair['first']} ile {pair['second']} arasında (r = {decimal(pair['spearman'], 3)})" for pair in pairs])
    lead = "tek güçlü ilişki" if len(pairs) == 1 else "güçlü ilişkiler"
    return (
        f"Sayısal değişkenler arasındaki {lead} (|r| ≥ 0,5) {described} görülmektedir. Güçlü ilişkili "
        "değişkenler büyük ölçüde aynı bilgiyi taşıdığından Lojistik Regresyon'da çoklu doğrusallık katsayıların yorumunu "
        "zorlaştırır. Bu nedenle L2 düzenlileştirme kullanılacak ve katsayılar bu bilgiyle yorumlanacaktır."
    )


def add_correlation_findings(report: WeeklyReport, summary: Summary) -> None:
    ranking = summary["target_correlation_ranking"]
    top = ranking[:3]
    age = next(item for item in ranking if item["column"] == "age")
    age_text = (
        f"Yaşın hedefle korelasyonu ({decimal(age['spearman'], 3)}) neredeyse sıfırdır. Bu, yaşın önemsiz olduğu anlamına gelmez. "
        "Bölüm 4.2'de görüldüğü gibi ilişki monoton değil U şeklindedir ve sıralama korelasyonu bu tür ilişkileri ölçemez."
        if abs(age["spearman"]) < NEGLIGIBLE_CORRELATION
        else f"Yaşın hedefle korelasyonu {decimal(age['spearman'], 3)} olarak ölçülmüştür."
    )
    top_text = join_names([f"{item['column']} ({decimal(item['spearman'], 3)})" for item in top])
    report.add_heading("4.4. Korelasyon Analizi")
    report.add_paragraph(
        f"{describe_strong_pairs(summary['strong_correlations'])} Hedef değişkenle mutlak değerce en yüksek korelasyonu "
        f"{top_text} göstermektedir. {age_text}"
    )
    report.add_figure(FIGURES_DIR / "h3_korelasyon.png", "Sayısal değişkenler ve hedef değişken arasındaki Spearman korelasyon matrisi", width=Cm(12))


def describe_chi_square_significance(chi_square: dict[str, Any]) -> str:
    not_significant = [column for column, values in chi_square.items() if values["p_value"] >= SIGNIFICANCE_LEVEL]
    if not not_significant:
        threshold = "p < 0,001" if all(values["p_value"] < 0.001 for values in chi_square.values()) else "p < 0,05"
        return f"Tüm kategorik değişkenler hedefle istatistiksel olarak anlamlı bir ilişki göstermektedir ({threshold})."
    return (
        f"{join_names(not_significant)} dışındaki kategorik değişkenler hedefle istatistiksel olarak anlamlı bir ilişki "
        "göstermektedir (p < 0,05)."
    )


def describe_mann_whitney(mann_whitney: dict[str, Any]) -> str:
    significant = {column: values for column, values in mann_whitney.items() if values["p_value"] < SIGNIFICANCE_LEVEL}
    higher = [column for column, values in significant.items() if values["rank_biserial"] > 0]
    lower = [column for column, values in significant.items() if values["rank_biserial"] < 0]
    not_significant = [column for column in mann_whitney if column not in significant]
    parts = []
    if higher:
        parts.append(f"{join_names(higher)} kabul edenlerde daha yüksek")
    if lower:
        parts.append(f"{join_names(lower)} kabul edenlerde daha düşük değerler almaktadır")
    sentence = f"{', '.join(parts)}." if parts else "Hiçbir sayısal değişkende anlamlı fark bulunmamıştır."
    if not_significant:
        details = join_names([
            f"{column} (p = {format_p_value(mann_whitney[column]['p_value'])}, r = {decimal(mann_whitney[column]['rank_biserial'], 3)})"
            for column in not_significant
        ])
        sentence += (
            f" {details} için fark anlamlı değildir. Ancak bu sonuç ilgili değişkenin kabulle ilişkisiz olduğu anlamına gelmez. "
            "Mann-Whitney testi grupların sıralamasındaki kaymayı ölçer; yaşta görülen ve iki uçta yükselen U şeklindeki "
            "ilişkiyi göremez."
        )
    return sentence


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
        "Hiçbir tabloda beklenen frekansı 5'in altında olan hücre bulunmadığından test varsayımları sağlanmaktadır."
        if low_expected == 0
        else f"Toplam {low_expected} hücrede beklenen frekans 5'in altındadır; bu değişkenlerin sonuçları dikkatle yorumlanmalıdır."
    )
    ordered = list(chi_square)
    report.add_paragraph(
        f"{describe_chi_square_significance(chi_square)} {low_expected_text} "
        f"İlişkinin gücüne bakıldığında en güçlü değişkenler {ordered[0]} (V = {decimal(chi_square[ordered[0]]['cramers_v'], 3)}) ve "
        f"{ordered[1]} (V = {decimal(chi_square[ordered[1]]['cramers_v'], 3)}), en zayıf değişken ise {ordered[-1]} "
        f"(V = {decimal(chi_square[ordered[-1]]['cramers_v'], 3)}) olmuştur."
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
        "Rank-biserial r değerinin pozitif olması, kabul eden müşterilerde o değişkenin daha yüksek değerler aldığını gösterir. "
        f"{describe_mann_whitney(mann_whitney)} Bu nedenle hiçbir değişken bu aşamada elenmemiştir. Değişkenlerin modele katkısı "
        "çapraz doğrulama ve permutation importance ile değerlendirilecektir."
    )
    report.add_figure(FIGURES_DIR / "h3_etki_buyuklukleri.png", "Değişkenlerin hedefle ilişkisinin etki büyüklükleri")


def add_preprocessing_decisions(report: WeeklyReport) -> None:
    report.add_heading("4.6. Ön İşleme Karar Taslağı")
    report.add_paragraph(
        "Bulgulara dayanarak dördüncü haftada uygulanacak ön işleme adımları belirlenmiştir. Taslağın ayrıntıları ve gerekçeleri "
        "proje deposundaki docs/on_isleme_kararlari.md dosyasındadır. Tüm dönüşümler yalnızca eğitim verisine uyarlanacak, test "
        "verisi son değerlendirmeye kadar kullanılmayacaktır."
    )
    report.add_table(
        "Ön işleme kararlarının özeti",
        ["Konu", "Karar"],
        [
            ["unknown değerler", "Silinmez; ayrı kategori olarak One-Hot Encoding ile kodlanır."],
            ["Aykırı değerler", "Silinmez; hedefle ilgili bilgi taşıdıkları için logaritmik dönüşümle etkileri yumuşatılır."],
            ["pdays = -1", "Yeni ikili özellik was_contacted_before eklenir; aranmamışlar için pdays 0'a çekilir."],
            ["age", "Sayısal değerin yanında yaş grubu özelliği denenir (U şeklindeki ilişki için)."],
            ["balance, campaign, previous", "Lojistik Regresyon için log dönüşümü ve standartlaştırma; ağaç modellerinde ham değer."],
            ["İkili değişkenler", "default, housing, loan 0/1 olarak kodlanır."],
            ["Değişken seçimi", "Bu aşamada eleme yapılmaz; karar etki büyüklüğü ve çapraz doğrulama sonuçlarına göre verilir."],
        ],
        [4.5, 11.5],
    )


def add_documents(report: WeeklyReport) -> None:
    report.add_heading("4.7. Proje Belgelerinin Güncellenmesi")
    report.add_paragraph(
        "Bu hafta proje belgeleri genişletilmiştir. Ürün gereksinim belgesinde (PRD) problem, çözüm, kullanıcı hikâyeleri, uygulama ve "
        "test kararları ile ölçülebilir kabul kriterleri tanımlanmıştır. Gereksinim analizinin ikinci sürümünde her gereksinime "
        "öncelik (MoSCoW), kabul kriteri, doğrulama yöntemi ve hedef hafta eklenmiş, veri sızıntısı analizi, risk kaydı ve "
        "gereksinim, modül, test ve hafta arasındaki izlenebilirlik matrisi oluşturulmuştur. 10 haftalık yol haritasında her hafta "
        "için hedefler, üye bazında görevler, teslim edilecek çıktılar ve takım kaptanı rotasyonu belirlenmiştir. Belgeler proje "
        "deposunun docs klasöründedir."
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
    add_method(report)
    add_project_management(report)
    report.add_heading("4. TEORİ VE SONUÇLAR")
    report.add_box([
        "Müşterilerin durumları, sosyoekonomik özellikleri ve geçmiş bankacılık alışkanlıkları pazarlama kampanyalarına verecekleri "
        "yanıtı doğrudan etkilemektedir. Bu hafta yapılan keşifsel analiz bu varsayımı sayılarla desteklemiştir. Önceki kampanya "
        "sonucu, iletişim ayı ve türü, yaş grubu ve arama sayısı kabul oranıyla güçlü ilişkiler göstermektedir. Bazı ilişkilerin "
        "doğrusal olmaması, ağaç tabanlı modellerin bu veride avantajlı olabileceğine işaret etmektedir. Bu haftanın sonuçları "
        "aşağıdaki alt başlıklarda verilmiştir.",
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
