"""3. hafta ilerleme raporunun kısa sürümünü üretir.

Kullanım: python reports/hafta3_rapor_kisa.py --kaptan 230260054
Tablolar, şekiller, form ve kararlar uzun sürümle aynıdır; yalnızca anlatım kısaltılmıştır. Sayılar ve sayıya dayalı hükümler
yine hafta3_ozet.json dosyasından türetilir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.hafta3_rapor import (
    NEGLIGIBLE_CORRELATION,
    OUTPUT_DIR,
    PROGRESS_CRITERIA,
    REFERENCES,
    SUMMARY_PATH,
    WEEK,
    Summary,
    add_chi_square_table,
    add_descriptive_table,
    add_mann_whitney_table,
    add_named_figure,
    add_outlier_table,
    add_preprocessing_table,
    add_project_management,
    count,
    decimal,
    describe_chi_square_significance,
    describe_mann_whitney,
    describe_strong_pairs,
    rate,
)
from src.formatting import format_percent
from src.report_docx import TEAM_PROCESS_CRITERIA, WeeklyReport, report_file_name

# Kısa sürümün yöntem bölümünde atıf yapılmayan kaynaklar kaynakçadan da çıkarılır.
UNCITED_IN_SHORT_VERSION = ("Wirth", "Cochran", "Kerby")
SHORT_REFERENCES = [reference for reference in REFERENCES if not reference.startswith(UNCITED_IN_SHORT_VERSION)]


def short_file_name() -> str:
    return report_file_name(WEEK).replace(".docx", "_kisa.docx")


def add_introduction(report: WeeklyReport) -> None:
    report.add_heading("1. GİRİŞ")
    report.add_box([
        "İkinci haftada proje altyapısı kurulmuş, veri seti doğrulanmış ve referans model oluşturulmuştu. Bu hafta planlandığı gibi "
        "keşifsel veri analizi yapıldı ve sonuçlara göre ön işleme için bir karar taslağı hazırlandı. Proje belgeleri de güncellendi.",
    ])


def add_goals(report: WeeklyReport) -> None:
    report.add_heading("1.1. Amaç ve Hedefler")
    report.add_labeled_box([
        ("Amaç:", "Bankanın hedef kitlesini doğru analiz ederek pazarlama kampanyalarının başarı oranını artıracak ve maliyetleri "
                  "düşürecek ölçülebilir bir makine öğrenmesi tahmin modeli ortaya koymaktır."),
        ("Hedefler:", "Veri setini doğrulamak (2. hafta, tamamlandı) ve değişkenlerin hedefle ilişkisini incelemek (3. hafta, "
                      "tamamlandı). Sonraki haftalarda ön işleme adımlarını kurmak, Lojistik Regresyon, Rastgele Orman ve XGBoost "
                      "modellerini çapraz doğrulama ile eğitmek ve test verisinde ROC-AUC değeri en az 0,75 olan bir modele ulaşmak."),
    ])


def add_method(report: WeeklyReport) -> None:
    report.add_heading("2. YÖNTEM")
    report.add_box([
        "Kategorik değişkenler için her kategorinin kabul oranı ve lift değeri hesaplandı. Sayısal değişkenlerde betimsel "
        "istatistiklere ve çarpıklığa bakıldı, aykırı değerler IQR yöntemiyle bulundu. Dağılımlar çarpık olduğu için Spearman "
        "korelasyonu kullanıldı (Spearman, 1904).",
        "Kategorik değişkenlerle hedef arasındaki ilişki ki-kare testiyle sınandı ve ilişkinin gücü Cramér's V ile ölçüldü (Cramér, "
        "1946). Sayısal değişkenlerde kabul eden ve etmeyen müşteriler Mann-Whitney U testiyle karşılaştırıldı (Mann ve Whitney, "
        "1947), farkın büyüklüğü için rank-biserial korelasyon kullanıldı. Veri büyük olduğu için küçük farklar bile anlamlı "
        "çıkabildiğinden p-değerinden çok etki büyüklüğüne baktık.",
    ])


def add_categorical_findings(report: WeeklyReport, summary: Summary) -> None:
    unknown_share = summary["contact_unknown_share_by_month"]
    other_months_max = max(share for month, share in unknown_share.items() if month not in ("may", "jun"))
    report.add_heading("4.1. Kategorik Değişkenlere Göre Kabul Oranları")
    report.add_paragraph(
        f"Genel kabul oranı {format_percent(summary['overall_acceptance_rate'])} ve grafiklerde kesikli çizgiyle gösteriliyor. "
        f"Önceki kampanyayı kabul etmiş müşterilerde oran {rate(summary, 'poutcome', 'success')}, bu da genel oranın yaklaşık "
        f"{decimal(summary['acceptance_rates']['poutcome']['success']['lift'], 1)} katı. İletişim türü unknown olan müşterilerde oran "
        f"{rate(summary, 'contact', 'unknown')}. İletişim türü kaydedilmemiş aramaların oranı mayısta "
        f"{format_percent(unknown_share['may'])}, haziranda {format_percent(unknown_share['jun'])}, diğer aylarda en fazla "
        f"{format_percent(other_months_max)} olduğu için unknown değerleri silmeyip ayrı kategori olarak tutacağız."
    )
    add_named_figure(report, "poutcome_contact_education")
    report.add_paragraph(
        f"Mart ({rate(summary, 'month', 'mar')}), eylül ({rate(summary, 'month', 'sep')}), ekim ({rate(summary, 'month', 'oct')}) "
        f"ve aralıkta ({rate(summary, 'month', 'dec')}) kabul oranı yüksek ama bu aylarda az arama yapılmış. En çok aramanın "
        f"yapıldığı mayısta ({count(summary, 'month', 'may')} arama) oran {rate(summary, 'month', 'may')}."
    )
    add_named_figure(report, "month")
    report.add_paragraph(
        f"Meslekler arasında en yüksek oran öğrencilerde ({rate(summary, 'job', 'student')}) ve emeklilerde "
        f"({rate(summary, 'job', 'retired')}), en düşük oran mavi yakalılarda ({rate(summary, 'job', 'blue-collar')})."
    )
    add_named_figure(report, "job")


def add_age_campaign_findings(report: WeeklyReport, summary: Summary) -> None:
    ages = summary["age_bin_rates"]
    campaigns = summary["campaign_bin_rates"]
    report.add_heading("4.2. Yaş, Arama Sayısı ve Önceki İletişim")
    report.add_paragraph(
        f"Kabul oranı 18-25 yaş grubunda {format_percent(ages['18-25']['acceptance_rate'])}, 61 yaş ve üstünde "
        f"{format_percent(ages['61+']['acceptance_rate'])}, orta yaşlarda ise {format_percent(ages['41-50']['acceptance_rate'])} ile "
        f"{format_percent(ages['31-40']['acceptance_rate'])} arasında. İlişki doğrusal olmadığı için dördüncü haftada yaş grubu "
        "özelliği de denenecek."
    )
    add_named_figure(report, "age")
    report.add_paragraph(
        f"Arama sayısı arttıkça kabul oranı düşüyor. Bir kez aranan müşterilerde oran {format_percent(campaigns['1']['acceptance_rate'])}, "
        f"11 ve daha fazla aranan müşterilerde {format_percent(campaigns['11+']['acceptance_rate'])}."
    )
    add_named_figure(report, "campaign")


def add_numeric_findings(report: WeeklyReport, summary: Summary) -> None:
    stats = summary["numeric_summary"]
    outlier_rates = summary["outlier_acceptance"]["balance"]
    report.add_heading("4.3. Sayısal Değişkenlerin Dağılımı ve Aykırı Değerler")
    add_descriptive_table(report, summary)
    add_outlier_table(report, summary)
    report.add_paragraph(
        f"balance ({decimal(stats['balance']['skewness'], 2)}), campaign ({decimal(stats['campaign']['skewness'], 2)}) ve previous "
        f"({decimal(stats['previous']['skewness'], 2)}) sağa çok çarpık, bu yüzden Lojistik Regresyon için log dönüşümü "
        "uygulanacak. Aykırı kayıtların kabul oranı diğerlerinden farklı olduğu için (balance değişkeninde "
        f"{format_percent(outlier_rates['outlier_rate'])} ve {format_percent(outlier_rates['inlier_rate'])}) aykırı değerler "
        "silinmeyecek. previous değişkeninde IQR sıfır çıktığı için yöntem yalnızca daha önce aranmış müşterilere uygulandı."
    )
    add_named_figure(report, "distributions")


def add_correlation_findings(report: WeeklyReport, summary: Summary) -> None:
    age = next(item for item in summary["target_correlation_ranking"] if item["column"] == "age")
    age_text = (
        f" Yaşın hedefle korelasyonu neredeyse sıfır ({decimal(age['spearman'], 3)}), çünkü ilişki doğrusal değil, U şeklinde."
        if abs(age["spearman"]) < NEGLIGIBLE_CORRELATION
        else ""
    )
    report.add_heading("4.4. Korelasyon Analizi")
    report.add_paragraph(f"{describe_strong_pairs(summary['strong_correlations'])}{age_text}")
    add_named_figure(report, "correlation")


def add_statistical_tests(report: WeeklyReport, summary: Summary) -> None:
    chi_square = summary["chi_square"]
    ordered = list(chi_square)
    mann_whitney_text = describe_mann_whitney(summary["mann_whitney"])
    report.add_heading("4.5. İstatistiksel Testler")
    add_chi_square_table(report, summary)
    add_mann_whitney_table(report, summary)
    report.add_paragraph(
        f"{describe_chi_square_significance(chi_square)} En güçlü ilişki {ordered[0]} "
        f"(V = {decimal(chi_square[ordered[0]]['cramers_v'], 3)}) ve {ordered[1]} (V = {decimal(chi_square[ordered[1]]['cramers_v'], 3)}) "
        f"değişkenlerinde görüldü. Mann-Whitney U testine göre {mann_whitney_text[0].lower()}{mann_whitney_text[1:]} Bu aşamada "
        "hiçbir değişken elenmedi."
    )
    add_named_figure(report, "effect_sizes")


def add_preprocessing_decisions(report: WeeklyReport) -> None:
    report.add_heading("4.6. Ön İşleme Karar Taslağı")
    report.add_paragraph(
        "Ön işleme için alınan kararlar aşağıdaki tabloda özetlenmiştir. Ayrıntılar depodaki docs/on_isleme_kararlari.md dosyasında."
    )
    add_preprocessing_table(report)


def add_documents(report: WeeklyReport) -> None:
    report.add_heading("4.7. Proje Belgelerinin Güncellenmesi")
    report.add_paragraph(
        "PRD, gereksinim analizinin ikinci sürümü ve 10 haftalık yol haritası hazırlandı. Belgeler deponun docs klasöründe."
    )


def build_short_report(summary: Summary, captain_id: str) -> WeeklyReport:
    report = WeeklyReport(week=WEEK, captain_id=captain_id)
    # Ekip süreci işaretleri taslaktır; kaptan teslimden önce gerçek duruma göre değiştirmelidir.
    report.add_captain_form(
        team_process=[(criterion, True) for criterion in TEAM_PROCESS_CRITERIA],
        progress=[(criterion, True) for criterion in PROGRESS_CRITERIA],
    )
    report.add_title_block()
    add_introduction(report)
    add_goals(report)
    add_method(report)
    add_project_management(report)
    report.add_heading("4. TEORİ VE SONUÇLAR")
    report.add_box([
        "Müşterilerin durumları, sosyoekonomik özellikleri ve geçmiş bankacılık alışkanlıkları, pazarlama kampanyalarına verecekleri "
        "yanıtı doğrudan etkilemektedir. Bu haftaki analizler bu ilişkilerin veride bulunduğunu gösterdi.",
    ])
    add_categorical_findings(report, summary)
    add_age_campaign_findings(report, summary)
    add_numeric_findings(report, summary)
    add_correlation_findings(report, summary)
    add_statistical_tests(report, summary)
    add_preprocessing_decisions(report)
    add_documents(report)
    report.add_links()
    report.add_references(SHORT_REFERENCES)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="3. hafta ilerleme raporunun kısa sürümünü .docx olarak üretir.")
    parser.add_argument("--kaptan", required=True, help="Bu haftanın takım kaptanının öğrenci numarası")
    args = parser.parse_args()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    output_path = build_short_report(summary, args.kaptan).save(OUTPUT_DIR / short_file_name())
    print(f"Rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    main()
