"""Streamlit arayüzü: tek müşteri için kabul olasılığı ve toplu CSV için sıralı liste.

İnce katmandır: tahmin ve doğrulama src/predict_service.py ile src/schemas.py içindedir, burada yalnızca form, gösterim ve Türkçe
etiketler vardır. Çalıştırma: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src import model_store
from src.config import CATEGORY_VALUES, CSV_SEP, FEATURE_COLUMNS, NUMERIC_RANGES
from src.formatting import format_number, format_percent
from src.model_store import ModelBundle
from src.modeling import LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST
from src.predict_service import BAND_HIGH, BAND_LOW, BAND_MEDIUM, PredictionResult, predict_batch, predict_customer
from src.schemas import format_validation_errors

MODEL_MISSING_MESSAGE = "Model bulunamadı. Önce `python notebooks/build_model_kaydi.py` çalıştırın."
PERCENT_POINT = 100
METRIC_DECIMALS = 3
BAND_LABELS = {BAND_HIGH: "Yüksek", BAND_MEDIUM: "Orta", BAND_LOW: "Düşük"}
BAND_HELP = {
    BAND_HIGH: "Kabul olasılığı karar eşiğinin üzerinde.",
    BAND_MEDIUM: "Kabul olasılığı genel kabul oranının üzerinde ama karar eşiğinin altında.",
    BAND_LOW: "Kabul olasılığı genel kabul oranının altında.",
}
# Renkler .streamlit/config.toml temasıyla aynı paletten gelir. Bant rengi her zaman metinle birlikte gösterilir.
BAND_COLORS = {BAND_HIGH: "#059669", BAND_MEDIUM: "#D97706", BAND_LOW: "#64748B"}
BAND_TEXT_COLORS = {BAND_HIGH: "green", BAND_MEDIUM: "orange", BAND_LOW: "gray"}
TRACK_COLOR = "#E4E7EB"
MARKER_COLOR = "#1E3A5F"
SCALE_TEXT_COLOR = "#64748B"
# Olasılık çubuğunun piksel ölçüleri
GAUGE_BAR_THICKNESS = 14
GAUGE_LABEL_ROW = 18
MODEL_LABELS = {XGBOOST: "XGBoost", RANDOM_FOREST: "Rastgele Orman", LOGISTIC_REGRESSION: "Lojistik Regresyon"}
FEATURE_LABELS = {
    "age": "Yaş", "job": "Meslek", "marital": "Medeni durum", "education": "Eğitim", "default": "Kredi temerrüdü",
    "balance": "Yıllık ortalama bakiye (euro)", "housing": "Konut kredisi", "loan": "Bireysel kredi", "contact": "İletişim türü",
    "day": "Ayın günü", "month": "Ay", "campaign": "Bu kampanyadaki arama sayısı", "pdays": "Önceki kampanyadan bu yana geçen gün",
    "previous": "Önceki kampanyalardaki arama sayısı", "poutcome": "Önceki kampanyanın sonucu",
}
VALUE_LABELS = {
    "job": {
        "admin.": "İdari personel", "blue-collar": "Mavi yaka", "entrepreneur": "Girişimci", "housemaid": "Ev hizmetleri",
        "management": "Yönetici", "retired": "Emekli", "self-employed": "Serbest meslek", "services": "Hizmet sektörü",
        "student": "Öğrenci", "technician": "Teknisyen", "unemployed": "İşsiz", "unknown": "Bilinmiyor",
    },
    "marital": {"divorced": "Boşanmış", "married": "Evli", "single": "Bekâr"},
    "education": {"primary": "İlköğretim", "secondary": "Ortaöğretim", "tertiary": "Yükseköğretim", "unknown": "Bilinmiyor"},
    "default": {"no": "Hayır", "yes": "Evet"},
    "housing": {"no": "Hayır", "yes": "Evet"},
    "loan": {"no": "Hayır", "yes": "Evet"},
    "contact": {"cellular": "Cep telefonu", "telephone": "Sabit telefon", "unknown": "Bilinmiyor"},
    "month": {
        "jan": "Ocak", "feb": "Şubat", "mar": "Mart", "apr": "Nisan", "may": "Mayıs", "jun": "Haziran",
        "jul": "Temmuz", "aug": "Ağustos", "sep": "Eylül", "oct": "Ekim", "nov": "Kasım", "dec": "Aralık",
    },
    "poutcome": {"failure": "Başarısız", "other": "Diğer", "success": "Başarılı", "unknown": "Bilinmiyor"},
}
NUMERIC_DEFAULTS = {"age": 40, "balance": 500, "day": 15, "campaign": 1, "pdays": -1, "previous": 0}
NUMERIC_HELP = {"pdays": "-1: müşteri daha önce hiçbir kampanyada aranmadı.", "balance": "Negatif değer, hesabın ekside olduğunu gösterir."}
CATEGORY_DEFAULTS = {"job": "management", "marital": "married", "education": "secondary", "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "may", "poutcome": "unknown"}
# Sonuç tablosunda gösterilen sütunlar ve başlıkları
BATCH_COLUMNS = {"probability": "Kabul olasılığı", "band": "Olasılık bandı", "will_accept": "Karar", "error": "Hata"}
DECISION_LABELS = {True: "Kabul eder", False: "Kabul etmez"}
TEMPLATE_FILE_NAME = "ornek_musteri_listesi.csv"
CSV_ENCODING = "utf-8-sig"


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_bundle() -> ModelBundle:
    return model_store.load_bundle()


def load_bundle_or_stop() -> ModelBundle:
    try:
        return get_bundle()
    except FileNotFoundError:
        st.error(MODEL_MISSING_MESSAGE)
        st.stop()


def category_input(column: str) -> str:
    options = CATEGORY_VALUES[column]
    return st.selectbox(FEATURE_LABELS[column], options, index=options.index(CATEGORY_DEFAULTS[column]), format_func=VALUE_LABELS[column].get, key=column)


def number_field(column: str) -> int:
    minimum, maximum = NUMERIC_RANGES[column]
    return int(st.number_input(FEATURE_LABELS[column], min_value=minimum, max_value=maximum, value=NUMERIC_DEFAULTS[column], step=1, help=NUMERIC_HELP.get(column), key=column))


def model_sidebar(bundle: ModelBundle) -> None:
    """Modelin karar eşiği, genel kabul oranı ve (pakette varsa) seçim özetindeki CV metrikleri. Sayılar model paketinden okunur."""
    with st.sidebar:
        st.subheader(":material/model_training: Model hakkında")
        model_name = bundle.metadata.get("model")
        if model_name is not None:
            suffix = ", olasılıklar kalibre edilmiş" if bundle.metadata.get("calibrated") else ""
            st.markdown(f"**Model:** {MODEL_LABELS.get(model_name, model_name)}{suffix}")
        threshold_column, base_rate_column = st.columns(2)
        threshold_column.metric("Karar eşiği", format_percent(bundle.threshold, 1), help="Bu olasılığın üzerindeki müşteriler için karar \"Kabul eder\" olur.")
        base_rate_column.metric("Genel oran", format_percent(bundle.base_rate, 1), help="Genel kabul oranı: eğitim verisinde kampanyayı kabul eden müşterilerin oranı.")
        oof_summary = bundle.metadata.get("selected", {}).get("oof_summary")
        if oof_summary is not None:
            roc_column, pr_column = st.columns(2)
            roc_column.metric("CV ROC-AUC", format_number(oof_summary["roc_auc"]["mean"], METRIC_DECIMALS))
            pr_column.metric("CV PR-AUC", format_number(oof_summary["pr_auc"]["mean"], METRIC_DECIMALS))
        st.divider()
        st.markdown("**Olasılık bantları**")
        for band, label in BAND_LABELS.items():
            st.markdown(f":{BAND_TEXT_COLORS[band]}-badge[{label}] {BAND_HELP[band]}")
        st.caption("Görüşme süresi (duration) görüşme bitmeden bilinemediği için modelde kullanılmaz.")


def customer_form() -> dict[str, int | str] | None:
    """Formu çizer. Gönderildiyse müşteri bilgilerini, gönderilmediyse None döndürür."""
    with st.form("customer_form", border=False):
        personal, financial, campaign = st.columns(3)
        with personal.container(border=True):
            st.markdown("##### :material/person: Müşteri")
            values = {"age": number_field("age"), "job": category_input("job"), "marital": category_input("marital"), "education": category_input("education")}
        with financial.container(border=True):
            st.markdown("##### :material/account_balance_wallet: Finansal durum")
            values |= {"balance": number_field("balance"), "default": category_input("default"), "housing": category_input("housing"), "loan": category_input("loan")}
        with campaign.container(border=True):
            st.markdown("##### :material/campaign: Kampanya")
            values |= {
                "contact": category_input("contact"), "day": number_field("day"), "month": category_input("month"), "campaign": number_field("campaign"),
                "pdays": number_field("pdays"), "previous": number_field("previous"), "poutcome": category_input("poutcome"),
            }
        submitted = st.form_submit_button("Tahmin et", type="primary", icon=":material/insights:")
    return {column: values[column] for column in FEATURE_COLUMNS} if submitted else None


def probability_gauge(result: PredictionResult, bundle: ModelBundle) -> str:
    """0-%100 çubuğunda müşterinin olasılığı, genel kabul oranı ve karar eşiği işaretleriyle (HTML).

    Etiketler çizginin sağına ve iki ayrı yüksekliğe yazılır, böylece birbirine binmez. Eşik ve genel oran çubuğun sol yarısında olduğu
    sürece dar ekranda da taşmaz. İşaret çizgileri yalnızca çubuğun boyunca iner, ölçek yazılarının üstünden geçmez. İçeriğe yalnızca koddaki sayılar girer.
    """
    def marker(position: float, label: str, rise: int) -> str:
        return (
            f'<div style="position:absolute;left:{position * PERCENT_POINT:.2f}%;top:-{rise}px;bottom:-4px;border-left:2px dashed {MARKER_COLOR}">'
            f'<span style="position:absolute;top:-2px;left:6px;font-size:12px;line-height:14px;white-space:nowrap;color:{MARKER_COLOR}">{label}</span></div>'
        )

    track = (
        f'<div style="height:{GAUGE_BAR_THICKNESS}px;background:{TRACK_COLOR};border-radius:999px;overflow:hidden">'
        f'<div style="width:{result.probability * PERCENT_POINT:.2f}%;height:100%;background:{BAND_COLORS[result.band]};border-radius:999px"></div></div>'
    )
    scale = f'<div style="display:flex;justify-content:space-between;font-size:12px;color:{SCALE_TEXT_COLOR};margin-top:6px"><span>%0</span><span>%50</span><span>%100</span></div>'
    markers = (
        marker(bundle.base_rate, f"Genel oran {format_percent(bundle.base_rate, 1)}", 2 * GAUGE_LABEL_ROW)
        + marker(bundle.threshold, f"Eşik {format_percent(bundle.threshold, 1)}", GAUGE_LABEL_ROW)
    )
    bar = f'<div style="position:relative">{track}{markers}</div>'
    return f'<div role="img" aria-label="Kabul olasılığı {format_percent(result.probability, 1)}" style="margin:{2 * GAUGE_LABEL_ROW + 4}px 0 4px">{bar}{scale}</div>'


def show_factors(result: PredictionResult) -> None:
    st.markdown("##### Kararı en çok etkileyen 3 etken")
    if not result.top_factors:
        st.write("Bu müşteri tüm değişkenlerde referans değerlere yakın, belirgin bir etken yok.")
        return
    for factor in result.top_factors:
        effect = f"{format_number(abs(factor.effect) * PERCENT_POINT, 1)} puan"
        color, icon = ("green", "trending_up") if factor.effect > 0 else ("red", "trending_down")
        st.markdown(f":{color}[:material/{icon}:] **{FEATURE_LABELS[factor.feature]}** olasılığı :{color}[**{effect} {factor.direction}**]")
    st.caption("Etkenler, ilgili bilgi eğitim verisindeki tipik değerle değiştirildiğinde olasılığın ne kadar değiştiğine bakılarak bulunur.")


def show_result(customer: dict[str, int | str], bundle: ModelBundle) -> None:
    try:
        result = predict_customer(customer, bundle)
    except ValidationError as error:
        st.error("Girilen bilgilerde sorun var:", icon=":material/error:")
        for message in format_validation_errors(error):
            st.markdown(f"- {message}")
        return
    with st.container(border=True):
        st.markdown("#### :material/analytics: Tahmin sonucu")
        probability_column, decision_column, band_column = st.columns(3)
        probability_column.metric("Kabul olasılığı", format_percent(result.probability, 1))
        decision_column.metric("Karar", DECISION_LABELS[result.will_accept])
        band_column.metric("Olasılık bandı", BAND_LABELS[result.band], help=BAND_HELP[result.band])
        st.html(probability_gauge(result, bundle))
        st.caption(BAND_HELP[result.band])
        st.divider()
        show_factors(result)


def tab_single_customer(bundle: ModelBundle) -> None:
    customer = customer_form()
    if customer is not None:
        show_result(customer, bundle)


def template_csv() -> bytes:
    """Formdaki varsayılan değerlerle tek satırlık, doğrudan yüklenebilir örnek dosya."""
    row = {column: {**NUMERIC_DEFAULTS, **CATEGORY_DEFAULTS}[column] for column in FEATURE_COLUMNS}
    return pd.DataFrame([row]).to_csv(sep=CSV_SEP, index=False).encode(CSV_ENCODING)


def read_uploaded_csv(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame | None:
    try:
        return pd.read_csv(uploaded, sep=CSV_SEP)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        st.error(f"Dosya okunamadı. CSV dosyası `{CSV_SEP}` ile ayrılmış ve bank-full.csv sütun düzeninde olmalı.")
        return None


def batch_upload_section() -> st.runtime.uploaded_file_manager.UploadedFile | None:
    upload_column, help_column = st.columns([3, 2])
    with upload_column:
        uploaded = st.file_uploader("Müşteri listesi (CSV)", type="csv")
    with help_column.container(border=True):
        st.markdown("##### :material/description: Dosya biçimi")
        st.markdown(f"Ayırıcı `{CSV_SEP}`, sütunlar bank-full.csv düzeninde. `duration` ve `y` sütunları varsa yok sayılır.")
        st.download_button("Örnek CSV şablonunu indir", template_csv(), file_name=TEMPLATE_FILE_NAME, mime="text/csv", icon=":material/download:")
    return uploaded


def tab_batch(bundle: ModelBundle) -> None:
    uploaded = batch_upload_section()
    if uploaded is None:
        return
    frame = read_uploaded_csv(uploaded)
    if frame is None:
        return
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        st.error(f"Dosyada şu sütunlar eksik: {', '.join(missing)}.")
        return
    with st.spinner("Tahminler hesaplanıyor..."):
        result = predict_batch(frame, bundle)
    invalid = result["error"].notna()
    ranked_column, accepting_column, invalid_column = st.columns(3)
    ranked_column.metric("Sıralanan müşteri", format_number(int((~invalid).sum())), border=True)
    accepting_column.metric("Kabul edecek", format_number(int(result.loc[~invalid, "will_accept"].astype(bool).sum())), border=True)
    invalid_column.metric("Geçersiz satır", format_number(int(invalid.sum())), border=True)
    shown = result.assign(will_accept=result["will_accept"].map(DECISION_LABELS), error=result["error"].fillna(""))[[*BATCH_COLUMNS, *FEATURE_COLUMNS]]
    # Ekranda bant ve kategoriler Türkçe görünen adlarıyla gösterilir, indirilen dosya ham değerleri korur.
    display_labels = {"band": BAND_LABELS, **VALUE_LABELS}
    translated = shown.assign(**{column: shown[column].map(labels).fillna(shown[column]) for column, labels in display_labels.items()})
    st.dataframe(
        translated.rename(columns={**BATCH_COLUMNS, **FEATURE_LABELS}),
        column_config={"Kabul olasılığı": st.column_config.ProgressColumn("Kabul olasılığı", min_value=0.0, max_value=1.0, format="percent")},
        hide_index=False,
    )
    if invalid.any():
        with st.expander("Geçersiz satırların nedenleri", icon=":material/warning:"):
            st.dataframe(result.loc[invalid, ["error"]].rename(columns={"error": "Neden"}))
    st.download_button(
        "Sonucu CSV olarak indir", shown.to_csv(sep=CSV_SEP, index=False).encode(CSV_ENCODING), file_name="tahmin_sonuclari.csv",
        mime="text/csv", type="primary", icon=":material/download:",
    )


def main() -> None:
    st.set_page_config(page_title="Müşteri Yanıt Tahmini", page_icon=":material/account_balance:", layout="wide")
    st.caption("BANKTECH · BANKA PAZARLAMA KAMPANYASI")
    st.title("Müşteri Yanıt Tahmini")
    st.markdown("Müşterinin kampanya teklifini kabul etme olasılığını tahmin eder. Tek bir müşteriyi değerlendirin ya da bir listeyi olasılığa göre sıralayın.")
    bundle = load_bundle_or_stop()
    model_sidebar(bundle)
    single_tab, batch_tab = st.tabs([":material/person: Tek müşteri", ":material/format_list_numbered: Liste sırala"])
    with single_tab:
        tab_single_customer(bundle)
    with batch_tab:
        tab_batch(bundle)


main()
