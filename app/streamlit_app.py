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
from src.predict_service import BAND_HIGH, BAND_LOW, BAND_MEDIUM, DIRECTION_INCREASES, predict_batch, predict_customer
from src.schemas import format_validation_errors

MODEL_MISSING_MESSAGE = "Model bulunamadı. Önce `python notebooks/build_model_kaydi.py` çalıştırın."
PERCENT_POINT = 100
BAND_BADGES = {BAND_HIGH: ":green-badge[Yüksek olasılık]", BAND_MEDIUM: ":orange-badge[Orta olasılık]", BAND_LOW: ":gray-badge[Düşük olasılık]"}
BAND_HELP = {
    BAND_HIGH: "Kabul olasılığı karar eşiğinin üzerinde.",
    BAND_MEDIUM: "Kabul olasılığı genel kabul oranının üzerinde ama karar eşiğinin altında.",
    BAND_LOW: "Kabul olasılığı genel kabul oranının altında.",
}
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


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_bundle() -> ModelBundle:
    return model_store.load_bundle()


def load_bundle_or_stop() -> ModelBundle:
    try:
        return get_bundle()
    except FileNotFoundError:
        st.error(MODEL_MISSING_MESSAGE)
        st.stop()


def feature_name(column: str) -> str:
    return FEATURE_LABELS[column]


def category_input(column: str) -> str:
    options = CATEGORY_VALUES[column]
    return st.selectbox(FEATURE_LABELS[column], options, index=options.index(CATEGORY_DEFAULTS[column]), format_func=VALUE_LABELS[column].get, key=column)


def number_field(column: str) -> int:
    minimum, maximum = NUMERIC_RANGES[column]
    return int(st.number_input(FEATURE_LABELS[column], min_value=minimum, max_value=maximum, value=NUMERIC_DEFAULTS[column], step=1, help=NUMERIC_HELP.get(column), key=column))


def customer_form() -> dict[str, int | str] | None:
    """Formu çizer. Gönderildiyse müşteri bilgilerini, gönderilmediyse None döndürür."""
    with st.form("customer_form"):
        personal, financial, campaign = st.columns(3)
        with personal:
            st.subheader("Müşteri")
            values = {"age": number_field("age"), "job": category_input("job"), "marital": category_input("marital"), "education": category_input("education")}
        with financial:
            st.subheader("Finansal durum")
            values |= {"balance": number_field("balance"), "default": category_input("default"), "housing": category_input("housing"), "loan": category_input("loan")}
        with campaign:
            st.subheader("Kampanya")
            values |= {
                "contact": category_input("contact"), "day": number_field("day"), "month": category_input("month"), "campaign": number_field("campaign"),
                "pdays": number_field("pdays"), "previous": number_field("previous"), "poutcome": category_input("poutcome"),
            }
        submitted = st.form_submit_button("Tahmin et", type="primary")
    return {column: values[column] for column in FEATURE_COLUMNS} if submitted else None


def show_result(customer: dict[str, int | str], bundle: ModelBundle) -> None:
    try:
        result = predict_customer(customer, bundle)
    except ValidationError as error:
        st.error("Girilen bilgilerde sorun var:")
        for message in format_validation_errors(error):
            st.markdown(f"- {message}")
        return
    with st.container(border=True):
        probability_column, decision_column = st.columns(2)
        probability_column.metric("Kabul olasılığı", format_percent(result.probability, 1))
        decision_column.markdown(f"**Karar:** {DECISION_LABELS[result.will_accept]}")
        decision_column.markdown(f"**Olasılık bandı:** {BAND_BADGES[result.band]}")
        decision_column.caption(BAND_HELP[result.band])
        st.markdown("**Kararı en çok etkileyen 3 etken**")
        if not result.top_factors:
            st.write("Bu müşteri tüm değişkenlerde referans değerlere yakın, belirgin bir etken yok.")
        for factor in result.top_factors:
            effect = f"{format_number(abs(factor.effect) * PERCENT_POINT, 1)} puan"
            st.markdown(f"- **{feature_name(factor.feature)}** olasılığı {effect} {factor.direction}")
        st.caption("Etkenler, ilgili bilgi eğitim verisindeki tipik değerle değiştirildiğinde olasılığın ne kadar değiştiğine bakılarak bulunur.")


def tab_single_customer(bundle: ModelBundle) -> None:
    customer = customer_form()
    if customer is not None:
        show_result(customer, bundle)


def read_uploaded_csv(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame | None:
    try:
        return pd.read_csv(uploaded, sep=CSV_SEP)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        st.error(f"Dosya okunamadı. CSV dosyası `{CSV_SEP}` ile ayrılmış ve bank-full.csv sütun düzeninde olmalı.")
        return None


def tab_batch(bundle: ModelBundle) -> None:
    st.write(f"CSV dosyasını yükleyin (ayırıcı `{CSV_SEP}`, bank-full.csv sütun düzeni). `duration` ve `y` sütunları varsa yok sayılır.")
    uploaded = st.file_uploader("CSV dosyası", type="csv")
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
    st.markdown(f"**{format_number(int((~invalid).sum()))}** satır sıralandı, **{format_number(int(invalid.sum()))}** satır geçersiz.")
    shown = result.assign(will_accept=result["will_accept"].map(DECISION_LABELS))[[*BATCH_COLUMNS, *FEATURE_COLUMNS]]
    st.dataframe(
        shown.rename(columns={**BATCH_COLUMNS, **FEATURE_LABELS}),
        column_config={"Kabul olasılığı": st.column_config.ProgressColumn("Kabul olasılığı", min_value=0.0, max_value=1.0, format="percent")},
        hide_index=False,
    )
    if invalid.any():
        with st.expander("Geçersiz satırların nedenleri"):
            st.dataframe(result.loc[invalid, ["error"]].rename(columns={"error": "Neden"}))
    st.download_button("Sonucu CSV olarak indir", shown.to_csv(sep=CSV_SEP, index=False).encode("utf-8-sig"), file_name="tahmin_sonuclari.csv", mime="text/csv")


def main() -> None:
    st.set_page_config(page_title="Müşteri Yanıt Tahmini", layout="wide")
    st.title("Banka Pazarlama Kampanyası: Müşteri Yanıtı Tahmini")
    st.caption("Müşterinin kampanya teklifini kabul etme olasılığını tahmin eder. Süre (duration) bilgisi kullanılmaz.")
    bundle = load_bundle_or_stop()
    single_tab, batch_tab = st.tabs(["Tek müşteri", "Liste sırala"])
    with single_tab:
        tab_single_customer(bundle)
    with batch_tab:
        tab_batch(bundle)


main()
