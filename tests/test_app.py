from dataclasses import replace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import model_store
from src.config import CSV_SEP, MODEL_PATH, PROJECT_ROOT
from src.modeling import XGBOOST
from tests.test_predict_service import PREDICT_CUSTOMER, make_bundle

APP_PATH = str(PROJECT_ROOT / "app" / "streamlit_app.py")
APP_TIMEOUT_SECONDS = 60
SUBMIT_LABEL = "Tahmin et"
TAB_LABELS = [":material/person: Tek müşteri", ":material/format_list_numbered: Liste sırala"]
TEMPLATE_LABEL = "Örnek CSV şablonunu indir"
CV_ROC_AUC = 0.8012
CV_PR_AUC = 0.4553


@pytest.fixture(autouse=True)
def clear_model_cache():
    # Arayüz modeli @st.cache_resource ile tutar, testler arasında farklı model kullanılabilmesi için temizlenir.
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture(scope="module")
def synthetic_bundle():
    return make_bundle()


def run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_SECONDS).run()


def metric_values(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


def batch_csv(rows: list[dict[str, int | str]]) -> bytes:
    header = CSV_SEP.join(rows[0])
    lines = [CSV_SEP.join(str(value) for value in row.values()) for row in rows]
    return "\n".join([header, *lines]).encode("utf-8")


def test_form_submission_shows_the_result_card(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)
    app = run_app()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == TAB_LABELS

    app.selectbox(key="poutcome").select("success")
    next(button for button in app.button if button.label == SUBMIT_LABEL).click()
    app.run()

    assert not app.exception
    values = metric_values(app)
    assert values["Kabul olasılığı"].startswith("%")
    assert values["Karar"] == "Kabul eder"
    assert values["Olasılık bandı"] == "Yüksek"
    text = " ".join(item.value for item in app.markdown)
    assert "Önceki kampanyanın sonucu" in text


def test_result_card_is_not_shown_before_the_form_is_submitted(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)

    app = run_app()

    assert "Kabul olasılığı" not in metric_values(app)


def test_sidebar_shows_threshold_and_base_rate_from_the_bundle(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)

    app = run_app()

    sidebar_metrics = {metric.label: metric.value for metric in app.sidebar.metric}
    assert sidebar_metrics["Karar eşiği"] == "%50,0"
    assert sidebar_metrics["Genel oran"] == "%12,0"
    # Sentetik pakette seçim özeti yok, CV metrikleri gösterilmez.
    assert "CV ROC-AUC" not in sidebar_metrics


def test_sidebar_shows_model_and_cv_metrics_when_the_bundle_has_a_selection_summary(monkeypatch, synthetic_bundle) -> None:
    summary = {"oof_summary": {"roc_auc": {"mean": CV_ROC_AUC}, "pr_auc": {"mean": CV_PR_AUC}}}
    bundle = replace(synthetic_bundle, metadata={**synthetic_bundle.metadata, "model": XGBOOST, "calibrated": True, "selected": summary})
    monkeypatch.setattr(model_store, "load_bundle", lambda: bundle)

    app = run_app()

    sidebar_metrics = {metric.label: metric.value for metric in app.sidebar.metric}
    assert "**Model:** XGBoost, olasılıklar kalibre edilmiş" in [item.value for item in app.sidebar.markdown]
    assert sidebar_metrics["CV ROC-AUC"] == "0,801"
    assert sidebar_metrics["CV PR-AUC"] == "0,455"


def test_batch_upload_shows_summary_metrics_and_the_ranked_table(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)
    rows = [PREDICT_CUSTOMER, {**PREDICT_CUSTOMER, "poutcome": "failure"}, {**PREDICT_CUSTOMER, "age": -5}]
    app = run_app()

    app.get("file_uploader")[0].upload("liste.csv", batch_csv(rows), "text/csv").run()

    assert not app.exception
    values = metric_values(app)
    assert values["Sıralanan müşteri"] == "2"
    assert values["Kabul edecek"] == "1"
    assert values["Geçersiz satır"] == "1"
    assert len(app.dataframe) >= 1


def test_batch_table_shows_turkish_category_names_and_no_error_text_for_valid_rows(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)
    app = run_app()

    app.get("file_uploader")[0].upload("liste.csv", batch_csv([PREDICT_CUSTOMER]), "text/csv").run()

    table = app.dataframe[0].value
    assert table["Meslek"].tolist() == ["Teknisyen"]
    assert table["Olasılık bandı"].tolist() == ["Yüksek"]
    assert table["Hata"].tolist() == [""]


def test_batch_tab_offers_a_csv_template(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)

    app = run_app()

    labels = [button.proto.label for button in app.get("download_button")]
    assert TEMPLATE_LABEL in labels


def test_missing_model_shows_a_clear_message_without_a_traceback(monkeypatch, tmp_path) -> None:
    original = model_store.load_bundle
    monkeypatch.setattr(model_store, "load_bundle", lambda: original(tmp_path / "yok.joblib"))

    app = run_app()

    assert not app.exception
    assert [error.value for error in app.error] == ["Model bulunamadı. Önce `python notebooks/build_model_kaydi.py` çalıştırın."]
    assert len(app.button) == 0


def test_app_works_with_the_saved_real_model() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH} yok, gerçek model ile arayüz denemesi atlandı. python notebooks/build_model_kaydi.py ile üretilir.")
    app = run_app()

    next(button for button in app.button if button.label == SUBMIT_LABEL).click()
    app.run()

    assert not app.exception and "Kabul olasılığı" in metric_values(app)
    assert {metric.label for metric in app.sidebar.metric} >= {"Karar eşiği", "CV ROC-AUC", "CV PR-AUC"}
