import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import model_store
from src.config import MODEL_PATH, PROJECT_ROOT
from tests.test_predict_service import make_bundle

APP_PATH = str(PROJECT_ROOT / "app" / "streamlit_app.py")
APP_TIMEOUT_SECONDS = 60
SUBMIT_LABEL = "Tahmin et"


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


def test_form_submission_shows_the_result_card(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)
    app = run_app()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Tek müşteri", "Liste sırala"]

    app.selectbox(key="poutcome").select("success")
    next(button for button in app.button if button.label == SUBMIT_LABEL).click()
    app.run()

    assert not app.exception
    assert app.metric[0].label == "Kabul olasılığı" and app.metric[0].value.startswith("%")
    text = " ".join(item.value for item in app.markdown)
    assert "Kabul eder" in text and "olasılık]" in text
    assert "Önceki kampanyanın sonucu" in text


def test_result_card_is_not_shown_before_the_form_is_submitted(monkeypatch, synthetic_bundle) -> None:
    monkeypatch.setattr(model_store, "load_bundle", lambda: synthetic_bundle)

    app = run_app()

    assert len(app.metric) == 0


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

    assert not app.exception and len(app.metric) == 1
