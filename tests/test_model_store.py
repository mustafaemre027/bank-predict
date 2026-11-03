import pytest

from src.model_store import ModelBundle, load_bundle, save_bundle
from tests.test_predict_service import PREDICT_CUSTOMER, make_bundle
from src.predict_service import predict_customer


def test_saved_and_reloaded_bundle_gives_the_same_probability(tmp_path) -> None:
    bundle = make_bundle()
    path = save_bundle(bundle, tmp_path / "model.joblib")

    reloaded = load_bundle(path)

    assert isinstance(reloaded, ModelBundle)
    assert predict_customer(PREDICT_CUSTOMER, reloaded).probability == predict_customer(PREDICT_CUSTOMER, bundle).probability
    assert reloaded.threshold == bundle.threshold and reloaded.metadata == bundle.metadata


def test_missing_model_file_error_names_the_command_to_create_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build_model_kaydi.py"):
        load_bundle(tmp_path / "yok.joblib")


def test_file_that_is_not_a_bundle_is_rejected(tmp_path) -> None:
    import joblib

    path = tmp_path / "yanlis.joblib"
    joblib.dump({"pipeline": 1}, path)

    with pytest.raises(ValueError, match="beklenen model paketini"):
        load_bundle(path)
