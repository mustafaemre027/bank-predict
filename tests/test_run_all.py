from run_all import ANALYSIS_SCRIPTS, ROOT


def test_every_listed_script_exists() -> None:
    assert [script for script in ANALYSIS_SCRIPTS if not (ROOT / script).exists()] == []


def test_scripts_run_in_week_order_and_cover_all_analysis_weeks() -> None:
    assert len(ANALYSIS_SCRIPTS) == 8
    assert ANALYSIS_SCRIPTS[-1].endswith("build_model_kaydi.py")
    assert ANALYSIS_SCRIPTS.index("notebooks/build_hiperparametre.py") < ANALYSIS_SCRIPTS.index("notebooks/build_final_degerlendirme.py")
