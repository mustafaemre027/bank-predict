from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bank-full.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
RANDOM_STATE = 42
TARGET = "y"
DROP_COLS = ["duration"]
CSV_SEP = ";"
TEST_SIZE = 0.2
# Modelde kullanılacak bağımsız değişkenler; DROP_COLS içindekiler (duration) bilinçli olarak yoktur.
CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
# pdays = -1 müşterinin önceki bir kampanyada hiç aranmadığını gösterir.
NOT_PREVIOUSLY_CONTACTED = -1

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "RANDOM_STATE",
    "TARGET",
    "DROP_COLS",
    "CSV_SEP",
    "TEST_SIZE",
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "NOT_PREVIOUSLY_CONTACTED",
]
