from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bank-full.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
RANDOM_STATE = 42
TARGET = "y"
POSITIVE_LABEL = "yes"
DROP_COLS = ["duration"]
CSV_SEP = ";"
TEST_SIZE = 0.2
# Modelde kullanılacak bağımsız değişkenler; DROP_COLS içindekiler (duration) bilinçli olarak yoktur.
CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
# pdays = -1 müşterinin önceki bir kampanyada hiç aranmadığını gösterir.
NOT_PREVIOUSLY_CONTACTED = -1
# Yalnızca yes/no değeri alan değişkenler 0/1 olarak kodlanır; diğer kategorikler One-Hot ile kodlanır.
BINARY_FEATURES = ["default", "housing", "loan"]
NOMINAL_FEATURES = [column for column in CATEGORICAL_FEATURES if column not in BINARY_FEATURES]
# Uç aralıklar açık uçludur; böylece beklenmeyen bir değer hiçbir aralığın dışında kalıp sessizce kaybolmaz.
AGE_BINS = [-np.inf, 25, 30, 40, 50, 60, np.inf]
AGE_LABELS = ["18-25", "26-30", "31-40", "41-50", "51-60", "61+"]
CV_FOLDS = 5
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "RANDOM_STATE",
    "TARGET",
    "POSITIVE_LABEL",
    "DROP_COLS",
    "CSV_SEP",
    "TEST_SIZE",
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "NOT_PREVIOUSLY_CONTACTED",
    "BINARY_FEATURES",
    "NOMINAL_FEATURES",
    "AGE_BINS",
    "AGE_LABELS",
    "CV_FOLDS",
    "MONTH_ORDER",
]
