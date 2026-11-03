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
# İş senaryosu için eşik seçiminde kabul edecek müşterilerin en az bu oranının yakalanması istenir.
TARGET_RECALL = 0.70
# FOG-1: test setinde ROC-AUC en az 0,75 ve PR-AUC en az 0,234 olmalı (0,234, referans modelin kabul oranı 0,117'nin iki katıdır).
FOG1_MIN_ROC_AUC = 0.75
FOG1_MIN_PR_AUC = 0.234
# Moro vd. (2014) çalışmasındaki AUC. Daha geniş bir veri sürümü ve kayan pencere kullandığı için yalnızca gösterge amaçlıdır.
LITERATURE_ROC_AUC = 0.794
# Tahmin servisinin kabul ettiği değerler. Elle yazılmıştır, tests/test_schemas.py bunları ham veriden hesaplanan değerlerle birebir karşılaştırır.
# Model girdisindeki sütun sırası veri dosyasındaki sıradır.
FEATURE_COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]
CATEGORY_VALUES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"],
    "poutcome": ["failure", "other", "success", "unknown"],
}
# Veri setinde gözlenen (en küçük, en büyük) değerler.
NUMERIC_RANGES = {
    "age": (18, 95),
    "balance": (-8019, 102127),
    "day": (1, 31),
    "campaign": (1, 63),
    "pdays": (-1, 871),
    "previous": (0, 275),
}
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
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
    "TARGET_RECALL",
    "FEATURE_COLUMNS",
    "CATEGORY_VALUES",
    "NUMERIC_RANGES",
    "MODEL_PATH",
    "FOG1_MIN_ROC_AUC",
    "FOG1_MIN_PR_AUC",
    "LITERATURE_ROC_AUC",
    "MONTH_ORDER",
]
