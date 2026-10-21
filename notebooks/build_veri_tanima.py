from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.data_loader import load_raw

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "01_veri_tanima.ipynb"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports" / "haftalik"
DOCS_DIR = ROOT / "docs"


def create_notebook() -> None:
    df = load_raw()

    target_counts = df["y"].value_counts()
    target_pct = df["y"].value_counts(normalize=True) * 100
    unknown_counts = {col: int((df[col] == "unknown").sum()) for col in df.columns}
    unknown_series = pd.Series(unknown_counts).sort_values(ascending=False)
    pdays_minus_one = int((df["pdays"] == -1).sum())
    duplicate_rows = int(df.duplicated().sum())
    duration_by_target = df.groupby("y")["duration"].mean()
    crosstab = pd.crosstab(df["pdays"] == -1, df["poutcome"], margins=True)

    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from src.config import DROP_COLS, RANDOM_STATE, TARGET, TEST_SIZE

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET].map({"yes": 1, "no": 0})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics = {
        "accuracy": round(float(accuracy_score(y_test, baseline_pred)), 4),
        "precision": round(float(precision_score(y_test, baseline_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, baseline_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, baseline_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, baseline_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, baseline_proba)), 4),
    }
    previous_zero_equals_pdays_minus_one = bool((df["previous"] == 0).equals(df["pdays"] == -1))

    summary = {
        "dataset_shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_pct": {str(k): round(float(v), 4) for k, v in target_pct.items()},
        "unknown_counts": {str(k): int(v) for k, v in unknown_series.items()},
        "pdays_minus_one_count": pdays_minus_one,
        "pdays_minus_one_pct": round((pdays_minus_one / len(df)) * 100, 4),
        "pdays_poutcome_crosstab": {
            "pdays_equal_minus_one": {str(k): int(v) for k, v in crosstab.loc[True].to_dict().items()},
            "pdays_not_minus_one": {str(k): int(v) for k, v in crosstab.loc[False].to_dict().items()},
            "all": {str(k): int(v) for k, v in crosstab.loc["All"].to_dict().items()},
        },
        "pdays_minus_one_poutcome_unknown_count": int(((df["pdays"] == -1) & (df["poutcome"] == "unknown")).sum()),
        "pdays_not_minus_one_poutcome_unknown_count": int(((df["pdays"] != -1) & (df["poutcome"] == "unknown")).sum()),
        "previous_zero_equals_pdays_minus_one": previous_zero_equals_pdays_minus_one,
        "baseline_model_metrics": baseline_metrics,
        "duplicate_rows": duplicate_rows,
        "duration_mean_by_target": {str(k): round(float(v), 2) for k, v in duration_by_target.items()},
    }
    (REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    (DOCS_DIR).mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "hafta2_ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    target_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868"])
    ax.set_title("Hedef Sınıf Dağılımı (y)", fontsize=12)
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Kayıt sayısı")
    ax.set_xticklabels(["no", "yes"], rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "h2_hedef_dagilimi.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    unknown_series.plot(kind="barh", ax=ax, color="#E76F51")
    ax.set_title('Her Sütundaki "unknown" Değerleri', fontsize=12)
    ax.set_xlabel("Sayı")
    ax.set_ylabel("Değişken")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "h2_unknown_dagilimi.png", dpi=150)
    plt.close(fig)

    glossary_lines = [
        "# Veri Sözlüğü",
        "",
        "| Değişken | Tür | Açıklama | Olası değerler veya aralık | \"unknown\" içeriyor mu? | Modelde kullanılacak mı? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric:
            dtype_label = "sayısal"
            if col == "duration":
                description = "Müşterinin arama sırasında konuşma süresi. Hedefe çok yakın olduğundan ve veri sızıntısı riski taşıdığından modelde kullanılmaz."
                values = f"{int(df[col].min())} - {int(df[col].max())}"
                model_use = "hayır (duration: hedefe çok yakın; veri sızıntısı riski)"
            else:
                description = f"{col} alanı için gözlenen değer aralığıdır."
                values = f"{int(df[col].min())} - {int(df[col].max())}"
                model_use = "evet"
        else:
            dtype_label = "kategorik"
            unique_values = sorted(df[col].dropna().astype(str).unique().tolist())
            description = "Kategorik özellik; müşteri profilini tanımlar."
            values = ", ".join(unique_values[:10]) + (" ..." if len(unique_values) > 10 else "")
            model_use = "evet"

        has_unknown = bool((df[col].astype(str) == "unknown").sum() > 0)
        glossary_lines.append(f"| {col} | {dtype_label} | {description} | {values} | {str(has_unknown).lower()} | {model_use} |")

    (DOCS_DIR / "veri_sozlugu.md").write_text("\n".join(glossary_lines) + "\n", encoding="utf-8")

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Bank Marketing Veri Tanıma\n\nBu not defteri hafta 2 için veri setinin ilk incelemesini yapar."),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "from src.data_loader import load_raw\n"
            "\n"
            "df = load_raw()\n"
            "print('Shape:', df.shape)\n"
            "print(df.dtypes)\n"
            "print(df.head())\n"
        ),
        nbf.v4.new_code_cell(
            "numeric_desc = df.describe(include=[np.number]).T\n"
            "print(numeric_desc)\n"
        ),
        nbf.v4.new_code_cell(
            "categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()\n"
            "for col in categorical_cols:\n"
            "    print(f'\\n{col}: {df[col].nunique()} unique values')\n"
            "    print(df[col].value_counts().head(10).to_string())\n"
        ),
        nbf.v4.new_code_cell(
            "target_counts = df['y'].value_counts()\n"
            "target_pct = df['y'].value_counts(normalize=True) * 100\n"
            "print('Target counts:', target_counts.to_dict())\n"
            "print('Target percentages:', target_pct.to_dict())\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "target_counts.plot(kind='bar', ax=ax, color=['#4C72B0', '#55A868'])\n"
            "ax.set_title('Hedef Sınıf Dağılımı (y)', fontsize=12)\n"
            "ax.set_xlabel('Sınıf')\n"
            "ax.set_ylabel('Kayıt sayısı')\n"
            "ax.set_xticklabels(['no', 'yes'], rotation=0)\n"
            "fig.tight_layout()\n"
            "fig.savefig('reports/figures/h2_hedef_dagilimi.png', dpi=150)\n"
            "plt.close(fig)\n"
        ),
        nbf.v4.new_code_cell(
            "unknown_counts = {col: int((df[col] == 'unknown').sum()) for col in df.columns}\n"
            "unknown_series = pd.Series(unknown_counts).sort_values(ascending=False)\n"
            "print('Unknown counts:', unknown_series.to_dict())\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "unknown_series.plot(kind='barh', ax=ax, color='#E76F51')\n"
            "ax.set_title('Her Sütundaki \"unknown\" Değerleri', fontsize=12)\n"
            "ax.set_xlabel('Sayı')\n"
            "ax.set_ylabel('Değişken')\n"
            "fig.tight_layout()\n"
            "fig.savefig('reports/figures/h2_unknown_dagilimi.png', dpi=150)\n"
            "plt.close(fig)\n"
        ),
        nbf.v4.new_code_cell(
            "pdays_minus_one = int((df['pdays'] == -1).sum())\n"
            "duplicate_rows = int(df.duplicated().sum())\n"
            "duration_by_target = df.groupby('y')['duration'].mean()\n"
            "pdays_poutcome = pd.crosstab(df['pdays'] == -1, df['poutcome'], margins=True)\n"
            "print('pdays_poutcome:\\n', pdays_poutcome)\n"
            "print('pdays == -1:', pdays_minus_one)\n"
            "print('duplicate rows:', duplicate_rows)\n"
            "print('duration_by_target:', duration_by_target.to_dict())\n"
        ),
        nbf.v4.new_code_cell(
            "from sklearn.dummy import DummyClassifier\n"
            "from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score\n"
            "from sklearn.model_selection import train_test_split\n"
            "from src.config import DROP_COLS, RANDOM_STATE, TARGET, TEST_SIZE\n"
            "X = df.drop(columns=DROP_COLS + [TARGET])\n"
            "y = df[TARGET].map({'yes': 1, 'no': 0})\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)\n"
            "baseline_model = DummyClassifier(strategy='most_frequent')\n"
            "baseline_model.fit(X_train, y_train)\n"
            "baseline_pred = baseline_model.predict(X_test)\n"
            "baseline_proba = baseline_model.predict_proba(X_test)[:, 1]\n"
            "metrics = {\n"
            "    'accuracy': accuracy_score(y_test, baseline_pred),\n"
            "    'precision': precision_score(y_test, baseline_pred, zero_division=0),\n"
            "    'recall': recall_score(y_test, baseline_pred, zero_division=0),\n"
            "    'f1': f1_score(y_test, baseline_pred, zero_division=0),\n"
            "    'roc_auc': roc_auc_score(y_test, baseline_proba),\n"
            "    'pr_auc': average_precision_score(y_test, baseline_proba),\n"
            "}\n"
            "print(metrics)\n"
        ),
        nbf.v4.new_code_cell(
            "summary = {\n"
            "    'dataset_shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},\n"
            "    'target_counts': {str(k): int(v) for k, v in target_counts.items()},\n"
            "    'target_pct': {str(k): round(float(v), 4) for k, v in target_pct.items()},\n"
            "    'unknown_counts': {str(k): int(v) for k, v in unknown_series.items()},\n"
            "    'pdays_minus_one_count': pdays_minus_one,\n"
            "    'pdays_minus_one_pct': round((pdays_minus_one / len(df)) * 100, 4),\n"
            "    'pdays_poutcome_crosstab': {\n"
            "        'pdays_equal_minus_one': {str(k): int(v) for k, v in pdays_poutcome.loc[True].to_dict().items()},\n"
            "        'pdays_not_minus_one': {str(k): int(v) for k, v in pdays_poutcome.loc[False].to_dict().items()},\n"
            "        'all': {str(k): int(v) for k, v in pdays_poutcome.loc['All'].to_dict().items()},\n"
            "    },\n"
            "    'pdays_minus_one_poutcome_unknown_count': int(((df['pdays'] == -1) & (df['poutcome'] == 'unknown')).sum()),\n"
            "    'pdays_not_minus_one_poutcome_unknown_count': int(((df['pdays'] != -1) & (df['poutcome'] == 'unknown')).sum()),\n"
            "    'previous_zero_equals_pdays_minus_one': bool((df['previous'] == 0).equals(df['pdays'] == -1)),\n"
            "    'baseline_model_metrics': {k: round(float(v), 4) for k, v in metrics.items()},\n"
            "    'duplicate_rows': duplicate_rows,\n"
            "    'duration_mean_by_target': {str(k): round(float(v), 2) for k, v in duration_by_target.items()},\n"
            "}\n"
            "json_path = Path('reports/haftalik/hafta2_ozet.json')\n"
            "json_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')\n"
            "print(json_path)\n"
        ),
        nbf.v4.new_code_cell(
            "glossary_lines = ['# Veri Sözlüğü', '', '| Değişken | Tür | Açıklama | Olası değerler veya aralık | \"unknown\" içeriyor mu? | Modelde kullanılacak mı? |', '| --- | --- | --- | --- | --- | --- |']\n"
            "for col in df.columns:\n"
            "    is_numeric = pd.api.types.is_numeric_dtype(df[col])\n"
            "    if is_numeric:\n"
            "        dtype_label = 'sayısal'\n"
            "        if col == 'duration':\n"
            "            description = 'Müşterinin arama sırasında konuşma süresi. Hedefe çok yakın olduğundan modelde kullanılmaz.'\n"
            "            values = f'{int(df[col].min())} - {int(df[col].max())}'\n"
            "            model_use = 'hayır (veri sızıntısı riski)'\n"
            "        else:\n"
            "            description = f'{col} alanı için gözlenen değer aralığıdır.'\n"
            "            values = f'{int(df[col].min())} - {int(df[col].max())}'\n"
            "            model_use = 'evet'\n"
            "    else:\n"
            "        dtype_label = 'kategorik'\n"
            "        unique_values = sorted(df[col].dropna().astype(str).unique().tolist())\n"
            "        description = 'Kategorik özellik; müşteri profilini tanımlar.'\n"
            "        values = ', '.join(unique_values[:10]) + (' ...' if len(unique_values) > 10 else '')\n"
            "        model_use = 'evet'\n"
            "    has_unknown = bool((df[col].astype(str) == 'unknown').sum() > 0)\n"
            "    glossary_lines.append(f'| {col} | {dtype_label} | {description} | {values} | {str(has_unknown).lower()} | {model_use} |')\n"
            "Path('docs/veri_sozlugu.md').write_text('\\n'.join(glossary_lines) + '\\n', encoding='utf-8')\n"
            "print('docs/veri_sozlugu.md created')\n"
        ),
    ]

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def execute_notebook() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(NOTEBOOK_PATH),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    create_notebook()
    execute_notebook()
    print(f"Notebook created and executed: {NOTEBOOK_PATH}")
