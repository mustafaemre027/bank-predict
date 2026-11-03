"""Tüm analiz betiklerini doğru sırayla çalıştırır: Hafta 2'den Hafta 9'a kadar özet JSON, grafik, notebook ve model dosyası üretilir.

Kullanım: python run_all.py
Toplam süre yaklaşık 10 dakikadır (en uzun adım Hafta 7 hiperparametre araması). Haftalık raporlar (.docx) ayrıca
python reports/haftaN_rapor.py --kaptan <öğrenci no> komutuyla üretilir.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Sıra önemlidir: her adım bir öncekinin özet JSON'unu okur.
ANALYSIS_SCRIPTS = [
    "notebooks/build_veri_tanima.py",
    "notebooks/build_kesifsel_analiz.py",
    "notebooks/build_ilk_model.py",
    "notebooks/build_model_karsilastirma.py",
    "notebooks/build_dengesizlik.py",
    "notebooks/build_hiperparametre.py",
    "notebooks/build_final_degerlendirme.py",
    "notebooks/build_model_kaydi.py",
]


def run_script(script: str) -> float:
    started = time.perf_counter()
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, script], check=True, cwd=ROOT)
    return time.perf_counter() - started


def main() -> None:
    for script in ANALYSIS_SCRIPTS:
        print(f"{script} tamamlandı ({run_script(script):.0f} sn)", flush=True)
    print("\nTüm analiz betikleri tamamlandı.")


if __name__ == "__main__":
    main()
