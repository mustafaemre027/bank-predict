# Banka Pazarlama Kampanyalarına Müşteri Yanıtının Tahmini

## Proje Tanımı
Bu proje, bankanın pazarlama kampanyasında müşteri yanıtını tahmin etmeyi amaçlayan bir ikili sınıflandırma problemidir. Hedef değişken `y` değeridir (`yes` / `no`).

## Ekip ve Roller
- Takım Adı: BankTech
- Takım Kaptanı: Ömer NARÇİÇEĞİ (230260054) – Makine Öğrenmesi Geliştirici
- Mesut ASLAN (230260173) – Veri Analisti / Yazılım Geliştirici
- Mustafa Emre KAPLAN (230260143) – Yazılım Geliştirici / Test ve Raporlama

## Klasör Yapısı
```text
bank-predict/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── haftalik/
├── src/
├── tests/
├── .github/
├── .gitignore
├── README.md
├── requirements.txt
└── .venv/
```

## Kurulum
### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Veriyi İndirme
- Veri kaynağı: UCI Bank Marketing Dataset
- Dosya: `bank-full.csv`
- İndirme komutu için `src/download_data.py` kullanılmalıdır.

## Testleri Çalıştırma
```bash
pytest
```

## Haftalık İlerleme
| Hafta | Durum |
|------|------|
| H1 | ✓ |
| H2 | Devam ediyor |
| H3 | Planlandı |
| H4 | Planlandı |
| H5 | Planlandı |
| H6 | Planlandı |
| H7 | Planlandı |
| H8 | Planlandı |
| H9 | Planlandı |
| H10 | Planlandı |
