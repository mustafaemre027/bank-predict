# Banka Pazarlama Kampanyalarına Müşteri Yanıtının Tahmini

## Proje Tanımı
Bu proje, bankanın pazarlama kampanyasında müşteri yanıtını tahmin etmeyi amaçlayan bir ikili sınıflandırma problemidir. Hedef değişken `y` değeridir (`yes` / `no`).

## Ekip ve Roller
- Takım Adı: BankTech
- Ömer NARÇİÇEĞİ (230260054) – Makine Öğrenmesi Geliştirici
- Mesut ASLAN (230260173) – Veri Analisti / Yazılım Geliştirici
- Mustafa Emre KAPLAN (230260143) – Yazılım Geliştirici / Test ve Raporlama
- Takım kaptanı haftalık değişir; güncel kaptan için [yol haritasındaki rotasyon tablosuna](docs/yol_haritasi.md#takım-kaptanı-rotasyonu) bakın.

## Proje Belgeleri
- [PRD — Ürün Gereksinim Belgesi](docs/prd.md)
- [Gereksinim Analizi (Sürüm 2)](docs/gereksinim_analizi.md)
- [10 Haftalık Yol Haritası](docs/yol_haritasi.md)
- [Ön İşleme Karar Taslağı](docs/on_isleme_kararlari.md)
- [Veri Sözlüğü](docs/veri_sozlugu.md)

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

## Analizleri ve Haftalık Raporu Üretme
```bash
# Hafta 3 keşifsel analiz: özet JSON, grafikler ve notebook
python notebooks/build_kesifsel_analiz.py

# Hafta 3 raporu (.docx); --kaptan o haftanın kaptanının öğrenci numarası
python reports/hafta3_rapor.py --kaptan 230260054
```
Rapor `reports/haftalik/` altına yazılır ve repoya eklenmez. Rapordaki tüm sayılar `reports/haftalik/hafta3_ozet.json` dosyasından okunur.

## Haftalık İlerleme
| Hafta | Durum |
|------|------|
| H1 | ✓ |
| H2 | ✓ |
| H3 | ✓ |
| H4 | Planlandı |
| H5 | Planlandı |
| H6 | Planlandı |
| H7 | Planlandı |
| H8 | Planlandı |
| H9 | Planlandı |
| H10 | Planlandı |
