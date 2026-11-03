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
- [9 Haftalık Yol Haritası](docs/yol_haritasi.md)
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
macOS'ta XGBoost'un çalışması için OpenMP kütüphanesi gerekir: `brew install libomp`.
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

# Aynı raporun kısa sürümü (tablo ve şekiller aynı, anlatım kısaltılmış)
python reports/hafta3_rapor_kisa.py --kaptan 230260054

# Hafta 4 ön işleme + ilk Lojistik Regresyon (5 katlı CV): özet JSON, PR/ROC grafiği ve notebook
python notebooks/build_ilk_model.py
python reports/hafta4_rapor.py --kaptan 230260054

# Hafta 5 Rastgele Orman, XGBoost ve ön işleme ablasyonu: özet JSON, grafikler ve notebook (yaklaşık 2 dakika)
python notebooks/build_model_karsilastirma.py
python reports/hafta5_rapor.py --kaptan 230260173

# Hafta 6 sınıf dengesizliği stratejileri ve eşik seçimi: özet JSON, grafikler ve notebook (yaklaşık 2 dakika)
python notebooks/build_dengesizlik.py
python reports/hafta6_rapor.py --kaptan 230260143

# Hafta 7 hiperparametre araması, kalibrasyon ve final aday seçimi: özet JSON, grafikler ve notebook (yaklaşık 5 dakika)
python notebooks/build_hiperparametre.py
python reports/hafta7_rapor.py --kaptan 230260173

# Hafta 8 final değerlendirme (test seti burada bir kez açılır), açıklanabilirlik ve hata analizi: özet JSON, grafikler ve notebook (yaklaşık 30 saniye)
python notebooks/build_final_degerlendirme.py
python reports/hafta8_rapor.py --kaptan 230260143

# Final raporu iskeleti (9. hafta sonuçları gelene kadar yalnızca taslak üretilir)
python reports/final_rapor.py --kaptan 230260054 --taslak
```
Rapor `reports/haftalik/` altına yazılır ve repoya eklenmez. Rapordaki tüm sayılar o haftanın `reports/haftalik/haftaN_ozet.json` dosyasından okunur.

## Haftalık İlerleme
| Hafta | Durum |
|------|------|
| H1 | ✓ |
| H2 | ✓ |
| H3 | ✓ |
| H4 | ✓ |
| H5 | ✓ |
| H6 | ✓ |
| H7 | ✓ |
| H8 | ✓ |
| H9 | Planlandı |
