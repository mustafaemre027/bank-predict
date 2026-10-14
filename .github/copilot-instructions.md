# BankTech Proje Kuralları

## Değişmez Proje Kuralları
1. `duration` değişkeni modelde kullanılmaz (görüşme bitmeden bilinemez → veri sızıntısı). Analizde gösterilebilir, ama `config.py` içindeki `DROP_COLS` listesinde olmalıdır.
2. Veride `NaN` yok; eksik bilgiler `unknown` kategorisi olarak kodlu. `pdays = -1` "daha önce aranmamış" demektir. Bunları "eksik veri" diye değil, ayrı ayrı raporlayın.
3. Sınıflar dengesizdir (~%11,7 `yes`). Accuracy tek başına metrik olarak sunulmaz; `F1`, `Recall`, `ROC-AUC` ve `PR-AUC` kullanılır.
4. SMOTE/oversampling ileride sadece eğitim setine, train/test ayrımından sonra uygulanır.
5. `random_state = 42` her yerde.
6. Yeniden kullanılan kod `src/` altında; notebook'lar sadece keşif ve sunum için ve `src/`'den import eder.
7. `data/raw/` altındaki dosyalar asla değiştirilmez.
8. Hiçbir sayıyı uydurmaya çalışma. Raporlara ve dokümanlara giren her sayı, çalıştırılan koddan üretilmiş olmalıdır.
9. Kod içi yorumlar ve dokümanlar Türkçe; değişken ve fonksiyon isimleri İngilizce.

## Proje Özeti
- Proje: Banka Pazarlama Kampanyalarına Müşteri Yanıtının Tahmini
- Hedef değişken: `y` (`yes` / `no`)
- Takım adı: BankTech
- Takım kaptanı: Ömer NARÇİÇEĞİ (230260054) – Makine Öğrenmesi Geliştirici
- Üyeler: Mesut ASLAN (230260173), Mustafa Emre KAPLAN (230260143)
- Veri kaynağı: UCI Machine Learning Repository – Bank Marketing (Moro, Cortez & Rita, 2014)
- Amaç: müşteri yanıtını tahmin etmek için ikili sınıflandırma modeli geliştirmek.

## Çalışma Prensibi
- Tüm haftalar için aynı kurallar geçerlidir.
- Kod ve raporlar Türkçe yazılır; teknik isimler İngilizcedir.
- Her hafta ilerleme ve dosya değişiklikleri net şekilde takip edilir.
