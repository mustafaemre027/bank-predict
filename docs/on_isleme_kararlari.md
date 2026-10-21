# Ön İşleme Karar Taslağı (Hafta 3 → Hafta 4)

> Hazırlayan: Ömer NARÇİÇEĞİ · Durum: **Taslak**, Hafta 4 başında ekipçe onaylanacak.
> Bu belgedeki tüm sayılar `notebooks/build_kesifsel_analiz.py` ile üretilen `reports/haftalik/hafta3_ozet.json` dosyasından alınmıştır.

## Değişmeyen ilkeler

1. `duration` modele girmez (görüşme bitmeden bilinemez, veri sızıntısı).
2. Tüm dönüşümler bir scikit-learn `Pipeline` / `ColumnTransformer` içinde tanımlanır ve **yalnızca eğitim verisine fit edilir**; test verisi yalnızca `transform` görür.
3. Test seti (%20, stratified, `random_state=42`) Hafta 8'deki final değerlendirmeye kadar kullanılmaz. Model seçimi, hiperparametre ve karar eşiği eğitim verisi içinde 5 katlı stratified çapraz doğrulama ile belirlenir.
4. SMOTE yalnızca çapraz doğrulamanın eğitim katmanlarında, `imblearn.pipeline.Pipeline` içinde uygulanır.

## Değişken bazında kararlar

| Değişken | EDA bulgusu | Karar |
|---|---|---|
| `poutcome` | En güçlü kategorik ilişki (Cramér's V = 0,312). `success` olanlarda kabul %64,73, `unknown` olanlarda %9,16. | One-Hot. `unknown` ayrı kategori olarak kalır, çünkü 36.959 kaydın 36.954'ü "önceki kampanya yok" anlamındadır (Hafta 2). |
| `month` | V = 0,260. Mart, eylül, ekim ve aralıkta kabul %43-52, ancak bu aylarda arama sayısı az (214-738). Mayısta 13.766 arama ve %6,72 kabul. | One-Hot. Az gözlemli ayların yüksek oranı aşırı öğrenmeye açık olduğundan etkisi CV'de izlenecek. |
| `contact` | V = 0,151. `unknown` olanlarda kabul %4,07. `unknown` payı mayısta %57,93, haziranda %84,85, diğer aylarda en fazla %8,12 (eylül). | One-Hot, `unknown` ayrı kategori. Bu değer büyük ölçüde veri toplamanın ilk dönemine ait bir kayıt eksikliğidir. Modelin bu dönemi ezberleme riski Hafta 8 hata analizinde kontrol edilecek. |
| `job`, `education`, `marital` | V sırasıyla 0,136 / 0,073 / 0,066. `student` %28,68, `retired` %22,79 kabul. | One-Hot. `unknown` ayrı kategori (az sayıda: job 288, education 1.857). |
| `housing`, `loan`, `default` | V sırasıyla 0,139 / 0,068 / 0,022. | İkili (0/1) kodlama. `default` zayıf ama bilgi taşıdığı için çıkarılmaz; önem analizinde tekrar bakılacak. |
| `age` | Mann-Whitney etkisi çok küçük (r = −0,016, p = 0,063), fakat ilişki **U şeklinde**: 18-25 yaş %23,95, 61+ yaş %42,26, 41-50 yaş %9,07. | Sayısal olarak kalır. Doğrusal olmayan ilişkiyi Lojistik Regresyon'un da yakalayabilmesi için ek olarak **yaş grubu** (6 aralık) One-Hot özelliği denenecek. |
| `balance` | Çok sağa çarpık (çarpıklık 8,36), negatif değerler var. IQR aykırı oranı %10,46. Aykırılarda kabul %16,22, diğerlerinde %11,17. | Aykırılar **silinmez** (bilgi taşıyor). LR için `sign(x)·log1p(|x|)` dönüşümü ve ardından `StandardScaler` uygulanır. Ağaç modelleri ham değeri kullanır. |
| `campaign` | Çarpıklık 4,90. Arama sayısı arttıkça kabul düzenli olarak düşüyor (1 arama %14,60 → 11+ arama %3,93). Aykırılarda (>6) kabul %5,25. | Silinmez. `log1p` + ölçekleme (LR). Monoton ilişki korunur. |
| `pdays` | Kayıtların %81,74'ünde −1 ("daha önce aranmamış"). Aranmış müşterilerde kabul %23,07, aranmamışlarda %9,16. | Yeni ikili özellik **`was_contacted_before`** (pdays ≠ −1). −1 değeri sayısal olarak anlamsız olduğundan aranmamışlar için `pdays` 0'a çekilir; ilişki bayrakla taşınır. |
| `previous` | `pdays` ile Spearman korelasyonu 0,986. IQR = 0 olduğu için aranmış her müşteri "aykırı" görünür (%18,26); yalnızca aranmışlar içinde aykırı oranı %5,49. | IQR ile aykırı ayıklaması bu değişkende **uygulanmaz**. `log1p` + ölçekleme. Çoklu doğrusallık nedeniyle LR'de `pdays`/`previous` birlikteliği katsayı yorumunda dikkate alınır; L2 düzenlileştirme kullanılır. |
| `day` | Zayıf etki (r = −0,053). Ayın gününe göre dalgalı ve doğrusal olmayan bir desen var. | Sayısal olarak kalır (ağaçlar için yeterli). LR'deki katkısı Hafta 5'te ablasyonla kontrol edilir. |

## Genel kararlar

- **Aykırı değerler:** Hiçbir değişkende satır silinmeyecek. Aykırı kayıtlar hedefle anlamlı biçimde farklı oranlar taşıyor; silmek bilgi kaybı ve seçim yanlılığı yaratır. Etkileri dönüşümle (log, ölçekleme) yumuşatılacak.
- **Kodlama:** `OneHotEncoder(handle_unknown="ignore")`, yani eğitimde görülmeyen bir kategori tahmin anında hata vermez.
- **Ölçekleme:** Yalnızca Lojistik Regresyon pipeline'ında. Rastgele Orman ve XGBoost ölçeklemeye duyarsız olduğundan dönüşümsüz kullanılır.
- **İstatistiksel testlerin yorumu:** 45.211 kayıtta p-değerleri neredeyse her değişken için 0,001'in altında çıkıyor (istisna `age`). Bu yüzden değişken eleme kararı p-değerine değil, **etki büyüklüğüne** (Cramér's V, rank-biserial r) ve CV'deki model katkısına göre verilecek. Bu aşamada hiçbir değişken elenmiyor.

## Hafta 4'te doğrulanacak açık sorular

1. Yaş grubu özelliği LR'nin PR-AUC değerini CV'de iyileştiriyor mu?
2. `contact = unknown` ile `month` birlikteliği modelde "dönem etkisi" yaratıyor mu? (Hafta 8 hata analizine taşınabilir.)
3. Az gözlemli aylar (aralık, n = 214) için kategori birleştirme gerekli mi?
