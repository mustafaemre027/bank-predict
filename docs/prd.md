# PRD — Banka Pazarlama Kampanyalarına Müşteri Yanıtının Tahmini

> Ekip: BankTech · Ömer NARÇİÇEĞİ (230260054), Mustafa Emre KAPLAN (230260143), Mesut ASLAN (230260173)
> Ders: Bilgisayar Mühendisliği Tasarımı, Fırat Üniversitesi · Sürüm 1.0 (Hafta 3)
> İlgili belgeler: [Gereksinim Analizi](gereksinim_analizi.md) · [Yol Haritası](yol_haritasi.md) · [Ön İşleme Kararları](on_isleme_kararlari.md) · [Veri Sözlüğü](veri_sozlugu.md)

## Problem

Bir bankanın pazarlama birimi, vadeli mevduat ürününü satmak için çağrı merkezi üzerinden binlerce müşteriyi telefonla arıyor. Aramaların büyük çoğunluğu sonuçsuz kalıyor: geçmiş kampanya verisinde müşterilerin yalnızca **%11,70**'i teklifi kabul etmiş. Hangi müşterinin olumlu yanıt vereceği önceden bilinmediği için çağrı merkezi zamanının yaklaşık %88'i teklifi reddedecek müşterilere harcanıyor. Aynı müşteriyi defalarca aramak da sonucu iyileştirmiyor: veride tek aramada kabul oranı %14,60 iken 11 ve üzeri aramada %3,93'e düşüyor. Bu durum hem işgücü maliyetini artırıyor hem de gereksiz yere rahatsız edilen müşterilerde memnuniyetsizlik yaratıyor.

Pazarlama yöneticisinin elinde arama listesini önceliklendirecek nesnel bir araç yok. Liste genellikle sezgiye ya da basit kurallara göre hazırlanıyor. Çağrı merkezi çalışanı da aradığı müşterinin kabul ihtimalinin yüksek mi düşük mü olduğunu bilmiyor.

## Çözüm

Arama **yapılmadan önce** bilinen müşteri bilgilerinden (demografi, finansal durum, önceki kampanya geçmişi, planlanan iletişim zamanı) yola çıkarak her müşterinin teklifi kabul etme **olasılığını** tahmin eden bir makine öğrenmesi modeli geliştirilecek. Model, aramanın süresi gibi ancak arama bittikten sonra öğrenilebilen bilgileri **kesinlikle kullanmayacak**.

Pazarlama yöneticisi modeli kullanarak müşteri listesini kabul olasılığına göre sıralayabilecek ve aramaları en olası müşterilerden başlatabilecek. Model ayrıca "listenin ilk %X'ini ararsak olası kabullerin ne kadarını yakalarız?" sorusunu yanıtlayacak. Basit bir web arayüzünde tek bir müşterinin bilgileri girilerek kabul olasılığı ve bu olasılığı en çok etkileyen etkenler görülebilecek. Teknik bilgisi olmayan bir çalışan da arayüzü kullanabilecek.

## Kullanıcı Hikayeleri

**Pazarlama yöneticisi**

1. Bir pazarlama yöneticisi olarak, müşteri listemin kabul olasılığına göre sıralanmasını istiyorum, böylece çağrı merkezinin kapasitesini en olası müşterilere ayırabilirim.
2. Bir pazarlama yöneticisi olarak, "listenin ilk %20'sini ararsam kabul edecek müşterilerin yüzde kaçına ulaşırım" sorusunun cevabını görmek istiyorum, böylece kampanya bütçesini gerçekçi planlayabilirim.
3. Bir pazarlama yöneticisi olarak, modelin hangi müşteri özelliklerine dayanarak karar verdiğini görmek istiyorum, böylece modele güvenip güvenmeyeceğime karar verebilirim.
4. Bir pazarlama yöneticisi olarak, modelin yalnızca arama öncesinde bilinen bilgileri kullandığından emin olmak istiyorum, böylece test sonuçlarının gerçek kampanyada da geçerli olacağını bilirim.
5. Bir pazarlama yöneticisi olarak, modelin "kabul eder" demesi için kullanılan olasılık eşiğinin neye göre seçildiğini bilmek istiyorum, böylece daha fazla müşteri yakalamak ile daha az boşa arama yapmak arasındaki dengeyi kurabilirim.
6. Bir pazarlama yöneticisi olarak, modelin başarısının yalnızca doğruluk oranıyla değil, kabul edenleri ne kadar yakaladığıyla raporlanmasını istiyorum, böylece "herkese hayır diyen" yanıltıcı bir modelle karşılaşmam.
7. Bir pazarlama yöneticisi olarak, aynı müşteriyi defalarca aramanın kabul ihtimalini nasıl etkilediğini görmek istiyorum, böylece müşteri başına arama sınırı koyabilirim.
8. Bir pazarlama yöneticisi olarak, önceki kampanyalarda olumlu yanıt vermiş müşterilerin ne kadar değerli olduğunu bilmek istiyorum, böylece onlara öncelik verebilirim.
9. Bir pazarlama yöneticisi olarak, kampanya zamanlamasının (ay) kabul oranını nasıl etkilediğini görmek istiyorum, böylece kampanyaları daha verimli dönemlere planlayabilirim.

**Çağrı merkezi çalışanı**

10. Bir çağrı merkezi çalışanı olarak, bir müşterinin bilgilerini basit bir forma girip kabul olasılığını görmek istiyorum, böylece görüşmeye nasıl yaklaşacağımı planlayabilirim.
11. Bir çağrı merkezi çalışanı olarak, olasılığın yanında "yüksek / orta / düşük" gibi anlaşılır bir etiket görmek istiyorum, böylece sayıları yorumlamakla vakit kaybetmem.
12. Bir çağrı merkezi çalışanı olarak, formda yalnızca geçerli seçenekleri (örneğin meslek listesini) seçebilmek istiyorum, böylece yanlış veri girip hatalı sonuç almam.
13. Bir çağrı merkezi çalışanı olarak, tahminin bir saniyeden kısa sürede gelmesini istiyorum, böylece arama akışım yavaşlamaz.
14. Bir çağrı merkezi çalışanı olarak, bilinmeyen bir bilgiyi (örneğin eğitim düzeyi) "bilinmiyor" olarak girebilmek istiyorum, böylece eksik bilgili müşteriler için de tahmin alabilirim.
15. Bir çağrı merkezi çalışanı olarak, olasılığı artıran ve azaltan başlıca etkenleri görmek istiyorum, böylece görüşmede hangi konulara odaklanacağımı bilirim.

**Proje ekibi (veri bilimci / geliştirici)**

16. Bir ekip üyesi olarak, veri setinin her yüklenişinde satır, sütun ve hedef sayılarının otomatik doğrulanmasını istiyorum, böylece bozuk ya da farklı bir veriyle çalışmadığımdan emin olurum.
17. Bir ekip üyesi olarak, ön işleme adımlarının tek bir pipeline içinde ve yalnızca eğitim verisine uyarlanarak çalışmasını istiyorum, böylece test verisinden bilgi sızmaz.
18. Bir ekip üyesi olarak, sınıf dengesizliği için SMOTE, sınıf ağırlıklandırma ve eşik ayarlamayı aynı çapraz doğrulama düzeninde karşılaştırmak istiyorum, böylece hangisinin gerçekten işe yaradığını adil biçimde görürüm.
19. Bir ekip üyesi olarak, Lojistik Regresyon, Rastgele Orman ve XGBoost modellerini aynı metriklerle yan yana görmek istiyorum, böylece en iyi modeli gerekçeli seçebilirim.
20. Bir ekip üyesi olarak, hiperparametre aramasının PR-AUC'yi optimize etmesini istiyorum, böylece dengesiz veride doğru hedefe göre ayar yapılır.
21. Bir ekip üyesi olarak, test setinin yalnızca final değerlendirmede bir kez kullanılmasını istiyorum, böylece raporlanan sonuçlar iyimser biçimde şişmez.
22. Bir ekip üyesi olarak, tüm deneylerin sabit rastgelelik tohumu ve kilitli kütüphane sürümleriyle tekrar üretilebilmesini istiyorum, böylece başka bir bilgisayarda aynı sonuçları alabilirim.
23. Bir ekip üyesi olarak, seçilen modeli diske kaydedip yeniden yükleyebilmek istiyorum, böylece arayüz her açılışta modeli yeniden eğitmez.
24. Bir ekip üyesi olarak, ortak fonksiyonların otomatik testlerle korunmasını istiyorum, böylece bir arkadaşımın değişikliği farkında olmadan analizimi bozmaz.
25. Bir ekip üyesi olarak, haftalık raporun sayılarının doğrudan koddan üretilmesini istiyorum, böylece raporla analiz arasında tutarsızlık oluşmaz.
26. Bir ekip üyesi (o haftanın takım kaptanı) olarak, rapor üreticiye kaptanı parametre olarak verebilmek istiyorum, böylece kaptan her hafta değiştiğinde kodu düzenlemem gerekmez.

**Ders sorumlusu öğretim üyesi**

27. Bir öğretim üyesi olarak, modelin literatürdeki çalışmalarla karşılaştırıldığını ve farkların gerekçelendirildiğini görmek istiyorum, böylece projenin bilimsel değerini ölçebilirim.
28. Bir öğretim üyesi olarak, proje deposundaki kodu tek komutla çalıştırıp rapordaki sonuçları yeniden üretebilmek istiyorum, böylece sonuçların gerçekliğini doğrulayabilirim.
29. Bir öğretim üyesi olarak, iş paketlerinin hangi üye tarafından ne zaman tamamlandığını haftalık raporlarda izlemek istiyorum, böylece takım çalışmasını değerlendirebilirim.

## Uygulama Kararları

**Modüller.** Her modül tek bir sorumluluğu sade bir arayüzün arkasında toplar ve izole test edilebilir.

| Modül | Sorumluluk | Arayüz (özet) | Hafta |
|---|---|---|---|
| Veri yükleme ve doğrulama | Ham veriyi okur ve beklenen şemaya karşı doğrular | Ham veri çerçevesi döndürür; doğrulama başarısızsa anlamlı hata verir | H2 ✓ |
| Keşifsel analiz | Kabul oranı, aykırı değer, ki-kare/Cramér's V, Mann-Whitney U | Veri çerçevesi + sütun listesi → özet tablo | H3 ✓ |
| Rapor üretici | Haftalık raporu ders şablonuna uygun .docx olarak üretir | Hafta + kaptan numarası + özet JSON → .docx | H3 ✓ |
| Ön işleme | Özellik mühendisliği + kodlama + ölçekleme; eğitim/test bölmesi | Model türüne göre yapılandırılmış, fit edilmemiş pipeline | H4 |
| Modeller | LR, RF, XGBoost için dengesizlik stratejisiyle birlikte pipeline kurar | Model adı + strateji → tahminci | H5-H6 |
| Değerlendirme | Metrikler, CV sonuç tablosu, eşik seçimi, lift/kazanç | Gerçek etiket + olasılık → metrik sözlüğü / eşik | H2 ✓ + H5-H8 |
| Açıklama | Permutation importance ve LR katsayıları | Eğitilmiş pipeline + veri → önem tablosu | H8 |
| Tahmin servisi | Kayıtlı modeli yükler, tek müşteri girdisini doğrular, olasılık + etiket döndürür | Müşteri sözlüğü → olasılık, etiket, başlıca etkenler | H9 |
| Arayüz | Streamlit formu ve sonuç ekranı | Tahmin servisini çağırır, iş mantığı içermez | H9 |

**Mimari ve veri akışı kararları**

- Ham veri hiçbir zaman değiştirilmez. Tüm dönüşümler kod içinde, tekrar üretilebilir biçimde yapılır.
- `duration` değişkeni modelden kalıcı olarak çıkarılmıştır (veri sızıntısı).
- `unknown` değerler eksik veri sayılmaz, ayrı kategori olarak kodlanır. `pdays = -1` "önceden aranmamış" anlamındadır; bundan `was_contacted_before` ikili özelliği türetilir.
- Aykırı değerler silinmez (hedefle ilgili bilgi taşıdıkları EDA'da gösterildi). Lojistik Regresyon için log dönüşümü ve standartlaştırma uygulanır, ağaç modelleri ham değeri kullanır.
- **Veri bölme:** %80 eğitim / %20 test, hedefe göre katmanlı, sabit tohum 42. Test seti Hafta 8'e kadar **kilitlidir**.
- **Model seçimi:** Eğitim verisi üzerinde 5 katlı katmanlı çapraz doğrulama. Birincil metrik PR-AUC, ikincil metrik ROC-AUC.
- **Dengesizlik:** Üç strateji karşılaştırılır: (a) sınıf ağırlığı (`class_weight` / `scale_pos_weight`), (b) SMOTE, yalnızca CV eğitim katmanlarında ve imbalanced-learn pipeline'ı içinde, (c) karar eşiği ayarlama.
- **Eşik seçimi:** Test setinde değil, eğitim verisinin çapraz doğrulama dışı (out-of-fold) tahminlerinde seçilir. Varsayılan kriter F1'i en büyüklemektir. İş senaryosu için "hedef Recall'a ulaşan en yüksek eşik" alternatifi de raporlanır.
- **Hiperparametre:** Rastgele arama, PR-AUC'ye göre, sabit iterasyon bütçesiyle (dizüstü bilgisayarda çalışabilir süre).
- **Açıklanabilirlik:** Permutation importance (modelden bağımsız) + LR katsayıları. SHAP kullanılmaz.
- **Kayıt:** Seçilen pipeline (ön işleme + model + eşik) tek bir artefakt olarak diske kaydedilir. Artefakt git'e eklenmez.
- **Arayüz:** Streamlit. Form alanları veri sözlüğündeki geçerli değerlerle sınırlıdır. Sayısal alanlar için alt/üst sınır doğrulaması yapılır. Hata durumunda kullanıcıya teknik ayrıntı değil anlaşılır bir mesaj gösterilir.
- **Rapor üretimi:** Her hafta bir analiz betiği özet JSON ve grafik üretir; rapor üretici metindeki sayıları bu JSON'dan okur. Kaptan komut satırı parametresidir.

**Tahmin servisi sözleşmesi (H9).** Girdi, modelde kullanılan 15 değişkenin değerlerini içeren bir sözlüktür. Kategorik değerler veri sözlüğündeki kümelerden biri olmalıdır, sayısal değerler veri setinde gözlenen aralıklarda olmalıdır. Çıktı alanları:
- Kabul olasılığı (0-1)
- Seçilen eşiğe göre "kabul eder / etmez" kararı
- Olasılık bandı (yüksek / orta / düşük)
- Kararı en çok etkileyen ilk 3 değişken

## Test Kararları

- **İyi test tanımı:** Yalnızca dışa dönük davranış test edilir (fonksiyonun girdisine karşı çıktısı), iç ayrıntılar test edilmez. Hesaplamalar, sonucu elle doğrulanabilen küçük sentetik veri çerçeveleriyle sınanır. Her test tek bir davranışı doğrular ve adı o davranışı anlatır. Rastgelelik sabit tohumla deterministiktir.
- **Mevcut örnekler:** Veri doğrulama testi (gerçek veri dosyasının şemasını doğrular), metrik testi (elle hesaplanmış örnekle), EDA testleri (tam ilişki → Cramér's V = 1, bağımsızlık → 0, IQR sınırları, sıfır ağırlıklı dağılım uç durumu), rapor üretici testleri (bölüm başlıkları, numaralandırma, JSON'daki sayıların metne aktarılması).
- **Test edilecek modüller ve kritik davranışlar:**
  - *Ön işleme (H4):*
    - `duration` çıktıda bulunmaz.
    - Pipeline yalnızca eğitim verisine fit edilir; test verisinin istatistikleri ölçekleyiciye sızmaz.
    - Eğitimde görülmeyen kategori hata vermez.
    - `was_contacted_before` doğru türetilir.
    - Bölme sınıf oranını korur.
  - *Modeller ve değerlendirme (H5-H8):*
    - SMOTE yalnızca eğitim katmanlarında çalışır (doğrulama katmanının boyutu ve sınıf oranı değişmez).
    - Eşik seçimi yalnızca out-of-fold tahminleri kullanır.
    - Aynı tohumla iki eğitim aynı skoru verir.
    - Lift/kazanç hesabı elle hesaplanmış örnekle uyuşur.
  - *Tahmin servisi (H9):*
    - Geçerli girdi 0-1 arası olasılık döndürür.
    - Geçersiz kategori ya da aralık dışı sayı anlamlı doğrulama hatası verir.
    - Kaydedilip yüklenen model, kaydetmeden önceki modelle aynı tahmini verir.
    - Tek tahmin 1 saniyenin altında tamamlanır.
  - *Arayüz:* İş mantığı tahmin servisinde olduğu için arayüz ince tutulur ve elle kabul testiyle doğrulanır (demo senaryosu).
- **Kapsam hedefi:** `src/` altındaki iş mantığında %80+ satır kapsamı. Kritik yollarda (sızıntı önleme, eşik seçimi, girdi doğrulama) eksiksiz kapsam.

## Kabul Kriterleri

1. Ham veri doğrulaması geçer: 45.211 satır, 17 sütun, 5.289 olumlu hedef.
2. Modele giren özellikler arasında `duration` yoktur; bu bir testle doğrulanır.
3. Tüm test paketi (`pytest`) temiz bir ortamda, `requirements.txt` kurulumundan sonra hatasız geçer.
4. LR, RF ve XGBoost için 5 katlı CV sonuçları (ortalama ± standart sapma: PR-AUC, ROC-AUC, F1, Recall, Precision) tek bir karşılaştırma tablosunda raporlanır. En az üç dengesizlik stratejisi karşılaştırılmıştır.
5. Seçilen model, kilitli test setinde **ROC-AUC ≥ 0,75** değerine ulaşır (FOG-1).
6. Seçilen modelin test PR-AUC değeri, referans modelin PR-AUC değerinin (0,117) **en az 2 katıdır** (≥ 0,234).
7. Seçilen eşikte test Recall'u referans modelin Recall'undan (0,00) yüksektir. Karmaşıklık matrisi raporlanır.
8. Kazanç (gain) tablosu raporlanır: olasılığa göre sıralı listenin ilk %10 / %20 / %30'u arandığında yakalanan kabullerin oranı.
9. En etkili ilk 10 değişken permutation importance ile grafikte gösterilir. LR katsayılarıyla tutarlılığı yorumlanır.
10. Kaydedilen model yeniden yüklendiğinde test setinde aynı skorları üretir.
11. Streamlit arayüzünde bir müşteri girilip olasılık, etiket ve ilk 3 etken 1 saniyenin altında görüntülenir. Geçersiz girdide kullanıcı dostu hata mesajı görünür.
12. Analiz ve rapor betikleri temiz bir klonda belgelenen komutlarla çalışır ve aynı sayıları üretir (sabit tohum, kilitli sürümler).
13. Her hafta raporu, sayıları o haftanın özet JSON'undan gelecek şekilde rapor üreticiyle oluşturulur.

## Kapsam Dışı

- Bankanın gerçek CRM, çağrı merkezi ya da veri tabanı sistemlerine entegrasyon; canlı (production) dağıtım, kimlik doğrulama ve kullanıcı yönetimi.
- `duration` değişkeninin modelde herhangi bir biçimde kullanılması (analizde yalnızca neden çıkarıldığını göstermek için yer alır).
- UCI'deki `bank-additional` sürümü (41.188 kayıt, sosyoekonomik göstergeler) ve dış veri kaynakları.
- SHAP gibi ek açıklanabilirlik kütüphaneleri; derin öğrenme modelleri.
- Gerçek maliyet/kazanç rakamlarıyla finansal getiri hesabı (banka verisi olmadığından). Kazanç analizi oransal yapılır.
- Zaman içinde model izleme (drift) ve otomatik yeniden eğitim.

## Notlar

- **Literatür karşılaştırması:** Moro vd. (2014) çalışması, `bank-full.csv`'den daha geniş bir veri sürümü kullanır: 2008-2013 arası 52.944 iletişim ve özellik seçiminden önce 150 değişken. Ayrıca zamana göre kayan pencereyle değerlendirme yapar. Bu nedenle 0,794 AUC değeriyle karşılaştırma **yalnızca göstergedir**. Final raporunda bu fark açıkça belirtilecektir.
- **Zaman sırası riski:** `bank-full.csv` kayıtları tarihe göre sıralıdır (Mayıs 2008 - Kasım 2010). `contact = unknown` değerinin mayıs-haziran aylarında yoğunlaşması (H3 bulgusu) bir dönem etkisine işaret eder. Rastgele katmanlı bölme bu etkiyi gizleyebilir. Hafta 8'de ek bir kontrol olarak "son %20 kayıt" zaman bazlı test kümesinde de skor raporlanacak (bilgilendirme amaçlı).
- **Kaptan rotasyonu:** Kaptan her hafta değişebildiğinden ekip bilgisi koda sabitlenmez. Rapor üreticiye `--kaptan` parametresiyle verilir. Rotasyon tablosu yol haritasındadır.
- **Proje kuralları** (`.github/copilot-instructions.md`): raporlara giren her sayı çalıştırılan koddan üretilir, `random_state = 42`, yeniden kullanılan kod `src/` altında durur.
