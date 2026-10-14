# Veri Sözlüğü

| Değişken | Tür | Açıklama | Olası değerler veya aralık | "unknown" içeriyor mu? | Modelde kullanılacak mı? |
| --- | --- | --- | --- | --- | --- |
| age | sayısal | age alanı için gözlenen değer aralığıdır. | 18 - 95 | false | evet |
| job | kategorik | Kategorik özellik; müşteri profilini tanımlar. | admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician ... | true | evet |
| marital | kategorik | Kategorik özellik; müşteri profilini tanımlar. | divorced, married, single | false | evet |
| education | kategorik | Kategorik özellik; müşteri profilini tanımlar. | primary, secondary, tertiary, unknown | true | evet |
| default | kategorik | Kategorik özellik; müşteri profilini tanımlar. | no, yes | false | evet |
| balance | sayısal | balance alanı için gözlenen değer aralığıdır. | -8019 - 102127 | false | evet |
| housing | kategorik | Kategorik özellik; müşteri profilini tanımlar. | no, yes | false | evet |
| loan | kategorik | Kategorik özellik; müşteri profilini tanımlar. | no, yes | false | evet |
| contact | kategorik | Kategorik özellik; müşteri profilini tanımlar. | cellular, telephone, unknown | true | evet |
| day | sayısal | day alanı için gözlenen değer aralığıdır. | 1 - 31 | false | evet |
| month | kategorik | Kategorik özellik; müşteri profilini tanımlar. | apr, aug, dec, feb, jan, jul, jun, mar, may, nov ... | false | evet |
| duration | sayısal | Müşterinin arama sırasında konuşma süresi. Hedefe çok yakın olduğundan modelde kullanılmaz. | 0 - 4918 | false | hayır (veri sızıntısı riski) |
| campaign | sayısal | campaign alanı için gözlenen değer aralığıdır. | 1 - 63 | false | evet |
| pdays | sayısal | pdays alanı için gözlenen değer aralığıdır. | -1 - 871 | false | evet |
| previous | sayısal | previous alanı için gözlenen değer aralığıdır. | 0 - 275 | false | evet |
| poutcome | kategorik | Kategorik özellik; müşteri profilini tanımlar. | failure, other, success, unknown | true | evet |
| y | kategorik | Kategorik özellik; müşteri profilini tanımlar. | no, yes | false | evet |
