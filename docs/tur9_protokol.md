## 22. 9. araştırma turu protokolü: büyüme-optimal kaldıraç ("Büyüme Motoru")

**Bu bölüm 2026-10-18 ~00:10 UTC'de yazıldı.** O sırada 8. turun verisi henüz indirilmemişti. 8. tur da bu tur da
hiçbir veride çalıştırılmamıştı ve bu turun kodu yazılmamıştı. Bu bölüm kodun ne yapacağını da tanımlar. Betik:
`arastirma_tur9.py`.

### Burak'ın isteği (2026-10-18)
50 USDT'yi "binlerce katına" çıkarma hedefine en çok yaklaşan, en güçlü sistem.

### Fikir: büyüme-optimal risk (Kelly)
- **Bileşik büyüme logaritmiktir.** Kaldıraç çarpanı k ile yıllık büyüme kabaca g(k) = k·μ − k²·σ²/2'dir.
  - μ: stratejinin maliyetler düşülmüş ortalama getirisi.
  - σ: stratejinin oynaklığı.
- **Zirve:** g(k), k\* = μ/σ²'de en yüksektir (Kelly, 1956).
  - Breiman (1961): bu strateji, büyük bir hedefe ulaşmanın beklenen süresini de uzun vadede en aza indirir.
  - Yani "binlerce kata en hızlı yol" k\*'dır, en yüksek kaldıraç değildir.
- **Zirveden sonra:**
  - 2k\*'da büyüme sıfırdır.
  - Daha yukarısında kârlı bir sistem bile uzun vadede neredeyse kesin sıfıra gider.
  - Fazla risk, az riskten çok daha ağır cezalandırılır (Thorp, 2006).
- **Yarım Kelly (k\*/2):**
  - Zirve büyümenin 3/4'ünü verir.
  - Başlangıç parasının yarısına hiç inme olasılığı tam Kelly'de 1/2, yarım Kelly'de 1/8'dir (sürekli zaman
    modeli; Thorp, 2006).
- **Tahmin hatası:**
  - μ belirsizdir. 97 yayınlanmış stratejide getiri, yayından sonra ortalama %58 düştü (McLean ve Pontiff, 2016).
  - μ yarıya inerse k\* da yarıya iner. Bu yüzden k\* "yarıya indirilmiş avantajla" hesaplanır.
- **Kripto getirileri normal dağılmaz.** Ani çöküşler formülü bozar. Bu yüzden formül yerine gerçek günlük
  getiriler (blok bootstrap) ve gün içi likidasyon kontrolü kullanılır.

### Önceden bilinenler (dürüstlük)
- **6. turdaki Kelly taraması yanıltıcıydı** (C1 ve R1).
  - Tarama şaltersizdi ve sınav dönemini içeriyordu.
  - Bu turda k\* yalnızca seçim döneminden (P1 + P2) hesaplanır ve P3'te doğrulanır.
- **Funding büyüktür.** 6. turda funding, trend takibinin avantajı kadar büyüktü. Kaldıraç funding'i de büyütür;
  gerçek funding geçmişi kullanılır.
- **8. turun sonucu bilinmiyordu.** Bu protokol yazıldığında 8. tur çalıştırılmamıştı.

### Temel strateji
8. turun (21. bölüm) hedef ağırlıkları, hiçbir değişiklik olmadan:
- **L (yalnızca long):** T1-25'in ağırlıkları.
- **LS (long/short):** T2-25'in ağırlıkları.

**Kaldıraç çarpanı:**
- Ağırlıklar 8. turdaki gibi (sınırlar dahil) hesaplanır, sonra k ile çarpılır.
- Izgara: k ∈ {1; 1,5; 2; 3; 4; 6; 8}. k = 1, 8. turun temel risk seviyesidir.
- Bütün koşular vadelide yapılır (USDⓈ-M, çapraz marjin), k = 1 dahil.

### Vadeli işlem modeli
- **8. turla aynı:**
  - Karar günlük kapanışta, işlem sonraki açılışta; işlem bandı %20.
  - %0,05 komisyon + %0,05 kayma.
  - Gerçek funding. Sözleşmenin ilk funding kaydından önce long 8 saatte %0,01 öder ve short açılmaz.
  - Ölçümler (getiri, CAGR, en kötü düşüş, Sharpe) 8. turdaki gibi hesaplanır.
- **Likidasyon (yeni; çapraz marjin, kötümser):**
  - Her gün, açılıştaki işlemlerden ve funding'den sonra pozisyonlar aynı anda değerlenir: long pozisyonlar günün en
    düşük fiyatıyla, short pozisyonlar günün en yüksek fiyatıyla.
  - Bu en kötü özsermaye bakım marjinine eşit ya da altındaysa hesap likide olur ve özsermayenin tamamı kaybedilir.
    Bakım marjini, o fiyatlarla pozisyon büyüklüğünün %2'sidir (6. turdaki gibi kötümser).
- **Tarihsel koşu:** Likidasyonda yol biter (özsermaye 0).
- **Bootstrap malzemesi:**
  - Likidasyon günü −%100 getiri olarak kaydedilir.
  - Seri, bir önceki kapanıştaki özsermayeyle ve pozisyonsuz devam eder.
  - Böylece likidasyon günü, örneklendiği her yolda batış olur.

### Dönemler
Dönemler 8. turla aynıdır. Her koşu 10.000 USDT ile ayrı başlar.

| Kod | Tarihler | Rolü |
|---|---|---|
| P1 | 2018-09-01 → 2021-12-31 | Düşüş ve likidasyon kontrolü |
| P2 | 2022-01-01 → 2024-12-31 | Düşüş ve likidasyon kontrolü |
| S (seçim) | 2018-09-01 → 2024-12-31 (P1 + P2, tek parça) | k\* yalnızca buradan hesaplanır |
| P3 | 2025-01-01 → verideki son tam gün | Doğrulama; seçim P3'ü görmez |
| TAM | 2018-09-01 → verideki son tam gün | Yalnızca projeksiyon (bilgi) |

### Büyüme ölçümü (seçim dönemi)
- **Malzeme:** Her yön ve k için S'nin günlük getirileri (bootstrap malzemesi).
- **Bootstrap:**
  - 20 günlük bloklar; 10.000 yol × 10 yıl (3.653 gün); tohum 20261018.
  - Bütün adaylar aynı blok sırasını kullanır.
- **Yarıya indirilmiş avantaj:**
  - Her günün getirisinden k × h çıkarılır (en fazla −%100'e kadar).
  - h, aynı yönde k = 1'in S'deki ortalama günlük getirisinin yarısıdır. Bu ortalama negatifse h = 0.
  - Likidasyon günleri −%100 kalır.
- **g(k):** Yolların yıllıklandırılmış logaritmik büyümesinin medyanı, yarıya indirilmiş avantajla. Batan yol −∞
  sayılır.
- **k\*:** g(k)'yı en yüksek yapan ızgara değeri. Eşitlikte düşük olan seçilir.

### Uygunluk (8. turdan)
Bir yön ancak 8. turdaki temel adayı "avantaj" ölçütlerini geçtiyse kaldıraçlanabilir:
- **L:** T1-25, 8. turun 1. (kâr), 3. (al-tut'a göre) ve 4. (trendin değeri) ölçütlerini geçmeli.
- **LS:** T2-25 aynı üç ölçütü ve 5. ölçütü (short'un değeri) geçmeli.

8. turun 2. ölçütü (düşüş) burada aranmaz; düşüşü profiller ele alır. Uygun olmayan yönün sonuçları yalnızca bilgi
olarak raporlanır.

### Profiller (her uygun yön için)
1. **Büyüme (yarım Kelly):** k\*/2'yi aşmayan en büyük ızgara değeri. En az 1.
2. **Korumalı (%25 sınırı):** Büyüme profilini aşmayan ızgara değerlerinden biridir. k = 1'den başlayarak şu iki
   koşulu kesintisiz sağlayan en büyük değer seçilir:
   - P1'de ve P2'de en kötü düşüş ≤ %25 (Burak'ın sınırı).
   - S'de likidasyon yok.

   k = 1 bile bu koşulları sağlamıyorsa Korumalı profil yoktur.

- İki profil aynı k'yi seçerse tek seçenek vardır.
- Profilin k'si 1 ise kaldıraç önerilmez; o yön için 8. turun kararı geçerlidir.

### Doğrulama ölçütleri (k > 1 olan her profil için)
1. P3'te net kâr > 0.
2. Hiçbir koşuda (P1, P2, S, P3) likidasyon yok.
3. **Kaldıraç karşılığını vermeli:** P3'teki CAGR, aynı yönde k = 1'in P3 CAGR'ından yüksek.
4. **Yalnızca Korumalı:** P3'te en kötü düşüş ≤ %25.

Ölçütleri geçemeyen profil benimsenmez. Daha düşük bir k'ye inilmez; P3'e bakarak ayar yapılmaz.

### Seçim ve karar
- **Yön:**
  - İki yönün aynı profili de geçerse, S'de g'si yüksek olan önerilir.
  - Fark yıllık 1 puandan azsa L (daha basit) önerilir.
- **Son karar Burak'ındır:** Korumalı profil, Büyüme profili ya da 8. turun kaldıraçsız sonucu.
  - Büyüme profili %25'ten büyük düşüş demek olabilir. Ancak Burak bunu açıkça kabul ederse kullanılır.
- Hiçbir profil geçmezse kaldıraç kullanılmaz.

### Bilgi amaçlı (karar değiştirmez)
- **Tam ızgara:** Her yön ve k için P1, P2, P3 ve S'de şunlar raporlanır: getiri, en kötü düşüş, Sharpe, ortalama
  brüt pozisyon, funding ve likidasyonlar.
- **g(k) eğrisi:**
  - Hem yarıya indirilmiş hem tarihsel avantajla, grafikle.
  - Ayrıca 10 yılda likidasyon ve %50 düşüş olasılıkları.
- **Projeksiyon (her yönde k = 1 ve geçen profiller için):**
  - Malzeme: TAM'ın günlük getirileri. 10.000 yol × 20 yıl (7.305 gün), aynı tohum.
  - 10, 100 ve 1.000 kata 10 ve 20 yılda ulaşma olasılığı.
  - 10 yılda likidasyon ve %50 düşüş olasılığı.
  - **Hedef:** 50 USDT'den başlayıp 30 günde bir 0, 50 ya da 100 USDT katkıyla 50.000 USDT'ye (1.000 kat) 5, 10,
    15 ve 20 yılda ulaşma olasılığı ve medyan süre. Likidasyondan sonra katkılar sıfırdan devam eder.
  - "Ulaşma", o süre içinde herhangi bir gün o seviyeye çıkmaktır. Yolların yarısından azı 20 yılda ulaşıyorsa
    medyan süre "20 yıldan uzun" yazılır.
  - Hepsi hem tarihsel hem yarıya indirilmiş avantajla. Projeksiyonda h, TAM'daki k = 1'den hesaplanır.
- **50 USDT ile gerçek vadeli kurallar:**
  - Her yönde k = 1 ve geçen profiller TAM'da, gerçek miktar adımı ve en küçük tutarla çalıştırılır.
  - Son bakiye, düşüş ve küçük olduğu için atlanan işlemlerin oranı raporlanır.

### Sonrası
- 8. turun "Sonrası" bölümü geçerlidir: en az 4 hafta sanal test, sonra yalnızca Burak'ın açık onayıyla küçük gerçek
  para.
- Vadeli işlem için API anahtarında "Enable Futures" açılmalıdır. Para çekme kapalı, IP kısıtlaması açık kalır.
- 50 USDT ile işlemlerin yarısından fazlası atlanıyorsa profil bu sermayeyle uygulanamaz. O zaman ya sermaye artar
  (katkı) ya da kaldıraçsız sonuç kullanılır.

### Dürüstlük notları (önceden)
- **Projeksiyon tahmin değildir.** Geçmişin aynı karışımla tekrarlanacağını varsayar. Yarıya indirilmiş avantaj daha
  gerçekçi senaryodur.
- **Bootstrap uzun bağımlılıkları kısmen bozar.** 20 günden uzun trendler ve rejimler bloklar arasında kopar.
- **Seçim dönemi iki boğa piyasası içerir:** 2019-2021 ve 2023-2024.
- **Likidasyon modeli kötümserdir.** İki coinin en kötü fiyatının aynı anda olduğu varsayılır. Ama ani çöküşlerde
  (10 Ekim 2025 gibi) borsa fiyatı daha da kötü olabilir.
- **Yalnızca iki coin.** Makaledeki 20 coinlik çeşitlendirme yok. Çeşitlendirme Sharpe'ı, dolayısıyla k\*'ı
  yükseltebilir; bu ayrı bir turun konusu.
- **Spot ile vadeli karışımı denenmedi.** İlk 1x'i spotta tutmak long tarafın funding maliyetini düşürebilir.
- **Kurallar sabittir.** Sonuçlar görüldükten sonra hiçbir parametre ya da ölçüt değiştirilmez. Kodda hata bulunursa
  düzeltilir; düzeltme ve etkisi sonuç bölümünde yazılır.

### Kaynaklar (bu bölüm)
- Kelly (1956): "A New Interpretation of Information Rate", *Bell System Technical Journal* 35(4).
- Breiman (1961): "Optimal Gambling Systems for Favorable Games", *4. Berkeley Sempozyumu*.
- Thorp (2006): "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market", *Handbook of Asset and
  Liability Management*, 1. cilt.
- MacLean, Thorp ve Ziemba (2010): "Good and Bad Properties of the Kelly Criterion".
- McLean ve Pontiff (2016): "Does Academic Research Destroy Stock Return Predictability?", *Journal of Finance*
  71(1).
