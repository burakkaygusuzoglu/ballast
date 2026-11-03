## 17. 6. araştırma turu protokolü: vadeli işlemler (futures), kaldıraç ve short

**Bu bölüm 2026-10-15 ~11:05 UTC'de yazıldı.** O sırada hiçbir vadeli işlem backtest'i çalıştırılmamıştı ve
vadeli işlem kodu yazılmamıştı. Bu bölüm kodun ne yapacağını da tanımlar. Betik: `arastirma_tur6.py`.

### Soru
Burak kaldıraç denemek istiyor: "risk daha yüksek olsa da daha iyi kazanç, ama en verimli şekilde." Soru şu:
botlarımızın vadeli işlem sürümü spot bottan daha hızlı büyüyebilir mi? Vadeli işlemin üç farkı var: daha düşük
komisyon, kaldıraç ve short. Vadeli işlem maliyetleri (komisyon, funding, likidasyon) düşüldükten sonra hangi bot,
hangi risk seviyesinde en verimli?

### Önceden bilinenler (dürüstlük)
- **Kelly taraması zaten görüldü.** Araştırma raporundaki tam dönem Kelly taraması (2026-10-10) spot işlemlerle ve
  sınav dönemi dahil yapıldı. Büyüme C1'de ~%1,2, R1'de ~%1,8 riskte tepe yaptı. Risk ızgarasının aralığı bundan
  etkilendi. Seçim kuralı yalnızca keşif dönemini kullanır.
- **Doğrulama dönemi temiz değil.** 3.-5. turlarda kullanıldı. Sınav dönemi de 5. turda kullanıldı. Tek temiz test
  ileriye dönük testtir.
- **Kaldıraç avantaj yaratmaz.** Stratejinin avantajını değiştirmez; yalnızca riski ölçekler, yani kazancı da kaybı
  da büyütür.

### Doğrulanan gerçekler (2026-10-15)
- **Erişim:**
  - Haziran 2026 tarihli kısıtlı ülke listelerinde kullanıcının ülkesi yok.
  - AB/AEA'da (MiCA) USDⓈ-M yerine "Futures Credits (BNFCR)" var. Kullanıcının ülkesi AEA'da değil.
  - Binance TR'de vadeli işlem yok.
  - Kesin kontrol Burak'ın hesabındaki Vadeli sekmesidir.
- **Komisyon (VIP 0, USDⓈ-M):** maker %0,02, taker %0,05 (spotta %0,1). BNB ile ödenirse %10 indirim var; bu
  indirim varsayılmaz.
- **Stop emirleri:**
  - 2025-12-09'dan beri şartlı emirler (STOP_MARKET) ayrı "Algo Order" uç noktasından verilir.
  - Tetiklenmemiş şartlı emir değiştirilemez; iptal edilip yenisi verilir.
  - Hesap başına en fazla 200 şartlı emir açık olabilir.
  - ccxt 4.5.85 bu emirleri otomatik olarak doğru uç noktaya gönderir.
- **Test ortamları:** Futures testnet ve Demo Trading. İkisi de ccxt 4.5.85'te var.
- **Funding:** Temel faiz bileşeni 8 saatte %0,01'dir. Ölçülen ortalamalar: 2024'te yılda ~%11, 2025'te ~%5,
  2026 başında ~%3,7.

### Vadeli işlem modeli (backtest ve paper aynı kodu kullanır)
- **Sözleşme ve kaldıraç:** USDⓈ-M sürekli sözleşme, tek yönlü mod, izole marjin. Her pozisyon 3x kaldıraçla
  açılır (marjin = pozisyon / 3).
- **Dolum:** Spotla aynı kötümser model.
  - Piyasa emri sonraki mumun açılışından dolar: %0,05 kayma ve %0,05 taker komisyonu.
  - Stop, stop fiyatından dolar; mum stopun ötesinde açıldıysa (boşluk) açılıştan. Kayma ve taker düşülür.
  - Short'ta yönler tersinedir.
- **Funding:** UTC 00:00, 08:00 ve 16:00'da o an açık pozisyonlardan alınır. Tutar = pozisyon büyüklüğü × mumun
  açılış fiyatı × oran.
  - Temel model: long 8 saatte %0,01 öder, short %0,01 alır.
  - **Stres modeli (kararlar bununla):** long 8 saatte %0,03 öder (yılda ~%33), short %0,01 öder.
- **Likidasyon (izole):** Likidasyonda pozisyonun marjininin tamamı kaybedilir. Bakım marjini kötümser olarak %2
  alınır:
  - LP_long = giriş × (1 − 1/3) / (1 − 0,02)
  - LP_short = giriş × (1 + 1/3) / (1 + 0,02)
  - Temel model: mum likidasyon fiyatının ötesinde açılırsa likidasyon olur.
  - **Stres modeli (kararlar bununla):** mumun dibi (long) ya da tepesi (short) likidasyon fiyatına değerse
    likidasyon olur. Borsadaki stopun çalışmadığı kabul edilir.
  - Stop mesafesi likidasyon mesafesinin %80'ini aşarsa işlem açılmaz.
- **Boyut:**
  - Formül spotla aynıdır: stop çalışırsa kayıp, maliyetler dahil, özsermayenin "risk" kadarı olur.
  - Tek pozisyon en fazla özsermayenin %100'ü olabilir (spotta %50).
  - Marjin ve komisyon serbest nakitten karşılanmalı. Böylece toplam pozisyon en fazla ~3 × özsermaye olur.
- **Risk ayarları:**
  - Risk seviyesiyle ölçeklenenler: günlük zarar limiti = 6 × risk (C1'de %3); düşüş şalteri = 40 × risk, en
    fazla %60 (C1'de %20).
  - C1 ile aynı kalanlar: en fazla 5 pozisyon, günde en fazla 4 işlem, çıkıştan sonra 4 mum bekleme.
  - Maliyet filtresi 0,25R'dir ve vadeli komisyonlarla hesaplanır.
- **Spot sonuçları değişmez.** C1'in parmak izi aynı kalmalı: 101 işlem, 8479.634912, ecd23151707a20cf.

### Adaylar
- **F-C1:** C1'in kuralları, yalnızca long, vadelide. Kurallar: 4 saatlik kanal kırılımı 20/10, 2×ATR(20) stop,
  12 coin.
- **F-R1:** R1, yani C1 + BTC rejim filtresi, yalnızca long. Yeni long yalnızca BTC'nin günlük kapanışı SMA200'ün
  üstündeyken açılır.
- **F-R1S:** F-R1'e aynalı bir short tarafı eklenir:
  - **Short girişi:** BTC'nin günlük kapanışı SMA200'ün altındayken, kapanış önceki 20 mumun en düşüğünün altına
    inerse.
  - **Short çıkışı:** kapanış önceki 10 mumun en yükseğini aşarsa, sonraki mumun açılışında.
  - **İlk stop:** giriş + 2×ATR(20).
  - **Rejim:** Yükseliş rejiminde short, düşüş rejiminde long açılmaz. Rejim değişince açık pozisyon kapatılmaz
    (R1 gibi).
- **Risk ızgarası:** %0,5, 0,75, 1,0, 1,25, 1,5, 2,0, 2,5.
- **Referans:** C1 ve R1 spotta, %0,5 riskle (canlı ve gölge ayarlar).

### Aşama A: geliştirme dönemleri (sınav verisi yüklenmez)
- **Dönemler:** keşif 2022-07-20 → 2024-12-31, doğrulama 2025-01-01 → 2026-03-31. Her dönem 10.000 USDT ile ayrı
  başlar.
- **Model:** Tüm ölçütler stres maliyet modeliyle hesaplanır.
- **CAGR:** yıllık bileşik getiri = (son / ilk)^(365,25 / gün) − 1.

**Risk seçimi (her aday için ayrı):**
1. **Büyüme tepesi f\*:** keşifte CAGR'ı en yüksek ızgara seviyesi. Eşitlikte düşük olan seçilir.
2. **Dengeli profil:** f\*/2'yi aşmayan en büyük ızgara seviyesi (yarım Kelly). En az %0,5.
3. **Agresif profil:** f\* × 0,75'i aşmayan en büyük ızgara seviyesi. En az %0,5.

Hiçbir profil f\*'ı aşamaz.

**Geçme ölçütleri (her profilin seçtiği seviyede):**
1. Keşifte ve doğrulamada net kâr > 0.
2. Maksimum düşüş, iki dönemde de: dengeli profilde ≤ %30, agresif profilde ≤ %45.
3. Doğrulamada kâr faktörü ≥ 1,1 (kapanmış işlemlerle).
4. **Risk karşılığını vermeli:** doğrulamada, seçilen seviyenin CAGR'ı aynı adayın %0,5 riskteki CAGR'ından
   yüksek olmalı. Seçilen seviye %0,5 ise bu ölçüt uygulanmaz.
5. **Yalnızca F-R1S:** short işlemlerin toplam net kârı keşifte de doğrulamada da > 0 olmalı. Değilse F-R1S kalır.
6. **Stres likidasyonları:** sayılır ve kayıpları sonuçlara dahil edilir. Ayrı bir ölçüt yoktur; 1. ve 2. ölçüt
   bunları kapsar.

**Seçim:**
- Her profil için, ölçütleri geçen adaylar arasından doğrulamada getiri/düşüşü en yüksek olan seçilir.
- Fark %5'ten azsa daha basit olan seçilir. Basitlik sırası: F-C1, F-R1, F-R1S.
- Hiçbiri geçmezse vadeli bot yapılmaz; spot C1 devam eder.

**Bilgi amaçlı (karar değiştirmez):**
- Temel maliyet modeli ve şaltersiz sonuçlar.
- Coin bazında sonuçlar.
- **Monte Carlo:** Seçilen ayarların keşif + doğrulama günlük getirileri, 20 günlük bloklarla yeniden örneklenir.
  10.000 adet 1 yıllık yol üretilir (tohum 20261015). Raporlanan: medyan, %10 ve %90 dilimleri, %30 ve %50 düşüş
  olasılığı, yılı zararla bitirme olasılığı.

### Aşama B: sınav dönemi (yalnızca bilgi)
- **Dönem:** 2026-04-01 → verideki son tam gün.
- **Zamanlama:** Aşama A'nın kararları yazıldıktan sonra bir kez çalıştırılır.
- **Rolü:** Bu dönem 5. turda kullanıldı, bu yüzden karar değiştiremez. Yalnızca ayarların son 6 ayda nasıl
  davranacağını gösterir.

### Aşama C: 56 USDT ve gerçek borsa verisi (bilgisayar açıkken indirilen verilerle)
- **İndirilecek veriler (API anahtarı gerekmez):**
  - USDⓈ-M sembol kuralları: miktar adımı ve en küçük tutar.
  - 12 coinin funding geçmişi (2022-07 → bugün).
  - Mark fiyatının 4 saatlik mumları.
- **Yeniden çalıştırma:** Seçilen ayarlar şu üç koşulla tekrar çalıştırılır:
  - gerçek funding geçmişi,
  - mark fiyatıyla likidasyon kontrolü,
  - 56 USDT ve gerçek sembol kuralları.
- **Funding kontrolü:** Gerçek funding ile doğrulama sonucu stres modelinden kötüyse, stres varsayımı yetersiz
  demektir; ayar kalır.
- **En küçük tutar:** Bu yüzden atlanan sinyaller raporlanır. Sinyallerin yarısından fazlası atlanıyorsa gerçek
  hesap için daha fazla sermaye ya da daha az coin gerekir. Bu durum raporlanır; strateji değiştirilmez.

### Aşama D: ileriye dönük test ve gerçek paraya geçiş
1. **Vadeli gölge bot (sanal para):** Seçilen ayarla çalışır ve teslim edildiği gün başlar.
2. **Test ağı / demo botu (sahte para, gerçek emir akışı):** Canlı vadeli broker yazılıp test edildikten sonra
   başlar.
3. **Mekanik doğrulama:** En az 14 gün ve en az 5 tamamlanmış işlem gerekir. Şartlar:
   - Her girişten sonra 60 saniye içinde borsada reduceOnly bir STOP_MARKET emri bulunur.
   - Çıkışlar gölge botun kararlarıyla aynıdır.
   - Funding ve komisyon kaydedilir.
   - Yakalanmamış hata olmaz.
4. **Küçük gerçek para:** Yalnızca Burak'ın açık onayıyla ve kaybetmeyi göze alabileceği bir tutarla.
   - API anahtarında "Enable Futures" açılır.
   - Para çekme kapalı kalır, IP kısıtlaması kalır.
5. **Performans değerlendirmesi:** 11. bölümdeki gibi, en erken 12 hafta sonra.

### Dürüstlük notları (önceden)
- **Short tarafı zayıf kalabilir.** Kripto piyasasındaki uzun vadeli yukarı eğilim ve ani short sıkışmaları buna yol
  açabilir. 5. ölçüt bunun için var.
- **Dolar karşılığı küçük kalır.** 56 USDT ile yılda %30 bile ~17 USDT eder.
- **Funding stres varsayımı bilinçli olarak kötümser.** Yılda ~%33, ölçülen ortalamaların çok üstündedir.
- **Spot verisi kullanılıyor.** Vadeli fiyatı spot fiyatından biraz farklıdır (baz). 4 saatlik trend takibinde
  etkisi küçüktür; Aşama C mark fiyatıyla kontrol eder.
- **Kurallar sabittir.** Sonuçlar görüldükten sonra hiçbir parametre ya da ölçüt değiştirilmez. Kodda hata bulunursa
  düzeltilir; düzeltme ve etkisi sonuç bölümünde yazılır.
