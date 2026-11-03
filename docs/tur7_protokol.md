## 19. 7. araştırma turu protokolü: korumalı büyüme sistemi (R1 + düşüşe göre ölçeklenen risk)

**Bu bölüm 2026-10-16 ~08:40 UTC'de yazıldı.** O sırada düşüşe göre ölçeklenen risk hiçbir veride
çalıştırılmamıştı ve kodu yazılmamıştı. Bu bölüm kodun ne yapacağını da tanımlar. Betik: `arastirma_tur7.py`.

### Burak'ın istekleri (2026-10-16)
- **Hedef:** Önceliği daha yüksek yıllık getiri olan, ama korumalı ve güvenli ilerleyen bir sistem.
- **Kayıp sınırı:** Tepe noktasından en fazla %25 düşüş (Burak seçti).
- **Gerçek para:** Yeni sistem onaylanana kadar gerçek parayla işlem yapılmaz (Burak seçti).

### Fikir
- **Çekirdek: R1 kuralları.** C1 + BTC rejim filtresi. Sınav döneminde en iyi sonuç buydu: +%14,7, düşüş %9,5,
  kâr faktörü 2,18.
- **Risk düşüşe göre ölçeklenir.** İşlem başına risk = taban risk × max(0, 1 − düşüş / 0,25).
  - Düşüş = 1 − özsermaye / en yüksek özsermaye.
  - Bot zirvedeyken taban riski kullanır. %12,5 düşüşte riskin yarısını kullanır. %25'te risk sıfırdır.
  - Bu, en yüksek düşüş sınırı altında büyümeyi en iyi yapan politikanın (Grossman ve Zhou, 1993) basit hâlidir:
    pozisyon, sınırın üstündeki "yastıkla" orantılı tutulur.
- **Şalter %25'te.** Düşüş %25'e ulaşırsa bot bütün pozisyonları kapatır ve durur; yalnızca elle sıfırlanır.
- **Spot, kaldıraç yok.** 6. tur kaldıraçlı sürümlerin kaybettiğini gösterdi. Spotta likidasyon ve funding yok.
  Toplam pozisyon en fazla özsermaye kadardır.

### Önceden bilinenler (dürüstlük)
- **Bu tasarım geçmiş bir gözleme dayanıyor.** 6. turda sabit yüksek riskin 2022-2023'te şaltere takıldığını
  gördük; ölçekleme bu sorunu çözmek için seçildi. Bu yüzden geçmiş veri testi iyimser olabilir. Asıl test
  ileriye dönük testtir.
- **Geçmiş veri yedinci kez kullanılıyor.** Keşif ve doğrulama dönemleri 3.-6. turlarda, sınav dönemi 5. turda
  kullanıldı.
- **Önceki taramalar biliniyor.** R1'in tam dönem Kelly taramasında büyüme ~%1,8 riskte tepe yapıyordu
  (şaltersiz, spot, sınav dahil).

### Adaylar
- **R1-K10, R1-K15, R1-K20:** R1 kuralları ve düşüşe göre ölçeklenen risk. Taban risk sırasıyla %1,0, %1,5,
  %2,0; ölçek sınırı ve şalter %25. Ayarlar:
  - Günlük zarar limiti: 6 × taban risk.
  - C1 ile aynı: tek pozisyon en fazla özsermayenin %50'si, en fazla 5 pozisyon, günde 4 işlem, 4 mum bekleme,
    maliyet filtresi 0,25R.
  - Spot komisyon ve kayma: %0,1 + %0,05.
- **Referans:** R1 sabit %0,5 risk, %20 şalter (gölge botun ayarı) ve C1 sabit %0,5 (gerçek botun ayarı).
- **Yalnızca bilgi:** R1 sabit %1,0 risk, ölçeklemesiz, %25 şalter. Ölçeklemenin etkisini göstermek içindir.

### Aşama A: geliştirme dönemleri (sınav verisi yüklenmez)
- **Dönemler:** keşif 2022-07-20 → 2024-12-31, doğrulama 2025-01-01 → 2026-03-31. Her dönem 10.000 USDT ile
  ayrı başlar.
- **Uygulanabilirlik:** Ayrıca 50 USDT ile, gerçek minimum işlem tutarlarıyla (data/markets.json) çalıştırılır.

**Geçme ölçütleri (her aday için):**
1. Keşifte ve doğrulamada net kâr > 0.
2. İki dönemde de şaltere takılmamalı (maksimum düşüş < %25).
3. Doğrulamada kâr faktörü ≥ 1,1 (kapanmış işlemlerle).
4. **Daha yüksek getiri:** İki dönemde de net getiri, referans R1 sabit %0,5'ten yüksek olmalı.
5. **50 USDT ile çalışabilmeli:** Doğrulamada sinyallerin en fazla %50'si minimum tutar yüzünden atlanmalı.
   Oran = atlanan / (atlanan + verilen emir).

**Seçim:**
- Ölçütlerin hepsini geçen adaylar arasından taban riski **en yüksek** olan seçilir. Burak'ın önceliği daha yüksek
  getiri; koruma 2. ölçütle sağlanır.
- Hiçbiri geçmezse sistem benimsenmez. R1 sabit %0,5 gölge testine 11. bölümdeki gibi devam eder.

**Bilgi amaçlı (karar değiştirmez):**
- **Monte Carlo:** Seçilen adayın keşif + doğrulama günlük getirileri, 20 günlük bloklarla yeniden örneklenir.
  10.000 adet 1 yıllık yol üretilir (tohum 20261016). Raporlanan: medyan, %10 ve %90 dilimleri, %25 düşüşe
  ulaşma olasılığı, yılı zararla bitirme olasılığı.
- Coin bazında sonuçlar ve ölçeklemenin ortalama etkisi (işlemlerin ortalama riski).

### Aşama B: sınav dönemi (yalnızca bilgi)
- **Dönem:** 2026-04-01 → verideki son tam gün.
- **Zamanlama:** Aşama A'nın kararları yazıldıktan sonra bir kez çalıştırılır.
- **Rolü:** Karar değiştiremez. Seçilen aday ve referanslar gösterilir.

### Aşama C: ileriye dönük sanal test ve gerçek paraya geçiş
1. **Sanal bot:** Seçilen aday, 50 USDT sanal bakiyeyle gölge bot olarak başlar.
2. **Gerçek para için en az 4 hafta.** Gerçek paraya geçmeden önce şunların hepsi sağlanmalı:
   - Sanal botun işlemleri aynı dönemin backtest'iyle birebir aynı olmalı (izleme panelindeki eşdeğerlik).
   - Şaltere takılmamalı.
   - Yakalanmamış hata olmamalı.
3. **4 hafta performansı değerlendirmek için yetmez.** Bu süre yalnızca sistemin canlıda doğru çalıştığını
   gösterir. Performans 11. bölümdeki ölçütlerle en erken 12 hafta sonra değerlendirilir.
4. **Gerçek paraya geçiş:** Yalnızca Burak'ın açık onayıyla olur. Gerçek botta da %25 şalter ve borsadaki stoplar
   çalışır.

### Dürüstlük notları (önceden)
- **Düşüş %25'i biraz aşabilir.** Ölçekleme düşüşü yavaşlatır ama kesin sınır şalterdir. Boşluklu düşüşlerde
  (ör. 10 Ekim 2025 çöküşü) şalter de %25'in biraz üstünde çalışabilir.
- **Ölçekleme bedelsiz değildir.** Düşüşten sonra risk azaldığı için toparlanma da yavaşlar. Trend takibinde
  kazancın büyük kısmı düşüşlerden hemen sonra gelebilir.
- **Kurallar sabittir.** Sonuçlar görüldükten sonra hiçbir parametre ya da ölçüt değiştirilmez. Kodda hata
  bulunursa düzeltilir; düzeltme ve etkisi sonuç bölümünde yazılır.
