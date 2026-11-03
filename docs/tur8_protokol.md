## 21. 8. araştırma turu protokolü: çok vadeli trend topluluğu (long ve long/short)

**Bu bölüm 2026-10-17 ~21:55 UTC'de yazıldı.** O sırada 8. turun verisi henüz indirilmemişti. Bu strateji ailesi hiçbir
veride çalıştırılmamıştı ve kodu yazılmamıştı. Bu bölüm kodun ne yapacağını da tanımlar. Betik: `arastirma_tur8.py`.

### Soru
Dünya araştırmasında (docs/dunya_arastirmasi_long_short.md) en güçlü kanıt, çok vadeli trend topluluğu ile volatilite
hedeflemenin birlikte kullanılmasıydı (Zarattini, Pagani ve Barbon, 2025). Sorular:
- **Long:** BTC ve ETH üzerinde bu sistem, kabul edilebilir düşüşle al-tut'tan ve R1'den daha iyi risk/getiri
  verir mi?
- **Short:** Aynalı short tarafı (vadeli) bir değer katar mı?

### Evren ve veri
- **Coinler:** BTC/USDT ve ETH/USDT, Binance spot günlük mumları (UTC 00:00), 2017-08-17 → verideki son tam gün.
- **Neden yalnızca BTC ve ETH:**
  - 2017-2026 boyunca en büyük iki coin; seçimlerinde hayatta kalma yanlılığı yok.
  - Likiditeleri en derin olanlar.
  - 50 USDT ile daha fazla coin uygulanamaz (en küçük işlem 5 USDT).
- **Funding:** BTCUSDT ve ETHUSDT sürekli sözleşmelerinin gerçek funding geçmişi.

### Strateji (yayınlanmış parametreler; ayar yapılmaz)
**Alt sinyaller.** Her coin ve her kanal uzunluğu L ∈ {5, 10, 20, 30, 60, 90, 150, 250, 360} gün için, günlük
kapanışlarla (C):
- H, son L kapanışın en yükseği; D, en düşüğü; M = (H + D) / 2. Bugünkü kapanış dahildir.
- **Long:** Kapanış önceki L kapanışın en yükseğini aşarsa (C > dünkü H) girilir; stop = bugünkü M.
  - Pozisyondayken her gün stop = max(dünkü stop, bugünkü M).
  - Kapanış stopun altına inerse çıkılır.
- **Short (yalnızca T2):** Aynası.
  - Kapanış önceki L kapanışın en düşüğünün altına inerse girilir.
  - Stop = min(dünkü stop, bugünkü M).
  - Kapanış stopun üstüne çıkarsa çıkılır.
- **Isınma:** Bir L, L+1 kapanış birikince devreye girer. Öncesinde o alt sinyal 0'dır.

**Topluluk sinyali:** S = 9 alt sinyalin ortalaması. Long 1, short -1, nakit 0; yani S, -1 ile 1 arasındadır.

**Volatilite hedefleme:**
- σ = son 90 günlük getirilerin standart sapması × √365.
- Coinin hedef pozisyonu e = S × min(V / σ, 2,0). V, hedef volatilitedir.

**Portföy:**
- Sermaye, sinyali sıfır olmayan coinler arasında eşit bölünür: ağırlık w = e / (açık coin sayısı).
- Toplam brüt ağırlık sınırı: spotta (T1) %100, kaldıraç yok. Vadelide (T2) %200. Sınır aşılırsa ağırlıklar
  orantılı küçültülür.

**İşlem zamanı:** Karar günlük kapanışta (UTC 00:00) verilir, işlem sonraki günün açılışında yapılır.

**İşlem bandı:** Bir coinde ancak şu durumlardan biri varsa işlem yapılır; aksi hâlde işlem yapılmaz:
- Hedef sıfıra iner: pozisyon kapatılır.
- Sıfırdan yeni pozisyon açılır.
- Yön değişir.
- Hedef ile mevcut ağırlık arasındaki fark, hedefin en az %20'si olur.

**Maliyetler:**
- **Spot:** işlem tutarı üzerinden %0,1 komisyon + %0,05 kayma.
- **Vadeli:** işlem tutarı üzerinden %0,05 komisyon + %0,05 kayma.
- **Funding (vadeli):** Günün gerçek funding oranları, açılış fiyatıyla pozisyon büyüklüğü üzerinden. Oran pozitifse
  long öder, short alır.
  - Sözleşmenin ilk funding kaydından önce long tarafı 8 saatte sabit %0,01 öder.
  - Bu tarihten önce short açılmaz.

### Adaylar
| Kod | Yön | Piyasa | Hedef volatilite (V) |
|---|---|---|---|
| T1-25 | Yalnızca long | Spot | %25 (yayınlanan değer) |
| T1-15 | Yalnızca long | Spot | %15 (korumacı sürüm) |
| T2-25 | Long/short | Vadeli | %25 |
| T2-15 | Long/short | Vadeli | %15 |

**Referanslar:**
- **Al-tut:** Başlangıçta yarı BTC, yarı ETH alınır ve tutulur.
- **VY (volatiliteye göre al-tut):** T1 ile aynı boyutlandırma, bant ve maliyetler, ama her zaman long (S = 1).
  Trend sinyalinin kendisinin değer katıp katmadığını ölçer.
- **R1 sabit %0,5:** Yalnızca bilgi; verisi 2022-07-20'den itibaren var.

### Dönemler
Her dönem 10.000 USDT ile ayrı başlar. Ayrıca tam dönem (2018-09-01 → son tam gün) tek parça çalıştırılır.

| Dönem | Tarihler | Piyasa |
|---|---|---|
| P1 | 2018-09-01 → 2021-12-31 | 2018 düşüşü, 2019-2021 yükselişi |
| P2 | 2022-01-01 → 2024-12-31 | 2022 düşüşü, 2023-2024 yükselişi |
| P3 | 2025-01-01 → verideki son tam gün | 2025-2026 |

Parametreler yayınlandığı gibi kullanıldığı için üç dönem de bu aile için ilk kez yapılan testtir.

### Ölçütler (her aday ayrı)
1. P1, P2 ve P3'ün her birinde net kâr > 0.
2. P1, P2 ve P3'ün her birinde maksimum düşüş ≤ %25. Bu Burak'ın belirlediği sınırdır.
3. Tam dönemde Sharpe al-tut'tan yüksek ve maksimum düşüş al-tut'un en kötü düşüşünün yarısından az.
4. Tam dönemde Sharpe, aynı V ile VY'den yüksek. Yani trend sinyali değer katmalı.
5. **Yalnızca T2:** Tam dönemde Sharpe aynı V'deki T1'den yüksek ve üç dönemin en az ikisinde net getiri T1'den
   yüksek. Yani short tarafı değer katmalı.
6. **Yalnızca T1:** Tam dönem, 50 USDT ile ve gerçek spot kurallarıyla (en küçük işlem, miktar adımı)
   çalıştırılır: net kâr > 0 ve maksimum düşüş ≤ %25.
   - T2, 50 USDT ile uygulanamayabilir (BTC vadelide en küçük işlem 100 USDT). Bunun sonucu yalnızca bilgi
     olarak raporlanır.

### Seçim
- **50 USDT için:** Ölçütlerin hepsini geçen T1 adaylarından V'si en yüksek olan.
- **Daha büyük sermaye için:** Geçen T2 adaylarından V'si en yüksek olan (5. ölçütü geçtiyse).
- Hiçbiri geçmezse sistem benimsenmez.

**Bilgi amaçlı (karar değiştirmez):**
- Monte Carlo (20 günlük bloklar, 10.000 yol, tohum 20261017).
- Coin bazında katkı, ortalama pozisyon, işlem sayısı, maliyet ve funding toplamı.

### Sonrası
- Geçen aday, günlük çalışan sanal bot olarak en az 4 hafta ileriye dönük test edilir. Şartlar:
  - Sanal botun işlemleri backtest'le birebir aynı olmalı.
  - Yakalanmamış hata olmamalı.
- Gerçek paraya geçiş yalnızca Burak'ın açık onayıyla olur. Hedef ağırlıklı canlı bot o zaman yazılır ve test edilir.

### Dürüstlük notları (önceden)
- **Seçimler sonuç görülmeden yapıldı.** V = %15 sürümü, %20'lik işlem bandı ve %25 düşüş sınırı bizim seçimimiz;
  sonuçlar görülmeden belirlendi.
- **Yalnızca iki coin.** Makaledeki çok coinli portföy (hayatta kalma yanlılığı olmayan veriyle ilk 20 coin) bire
  bir kopyalanmadı. Sonuçlar ondan farklı olabilir.
- **Short ancak sözleşme varken.** Short tarafı yalnızca sürekli sözleşmenin var olduğu tarihten itibaren
  simüle edilir.
- **Veri yeni değil, aile yeni.** 2022-07 → 2026-10 dönemi önceki turlarda başka stratejiler için kullanıldı; bu aile
  için kullanılmadı.
- **Kurallar sabittir.** Sonuçlar görüldükten sonra hiçbir parametre ya da ölçüt değiştirilmez. Kodda hata bulunursa
  düzeltilir; düzeltme ve etkisi sonuç bölümünde yazılır.
