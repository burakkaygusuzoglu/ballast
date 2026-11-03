# Daha gelişmiş long/short bot: dünya kanıtları

**Tarih:** 2026-10-17. **Soru:** Kriptoda long ve short pozisyonlarla daha yüksek kazanç hedefleyen, en iyi kararı
veren bir bot için dünyadaki kanıtlar ne diyor? **Not:** Bu bir yatırım tavsiyesi değildir.

## Kısa cevap

| Yaklaşım | Kanıt | Bizim için anlamı |
|---|---|---|
| **Trend takibi, long** (çok vadeli + volatiliteye göre boyut) | **Güçlü.** Hakemli çalışmalar ve 2025 tarihli kapsamlı bir test | En iyi aday. Şimdiye kadar test ettiğimiz tek kanallı 4 saatlik C1/R1'den yapısal olarak farklı. |
| **Trend takibi, short** | **Zayıf, belirsiz.** Satılan ürünlerin hepsi long/flat. Bizim 6. tur testinde short tarafı her seviyede kaybettirdi. | Yalnızca ayrı bir test kolu olarak ve düşük beklentiyle. |
| **Volatilite hedefleme** | **Güçlü.** Riski en çok düşüren bileşen | Her adayda olmalı. |
| **Funding taşıma** (spot long + vadeli short) | **Orta.** Ortalama pozitif ama çökme ve likidasyon riski var | Küçük sermayede uygulanamaz; sermaye büyürse ikinci kol olabilir. |
| **Aşırı funding'e karşı işlem** | **Zayıf.** Küçük örnekler, maliyetler yok sayılmış | Önermiyorum. |
| **Yapay zekâ / LLM ile karar** | **Olumsuz.** Gerçek parayla çoğu kaybetti; backtest başarısı ileriye taşınmadı | Emir kararında kullanmıyoruz. |
| **Kesitsel momentum** (coinleri sıralama) | **Zayıf.** Gerçekçi varsayımlarla "neredeyse yok" | Önermiyorum. Bizim R3 ve 4. tur bulgularıyla da tutarlı. |

## 1. Trend takibi: en sağlam kanıt long tarafta
- **Gerçekçi varsayımlarla momentum (Han, Kang ve Ryu; *Review of Asset Pricing Studies*):**
  - Zaman serisi momentumu "güçlü biçimde destekleniyor"; kesitsel momentum ise "neredeyse yok".
  - Günlük fiyat oynaklığı hesaba katılınca "birçok momentum portföyü likide oluyor".
  - Ortalama getiri yanıltıcı olabilir: getiriler çok çarpık ve uç değerli.
- **Catching Crypto Trends (Zarattini, Pagani ve Barbon, Nisan 2025):**
  - Kurulum: long-only, ilk 20 likit coin.
  - Sinyal: 5, 10, 20, 30, 60, 90, 150, 250 ve 360 günlük kanalların topluluğu; iz süren stop.
  - Boyut: her pozisyon %25 yıllık hedef volatiliteye göre, kaldıraç en fazla %200.
  - Sonuç: komisyon sonrası Sharpe 1,5'in üstünde, Bitcoin'e göre yıllık %10,8 alfa. Hayatta kalma yanlılığı
    olmayan veriyle, 2015'ten itibaren.
  - Dikkat: dönem 2017 ve 2020-21 boğalarını içeriyor. CAGR ve en kötü düşüş özette yok.
- **Ayık bir pratisyen testi (BTC/ETH/SOL, 2017 → 2025 ortası):**

  | Portföy | Yıllık getiri | Sharpe | En kötü düşüş |
  |---|---|---|---|
  | Trend (1/3/12 ay, long/short) | %6,3 | 0,79 | -%14,6 |
  | Volatiliteye göre eşit ağırlık (trend sinyali yok) | %12,9 | 1,24 | -%19,9 |
  | Bitcoin al-tut | %73,5 | 1,13 | -%83,8 |

  - Yazarın sonucu: trend takibi "getiriyi artırmaktan çok riskin şeklini değiştirdi".
  - Volatilite ölçekleme işin çoğunu yaptı; trend sinyali Sharpe'a yalnızca ~0,05 ekledi.
  - Short tarafı o örnekte kârın neredeyse hiçbirini sağlamadı.

## 2. Short tarafı neden zayıf
- Kriptoda uzun vadeli yukarı eğilim var, short pozisyonu taşımanın maliyeti var ve ani short sıkışmaları oluyor.
- Piyasadaki trend ürünlerinin hepsi long/flat (long ya da nakit) tasarlanmış.
- Kanıt konuyu kesin çözmüyor. Ama bizim testimizde (6. tur, R1 + short) short tarafı her risk seviyesinde, iki
  dönemde de zarar etti.

## 3. Funding taşıma ve funding sinyalleri
- **BIS çalışması "Crypto carry" (2023):**
  - Vadeli ile spot arasındaki fark (taşıma) 2019-2022'de yılda ortalama ~%10; zirvelerde %60'a kadar çıktı.
  - Taşıma yüksekken sonraki ay satış yönlü likidasyonlar artıyor: taşımadaki %10'luk artış, likidasyonlarda
    %44'lük artışla birlikte görülüyor.
  - Maliyet öncesi Sharpe ~0,59. Yazarların ifadesiyle işlem "önemli riskler taşıyor".
  - Kaldıraçlı vadeli bacağın simülasyonda ayların %33'ünde (5x) ile %67'sinde (20x) likide olduğu görüldü.
- **Funding'in kendisi kısmen tahmin edilebilir** (Inan, 2025). Ama bunun kârlı bir işlem stratejisine dönüştüğü
  gösterilmedi.
- **Aşırı funding'de ters yönde işlem:** BitMEX'in testi küçük örnekli, hesap tablosunda işaret hataları var ve
  komisyonları içermiyor. Kanıt zayıf.

## 4. Yapay zekâ ile karar veren botlar
- **Alpha Arena 1. sezon (gerçek para, kişi başı 10.000 $, Hyperliquid vadeli):**

  | Model | Getiri |
  |---|---|
  | Qwen3 Max | +%22,3 |
  | DeepSeek V3.1 | +%4,9 |
  | Claude Sonnet 4.5 | -%30,8 |
  | Grok 4 | -%45,3 |
  | Gemini 2.5 Pro | -%56,7 |
  | GPT-5 | -%62,7 |

  - 6 modelin 4'ü büyük kayıp verdi.
  - Düzenleyenlerin dersi: tahmin, uygulama, boyutlandırma ve risk kontrolü birlikte değerlendirilmeli.
- **"Can AI Make Money in Crypto?" (Eylül 2026, 32 yöntem):**
  - Backtest'te ML modellerinin neredeyse hepsi kârlıydı. LLM, pekiştirmeli öğrenme ve ajan yöntemlerinin hiçbiri
    backtest'te bile kârlı değildi.
  - Sanal işlemde ML'nin medyan getirisi +%36'dan -%9,8'e düştü.
  - Sonuç: backtest sıralamaları ileriye dönük sonucu güvenilir biçimde göstermiyor.
- **LiveTradeBench (2025, canlı):** Yüksek genel zekâ puanı daha iyi işlem sonucu anlamına gelmiyor. Büyük akıl
  yürütme modelleri bir avantaj sağlamadı.

## 5. Profesyoneller gerçekte ne kazanıyor (2025)
- Kripto fonları ortalama -%7,2 (medyan -%5,2); fonların yalnızca %37'si kârda.
- Algoritmik/kantitatif fonlar ortalama +%0,4, medyan +%3,2.
- Bitcoin aynı yıl -%6,3.

## Bizim için sonuç
1. **"Daha gelişmiş" = daha sağlam.** Çok vadeli trend topluluğu, volatilite hedefleme ve günlük karar: dünyada en
   iyi belgelenmiş yapı bu ve biz henüz test etmedik.
2. **Short ayrı bir kol olarak test edilmeli.** Long tarafı bozmamalı ve kendi kârını kanıtlamalı.
3. **Yapay zekâ emir vermemeli.** İzleme ve özet gibi kritik olmayan işlerde kullanılabilir.
4. **Beklenti gerçekçi olmalı.** En iyi kanıtlı sistemler bile riski düzenler; büyük ve hızlı getiri vaat etmez.

## Önerilen 8. tur (taslak; kurallar test öncesinde ayrıca yazılacak)
- **T1, trend topluluğu (long-only, spot):** Zarattini ve ark.'nın yayınlanmış parametreleriyle, ayar yapmadan.
- **T2, aynı sistem long/short (vadeli):** Short tarafı aynalı; kaldıraç en fazla 2x.
- **Referans:** R1 sabit %0,5.
- **Veri:** 2018'den bugüne günlük mumlar. 360 günlük kanalın ısınması ve 2018/2022 düşüş yıllarını kapsamak için
  gerekli. Senin bilgisayarından indirilecek (API anahtarı gerekmez).
- **Sonra:** Geçen aday, önce sanal parayla ileriye dönük test.

## Kaynaklar
- [Han, Kang ve Ryu: Kripto piyasasında momentum, gerçekçi varsayımlarla (SSRN; Review of Asset Pricing Studies)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565)
- [Zarattini, Pagani ve Barbon (2025): Catching Crypto Trends (Concretum)](https://concretumgroup.com/catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/) · [CXO Advisory özeti](https://www.cxoadvisory.com/technical-trading/crypto-asset-trend-following-strategies/)
- [Summitward: Kriptoda trend takibi işe yarıyor mu? (pratisyen testi)](https://summitward.com/learn/crypto-trend-following)
- [BIS Working Paper 1087: Crypto carry (2023)](https://www.bis.org/publ/work1087.pdf)
- [Inan (2025): Funding oranlarının tahmin edilebilirliği (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424)
- [BitMEX: XBTUSD funding'e karşı işlem stratejisi](https://www.bitmex.com/blog/xbtusd-funding-mean-reversion-strategy)
- [Alpha Arena 1. sezon sonuçları](https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/)
- [Can AI Make Money in Crypto? (arXiv 2609.34510, Eylül 2026)](https://arxiv.org/html/2609.34510v1)
- [LiveTradeBench (arXiv 2511.03628)](https://arxiv.org/html/2511.03628)
- [Crypto Fund Research: 2025 kripto fonu performansı](https://cryptofundresearch.com/crypto-hedge-fund-performance/)
