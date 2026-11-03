"""4. araştırma turu için GENİŞ coin evreni: kurala göre seçim ve 1 saatlik veri (docs/ARASTIRMA.md, 13. bölüm).

Kural (sonuçlar görülmeden önce yazıldı): 2. turdaki kuralın aynısı, daha uzun aday listesiyle.
CoinMarketCap'in 2 Ocak 2022 sıralaması sırayla denenir (BTC, ETH, sabit coinler ve sarılı tokenlar
hariç); veri koşulunu sağlayan ilk 25 coin seçilir. BTC ve ETH her zaman evrendedir. Koşul yalnızca
verinin VARLIĞINA bakar, fiyat performansına bakmaz (bkz. arastirma_verisi.py).

Seçim data/evren_tur4.json dosyasına yazılır. İndirme kaldığı yerden devam eder; tekrar çalıştırmak güvenlidir.

Kullanım:
    python tur4_veri.py
"""
from __future__ import annotations

import sys
import time

import ccxt

import arastirma_verisi as V
from bot.cli import disable_quick_edit, load_config_or_exit, setup_console
from bot.data import save_market_rules, update_candles
from bot.exchange import ExchangeClient
from bot.logger import close_logging, setup_logging
from bot.timeutil import parse_utc, timeframe_ms

# 2. turun 22 adayı + CoinMarketCap 2 Ocak 2022 listesindeki sonraki adaylar, aynı sırayla
# (TerraUSD ve Dai sabit coin, Bitcoin BEP2 sarılı token olduğu için yok).
TUR4_CANDIDATES = V.CANDIDATES + ("AXS", "HBAR", "VET", "FTT", "SAND", "ICP", "FIL", "THETA", "EGLD", "ETC", "HNT")
TUR4_SIZE = 25
MIN_SIZE = 18                     # bundan azı bulunursa araştırma anlamlı değil
ALWAYS = ("BTC/USDT", "ETH/USDT")
UNIVERSE_FILE = "evren_tur4.json"


def configure_round4() -> None:
    """arastirma_verisi.py'deki seçim fonksiyonları bu turun aday listesini ve dosyasını kullansın."""
    V.CANDIDATES = TUR4_CANDIDATES
    V.UNIVERSE_SIZE = TUR4_SIZE
    V.UNIVERSE_FILE = UNIVERSE_FILE


def download_round4(client, cfg, log=None, echo=print) -> dict:
    configure_round4()
    start_ms = parse_utc(V.DATA_START)
    tf_ms = timeframe_ms(V.TIMEFRAME)
    for symbol in ALWAYS:
        update_candles(client, symbol, V.TIMEFRAME, tf_ms, start_ms, cfg.data_dir, log)
        echo(f"[OK]      {symbol:<11} (her zaman evrende)")
    universe = V.select_universe(client, cfg, log, echo)
    universe["kural"] = "docs/ARASTIRMA.md, 13. bölüm"
    universe["her_zaman"] = list(ALWAYS)
    return universe


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()                 # pencereye tıklamak indirmeyi dondurmasın
    cfg = load_config_or_exit()
    log = setup_logging(cfg.log_dir, cfg.logging.level, "veri_tur4.log")
    print(f"Veri klasörü : {cfg.data_dir}")
    print(f"Kural        : {len(TUR4_CANDIDATES)} aday sırayla denenir; koşulu sağlayan ilk {TUR4_SIZE} coin "
          "seçilir (+ BTC, ETH).")
    print("İndirme 10-15 dakika sürebilir; pencereyi kapatmayın. Çalışan botlara dokunmaz.\n")
    t0 = time.perf_counter()
    try:
        client = ExchangeClient(cfg, logger=log)
        client.refresh_time_offset()
        universe = download_round4(client, cfg, log)
        if len(universe["secilen"]) < MIN_SIZE:
            print(f"\n[HATA] Koşulu sağlayan yalnızca {len(universe['secilen'])} coin bulundu (en az {MIN_SIZE}).")
            return 1
        rules = client.market_rules(list(ALWAYS) + universe["secilen"], cfg.market_rules_default)
        save_market_rules(cfg.data_dir, rules)
        path = V.save_universe(cfg, universe)
    except ccxt.BaseError as exc:
        print(f"\n[HATA] Borsa/ağ hatası: {exc}")
        print("İnternet bağlantınızı kontrol edip tekrar çalıştırın; kaldığı yerden devam eder.")
        return 1
    finally:
        close_logging()
    coins = [s.split("/")[0] for s in list(ALWAYS) + universe["secilen"]]
    print(f"\nEvren ({len(coins)} coin): {', '.join(coins)}")
    print(f"Kayıt: {path}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    print("[OK] Veri hazır. Sonraki adım: araştırma betiğini çalıştırın.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
