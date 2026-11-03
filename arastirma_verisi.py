"""2. araştırma turu için coin evrenini kurala göre seçer ve 1 saatlik verisini indirir.

Kural (docs/ARASTIRMA.md, 6. bölüm): CoinMarketCap'in 2 Ocak 2022 sıralamasındaki adaylar sırayla
denenir; veri koşulunu sağlayan ilk 10 coin seçilir. Koşul yalnızca verinin VARLIĞINA bakar, fiyat
performansına bakmaz:
  - Binance'te aynı sembolle aktif bir USDT spot paritesi olmalı,
  - 1 saatlik veri 2022-01-01'de ya da daha önce başlamalı,
  - sınav başlangıcına (data.exam_start) kadar 7 günden uzun boşluk olmamalı.

Seçim data/evren_tur2.json dosyasına yazılır; arastirma_tur2.py evreni bu dosyadan okur.

Kullanım:
    python arastirma_verisi.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from bot.cli import load_config_or_exit, setup_console
from bot.config import Config
from bot.data import save_market_rules, update_candles
from bot.exchange import ExchangeClient
from bot.logger import close_logging, setup_logging
from bot.timeutil import DAY_MS, fmt_utc, parse_utc, timeframe_ms

# CoinMarketCap 2 Ocak 2022 sıralaması; BTC, ETH, sabit coinler ve sarılı tokenlar çıkarılmış hâli.
CANDIDATES = ("BNB", "SOL", "ADA", "XRP", "LUNA", "DOT", "AVAX", "DOGE", "SHIB", "MATIC", "CRO", "UNI", "ALGO",
              "LTC", "LINK", "NEAR", "BCH", "ATOM", "TRX", "XLM", "FTM", "MANA")
UNIVERSE_SIZE = 10
DATA_START = "2022-01-01"
MAX_GAP_DAYS = 7
TIMEFRAME = "1h"
UNIVERSE_FILE = "evren_tur2.json"


def market_problem(markets: dict, symbol: str) -> str | None:
    """Parite koşulu: aynı sembolle aktif bir USDT spot paritesi. Sorun yoksa None."""
    market = markets.get(symbol)
    if market is None:
        return "Binance'te USDT paritesi yok"
    if market.get("spot") is False:
        return "spot paritesi değil"
    if market.get("active") is False:
        return "parite aktif değil (işlemden kaldırılmış)"
    return None


def data_problem(df: pd.DataFrame, start_ms: int, end_ms: int, tf_ms: int, max_gap_ms: int) -> str | None:
    """Veri koşulu, yalnızca zaman damgalarına bakar. Sorun yoksa None."""
    if df.empty:
        return "veri yok"
    first = int(df["ts"].iloc[0])
    if first > start_ms:
        return f"veri {fmt_utc(first, False)} tarihinde başlıyor"
    ts = df.loc[df["ts"] + tf_ms <= end_ms, "ts"].to_numpy(dtype="int64")
    # Sona kadar kesintisiz mi: son mumdan kontrol sonuna kadar olan boşluk da sayılır.
    edges = np.concatenate([ts, [end_ms]])
    gaps = np.diff(edges) - tf_ms
    worst = int(np.argmax(gaps))
    if gaps[worst] > max_gap_ms:
        days = gaps[worst] / DAY_MS
        if worst == len(gaps) - 1:
            return f"veri {fmt_utc(int(edges[worst]) + tf_ms, False)} tarihinde bitiyor"
        return f"{days:.0f} günlük boşluk ({fmt_utc(int(edges[worst]) + tf_ms, False)} sonrası)"
    return None


def universe_path(cfg: Config) -> Path:
    return cfg.data_dir / UNIVERSE_FILE


def select_universe(client, cfg: Config, log=None, echo=print) -> dict:
    """Adayları sırayla dener; koşulu sağlayan ilk UNIVERSE_SIZE coini seçer ve verisini kaydeder."""
    start_ms = parse_utc(DATA_START)
    end_ms = parse_utc(cfg.data.exam_start)
    tf_ms = timeframe_ms(TIMEFRAME)
    markets = client.markets()
    selected: list[str] = []
    rejected: dict[str, str] = {}
    for coin in CANDIDATES:
        if len(selected) == UNIVERSE_SIZE:
            break
        symbol = f"{coin}/USDT"
        problem = market_problem(markets, symbol)
        if problem is None:
            df = update_candles(client, symbol, TIMEFRAME, tf_ms, start_ms, cfg.data_dir, log)
            problem = data_problem(df, start_ms, end_ms, tf_ms, MAX_GAP_DAYS * DAY_MS)
        if problem is None:
            selected.append(symbol)
            echo(f"[OK]      {symbol:<11} seçildi ({len(selected)}/{UNIVERSE_SIZE})")
        else:
            rejected[symbol] = problem
            echo(f"[ELENDİ]  {symbol:<11} {problem}")
    return {
        "kural": "docs/ARASTIRMA.md, 6. bölüm",
        "olusturma": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "zaman_dilimi": TIMEFRAME,
        "secilen": selected,
        "elenen": rejected,
        "denenmeyen": [f"{c}/USDT" for c in CANDIDATES if f"{c}/USDT" not in selected and f"{c}/USDT" not in rejected],
    }


def save_universe(cfg: Config, universe: dict) -> Path:
    path = universe_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(universe, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_universe(cfg: Config) -> dict:
    path = universe_path(cfg)
    if not path.exists():
        raise FileNotFoundError(path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    setup_console()
    cfg = load_config_or_exit()
    log = setup_logging(cfg.log_dir, cfg.logging.level, "veri.log")
    print(f"Veri klasörü : {cfg.data_dir}")
    print(f"Kural        : {len(CANDIDATES)} aday sırayla denenir; koşulu sağlayan ilk {UNIVERSE_SIZE} coin seçilir.")
    print("İndirme birkaç dakika sürebilir; pencereyi kapatmayın.\n")
    t0 = time.perf_counter()
    try:
        client = ExchangeClient(cfg, logger=log)
        client.refresh_time_offset()
        universe = select_universe(client, cfg, log)
        if len(universe["secilen"]) < UNIVERSE_SIZE:
            print(f"\n[HATA] Koşulu sağlayan yalnızca {len(universe['secilen'])} coin bulundu.")
            return 1
        rules = client.market_rules(universe["secilen"], cfg.market_rules_default)
        save_market_rules(cfg.data_dir, rules)
        path = save_universe(cfg, universe)
    except ccxt.BaseError as exc:
        print(f"\n[HATA] Borsa/ağ hatası: {exc}")
        print("İnternet bağlantınızı kontrol edip tekrar çalıştırın; kaldığı yerden devam eder.")
        return 1
    finally:
        close_logging()
    print(f"\nSeçilen {UNIVERSE_SIZE} coin: {', '.join(s.split('/')[0] for s in universe['secilen'])}")
    print(f"Kayıt: {path}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    print("[OK] Veri hazır. Sıradaki adım: 8_tur2_arastirma.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
