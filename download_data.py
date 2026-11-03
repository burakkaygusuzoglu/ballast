"""Binance'ten geçmiş mum verisini indirir ya da günceller (API anahtarı gerekmez).

Kullanım:
    python download_data.py                    # config.yaml'daki semboller, data.start'tan bugüne
    python download_data.py --start 2023-01-01

İlk indirme birkaç dakika sürer; sonraki çalıştırmalar yalnızca eksik kısmı indirir.
"""
from __future__ import annotations

import argparse
import sys
import time

import ccxt

from bot.cli import load_config_or_exit, setup_console
from bot.data import describe, save_market_rules, update_candles
from bot.exchange import ExchangeClient
from bot.logger import close_logging, setup_logging
from bot.timeutil import parse_utc, timeframe_ms


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Binance'ten geçmiş mum verisini indirir/günceller.")
    parser.add_argument("--start", help="Başlangıç tarihi (YYYY-AA-GG). Varsayılan: config.yaml -> data.start")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    log = setup_logging(cfg.log_dir, cfg.logging.level, "veri.log")
    start = args.start or cfg.data.start
    try:
        start_ms = parse_utc(start)
    except ValueError:
        print(f"[HATA] Geçersiz tarih: {start!r} (örnek: 2022-01-01)")
        return 2

    client = ExchangeClient(cfg, logger=log)
    print(f"Veri klasörü : {cfg.data_dir}")
    print(f"Başlangıç    : {start} | Semboller: {', '.join(cfg.symbols)} | "
          f"Zaman dilimleri: {cfg.timeframes.entry}, {cfg.timeframes.trend}")
    print("İlk indirme birkaç dakika sürebilir; pencereyi kapatmayın.\n")
    t0 = time.perf_counter()
    empty = 0
    try:
        offset = client.refresh_time_offset()
        print(f"[OK]    Binance saati alındı (bilgisayar saati farkı {offset / 1000:+.2f} sn)")
        for symbol in cfg.symbols:
            for timeframe in (cfg.timeframes.entry, cfg.timeframes.trend):
                tf_ms = timeframe_ms(timeframe)
                df = update_candles(client, symbol, timeframe, tf_ms, start_ms, cfg.data_dir, log)
                info = describe(df, tf_ms)
                empty += info["rows"] == 0
                status = "[OK]   " if info["rows"] > 0 else "[HATA] "
                print(f"{status} {symbol:<9} {timeframe:>4}: {info['rows']:>8,} mum | {info['first']} -> {info['last']} "
                      f"| boşluk: {info['gaps']} ({info['missing_bars']} eksik mum)")
        rules = client.market_rules(cfg.symbols, cfg.market_rules_default)
        save_market_rules(cfg.data_dir, rules)
        for symbol, rule in rules.items():
            print(f"[OK]    {symbol:<9} kurallar: miktar adımı {rule.qty_step:g}, min. işlem {rule.min_notional:g} USDT")
    except ccxt.BaseError as exc:
        print(f"\n[HATA] Borsa/ağ hatası: {exc}")
        print("İnternet bağlantınızı kontrol edip komutu tekrar çalıştırın; kaldığı yerden devam eder.")
        return 1
    finally:
        close_logging()

    print(f"\nSüre: {time.perf_counter() - t0:.0f} sn")
    print("Not: Birkaç küçük boşluk normaldir (Binance bakım aralıkları); motor bunları atlar.")
    if empty:
        print("[HATA] Bazı veriler indirilemedi.")
        return 1
    print("[OK] Veri hazır. Sıradaki adım: 3_backtest.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
