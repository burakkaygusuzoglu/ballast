"""8. araştırma turu için veri: BTC ve ETH'nin 2017'den bugüne günlük mumları, vadeli funding geçmişi ve vadeli
sembol kuralları. Hepsi Binance'in herkese açık verisi: API anahtarı kullanılmaz, hiçbir işlem yapılmaz, çalışan
botlara dokunulmaz.

- Günlük spot mumlar: data/binance_BTC-USDT_1d.csv, data/binance_ETH-USDT_1d.csv (2017-08-01 sonrası).
- Funding geçmişi: data/tur8/funding_BTCUSDT.csv, data/tur8/funding_ETHUSDT.csv (sürekli sözleşmenin ilk gününden).
- Vadeli sembol kuralları (en küçük tutar, miktar adımı): data/tur8/vadeli_kurallar.json.

İndirme kaldığı yerden devam etmez ama kısa sürer (birkaç dakika); tekrar çalıştırmak güvenlidir.

Kullanım:
    python tur8_veri.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import ccxt

from bot.cli import disable_quick_edit, load_config_or_exit, setup_console
from bot.data import update_candles
from bot.exchange import ExchangeClient
from bot.logger import close_logging, setup_logging
from bot.timeutil import DAY_MS, fmt_utc, parse_utc

SPOT_SYMBOLS = ("BTC/USDT", "ETH/USDT")
PERPS = {"BTC/USDT": "BTC/USDT:USDT", "ETH/USDT": "ETH/USDT:USDT"}
SPOT_START = "2017-08-01"                    # Binance spot'un açılışından önce: ilk mumdan itibaren alınır
FUNDING_START = "2019-09-01"                 # BTCUSDT sürekli sözleşmesi 2019-09'da açıldı
FUNDING_LIMIT = 1000
OUT_DIR = "tur8"


def funding_history(api: Any, perp: str, since_ms: int, until_ms: int) -> list[tuple[int, float]]:
    """Funding geçmişi (zaman, oran), sayfa sayfa. api: fetch_funding_rate_history destekleyen ccxt nesnesi."""
    rows: dict[int, float] = {}
    since = since_ms
    while since < until_ms:
        batch = api.fetch_funding_rate_history(perp, since=since, limit=FUNDING_LIMIT)
        if not batch:
            break
        for item in batch:
            ts = int(item["timestamp"])
            if ts < until_ms:
                rows[ts] = float(item["fundingRate"])
        last = int(batch[-1]["timestamp"])
        if last + 1 <= since:                 # ilerleme yoksa dur (sonsuz döngü olmasın)
            break
        since = last + 1
        if len(batch) < FUNDING_LIMIT:
            break
    return sorted(rows.items())


def save_funding(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "rate"])
        w.writerows([(ts, repr(rate)) for ts, rate in rows])
    tmp.replace(path)


def perp_rules(api: Any) -> dict[str, dict]:
    """Vadeli sembol kuralları: en küçük tutar (USDT), en küçük miktar ve miktar adımı."""
    api.load_markets()
    out = {}
    for spot, perp in PERPS.items():
        m = api.market(perp)
        out[spot] = {
            "vadeli_sembol": m["id"],
            "en_kucuk_tutar": (m.get("limits", {}).get("cost") or {}).get("min"),
            "en_kucuk_miktar": (m.get("limits", {}).get("amount") or {}).get("min"),
            "miktar_adimi": (m.get("precision") or {}).get("amount"),
        }
    return out


def download(client: Any, futures: Any, cfg: Any, log: Any = None, echo=print) -> dict:
    out_dir = Path(cfg.data_dir) / OUT_DIR
    summary: dict[str, Any] = {"spot": {}, "funding": {}}
    for symbol in SPOT_SYMBOLS:
        df = update_candles(client, symbol, "1d", DAY_MS, parse_utc(SPOT_START), cfg.data_dir, log)
        first, last = int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])
        summary["spot"][symbol] = {"mum": len(df), "ilk": fmt_utc(first, False), "son": fmt_utc(last, False)}
        echo(f"[OK] {symbol:<9} günlük: {len(df):,} mum ({fmt_utc(first, False)} → {fmt_utc(last, False)})")
    now = client.server_now()
    for spot, perp in PERPS.items():
        rows = funding_history(futures, perp, parse_utc(FUNDING_START), now)
        name = perp.split(":")[0].replace("/", "")
        save_funding(out_dir / f"funding_{name}.csv", rows)
        summary["funding"][spot] = {"kayit": len(rows), "ilk": fmt_utc(rows[0][0], False) if rows else None,
                                    "son": fmt_utc(rows[-1][0], False) if rows else None}
        echo(f"[OK] {name:<9} funding: {len(rows):,} kayıt")
    rules = perp_rules(futures)
    (out_dir / "vadeli_kurallar.json").write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["vadeli_kurallar"] = rules
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    limits = ", ".join(f"{sym} en küçük {info['en_kucuk_tutar']} USDT" for sym, info in rules.items())
    echo(f"[OK] Vadeli sembol kuralları: {limits}")
    return summary


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()                 # pencereye tıklamak indirmeyi dondurmasın
    cfg = load_config_or_exit()
    log = setup_logging(cfg.log_dir, cfg.logging.level, "veri_tur8.log")
    print(f"Veri klasörü : {cfg.data_dir}")
    print("İndirilecek  : BTC ve ETH günlük mumları (2017'den bugüne), vadeli funding geçmişi, vadeli sembol kuralları.")
    print("Herkese açık veri: API anahtarı kullanılmaz, işlem yapılmaz. Birkaç dakika sürer; pencereyi kapatmayın.\n")
    t0 = time.perf_counter()
    try:
        client = ExchangeClient(cfg, logger=log)
        client.refresh_time_offset()
        futures = ccxt.binanceusdm({"enableRateLimit": True, "timeout": cfg.exchange.timeout_ms})
        download(client, futures, cfg, log)
    except ccxt.BaseError as exc:
        print(f"\n[HATA] Borsa/ağ hatası: {exc}")
        print("İnternet bağlantınızı kontrol edip tekrar çalıştırın.")
        return 1
    finally:
        close_logging()
    print(f"\nSüre: {time.perf_counter() - t0:.0f} sn")
    print("[OK] Veri hazır. Sonraki adım: araştırma betiğini çalıştırın.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
