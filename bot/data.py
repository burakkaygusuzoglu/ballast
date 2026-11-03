"""Mum verisi: indirme/güncelleme, yerel CSV önbelleği ve doğrulama.

Dosya biçimi: data/<borsa>_<SEMBOL>_<zaman dilimi>.csv  (ts = mum açılış zamanı, UTC ms)
Yalnızca KAPANMIŞ mumlar saklanır. Güncelleme artımlıdır: sadece eksik kısım indirilir.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from bot.config import Config, MarketRules
from bot.timeutil import TIMEFRAME_MS, fmt_utc

if TYPE_CHECKING:
    from bot.exchange import ExchangeClient

CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
_DTYPES = {"ts": "int64", "open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"}
MARKETS_FILE = "markets.json"


def candle_path(data_dir: Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    return Path(data_dir) / f"{exchange_id}_{symbol.replace('/', '-')}_{timeframe}.csv"


def empty_candles() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DTYPES.items()})


def candles_from_rows(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return empty_candles()
    df = pd.DataFrame([r[:6] for r in rows], columns=CANDLE_COLUMNS).astype(_DTYPES)
    return df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


def merge_candles(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        merged = new
    elif new.empty:
        merged = old
    else:
        merged = pd.concat([old, new], ignore_index=True)
    return merged.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True).astype(_DTYPES)


def drop_unclosed(df: pd.DataFrame, tf_ms: int, now_ms: int) -> pd.DataFrame:
    """Kapanış zamanı henüz gelmemiş (oluşmakta olan) mumu atar."""
    return df[df["ts"] + tf_ms <= now_ms].reset_index(drop=True)


def load_candles(path: Path) -> pd.DataFrame:
    # round_trip: kaydedilen her ondalık sayı bit bit aynı geri okunur. Varsayılan okuyucu
    # son basamakta sapabiliyor; bu da backtest ile paper trading arasında fark yaratırdı.
    df = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: eksik sütun(lar): {', '.join(missing)}")
    return merge_candles(None, df[CANDLE_COLUMNS].astype(_DTYPES))


def save_candles(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[CANDLE_COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)   # yarım yazılmış dosya asla kalmaz


def find_gaps(df: pd.DataFrame, tf_ms: int) -> list[tuple[int, int, int]]:
    """(önceki mum, sonraki mum, eksik mum sayısı) listesi."""
    ts = df["ts"].to_numpy()
    if len(ts) < 2:
        return []
    diffs = np.diff(ts)
    idx = np.flatnonzero(diffs > tf_ms)
    return [(int(ts[i]), int(ts[i + 1]), int(diffs[i] // tf_ms - 1)) for i in idx]


def describe(df: pd.DataFrame, tf_ms: int) -> dict:
    gaps = find_gaps(df, tf_ms)
    return {
        "rows": len(df),
        "first": fmt_utc(int(df["ts"].iloc[0])) if len(df) else "-",
        "last": fmt_utc(int(df["ts"].iloc[-1])) if len(df) else "-",
        "gaps": len(gaps),
        "missing_bars": sum(g[2] for g in gaps),
        "largest_gap": max((g[2] for g in gaps), default=0),
    }


def load_pair(cfg: Config, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bir sembolün giriş ve trend zaman dilimi mumlarını yükler."""
    return load_timeframe(cfg, symbol, cfg.timeframes.entry), load_timeframe(cfg, symbol, cfg.timeframes.trend)


def load_timeframe(cfg: Config, symbol: str, timeframe: str) -> pd.DataFrame:
    """Dosya yoksa ve 1 saatlik veri varsa, UTC'ye hizalı mumlar 1 saatliklerden türetilir (borsanınkiyle aynı)."""
    path = candle_path(cfg.data_dir, cfg.exchange.id, symbol, timeframe)
    if path.exists():
        return load_candles(path)
    source = candle_path(cfg.data_dir, cfg.exchange.id, symbol, "1h")
    if TIMEFRAME_MS[timeframe] > TIMEFRAME_MS["1h"] and source.exists():
        from bot.candidates import resample_candles  # döngüsel içe aktarmayı önlemek için burada
        return resample_candles(load_candles(source), TIMEFRAME_MS["1h"], TIMEFRAME_MS[timeframe])
    raise FileNotFoundError(path.name)


def update_candles(
    client: ExchangeClient,
    symbol: str,
    timeframe: str,
    tf_ms: int,
    start_ms: int,
    data_dir: Path,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Yerel dosyayı borsadan günceller (artımlı) ve birleşik veriyi döndürür."""
    log = logger or logging.getLogger("bot.data")
    path = candle_path(data_dir, client.id, symbol, timeframe)
    existing = load_candles(path) if path.exists() else empty_candles()
    now = client.server_now()
    pieces = []

    if existing.empty or int(existing["ts"].iloc[0]) > start_ms + tf_ms:
        until = int(existing["ts"].iloc[0]) if not existing.empty else now
        log.info(f"{symbol} {timeframe}: {fmt_utc(start_ms, False)} sonrası indiriliyor...")
        pieces.append(client.fetch_ohlcv_range(symbol, timeframe, start_ms, until, label=f"{symbol} {timeframe}"))
    if not existing.empty:
        since = int(existing["ts"].iloc[-1]) + tf_ms
        if since + tf_ms <= now:
            pieces.append(client.fetch_ohlcv_range(symbol, timeframe, since, now, label=f"{symbol} {timeframe}"))

    fresh = candles_from_rows([row for piece in pieces for row in piece])
    merged = merge_candles(existing, drop_unclosed(fresh, tf_ms, now))
    merged = merged[merged["ts"] >= start_ms].reset_index(drop=True)
    save_candles(merged, path)
    return merged


def save_market_rules(data_dir: Path, rules: dict[str, MarketRules]) -> Path:
    """Kuralları kaydeder; dosyada olan diğer sembollerin kuralları korunur."""
    path = Path(data_dir) / MARKETS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update({s: {"qty_step": r.qty_step, "min_notional": r.min_notional} for s, r in rules.items()})
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_market_rules(data_dir: Path, symbols: list[str], default: MarketRules) -> dict[str, MarketRules]:
    path = Path(data_dir) / MARKETS_FILE
    stored: dict = {}
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
    return {s: MarketRules(**stored[s]) if s in stored else default for s in symbols}
