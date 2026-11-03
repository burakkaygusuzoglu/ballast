"""Test yardımcıları: internete bağlanmadan gerçekçi sentetik mumlar ve sahte veri kaynakları."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from bot.config import Config, MarketRules
from bot.timeutil import parse_utc

M15 = 900_000
H1 = 3_600_000
H4 = 4 * H1
D1 = 24 * H1
TF_MS = {"15m": M15, "1h": H1, "4h": H4, "1d": D1}
TEST_CONFIG_PATH = Path(__file__).resolve().parent / "config_test.yaml"   # testlerin sabit (pullback) ayarı
COLS = ["ts", "open", "high", "low", "close", "volume"]


def make_candles(n_bars: int, seed: int = 0, start: str = "2023-01-01", base: float = 100.0,
                 tf_ms: int = M15, vol: float = 0.004) -> pd.DataFrame:
    """Yükseliş / düşüş / yatay rejimler arasında geçiş yapan rastgele yürüyüş."""
    rng = np.random.default_rng(seed)
    drifts: list[float] = []
    while len(drifts) < n_bars:
        drifts.extend([float(rng.choice([0.0004, -0.0003, 0.0]))] * int(rng.integers(300, 1200)))
    rets = np.array(drifts[:n_bars]) + vol * rng.standard_normal(n_bars)
    close = base * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.6, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.6, n_bars)))
    volume = rng.lognormal(3.0, 0.5, n_bars) * (1 + 60 * np.abs(rets))
    ts = parse_utc(start) + np.arange(n_bars, dtype="int64") * tf_ms
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def resample(df: pd.DataFrame, to_ms: int, from_ms: int = M15) -> pd.DataFrame:
    """15m mumları tam saatlik mumlara birleştirir (eksik kalan son saat atılır)."""
    key = df["ts"] // to_ms * to_ms
    grouped = df.groupby(key)
    out = pd.DataFrame({
        "ts": grouped["ts"].first().index.to_numpy(dtype="int64"),
        "open": grouped["open"].first().to_numpy(),
        "high": grouped["high"].max().to_numpy(),
        "low": grouped["low"].min().to_numpy(),
        "close": grouped["close"].last().to_numpy(),
        "volume": grouped["volume"].sum().to_numpy(),
        "n": grouped["ts"].count().to_numpy(),
    })
    out = out[out["n"] == to_ms // from_ms].drop(columns="n")
    return out.reset_index(drop=True)


def with_overrides(cfg: Config, **sections: dict) -> Config:
    """cfg.risk.risk_per_trade gibi değerleri testte değiştirmek için."""
    changes = {name: dataclasses.replace(getattr(cfg, name), **values) for name, values in sections.items()}
    return dataclasses.replace(cfg, **changes)


def loose_rules(symbols) -> dict[str, MarketRules]:
    return {s: MarketRules(qty_step=1e-8, min_notional=1.0) for s in symbols}


class FakeSource:
    """Borsa yerine geçen veri kaynağı.

    - Oluşmakta olan (kapanmamış) mumu da döndürür, ama ZEHİRLİ değerlerle: kod onu
      yanlışlıkla kullanırsa sonuçlar backtest'ten sapar ve testler yakalar.
    - lag_ms: mum kapandıktan sonra verinin yayınlanma gecikmesi. Anahtar zaman dilimi
      ("1h") ya da (sembol, zaman dilimi) olabilir; ikincisi önceliklidir.
    """

    def __init__(self, data: dict[tuple[str, str], pd.DataFrame], lag_ms: dict | None = None):
        self.data = data
        self.lag_ms = lag_ms or {}
        self.calls = 0

    def fetch_ohlcv_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, label: str = "") -> list[list]:
        self.calls += 1
        tf = TF_MS[timeframe]
        df = self.data[(symbol, timeframe)]
        lag = self.lag_ms.get((symbol, timeframe), self.lag_ms.get(timeframe, 0))
        published = df[(df["ts"] >= since_ms) & (df["ts"] + tf + lag <= until_ms)]
        rows = published[COLS].values.tolist()
        forming = df[(df["ts"] >= since_ms) & (df["ts"] <= until_ms) & (df["ts"] + tf > until_ms)]
        for ts, o, h, low, c, v in forming[COLS].values.tolist():
            rows.append([ts, o, h * 1.5, low * 0.5, c * 1.3, v * 9])
        return rows


class FakeClient:
    """ExchangeClient yerine geçer: saati test belirler, veri FakeSource'tan gelir."""

    def __init__(self, data: dict[tuple[str, str], pd.DataFrame], now_ms: int, lag_ms: dict | None = None):
        self.id = "binance"
        self.source = FakeSource(data, lag_ms)
        self.now = now_ms
        self.time_offset_ms = 0

    def milliseconds(self) -> int:
        return self.now

    def server_now(self) -> int:
        return self.now

    def refresh_time_offset(self) -> int:
        return 0

    def fetch_ohlcv_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, label: str = "") -> list[list]:
        return self.source.fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms, label)

    def market_rules(self, symbols, default: MarketRules) -> dict[str, MarketRules]:
        return {s: MarketRules(qty_step=0.00001, min_notional=5.0) for s in symbols}
