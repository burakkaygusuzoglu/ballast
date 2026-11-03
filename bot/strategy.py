"""3 katmanlı sinyal motoru - sadece long, spot.

Katman 1 · Rejim (1h, yalnızca KAPANMIŞ mumlar): kapanış > EMA200 ve EMA50 > EMA200.
Katman 2 · Momentum (15m): son N mumda RSI geri çekilme seviyesinin altına inmiş, şimdi
          tetik seviyesini yukarı kesiyor ve kapanış EMA20'nin üzerinde.
Katman 3 · Hacim ve oynaklık: hacim ortalamanın, OBV kendi ortalamasının üzerinde;
          kapanış üst Bollinger bandının altında (yükselişi kovalamıyoruz).
Maliyet filtresi: gidiş-dönüş maliyeti, riske edilen tutarın (1R) en fazla max_cost_r'si.

Bu modül dış dünyaya bağlanmaz: mum alır, sinyal döndürür. Backtest ve paper trading
aynı fonksiyonu kullanır. Geleceği görme hatasına karşı testlerle korunur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import indicators as ind
from bot.config import Config

CANDLE_COLUMNS = ("ts", "open", "high", "low", "close", "volume")

# Sıralı koşullar: (sütun, rapordaki açıklama). Sinyal = hepsi aynı anda doğru.
CONDITIONS: tuple[tuple[str, str], ...] = (
    ("c_ready", "Göstergeler hazır"),
    ("c_regime", "1h rejim yükselişte"),
    ("c_pullback", "RSI geri çekilmesi"),
    ("c_trigger", "RSI tetik kesişimi"),
    ("c_above_fast", "Kapanış EMA20 üstünde"),
    ("c_volume", "Hacim ortalama üstünde"),
    ("c_obv", "OBV ortalama üstünde"),
    ("c_not_extended", "Üst bandın altında"),
    ("c_cost", "Maliyet filtresi"),
)
CONDITION_COLUMNS = tuple(column for column, _ in CONDITIONS)


def warmup_bars(cfg: Config) -> tuple[int, int]:
    """Göstergelerin güvenilir hale gelmesi için gereken mum sayısı (giriş, trend)."""
    s = cfg.strategy
    entry = 3 * max(
        s.ema_fast, s.rsi_period, s.bb_period, s.atr_period,
        s.volume_ma_period, s.obv_ema_period, s.pullback_lookback,
    )
    return entry, 3 * s.ema_trend_slow


def validate_candles(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: eksik sütun(lar): {', '.join(missing)}")
    if len(df) == 0:
        raise ValueError(f"{name}: veri boş.")
    if len(df) > 1 and not (np.diff(df["ts"].to_numpy()) > 0).all():
        raise ValueError(f"{name}: zaman damgaları artan sırada ve tekil olmalı.")
    if df[["open", "high", "low", "close", "volume"]].isna().to_numpy().any():
        raise ValueError(f"{name}: boş (NaN) fiyat/hacim değeri var.")


def _trend_frame(df1h: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    s = cfg.strategy
    close = df1h["close"].reset_index(drop=True).astype("float64")
    return pd.DataFrame(
        {
            # Mumun KAPANIŞ zamanı: ancak bu andan sonra bilinir.
            "trend_close_ts": df1h["ts"].to_numpy(dtype="int64") + cfg.trend_ms,
            "trend_close": close.to_numpy(),
            "trend_ema_mid": ind.ema(close, s.ema_trend_mid).to_numpy(),
            "trend_ema_slow": ind.ema(close, s.ema_trend_slow).to_numpy(),
            "trend_bar_no": np.arange(len(close), dtype="int64"),
        }
    )


def compute_signals(df15: pd.DataFrame, df1h: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Giriş zaman dilimindeki her KAPANMIŞ mum için göstergeleri, koşulları ve sinyali döndürür."""
    validate_candles(df15, "giriş mumları")
    validate_candles(df1h, "trend mumları")
    s = cfg.strategy

    d = df15.loc[:, list(CANDLE_COLUMNS)].reset_index(drop=True).copy()
    d["ts"] = d["ts"].astype("int64")
    close, volume = d["close"], d["volume"]

    d["atr"] = ind.atr(d["high"], d["low"], close, s.atr_period)
    d["ema_fast"] = ind.ema(close, s.ema_fast)
    d["rsi"] = ind.rsi(close, s.rsi_period)
    d["bb_lower"], d["bb_mid"], d["bb_upper"] = ind.bollinger(close, s.bb_period, s.bb_std)
    d["vol_ma"] = ind.sma(volume, s.volume_ma_period)
    d["obv"] = ind.obv(close, volume)
    d["obv_ema"] = ind.ema(d["obv"], s.obv_ema_period)

    # Çoklu zaman dilimi: 15m mum kapandığında yalnızca o ana kadar KAPANMIŞ son 1h mum görülür.
    d["close_ts"] = d["ts"] + cfg.entry_ms
    d = pd.merge_asof(
        d, _trend_frame(df1h, cfg),
        left_on="close_ts", right_on="trend_close_ts", direction="backward",
    )

    entry_warmup, trend_warmup = warmup_bars(cfg)
    prev_rsi = d["rsi"].shift(1)
    d["c_ready"] = (np.arange(len(d)) >= entry_warmup) & (d["trend_bar_no"].to_numpy() >= trend_warmup)
    d["c_regime"] = (d["trend_close"] > d["trend_ema_slow"]) & (d["trend_ema_mid"] > d["trend_ema_slow"])
    d["c_pullback"] = prev_rsi.rolling(s.pullback_lookback, min_periods=1).min() < s.rsi_pullback_level
    d["c_trigger"] = (d["rsi"] > s.rsi_trigger_level) & (prev_rsi <= s.rsi_trigger_level)
    d["c_above_fast"] = close > d["ema_fast"]
    d["c_volume"] = volume > d["vol_ma"]
    d["c_obv"] = d["obv"] > d["obv_ema"]
    d["c_not_extended"] = close < d["bb_upper"]
    # Maliyetin R cinsinden değeri: gidiş-dönüş maliyeti / stop mesafesi (oran olarak).
    d["cost_r"] = cfg.round_trip_cost * close / (cfg.risk.stop_atr_mult * d["atr"])
    d["c_cost"] = d["cost_r"] <= s.max_cost_r

    d["entry_signal"] = d[list(CONDITION_COLUMNS)].all(axis=1)
    d["regime_ok"] = d["c_regime"].astype(bool)
    d["exit_signal"] = False   # bu strateji stop, iz süren stop ve rejim çıkışıyla çıkar
    return d


def signal_funnel(signals: pd.DataFrame, start_ts: int, end_ts: int,
                  conditions: tuple[tuple[str, str], ...] = CONDITIONS) -> list[tuple[str, int]]:
    """Her filtreden sonra kalan mum sayısı: hangi katmanın en çok elediğini gösterir."""
    window = signals[(signals["ts"] >= start_ts) & (signals["ts"] < end_ts)]
    mask = pd.Series(True, index=window.index)
    rows = [("Toplam mum", len(window))]
    for column, label in conditions:
        mask &= window[column].astype(bool)
        rows.append((label, int(mask.sum())))
    return rows


def first_ready_ts(signals: pd.DataFrame) -> int | None:
    ready = signals.loc[signals["c_ready"].astype(bool), "ts"]
    return int(ready.iloc[0]) if len(ready) else None
