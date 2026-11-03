"""Aday stratejiler - docs/ARASTIRMA.md, 1. araştırma turu (S2, S3, S4).

Her aday compute_signals ile aynı sütunları üretir (atr, entry_signal, exit_signal,
regime_ok, cost_r, c_*). Böylece mevcut motor, aynı kötümser dolum modeli ve aynı risk
yöneticisiyle, birebir aynı koşullarda test edilir.

Hepsi nedenseldir: t anındaki sinyal yalnızca t ve öncesindeki kapanmış mumlarla hesaplanır
(testlerle korunur). Kırılım ve çıkış seviyeleri "önceki N mum" üzerinden hesaplanır; içinde
bulunulan mum dahil edilmez.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot import indicators as ind
from bot.config import Config
from bot.strategy import CANDLE_COLUMNS, validate_candles


def resample_candles(df: pd.DataFrame, from_ms: int, to_ms: int) -> pd.DataFrame:
    """Kısa zaman dilimindeki mumlardan UTC'ye hizalı uzun mumlar üretir.

    Binance'in 4 saatlik ve günlük mumları da UTC 00:00'a hizalıdır; bu yüzden 1 saatlik
    mumlardan birleştirilen değerler borsanınkiyle aynıdır. Borsa kesintisi nedeniyle alt
    mumu eksik olan dönem (borsanın yaptığı gibi) mevcut işlemlerle oluşturulur. Veri sonunda
    henüz tamamlanmamış son dönem atılır.
    """
    if to_ms % from_ms != 0 or to_ms <= from_ms:
        raise ValueError("Hedef zaman dilimi, kaynağın tam katı ve ondan büyük olmalı.")
    validate_candles(df, "kaynak mumlar")
    key = df["ts"] // to_ms * to_ms
    grouped = df.groupby(key, sort=True)
    out = pd.DataFrame({
        "ts": grouped["ts"].first().index.to_numpy(dtype="int64"),
        "open": grouped["open"].first().to_numpy(),
        "high": grouped["high"].max().to_numpy(),
        "low": grouped["low"].min().to_numpy(),
        "close": grouped["close"].last().to_numpy(),
        "volume": grouped["volume"].sum().to_numpy(),
    })
    last_source_end = int(df["ts"].iloc[-1]) + from_ms
    if len(out) and int(out["ts"].iloc[-1]) + to_ms > last_source_end:
        out = out.iloc[:-1]
    return out.reset_index(drop=True)


def _base_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_candles(df, "mumlar")
    d = df.loc[:, list(CANDLE_COLUMNS)].reset_index(drop=True).copy()
    d["ts"] = d["ts"].astype("int64")
    return d


def _cost_columns(d: pd.DataFrame, cfg: Config) -> None:
    """Maliyet filtresi: gidiş-dönüş maliyeti / stop mesafesi (1R) en fazla max_cost_r."""
    d["cost_r"] = cfg.round_trip_cost * d["close"] / (cfg.risk.stop_atr_mult * d["atr"])
    d["c_cost"] = d["cost_r"] <= cfg.strategy.max_cost_r


# ---------------------------------------------------------------------------
# S2 / S3 - Kanal kırılımı (Kaplumbağa Sistem 1 mantığı)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BreakoutParams:
    entry_n: int = 20     # kapanış, önceki 20 mumun en yükseğini aşarsa al
    exit_n: int = 10      # kapanış, önceki 10 mumun en düşüğünün altına inerse sat
    atr_n: int = 20       # ilk stop = giriş - stop_atr_mult x ATR(20)


def breakout_signals(df: pd.DataFrame, params: BreakoutParams, cfg: Config) -> pd.DataFrame:
    d = _base_frame(df)
    p = params
    d["atr"] = ind.atr(d["high"], d["low"], d["close"], p.atr_n)
    d["upper"] = d["high"].shift(1).rolling(p.entry_n, min_periods=p.entry_n).max()
    d["lower"] = d["low"].shift(1).rolling(p.exit_n, min_periods=p.exit_n).min()
    d["c_ready"] = np.arange(len(d)) >= 3 * max(p.entry_n, p.exit_n, p.atr_n)
    d["c_breakout"] = d["close"] > d["upper"]
    _cost_columns(d, cfg)
    d["entry_signal"] = d["c_ready"] & d["c_breakout"] & d["c_cost"]
    d["exit_signal"] = (d["close"] < d["lower"]).fillna(False).astype(bool)
    d["regime_ok"] = True
    # Kırılımın gücü (ATR cinsinden): çok coinli hesapta yer azsa güçlü kırılım önce alınır.
    strength = (d["close"] - d["upper"]) / d["atr"]
    d["priority"] = strength.where(d["entry_signal"], 0.0).fillna(0.0)
    return d


def add_short_side(sig: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
    """Vadeli işlem (6. tur): kanal kırılımının aynası, breakout_signals çıktısına eklenir.

    Kapanış önceki entry_n mumun en düşüğünün altına inerse short; kapanış önceki exit_n mumun en yükseğini
    aşarsa short sonraki açılışta kapanır. Hazırlık (c_ready) ve maliyet filtresi (c_cost) long ile aynıdır.
    """
    d = sig.copy()
    p = params
    d["lower_entry"] = d["low"].shift(1).rolling(p.entry_n, min_periods=p.entry_n).min()
    d["upper_exit"] = d["high"].shift(1).rolling(p.exit_n, min_periods=p.exit_n).max()
    d["c_breakdown"] = d["close"] < d["lower_entry"]
    d["short_signal"] = (d["c_ready"].to_numpy(dtype=bool) & d["c_breakdown"].to_numpy(dtype=bool)
                         & d["c_cost"].to_numpy(dtype=bool))
    d["short_exit_signal"] = (d["close"] > d["upper_exit"]).fillna(False).astype(bool)
    strength = (d["lower_entry"] - d["close"]) / d["atr"]
    d["short_priority"] = strength.where(d["short_signal"], 0.0).fillna(0.0)
    return merge_priority(d)


def merge_priority(d: pd.DataFrame) -> pd.DataFrame:
    """Öncelik sütunu: long sinyalinde long gücü, short sinyalinde short gücü (ikisi aynı mumda olamaz)."""
    long_p = d["priority"].where(d["entry_signal"].to_numpy(dtype=bool), 0.0)
    d["priority"] = long_p.where(~d["short_signal"].to_numpy(dtype=bool), d["short_priority"]).fillna(0.0)
    return d


def apply_regime_two_sided(sig: pd.DataFrame, regime: pd.DataFrame, entry_ms: int) -> pd.DataFrame:
    """İki yönlü rejim (6. tur, F-R1S): long yalnızca BTC rejimi olumluyken, short yalnızca rejim hesaplanabilir
    ve OLUMSUZken. Açık pozisyon rejim değişince kapatılmaz (R1 gibi)."""
    d = apply_regime(sig, regime, entry_ms)
    bear = d["c_regime_ready"].to_numpy(dtype=bool) & ~d["c_regime"].to_numpy(dtype=bool)
    d["c_short_regime"] = bear
    d["short_signal"] = d["short_signal"].to_numpy(dtype=bool) & bear
    d["short_priority"] = d["short_priority"].where(d["short_signal"], 0.0)
    return merge_priority(d)


# ---------------------------------------------------------------------------
# S4 - Ortalamaya dönüş (Connors RSI-2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MeanRevParams:
    rsi_n: int = 2
    rsi_entry: float = 10.0   # RSI(2) bunun altındaysa aşırı satım
    trend_n: int = 200        # yalnızca kapanış SMA(200) üzerindeyken (yükseliş eğilimi) al
    exit_n: int = 5           # kapanış SMA(5)'i aşınca çık
    atr_n: int = 14           # koruyucu stop = giriş - stop_atr_mult x ATR(14)


def meanrev_signals(df: pd.DataFrame, params: MeanRevParams, cfg: Config) -> pd.DataFrame:
    d = _base_frame(df)
    p = params
    d["atr"] = ind.atr(d["high"], d["low"], d["close"], p.atr_n)
    d["rsi"] = ind.rsi(d["close"], p.rsi_n)
    d["sma_trend"] = ind.sma(d["close"], p.trend_n)
    d["sma_exit"] = ind.sma(d["close"], p.exit_n)
    d["c_ready"] = np.arange(len(d)) >= 3 * max(p.trend_n, p.exit_n, p.atr_n, p.rsi_n)
    d["c_trend"] = d["close"] > d["sma_trend"]
    d["c_oversold"] = d["rsi"] < p.rsi_entry
    _cost_columns(d, cfg)
    d["entry_signal"] = d["c_ready"] & d["c_trend"] & d["c_oversold"] & d["c_cost"]
    d["exit_signal"] = (d["close"] > d["sma_exit"]).fillna(False).astype(bool)
    d["regime_ok"] = True
    return d


# ---------------------------------------------------------------------------
# 3. tur: BTC rejim filtresi (R1, R2, R3) - docs/ARASTIRMA.md, 9. bölüm
# ---------------------------------------------------------------------------
DAY_MS = 86_400_000


def btc_regime(btc_daily: pd.DataFrame, sma_n: int = 200) -> pd.DataFrame:
    """Günlük BTC rejimi: kapanış > SMA(sma_n). Her değer, günlük mum KAPANDIKTAN sonra kullanılabilir.

    Dönüş: avail_ts (değerin bilindiği an = günlük mumun kapanışı), regime (bool), ready (SMA hesaplanabildi).
    """
    d = _base_frame(btc_daily)
    sma = ind.sma(d["close"], sma_n)
    return pd.DataFrame({
        "avail_ts": (d["ts"] + DAY_MS).astype("int64"),
        "regime": (d["close"] > sma).to_numpy(dtype=bool),
        "ready": sma.notna().to_numpy(dtype=bool),
    })


def apply_regime(sig: pd.DataFrame, regime: pd.DataFrame, entry_ms: int, exit_on_break: bool = False) -> pd.DataFrame:
    """Sinyal tablosuna rejimi NEDENSEL olarak ekler: her mum, kendi kapanışında bilinen son günlük rejimi görür.

    Yeni alım yalnızca rejim olumluyken. exit_on_break=True ise rejim bozulunca açık pozisyon da kapanır.
    Rejim henüz hesaplanamıyorsa (ısınma) mum hazır sayılmaz.
    """
    d = sig.copy()
    close_ts = (d["ts"] + entry_ms).astype("int64")
    keyed = pd.DataFrame({"close_ts": close_ts.to_numpy(), "_row": np.arange(len(d))})
    merged = pd.merge_asof(keyed.sort_values("close_ts"), regime.sort_values("avail_ts"),
                           left_on="close_ts", right_on="avail_ts", direction="backward")
    merged = merged.sort_values("_row")
    ready = merged["ready"].fillna(False).to_numpy(dtype=bool)
    ok = merged["regime"].fillna(False).to_numpy(dtype=bool) & ready
    d["c_regime_ready"] = ready
    d["c_regime"] = ok
    d["regime_ok"] = ok
    d["c_ready"] = d["c_ready"].to_numpy(dtype=bool) & ready
    d["entry_signal"] = d["entry_signal"].to_numpy(dtype=bool) & ok & ready
    if exit_on_break:
        d["exit_signal"] = d["exit_signal"].to_numpy(dtype=bool) | (ready & ~ok)
    if "priority" in d.columns:
        d["priority"] = d["priority"].where(d["entry_signal"], 0.0).fillna(0.0)
    return d


# ---------------------------------------------------------------------------
# 3. tur: momentum rotasyonu (R3) - günlük, coinler arası sıralama
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MomentumParams:
    lookback: int = 28     # gün: getiri penceresi
    top_n: int = 3         # en güçlü ilk 3 coin alınır
    exit_rank: int = 5     # ilk 5'ten düşen satılır
    atr_n: int = 20        # koruyucu stop = giriş - stop_atr_mult x ATR(20)
    sma_n: int = 200       # BTC rejim filtresi


def momentum_signals(daily: dict[str, pd.DataFrame], btc_daily: pd.DataFrame, params: MomentumParams,
                     cfg: Config) -> dict[str, pd.DataFrame]:
    """Her gün kapanışta coinler lookback günlük getiriye göre sıralanır (yalnızca o ana kadarki veriyle).

    Giriş: sıra <= top_n, getiri > 0, BTC rejimi olumlu. Çıkış: sıra > exit_rank ya da getiri <= 0 ya da
    rejim bozuldu. Sıralama yalnızca o gün getirisi hesaplanabilen coinler arasında yapılır.
    """
    p = params
    frames = {s: _base_frame(df) for s, df in daily.items()}
    returns = pd.DataFrame({
        s: pd.Series((d["close"] / d["close"].shift(p.lookback) - 1.0).to_numpy(), index=d["ts"].to_numpy())
        for s, d in frames.items()
    })
    ranks = returns.rank(axis=1, ascending=False, method="first")
    regime = btc_regime(btc_daily, p.sma_n)
    out = {}
    for s, d in frames.items():
        d["atr"] = ind.atr(d["high"], d["low"], d["close"], p.atr_n)
        ret = returns[s].reindex(d["ts"].to_numpy()).to_numpy()
        rank = ranks[s].reindex(d["ts"].to_numpy()).to_numpy()
        d["mom"] = ret
        d["rank"] = rank
        d["c_ready"] = (np.arange(len(d)) >= 3 * max(p.lookback, p.atr_n)) & np.isfinite(ret)
        d["c_top"] = np.nan_to_num(rank, nan=np.inf) <= p.top_n
        d["c_positive"] = np.nan_to_num(ret, nan=-1.0) > 0
        _cost_columns(d, cfg)
        d["entry_signal"] = d["c_ready"] & d["c_top"] & d["c_positive"] & d["c_cost"]
        drop = (np.nan_to_num(rank, nan=np.inf) > p.exit_rank) | (np.nan_to_num(ret, nan=0.0) <= 0)
        d["exit_signal"] = pd.Series(drop & np.isfinite(ret), index=d.index).astype(bool)
        d["regime_ok"] = True
        d["priority"] = pd.Series(np.nan_to_num(ret, nan=0.0), index=d.index).where(d["entry_signal"], 0.0)
        out[s] = apply_regime(d, regime, DAY_MS, exit_on_break=True)
    return out


# ---------------------------------------------------------------------------
# 5. tur: uzun vadeli trend (L1) - her coin kendi 200 günlük ortalamasına göre (docs/ARASTIRMA.md, 15. bölüm)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrendParams:
    sma_n: int = 200       # kapanış SMA(200) üstündeyse tut, altına inince sat
    atr_n: int = 20        # felaket stopu = giriş - stop_atr_mult x ATR(20)


def trend_signals(df: pd.DataFrame, params: TrendParams, cfg: Config) -> pd.DataFrame:
    """Zaman serisi trendi (Faber 2007): her coin yalnızca KENDİ kapanışı ve ortalamasıyla karar verir.

    Giriş: kapanış > SMA(sma_n). Çıkış: kapanış < SMA(sma_n), sonraki açılışta. Nedenseldir: t anındaki karar
    yalnızca t ve öncesinde kapanmış mumlarla verilir. Boyut stratejide değil risk ayarında (risk.sizing).
    """
    d = _base_frame(df)
    p = params
    d["atr"] = ind.atr(d["high"], d["low"], d["close"], p.atr_n)
    d["sma"] = ind.sma(d["close"], p.sma_n)
    d["c_ready"] = d["sma"].notna().to_numpy() & (np.arange(len(d)) >= 3 * p.atr_n)
    d["c_trend"] = (d["close"] > d["sma"]).to_numpy(dtype=bool)
    _cost_columns(d, cfg)
    d["entry_signal"] = d["c_ready"] & d["c_trend"] & d["c_cost"]
    d["exit_signal"] = (d["c_ready"] & (d["close"] < d["sma"])).astype(bool)
    d["regime_ok"] = True
    d["priority"] = 0.0
    return d
