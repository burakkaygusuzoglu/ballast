"""Çok vadeli trend topluluğu ve volatilite hedefleme (8. tur; Zarattini, Pagani ve Barbon, 2025).

Her coin ve her kanal uzunluğu L için, günlük kapanışlarla:
- H / D: son L kapanışın en yükseği / en düşüğü (bugün dahil), M = (H + D) / 2.
- Long: kapanış dünkü H'yi aşarsa girilir; stop = M, her gün max(dünkü stop, bugünkü M); kapanış stopun altına
  inerse çıkılır.
- Short (isteğe bağlı): aynası (kapanış dünkü D'nin altına inerse; stop = min(dünkü stop, M); üstüne çıkarsa çıkış).
Topluluk sinyali = alt sinyallerin ortalaması (-1 ile 1 arası). Hedef pozisyon = sinyal x min(V / σ, kaldıraç tavanı),
σ = son 90 günlük getirilerin standart sapması x √365.

Bütün hesaplar NEDENSELDİR: t günündeki değer yalnızca t günü ve öncesinin kapanışlarını kullanır (testlerle doğrulanır).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
VOL_WINDOW = 90
DAYS_PER_YEAR = 365
LEVERAGE_CAP = 2.0


def channel_states(close: np.ndarray, lookback: int, allow_short: bool = False) -> np.ndarray:
    """Tek kanal uzunluğunun alt sinyali her gün için: 1 long, -1 short, 0 nakit."""
    c = np.asarray(close, dtype="float64")
    n = len(c)
    out = np.zeros(n, dtype="int8")
    if n <= lookback:
        return out
    s = pd.Series(c)
    high = s.rolling(lookback, min_periods=lookback).max().to_numpy()
    low = s.rolling(lookback, min_periods=lookback).min().to_numpy()
    mid = (high + low) / 2
    state, stop = 0, math.nan
    for t in range(lookback, n):          # dünkü H/D için t-1 >= lookback-1: L+1 kapanış birikince
        price = c[t]
        if state == 1:
            stop = max(stop, mid[t])
            if price < stop:
                state = 0
        elif state == -1:
            stop = min(stop, mid[t])
            if price > stop:
                state = 0
        if state == 0:
            if price > high[t - 1]:
                state, stop = 1, mid[t]
            elif allow_short and price < low[t - 1]:
                state, stop = -1, mid[t]
        out[t] = state
    return out


def ensemble_signal(close: np.ndarray, allow_short: bool = False, lookbacks=LOOKBACKS) -> np.ndarray:
    """Alt sinyallerin ortalaması; henüz devreye girmemiş kanallar 0 sayılır."""
    states = np.vstack([channel_states(close, lb, allow_short) for lb in lookbacks]).astype("float64")
    return states.mean(axis=0)


def annual_volatility(close: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    """Son `window` günlük basit getirilerin yıllıklandırılmış standart sapması (yeterli veri yoksa NaN)."""
    rets = pd.Series(np.asarray(close, dtype="float64")).pct_change()
    return (rets.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(DAYS_PER_YEAR)).to_numpy()


def coin_exposure(close: np.ndarray, target_vol: float, allow_short: bool = False,
                  cap: float = LEVERAGE_CAP, always_long: bool = False) -> np.ndarray:
    """Coin başına hedef pozisyon (özsermayenin katı olarak): sinyal x min(V / σ, tavan). σ yoksa 0.

    always_long=True: trend sinyali yerine her gün tam long (VY referansı: volatiliteye göre al-tut).
    """
    signal = np.ones(len(close)) if always_long else ensemble_signal(close, allow_short)
    vol = annual_volatility(close)
    scale = np.where(np.isfinite(vol) & (vol > 0), np.minimum(target_vol / np.where(vol > 0, vol, 1.0), cap), 0.0)
    return signal * scale


def portfolio_weights(exposures: dict[str, np.ndarray], gross_cap: float) -> dict[str, np.ndarray]:
    """Sermaye, sinyali sıfır olmayan coinler arasında eşit bölünür; toplam brüt ağırlık gross_cap'i aşamaz."""
    names = list(exposures)
    e = np.vstack([exposures[n] for n in names])
    active = (e != 0).sum(axis=0)
    w = e / np.maximum(active, 1)
    gross = np.abs(w).sum(axis=0)
    scale = np.where(gross > gross_cap, gross_cap / np.where(gross > 0, gross, 1.0), 1.0)
    w = w * scale
    return {n: w[i] for i, n in enumerate(names)}
