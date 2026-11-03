"""Teknik göstergeler - saf pandas/numpy.

Tanımlar TradingView (Pine) formülleriyle aynıdır, böylece değerler grafikte doğrulanabilir:
- ema      : ta.ema   (ilk değerle başlar)
- rma      : ta.rma   (Wilder yumuşatması; ilk değer = ilk n değerin basit ortalaması)
- rsi      : ta.rsi   (rma tabanlı)
- atr      : ta.atr   (true range'in rma'sı)
- bollinger: ta.bb    (popülasyon standart sapması, ddof=0)
- obv      : ta.obv

Hepsi nedenseldir: t anındaki değer yalnızca t ve öncesindeki verilerle hesaplanır.
Bu özellik testlerle (geleceği görme testi) korunur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder hareketli ortalaması. İlk geçerli değerden itibaren `period` değerin
    ortalamasıyla başlar, sonra y_t = y_(t-1) + (x_t - y_(t-1)) / period."""
    values = series.to_numpy(dtype="float64")
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size < period:
        return pd.Series(np.full(values.shape, np.nan), index=series.index)
    start = int(valid[0])
    seed_idx = start + period - 1
    if np.isnan(values[start:]).any():
        raise ValueError("rma: veri ortasında boş (NaN) değer var.")
    seeded = np.full(values.shape, np.nan)
    seeded[seed_idx] = values[start : seed_idx + 1].mean()
    seeded[seed_idx + 1 :] = values[seed_idx + 1 :]
    # Baştaki NaN'lardan sonra ilk değerle başlayan özyinelemeli yumuşatma (C hızında).
    out = pd.Series(seeded, index=series.index).ewm(alpha=1.0 / period, adjust=False).mean()
    out.iloc[:seed_idx] = np.nan
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    avg_gain = rma(change.clip(lower=0.0), period)
    avg_loss = rma((-change).clip(lower=0.0), period)
    result = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    # TradingView: kayıp 0 ise 100, kazanç 0 ise 0.
    result = result.mask(avg_loss == 0, 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return result


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1)  # ilk mumda önceki kapanış yok -> high - low


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), period)


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid - num_std * std, mid, mid + num_std * std


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()
