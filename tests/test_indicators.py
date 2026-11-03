"""Göstergeler: dış referans değerler ve TradingView tanımlarının elle yazılmış karşılıkları."""
import math

import numpy as np
import pandas as pd
import pytest

from bot import indicators as ind

# StockCharts ChartSchool RSI hesaplama örneği (14 periyot) - yayımlanmış referans değerler.
STOCKCHARTS_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245, 45.8433, 46.0826,
    45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028, 46.0328, 46.4116, 46.2222, 45.6439,
    46.2122, 46.2521, 45.7137, 46.4515, 45.7835, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672,
    43.4205, 42.6628, 43.1314,
]
STOCKCHARTS_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38,
    54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]


def test_rsi_matches_stockcharts_reference():
    rsi = ind.rsi(pd.Series(STOCKCHARTS_CLOSES), 14)
    assert rsi.first_valid_index() == 14
    assert [round(v, 2) for v in rsi.dropna()] == STOCKCHARTS_RSI


def _pine_ema(values, period):
    alpha, out, prev = 2 / (period + 1), [], None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def _pine_rma(values, period):
    out, prev = [], None
    for i, v in enumerate(values):
        if i < period - 1:
            out.append(math.nan)
        elif i == period - 1:
            prev = sum(values[:period]) / period
            out.append(prev)
        else:
            prev = prev + (v - prev) / period
            out.append(prev)
    return out


def test_ema_matches_tradingview_recursion():
    values = list(np.random.default_rng(1).normal(100, 5, 300))
    got = ind.ema(pd.Series(values), 20).to_numpy()
    assert np.allclose(got, _pine_ema(values, 20), rtol=0, atol=1e-9)


def test_rma_matches_tradingview_recursion():
    values = list(np.random.default_rng(2).uniform(0, 3, 300))
    got = ind.rma(pd.Series(values), 14).to_numpy()
    expected = np.array(_pine_rma(values, 14))
    assert np.isnan(got[:13]).all()
    assert np.allclose(got[13:], expected[13:], rtol=0, atol=1e-9)


def test_rma_rejects_gaps_in_the_middle():
    with pytest.raises(ValueError):
        ind.rma(pd.Series([1.0, 2.0, np.nan, 4.0, 5.0]), 2)


def test_atr_first_value_is_mean_true_range():
    high = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0])
    low = pd.Series([9.0, 10.0, 10.5, 10.0, 11.0])
    close = pd.Series([9.5, 10.5, 11.5, 10.5, 12.5])
    tr = ind.true_range(high, low, close)
    assert tr.tolist() == pytest.approx([1.0, 1.5, 1.5, 1.5, 2.5])  # ilk mum: high - low
    atr = ind.atr(high, low, close, 3)
    assert atr.iloc[2] == pytest.approx((1.0 + 1.5 + 1.5) / 3)
    assert atr.iloc[3] == pytest.approx(atr.iloc[2] + (1.5 - atr.iloc[2]) / 3)


def test_bollinger_uses_population_std():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    lower, mid, upper = ind.bollinger(close, 5, 2.0)
    std = np.std([1, 2, 3, 4, 5])  # ddof=0
    assert mid.iloc[-1] == pytest.approx(3.0)
    assert upper.iloc[-1] == pytest.approx(3.0 + 2 * std)
    assert lower.iloc[-1] == pytest.approx(3.0 - 2 * std)


def test_obv_accumulates_signed_volume():
    close = pd.Series([10.0, 11.0, 10.5, 10.5, 12.0])
    volume = pd.Series([100.0, 50.0, 30.0, 20.0, 10.0])
    assert ind.obv(close, volume).tolist() == [0.0, 50.0, 20.0, 20.0, 30.0]


def test_rsi_extremes_follow_tradingview_rules():
    up = pd.Series(np.linspace(1, 50, 60))
    down = pd.Series(np.linspace(50, 1, 60))
    assert ind.rsi(up, 14).dropna().eq(100.0).all()
    assert ind.rsi(down, 14).dropna().eq(0.0).all()
