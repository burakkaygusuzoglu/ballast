"""Strateji: geleceği görme hatasına karşı koruma ve filtrelerin doğru çalışması."""
import numpy as np
import pytest

from bot.strategy import CONDITION_COLUMNS, compute_signals, signal_funnel
from helpers import H1, M15, make_candles, with_overrides

COMPARE = ["atr", "rsi", "ema_fast", "cost_r", "trend_close", "trend_ema_slow", "entry_signal", "regime_ok",
           *CONDITION_COLUMNS]


def test_15m_bar_only_sees_closed_1h_bar(cfg, market_15m, market_1h):
    by_ts = compute_signals(market_15m, market_1h, cfg).set_index("ts")
    closes_1h = market_1h.set_index("ts")["close"]
    hour0 = int(market_1h["ts"].iloc[100])           # 1h mum: hour0 -> hour0 + 1h
    # 15m mumunun kapanışı = açılış + 15dk. Saatlik mum ancak saat dolunca kapanır.
    for offset, expected_hour in [(0, hour0 - H1), (M15, hour0 - H1), (2 * M15, hour0 - H1), (3 * M15, hour0)]:
        row = by_ts.loc[hour0 + offset]
        assert row["trend_close"] == closes_1h.loc[expected_hour], f"offset={offset}: kapanmamış 1h mum görüldü"


def test_signals_have_no_lookahead(cfg, market_15m, market_1h):
    """t anındaki sinyal, t sonrası veri hiç yokken hesaplananla birebir aynı olmalı."""
    full = compute_signals(market_15m, market_1h, cfg).set_index("ts")
    rng = np.random.default_rng(3)
    for cut in sorted(rng.integers(2600, len(market_15m) - 1, 12)):
        cut_ts = int(market_15m["ts"].iloc[cut])
        part15 = market_15m.iloc[: cut + 1]
        part1h = market_1h[market_1h["ts"] + H1 <= cut_ts + M15]   # o an kapanmış 1h mumlar
        part = compute_signals(part15, part1h, cfg).set_index("ts")
        a, b = full.loc[cut_ts, COMPARE], part.loc[cut_ts, COMPARE]
        for column in COMPARE:
            if isinstance(a[column], (bool, np.bool_)):
                assert bool(a[column]) == bool(b[column]), f"{column} @ {cut}"
            else:
                assert np.isclose(a[column], b[column], equal_nan=True, rtol=0, atol=1e-9), f"{column} @ {cut}"


def test_synthetic_market_produces_some_signals(cfg, market_15m, market_1h):
    sig = compute_signals(market_15m, market_1h, cfg)
    assert sig["entry_signal"].sum() >= 5
    assert not sig.loc[~sig["c_ready"], "entry_signal"].any()


def test_cost_filter_formula(cfg, market_15m, market_1h):
    sig = compute_signals(market_15m, market_1h, cfg)
    row = sig.dropna(subset=["atr"]).iloc[-1]
    expected = cfg.round_trip_cost / (cfg.risk.stop_atr_mult * row["atr"] / row["close"])
    assert row["cost_r"] == pytest.approx(expected)
    assert bool(row["c_cost"]) == (expected <= cfg.strategy.max_cost_r)


def test_cost_filter_blocks_everything_when_costs_are_huge(cfg, market_15m, market_1h):
    expensive = with_overrides(cfg, costs={"maker_fee": 0.009, "taker_fee": 0.009, "slippage": 0.009})
    sig = compute_signals(market_15m, market_1h, expensive)
    assert sig["entry_signal"].sum() == 0
    assert sig["c_cost"].sum() == 0


def test_signal_funnel_is_monotonic(cfg, market_15m, market_1h):
    sig = compute_signals(market_15m, market_1h, cfg)
    rows = signal_funnel(sig, int(sig["ts"].iloc[0]), int(sig["ts"].iloc[-1]) + 1)
    counts = [count for _, count in rows]
    assert counts[0] == len(sig)
    assert all(a >= b for a, b in zip(counts, counts[1:], strict=False))
    assert counts[-1] == int(sig["entry_signal"].sum())


def test_bad_candles_are_rejected(cfg, market_1h):
    bad = make_candles(200, seed=1)
    bad.loc[50, "close"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_signals(bad, market_1h, cfg)
    unsorted = make_candles(200, seed=1).iloc[::-1]
    with pytest.raises(ValueError, match="artan"):
        compute_signals(unsorted, market_1h, cfg)
