"""Backtest motoru: test aracının kendisinin doğruluğu ve uçtan uca geleceği görmeme."""
import math

import numpy as np
import pandas as pd
import pytest
from helpers import H1, M15, loose_rules, make_candles, resample, with_overrides

from backtest import split_periods
from bot.backtester import run_backtest
from bot.metrics import buy_and_hold, compute_metrics, evaluate_gate, max_drawdown
from bot.strategy import compute_signals
from bot.timeutil import parse_utc

RULES = loose_rules(["BTC/USDT"])


def forced_signals(candles: pd.DataFrame, signal_at: list[int], atr: float, regime_ok: bool = True) -> pd.DataFrame:
    df = candles.copy()
    df["atr"] = atr
    df["entry_signal"] = False
    df.loc[signal_at, "entry_signal"] = True
    df["regime_ok"] = regime_ok
    return df


def span(candles: pd.DataFrame) -> tuple[int, int]:
    return int(candles["ts"].iloc[0]), int(candles["ts"].iloc[-1]) + M15


def test_never_trading_keeps_equity_constant(cfg):
    candles = make_candles(800, seed=11)
    res = run_backtest(cfg, {"BTC/USDT": forced_signals(candles, [], atr=1.0)}, *span(candles), 10_000, RULES)
    assert res.equity.eq(10_000).all()
    assert res.total_fees == 0 and res.trades == []


def test_single_buy_and_hold_accounting_is_exact(cfg):
    """Tek giriş, hiç çıkış yok: son özsermaye elle hesaplananla birebir aynı olmalı."""
    candles = make_candles(500, seed=12, vol=0.002)
    market_cfg = with_overrides(
        cfg,
        execution={"entry_order": "market"},
        risk={"risk_per_trade": 0.05, "max_position_pct": 1.0},
        strategy={"exit_on_regime_break": False},
    )
    atr = float(candles["close"].iloc[0]) / 4           # stop çok aşağıda: hiç tetiklenmez
    res = run_backtest(market_cfg, {"BTC/USDT": forced_signals(candles, [0], atr=atr)}, *span(candles), 10_000, RULES)
    assert len(res.trades) == 0 and len(res.open_positions) == 1
    pos = res.open_positions[0]
    fill_price = candles["open"].iloc[1] * (1 + cfg.costs.slippage)
    assert pos.entry_price == pytest.approx(fill_price)
    expected_final = 10_000 - pos.qty * fill_price * (1 + cfg.costs.taker_fee) + pos.qty * candles["close"].iloc[-1]
    assert res.equity.iloc[-1] == pytest.approx(expected_final, rel=1e-12)
    assert res.total_fees == pytest.approx(pos.qty * fill_price * cfg.costs.taker_fee)


def test_engine_has_no_lookahead_end_to_end(cfg, market_15m, market_1h):
    """Veriyi ortadan kesmek, kesimden önceki hiçbir işlemi ve özsermaye değerini değiştirmemeli."""
    sig = compute_signals(market_15m, market_1h, cfg)
    start = int(sig.loc[sig["c_ready"], "ts"].iloc[0])
    end = int(sig["ts"].iloc[-1]) + M15
    cut = int(sig["ts"].iloc[4500])
    full = run_backtest(cfg, {"BTC/USDT": sig}, start, end, 10_000, RULES)
    part = run_backtest(cfg, {"BTC/USDT": sig}, start, cut, 10_000, RULES)
    assert len(full.trades) > 0, "test anlamlı olsun diye işlem olmalı"
    closed_before = [t for t in full.trades if t.exit_ts < cut - M15]
    assert closed_before == [t for t in part.trades if t.exit_ts < cut - M15]
    common = part.equity.index
    assert np.allclose(full.equity.loc[common].to_numpy(), part.equity.to_numpy(), rtol=0, atol=1e-9)


def test_stop_losses_respect_risk_budget(cfg, market_15m, market_1h):
    sig = compute_signals(market_15m, market_1h, cfg)
    start = int(sig.loc[sig["c_ready"], "ts"].iloc[0])
    res = run_backtest(cfg, {"BTC/USDT": sig}, start, int(sig["ts"].iloc[-1]) + M15, 10_000, RULES)
    for t in res.trades:
        assert t.r_multiple >= -1.5, t          # kötümser kayma/boşluk payı ile bile ~-1R civarı
        assert t.risk_amount <= 10_000 * 2 * cfg.risk.risk_per_trade


def test_metrics_on_known_series():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0], index=pd.Index(np.arange(5) * 86_400_000 + 1))
    assert max_drawdown(equity) == pytest.approx(0.25)
    m = compute_metrics(equity, [], 100.0, 0.0)
    assert m["total_return"] == pytest.approx(0.30)
    assert m["return_to_drawdown"] == pytest.approx(1.2)
    assert m["profit_factor"] == 0.0 and m["n_trades"] == 0 and m["cost_share"] is None


def test_buy_and_hold_benchmark_math():
    a = make_candles(200, seed=1)
    b = make_candles(200, seed=2, base=50.0)
    start, end = span(a)
    bh = buy_and_hold({"A": a, "B": b}, start, end, 10_000, 0.001, 0.0005, M15)
    qa = 5_000 / (a["open"].iloc[0] * 1.0005 * 1.001)
    qb = 5_000 / (b["open"].iloc[0] * 1.0005 * 1.001)
    assert bh.iloc[-1] == pytest.approx(qa * a["close"].iloc[-1] + qb * b["close"].iloc[-1])
    assert not math.isnan(bh.iloc[0])


def test_split_periods_uses_fixed_exam_date(cfg):
    m15 = make_candles(20_000, seed=5, start="2023-01-01")          # ~208 gün
    m1h = resample(m15, H1)
    c = with_overrides(cfg, data={"exam_start": "2023-05-01"})
    sig = {"BTC/USDT": compute_signals(m15, m1h, c)}
    periods = split_periods(c, sig)
    exam = parse_utc("2023-05-01")
    assert periods["dev"][1] == exam and periods["exam"][0] == exam
    assert periods["dev"][0] >= int(sig["BTC/USDT"].loc[sig["BTC/USDT"]["c_ready"], "ts"].iloc[0])
    # Veri sınav tarihinden önce bitiyorsa sınav dönemi yoktur (geliştirme yine de çalışır).
    short = {"BTC/USDT": sig["BTC/USDT"][sig["BTC/USDT"]["ts"] < exam - 10 * M15]}
    assert "exam" not in split_periods(c, short) and "dev" in split_periods(c, short)


def test_gate_checks(cfg):
    good = {"profit_factor": 1.5, "max_drawdown": 0.1, "n_trades": 150, "net_profit": 100.0, "return_to_drawdown": 2.0}
    bench = {"return_to_drawdown": 0.5}
    assert all(c.passed for c in evaluate_gate(good, bench, cfg.gate, "dev"))
    bad = dict(good, n_trades=20, profit_factor=0.9)
    failed = [c.name for c in evaluate_gate(bad, bench, cfg.gate, "dev") if not c.passed]
    assert len(failed) == 2
    exam_checks = evaluate_gate(dict(good, net_profit=-1.0), bench, cfg.gate, "exam")
    assert not all(c.passed for c in exam_checks)
