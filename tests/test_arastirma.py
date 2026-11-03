"""Araştırma betiği: sınav verisini asla görmez; muhasebe, ölçütler ve özetler doğru çalışır."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from helpers import H1, loose_rules, make_candles, with_overrides

import arastirma as A
from bot.candidates import BreakoutParams, MeanRevParams
from bot.timeutil import DAY_MS, TIMEFRAME_MS, parse_utc

EXAM = "2025-03-01"
SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def raw_hourly(start: str, days: int) -> dict:
    return {s: {"1h": make_candles(days * 24, seed=seed, start=start, tf_ms=H1, vol=0.006)}
            for s, seed in zip(SYMBOLS, (41, 42), strict=True)}


@pytest.fixture(scope="module")
def study_cfg(cfg):
    return with_overrides(cfg, data={"exam_start": EXAM})


@pytest.fixture(scope="module")
def market():
    return A.prepare_market(raw_hourly("2024-03-01", 400), parse_utc(EXAM))    # veri sınavdan sonrasına uzanır


@pytest.fixture(scope="module")
def study(study_cfg, market):
    chosen = [c for c in A.protocol_candidates() if c.code in ("S2", "S4")]
    return A.run_study(study_cfg, market, loose_rules(SYMBOLS), candidates=chosen)


def test_exam_data_is_dropped_before_anything_else():
    exam = parse_utc(EXAM)
    raw = raw_hourly("2024-12-01", 120)
    for frames in raw.values():
        h1 = frames["1h"]
        h1.loc[h1["ts"] + H1 > exam, ["open", "high", "low", "close"]] = np.nan   # zehirli sınav verisi
    market = A.prepare_market(raw, exam)                    # sızsaydı mum doğrulaması NaN yüzünden patlardı
    for frames in market.values():
        for tf, df in frames.items():
            assert int(df["ts"].iloc[-1]) + TIMEFRAME_MS[tf] == exam, tf   # son mum tam sınav anında kapanır


def test_study_respects_protocol_periods(study, study_cfg):
    split, exam = parse_utc(A.VALIDATION_START), parse_utc(EXAM)
    for r in study.reports:
        assert r.runs["kesif"].end == split and r.runs["dogrulama"].start == split
        assert r.runs["dogrulama"].end == exam
        assert r.runs["kesif"].metrics["n_trades"] > 0, "test anlamlı olsun diye işlem olmalı"
        assert set(r.checks) == {"kesif", "dogrulama", "islem", "saglamlik"}
    assert [len(r.robustness) for r in study.reports] == [6, 8]
    assert set(study.portfolios["tum"]) == {"kesif", "dogrulama"}


def test_portfolio_aligns_members_to_common_window(study):
    p = study.portfolios["tum"]["kesif"]
    assert p.start == max(r.runs["kesif"].start for r in study.reports)
    assert p.equity.iloc[0] == study.initial_equity
    assert list(p.correlation.columns) == ["S2", "S4"]


def test_report_and_summary_are_complete(study, tmp_path):
    text = A.format_markdown(study)
    for fragment in ("Keşif dönemi", "Doğrulama dönemi", "Ölçütler", "Sağlamlık", "Monte Carlo", "S2", "S4"):
        assert fragment in text
    json.dumps(A.study_summary(study), allow_nan=False)
    A.save_study(tmp_path, study, text, plot=False)
    trades = pd.read_csv(tmp_path / "islemler.csv")
    expected = sum(len(run.result.trades) for r in study.reports for run in r.runs.values())
    assert len(trades) == expected


def test_variants_change_exactly_one_parameter():
    by_code = {c.code: c for c in A.protocol_candidates()}
    assert A.robustness_variants(by_code["S0"]) == []
    for code, n in (("S1", 6), ("S2", 6), ("S3", 6), ("S4", 8)):
        base = by_code[code]
        variants = A.robustness_variants(base)
        assert len(variants) == n
        for label, v in variants:
            changed = (v.params != base.params) + sum(
                v.overrides.get(sec, {}).get(key) != base.overrides.get(sec, {}).get(key)
                for sec in v.overrides for key in v.overrides[sec])
            assert changed == 1, label
    s2 = dict(A.robustness_variants(by_code["S2"]))
    assert s2["giriş kanalı 15"].params == BreakoutParams(entry_n=15, exit_n=10, atr_n=20)
    assert s2["stop 2,5 ATR"].overrides["risk"] == {"stop_atr_mult": 2.5, "trail_atr_mult": 0.0}
    s4 = dict(A.robustness_variants(by_code["S4"]))
    assert s4["çıkış SMA 6"].params == MeanRevParams(exit_n=6)


def test_pullback_candidate_uses_hourly_entry_and_4h_trend(study_cfg, market):
    s1 = next(c for c in A.protocol_candidates() if c.code == "S1")
    cfg1 = A.apply_overrides(study_cfg, s1.overrides)
    assert cfg1.entry_ms == H1 and cfg1.trend_ms == 4 * H1
    signals = A.build_signals(s1, cfg1, market)
    start, _ = A.trade_window(signals, cfg1.entry_ms)
    first = int(market["BTC/USDT"]["4h"]["ts"].iloc[0])
    assert start >= first + 3 * study_cfg.strategy.ema_trend_slow * 4 * H1     # 4h EMA200 ısınması


def test_evaluate_uses_protocol_thresholds():
    c = next(c for c in A.protocol_candidates() if c.code == "S2")

    def run(pf, net, dd, n, nets=(1.0, 1.0)):
        return SimpleNamespace(metrics={"profit_factor": pf, "net_profit": net, "max_drawdown": dd, "n_trades": n},
                               symbol_net=dict(zip(SYMBOLS, nets, strict=True)))

    good = {"kesif": run(1.2, 10.0, 0.20, 20), "dogrulama": run(1.0, 1.0, 0.20, 10)}
    checks = A.evaluate(c, good, [("x", {"profit_factor": 1.0})])
    assert all(ok for ok, _ in checks.values())
    bad = {"kesif": run(1.19, 10.0, 0.10, 20, nets=(11.0, 0.0)), "dogrulama": run(0.99, 1.0, 0.21, 9)}
    checks = A.evaluate(c, bad, [("x", {"profit_factor": 0.99})])
    assert not any(ok for ok, _ in checks.values())
    assert "ETH" in checks["kesif"][1] and "PF" in checks["kesif"][1]


def test_monte_carlo_and_drawdown():
    assert A.path_drawdown(np.array([-1.0, -1.0, 2.0]), 0.5) == pytest.approx(0.75)
    r = list(np.random.default_rng(0).normal(0.1, 1.5, 200))
    a, b = A.monte_carlo(r, 0.01), A.monte_carlo(r, 0.01)
    assert a == b and a["n"] == 200
    assert 0 < a["median_dd"] <= a["p95_dd"] < 1
    assert A.monte_carlo([0.5], 0.01)["p95_dd"] is None


def test_combine_equal_averages_normalized_curves():
    day0 = parse_utc("2024-01-01")
    index = pd.Index([day0 + DAY_MS, day0 + 2 * DAY_MS], dtype="int64")
    up = pd.Series([10_500.0, 11_000.0], index=index)
    down = pd.Series([9_500.0, 9_000.0], index=index)
    combined = A.combine_equal([up, down], 10_000.0, day0)
    assert combined.tolist() == [10_000.0, 10_000.0, 10_000.0]
    assert combined.index.tolist() == [day0, day0 + DAY_MS, day0 + 2 * DAY_MS]


def test_pre_cost_profit_factor_adds_back_slippage_but_not_fees():
    def trade(gross, slip, fees):
        return SimpleNamespace(gross_pnl=gross, slippage_cost=slip, net_pnl=gross - fees)
    trades = [trade(10.0, 1.0, 2.0), trade(-5.0, 0.5, 2.0), trade(-0.5, 0.5, 2.0)]   # maliyetsiz: +11, -4.5, 0
    assert A.pre_cost_profit_factor(trades) == pytest.approx(11.0 / 4.5)
    assert A.pre_cost_profit_factor([]) == 0.0
