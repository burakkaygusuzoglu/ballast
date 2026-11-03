"""5. araştırma turu (docs/ARASTIRMA.md, 15. bölüm): L1 sinyali, tahsis boyutlandırması ve protokol."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from helpers import D1, H1, make_candles, with_overrides

import arastirma as A
import arastirma_tur3 as T3
import arastirma_tur4 as T4
import arastirma_tur5 as T5
from bot.candidates import TrendParams, trend_signals
from bot.config import ConfigError, MarketRules
from bot.risk import RiskManager
from bot.timeutil import parse_utc

LOOSE = MarketRules(qty_step=1e-8, min_notional=1.0)


def daily_from_closes(closes, start="2023-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype="float64")
    ts = parse_utc(start) + np.arange(len(closes), dtype="int64") * D1
    return pd.DataFrame({"ts": ts, "open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes,
                         "volume": np.ones(len(closes))})


# ---------------------------------------------------------------------------
# L1 sinyali
# ---------------------------------------------------------------------------
def test_trend_signals_have_no_lookahead(cfg):
    df = make_candles(700, seed=501, start="2022-01-01", tf_ms=D1, vol=0.03)
    full = trend_signals(df, TrendParams(), cfg).set_index("ts")
    assert full["entry_signal"].sum() > 0 and full["exit_signal"].sum() > 0
    rng = np.random.default_rng(5)
    for cut in sorted(rng.integers(260, 699, 10)):
        cut_ts = int(df["ts"].iloc[cut])
        part = trend_signals(df[df["ts"] <= cut_ts], TrendParams(), cfg).set_index("ts")
        a, b = full.loc[cut_ts], part.loc[cut_ts]
        for column in ("sma", "atr", "entry_signal", "exit_signal", "c_ready"):
            if isinstance(a[column], (bool, np.bool_)):
                assert bool(a[column]) == bool(b[column]), f"{column} @ {cut}"
            else:
                assert np.isclose(a[column], b[column], equal_nan=True, atol=1e-12), f"{column} @ {cut}"


def test_trend_signals_hold_above_and_exit_below_the_average(cfg):
    closes = [100.0] * 250 + [100.0 * 1.01 ** i for i in range(1, 61)] + [180.0 * 0.98 ** i for i in range(1, 81)]
    sig = trend_signals(daily_from_closes(closes), TrendParams(sma_n=200), cfg)
    ready = sig[sig["c_ready"]]
    assert ready.index[0] == 199                                   # 200. mumda ortalama hazır
    above = ready["close"] > ready["sma"]
    assert (ready["entry_signal"] == above).all()                  # maliyet filtresi geniş stopla hep geçer
    assert (ready["exit_signal"] == (ready["close"] < ready["sma"])).all()
    assert sig["entry_signal"].iloc[300] and not sig["exit_signal"].iloc[300]    # yükselişte: tut
    assert sig["exit_signal"].iloc[-1] and not sig["entry_signal"].iloc[-1]      # düşüşte: sat
    assert not sig.loc[:198, ["entry_signal", "exit_signal"]].to_numpy().any()   # ısınmada karar yok


# ---------------------------------------------------------------------------
# Tahsis boyutlandırması
# ---------------------------------------------------------------------------
def test_allocation_sizing_takes_fixed_share_regardless_of_stop(cfg):
    c = with_overrides(cfg, risk={"sizing": "allocation", "max_position_pct": 0.5},
                       execution={"entry_order": "market"})
    rm = RiskManager(c, 10_000.0)
    for stop in (90.0, 50.0, 1.0):                                 # stop mesafesi boyutu değiştirmez
        qty, risk = rm.size(10_000.0, 10_000.0, 100.0, stop, LOOSE)
        assert qty * 100.0 == pytest.approx(5_000.0, rel=1e-9)
        loss = (100.0 - stop) + 100.0 * c.entry_fee_rate + stop * c.exit_fee_rate
        assert risk == pytest.approx(qty * loss)
    qty, _ = rm.size(10_000.0, 4_000.0, 100.0, 90.0, LOOSE)        # nakit yetmezse nakitle sınırlı
    assert qty * 100.0 == pytest.approx(4_000.0 / (1 + c.entry_fee_rate) * 0.98)


def test_default_sizing_is_unchanged_risk_based(cfg):
    assert cfg.risk.sizing == "risk"
    c = with_overrides(cfg, risk={"risk_per_trade": 0.01, "max_position_pct": 1.0})
    rm = RiskManager(c, 10_000.0)
    qty, risk = rm.size(10_000.0, 10_000.0, 100.0, 95.0, LOOSE)
    loss = 5.0 + 100.0 * c.entry_fee_rate + 95.0 * c.exit_fee_rate
    assert qty == pytest.approx(100.0 / loss, rel=1e-6) and risk == pytest.approx(100.0, rel=1e-6)


def test_sizing_mode_is_validated(tmp_path):
    from pathlib import Path

    from bot.config import load_config
    text = (Path(__file__).resolve().parent / "config_test.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("risk:\n", "risk:\n  sizing: yarim\n", 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="risk.sizing"):
        load_config(bad)
    good = tmp_path / "good.yaml"
    good.write_text(text.replace("risk:\n", "risk:\n  sizing: allocation\n", 1), encoding="utf-8")
    assert load_config(good).risk.sizing == "allocation"
    assert dataclasses.replace(load_config(good).risk, sizing="risk") == load_config(
        Path(__file__).resolve().parent / "config_test.yaml").risk


# ---------------------------------------------------------------------------
# Protokol (docs/ARASTIRMA.md, 15. bölüm)
# ---------------------------------------------------------------------------
def test_members_are_the_locked_bots_of_earlier_rounds():
    team = T5.members()
    t3 = {c.code: c for c in T3.tur3_candidates()}
    w0a = next(c for c in T4.tur4_candidates() if c.code == "W0A")
    assert list(team) == ["C1", "R1", "R3", "R3A", "L1"]
    assert team["C1"].overrides == t3["R0"].overrides and team["R1"].overrides == t3["R1"].overrides
    assert team["R3"].overrides == t3["R3"].overrides and team["R3A"].overrides == w0a.overrides
    assert all(team[c].symbols == T4.C1_SYMBOLS for c in ("C1", "R1", "R3", "R3A"))
    assert team["L1"].symbols == ("BTC/USDT", "ETH/USDT") and team["L1"].entry_tf == "1d"
    assert T5.L1_PARAMS == TrendParams(sma_n=200, atr_n=20) and T5.L1_ROBUST_SMA == (150, 250)
    risk = T5.L1_RULES["risk"]
    assert (risk["sizing"], risk["max_position_pct"], risk["max_open_positions"]) == ("allocation", 0.5, 2)
    assert (risk["stop_atr_mult"], risk["cooldown_bars"], risk["daily_loss_limit"], risk["max_drawdown_halt"]) == (
        10.0, 0, 0.99, 0.40)
    assert T5.EXAM_MAX_DD == {"C1": 0.20, "R1": 0.20, "R3": 0.20, "P2": 0.30, "R3A": 0.40, "L1": 0.40}
    assert T5.EXAM_PF_EXEMPT == ("L1", "P2") and T5.EXAM_START == "2026-04-01"


def test_exam_status_follows_protocol():
    good = {"net_profit": 100.0, "max_drawdown": 0.10, "profit_factor": 1.5, "return_to_drawdown": 2.0,
            "n_trades": 40}
    assert T5.exam_status("C1", good, 1.0, 0.5) == ("GEÇTİ", "tümü sağlandı")
    assert T5.exam_status("R3", {**good, "n_trades": 9}, 1.0, 0.5)[0] == "sonuçsuz"
    assert T5.exam_status("L1", good, 1.0, 0.05)[0] == "sonuçsuz"
    assert T5.exam_status("L1", {**good, "profit_factor": 0.0, "n_trades": 0}, 1.0, 0.5)[0] == "GEÇTİ"  # KF muaf
    assert T5.exam_status("R1", {**good, "profit_factor": 1.1}, 1.0, 0.5)[0] == "KALDI"
    assert T5.exam_status("R3A", {**good, "max_drawdown": 0.35}, 1.0, 0.5)[0] == "GEÇTİ"      # %40 sınıfı
    assert T5.exam_status("R3", {**good, "max_drawdown": 0.35}, 1.0, 0.5)[0] == "KALDI"       # %20 sınıfı
    assert T5.exam_status("P2", good, 1.0, 1.0, c1_rd=2.5)[0] == "KALDI"                       # C1'den iyi olmalı
    status, detail = T5.exam_status("C1", {**good, "return_to_drawdown": 0.5}, 1.0, 0.5)
    assert status == "KALDI" and "al-tut" in detail


def test_decisions_follow_protocol():
    passed = T5.StageA("", {}, {}, {}, {}, [], {}, {}, l1_checks={"x": (True, "")}, p2_checks={"x": (True, "")})
    failed = T5.StageA("", {}, {}, {}, {}, [], {}, {}, l1_checks={"x": (False, "")}, p2_checks={"x": (False, "")})
    assert T5.decide("L1", "GEÇTİ", passed).startswith("gölge bot olarak eklenir")
    assert T5.decide("L1", "GEÇTİ", failed).startswith("eklenmez")
    assert T5.decide("L1", "KALDI", passed).startswith("eklenmez")
    assert T5.decide("R3", "KALDI", passed) == "gölge bot durdurulur"
    assert T5.decide("R3", "sonuçsuz", passed).startswith("gölge test sürer")
    assert T5.decide("R1", "GEÇTİ", failed) == "gölge test sürer"
    assert T5.decide("C1", "KALDI", failed).startswith("değişmez")


# ---------------------------------------------------------------------------
# Uçtan uca (sentetik veri, 12 coin)
# ---------------------------------------------------------------------------
EXAM = "2026-04-01"


@pytest.fixture(scope="module")
def tur5_market(cfg):
    n = (parse_utc("2026-05-01") - parse_utc("2022-01-01")) // H1
    raw = {s: {"1h": make_candles(n, seed=700 + i, start="2022-01-01", base=30.0 * (i + 1), tf_ms=H1, vol=0.008)}
           for i, s in enumerate(T4.C1_SYMBOLS)}
    c = with_overrides(cfg, data={"exam_start": EXAM})
    return c, raw


@pytest.fixture(scope="module")
def stage_a(tur5_market):
    c, raw = tur5_market
    rules = {s: LOOSE for s in T4.C1_SYMBOLS}
    return T5.run_stage_a(c, A.prepare_market(raw, parse_utc(EXAM)), rules, rules)


def test_stage_a_refuses_exam_data(tur5_market):
    c, raw = tur5_market
    rules = {s: LOOSE for s in T4.C1_SYMBOLS}
    with pytest.raises(AssertionError, match="sınav"):
        T5.run_stage_a(c, A.prepare_market(raw, T5.last_full_day(raw)), rules, rules)


def test_stage_a_runs_protocol_periods_and_checks(stage_a):
    s = stage_a
    assert s.bounds == {"kesif": (parse_utc("2022-07-20"), parse_utc("2025-01-01")),
                        "dogrulama": (parse_utc("2025-01-01"), parse_utc(EXAM))}
    assert set(s.l1_checks) == {"kesif", "dogrulama", "saglamlik", "islem"}
    assert set(s.p2_checks) == {"kesif", "dogrulama", "iyilesme", "l1"}
    assert s.p2_checks["l1"][0] == s.l1_passed
    assert [label for label, _ in s.robustness] == ["150 günlük ortalama", "250 günlük ortalama"]
    assert s.robustness[1][1].start > s.bounds["kesif"][0]           # 250 gün daha geç ısınır
    for name in ("kesif", "dogrulama"):
        trades = s.l1[name].result.trades
        assert trades and {t.symbol for t in trades} <= {"BTC/USDT", "ETH/USDT"}
        peak = float(s.l1[name].result.equity.max())
        assert max(t.qty * t.entry_price for t in trades) <= 0.5 * peak * (1 + 1e-9)   # coin başına en fazla %50
        assert all(t.entry_ts >= s.bounds[name][0] and t.exit_ts < s.bounds[name][1] for t in trades)
        # P2 = iki botun eşit sermayeli birleşimi (yeniden dengeleme yok)
        c1 = s.c1[name].result.equity
        l1 = s.l1[name].result.equity
        expected = (T5.daily_equity(c1).iloc[-1] + T5.daily_equity(l1).iloc[-1]) / 2
        assert s.p2[name].equity.iloc[-1] == pytest.approx(expected)
    text = T5.format_stage_a(s)
    assert "Aşama A kararı" in text and "Uygulanabilirlik" in text


def test_stage_b_runs_every_candidate_once_on_exam_window(tur5_market, stage_a, tmp_path):
    c, raw = tur5_market
    rules = {s: LOOSE for s in T4.C1_SYMBOLS}
    end = T5.last_full_day(raw)
    assert end % D1 == 0 and end <= parse_utc("2026-05-01")
    b = T5.run_stage_b(c, A.prepare_market(raw, end), parse_utc(EXAM), end, rules, rules, stage_a)
    assert list(b.results) == list(T5.EXAM_ORDER)
    for code, r in b.results.items():
        assert r.status in ("GEÇTİ", "KALDI", "sonuçsuz")
        assert r.decision == T5.decide(code, r.status, stage_a)
    for run in b.runs.values():
        assert all(t.entry_ts >= parse_utc(EXAM) for t in run.result.trades)
    with pytest.raises(ValueError):
        T5.run_stage_b(c, A.prepare_market(raw, end), parse_utc("2026-03-01"), end, rules, rules, stage_a)
    text = T5.format_stage_b(b)
    assert "Sınav dönemi" in text and "Gölge botun sermayesiyle" in text
    out = T5.save_stage(tmp_path / "B", T5.stage_b_summary(b), text,
                        [(code, "sinav", run.result.trades) for code, run in b.runs.items()], None)
    assert out is None and (tmp_path / "B" / "ozet.json").exists() and (tmp_path / "B" / "sonuc.md").exists()
