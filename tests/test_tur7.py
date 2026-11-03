"""7. tur: düşüşe göre ölçeklenen risk ve protokol kurallarının kodu; sentetik veriyle uçtan uca kuru çalıştırma."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from helpers import H1, H4, loose_rules, make_candles

import arastirma as A
import arastirma_tur7 as T7
from bot.backtester import run_backtest
from bot.candidates import BreakoutParams, breakout_signals
from bot.config import DEFAULT_CONFIG_PATH, ConfigError, MarketRules, load_config
from bot.risk import RiskManager

RULES = MarketRules(qty_step=1e-8, min_notional=1.0)


def scaled_cfg(base=0.01, limit=0.25):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    return A.apply_overrides(cfg, {"risk": {"risk_per_trade": base, "dd_scaling": limit, "max_drawdown_halt": 0.25,
                                            "max_position_pct": 1.0}})


# ---- risk ölçekleme ----------------------------------------------------------------------------------------------
def test_effective_risk_falls_linearly_with_drawdown():
    rm = RiskManager(scaled_cfg(), 10_000.0)
    assert rm.effective_risk(10_000.0) == pytest.approx(0.01)
    rm.state.peak_equity = 12_000.0
    assert rm.effective_risk(12_000.0 * 0.875) == pytest.approx(0.005)      # %12,5 düşüş: yarı risk
    assert rm.effective_risk(12_000.0 * 0.75) == pytest.approx(0.0)         # %25: sıfır
    assert rm.effective_risk(12_000.0 * 0.70) == 0.0                         # sınırın altı: negatif olmaz
    assert rm.effective_risk(13_000.0) == pytest.approx(0.01)               # yeni zirve (henüz kaydedilmemiş)


def test_scaling_off_keeps_fixed_risk_and_size_uses_effective_risk():
    off = RiskManager(load_config(DEFAULT_CONFIG_PATH), 10_000.0)
    off.state.peak_equity = 20_000.0
    assert off.effective_risk(10_000.0) == 0.005                             # C1: düşüş riski değiştirmez
    rm = RiskManager(scaled_cfg(), 10_000.0)
    q_peak, r_peak = rm.size(10_000.0, 10_000.0, 100.0, 95.0, RULES)
    rm.state.peak_equity = 10_000.0 / 0.875                                  # aynı özsermaye, %12,5 düşüşte
    q_dd, r_dd = rm.size(10_000.0, 10_000.0, 100.0, 95.0, RULES)
    assert r_peak == pytest.approx(100.0, rel=1e-3) and r_dd == pytest.approx(50.0, rel=1e-3)
    assert q_dd == pytest.approx(q_peak / 2, rel=1e-6)
    rm.state.peak_equity = 10_000.0 / 0.75
    assert rm.size(10_000.0, 10_000.0, 100.0, 95.0, RULES) == (0.0, 0.0)


def write(tmp_path, mutate) -> str:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_dd_scaling_is_validated(tmp_path):
    cfg = load_config(write(tmp_path, lambda d: d["risk"].update(dd_scaling=0.2)))   # şalter 0.20: olur
    assert cfg.risk.dd_scaling == 0.2
    with pytest.raises(ConfigError, match="dd_scaling"):
        load_config(write(tmp_path, lambda d: d["risk"].update(dd_scaling=0.3)))     # şalterden büyük
    with pytest.raises(ConfigError, match="dd_scaling"):
        load_config(write(tmp_path, lambda d: d["risk"].update(dd_scaling=-0.1)))


def test_backtest_sizes_every_trade_by_drawdown_at_entry():
    cfg = scaled_cfg(base=0.02)
    candles = {s: make_candles(2500, seed=60 + i, tf_ms=H4, vol=0.02, start="2023-01-01")
               for i, s in enumerate(("AAA/USDT", "BBB/USDT", "CCC/USDT"))}
    sig = {s: breakout_signals(df, BreakoutParams(), cfg) for s, df in candles.items()}
    start, end = A.trade_window(sig, cfg.entry_ms)
    res = run_backtest(cfg, sig, start, end, 10_000.0, loose_rules(candles))
    eq = res.equity
    scaled = 0
    for t in res.trades:
        before = eq[eq.index <= t.entry_ts]
        equity = float(before.iloc[-1])
        peak = max(10_000.0, float(before.max()))
        allowed = 0.02 * max(0.0, 1 - (1 - equity / peak) / 0.25)
        assert t.risk_amount <= equity * allowed * (1 + 1e-9) + 1e-9
        scaled += allowed < 0.02 * 0.99
    assert len(res.trades) > 20 and scaled > 0


# ---- protokol kuralları -------------------------------------------------------------------------------------------
def test_protocol_file_is_unchanged():
    assert T7.protocol_check() == T7.PROTOCOL_SHA256


def test_setups_follow_protocol():
    s = T7.setups()
    assert set(s) == {"R1-K10", "R1-K15", "R1-K20", "R1", "C1", "R1-S10"}
    k15 = s["R1-K15"].overrides["risk"]
    assert (k15["risk_per_trade"], k15["dd_scaling"], k15["max_drawdown_halt"]) == (0.015, 0.25, 0.25)
    assert k15["daily_loss_limit"] == pytest.approx(0.09) and k15["stop_atr_mult"] == 2.0
    assert s["R1"].overrides["risk"]["risk_per_trade"] == 0.005 and "dd_scaling" not in s["R1"].overrides["risk"]
    assert s["R1-S10"].overrides["risk"]["max_drawdown_halt"] == 0.25 and not s["R1-S10"].candidate
    assert [c for c, x in s.items() if x.candidate] == ["R1-K10", "R1-K15", "R1-K20"]
    cfg = A.apply_overrides(load_config(DEFAULT_CONFIG_PATH), s["R1-K20"].overrides)
    assert cfg.risk.max_position_pct == 0.5 and not cfg.is_futures                    # spot, tek pozisyon en çok %50


def run_(ret, dd, pf=1.5, halted=None, skipped=0, orders=10):
    return SimpleNamespace(metrics={"net_profit": ret, "total_return": ret, "max_drawdown": dd, "profit_factor": pf},
                           result=SimpleNamespace(halted_at=halted, skipped=Counter({T7.MIN_NOTIONAL_SKIP: skipped}),
                                                  orders_submitted=orders))


def test_gates_and_selection_prefer_highest_passing_risk():
    runs = {("R1", "kesif"): run_(-0.10, 0.20), ("R1", "dogrulama"): run_(0.10, 0.18)}
    real = {}
    for code in T7.BASE_RISKS:
        runs[(code, "kesif")] = run_(0.30, 0.20)
        runs[(code, "dogrulama")] = run_(0.20, 0.22)
        real[(code, "dogrulama")] = run_(0.1, 0.1, skipped=2, orders=8)
    runs[("R1-K20", "dogrulama")] = run_(0.25, 0.26, halted=123)              # %25 şaltere takıldı
    real[("R1-K15", "dogrulama")] = run_(0.1, 0.1, skipped=11, orders=9)       # %55 sinyal atlandı
    s = T7.StageA("t", {}, runs, real)
    for code in T7.BASE_RISKS:
        s.checks[code] = T7.evaluate(s, code)
    assert s.passed("R1-K10") and not s.passed("R1-K15") and not s.passed("R1-K20")
    assert s.checks["R1-K20"]["2_salter"][0] is False and "şalter" in s.checks["R1-K20"]["2_salter"][1]
    assert s.checks["R1-K15"]["5_56usdt"][0] is False
    runs[("R1-K10", "dogrulama")] = run_(0.05, 0.10)                            # referanstan (%10) düşük
    s.checks["R1-K10"] = T7.evaluate(s, "R1-K10")
    assert s.checks["R1-K10"]["4_getiri"][0] is False


def test_dry_run_on_synthetic_market(tmp_path, monkeypatch):
    """Gerçek veriye dokunmadan bütün kod yolları: koşular, ölçütler, Monte Carlo, Aşama B, rapor, grafik."""
    monkeypatch.setattr(T7, "MC_PATHS", 300)
    cfg = load_config(DEFAULT_CONFIG_PATH)
    symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    exam, end = 1_775_001_600_000, 1_777_593_600_000                          # 2026-04-01, 2026-05-01
    n = (end - 1_640_995_200_000) // H1
    raw = {s: {"1h": make_candles(n, seed=70 + i, start="2022-01-01", tf_ms=H1, vol=0.008)}
           for i, s in enumerate(symbols)}
    s = T7.run_stage_a(cfg, A.prepare_market(raw, exam), loose_rules(symbols))
    assert len(s.runs) == len(T7.setups()) * 2 and len(s.real) == len(s.runs)
    assert s.chosen is None or s.chosen in T7.BASE_RISKS
    b = T7.run_stage_b(cfg, A.prepare_market(raw, end), exam, end, loose_rules(symbols), s)
    assert {"R1", "C1"} <= set(b.runs)
    text = T7.format_stage_a(s) + "\n\n" + T7.format_stage_b(b)
    assert "Ölçütler" in text and "Aşama B" in text and "50 USDT" in text
    chart = T7.save_report(tmp_path / "rapor", {"asama_a": T7.stage_a_summary(s)}, text,
                           [("R1", "kesif", s.runs[("R1", "kesif")].result.trades)], lambda p: T7.plot_stage(p, s, b))
    assert chart.exists() and (tmp_path / "rapor" / "islemler.csv").read_text(encoding="utf-8").startswith("kurulum")
    assert np.isfinite(T7.mean_trade_risk(s.runs[("R1", "kesif")]))
