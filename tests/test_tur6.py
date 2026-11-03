"""6. tur: protokol kurallarının kodu (risk seçimi, ölçütler, seçim) ve sentetik veriyle uçtan uca kuru çalıştırma."""
from types import SimpleNamespace

import pytest
from helpers import H1, loose_rules, make_candles

import arastirma as A
import arastirma_tur6 as T6
from bot.config import DEFAULT_CONFIG_PATH, load_config


def test_protocol_file_is_unchanged():
    assert T6.protocol_check() == T6.PROTOCOL_SHA256


@pytest.mark.parametrize("f_star, dengeli, agresif", [
    (0.015, 0.0075, 0.01),       # 0,5 x 1,5 = 0,75; 0,75 x 1,5 = 1,125 -> 1,0
    (0.005, 0.005, 0.005),       # tepe en düşük seviyede: ikisi de %0,5
    (0.025, 0.0125, 0.015),      # 0,75 x 2,5 = 1,875 -> 1,5
    (0.0075, 0.005, 0.005),      # 0,375 -> en az %0,5; 0,5625 -> %0,5
    (0.02, 0.01, 0.015),
])
def test_pick_risk_follows_protocol(f_star, dengeli, agresif):
    assert T6.pick_risk(f_star, T6.PROFILES["dengeli"][0]) == dengeli
    assert T6.pick_risk(f_star, T6.PROFILES["agresif"][0]) == agresif
    assert T6.pick_risk(f_star, 0.75) <= f_star


def test_switches_scale_with_risk():
    o = T6.futures_overrides(0.01, "stres", allow_short=True)
    assert o["risk"]["daily_loss_limit"] == pytest.approx(0.06) and o["risk"]["max_drawdown_halt"] == pytest.approx(0.4)
    assert T6.futures_overrides(0.025, "stres", False)["risk"]["max_drawdown_halt"] == T6.HALT_CAP
    assert T6.futures_overrides(0.005, "temel", False)["risk"]["max_drawdown_halt"] == pytest.approx(0.20)
    assert o["futures"]["liquidation"] == "touch" and o["futures"]["funding_long_8h"] == 0.0003
    assert o["exchange"]["market_type"] == "futures" and o["strategy"]["allow_short"] is True
    assert o["risk"]["stop_atr_mult"] == 2.0 and o["timeframes"]["entry"] == "4h"      # C1 kuralları korunur
    cfg = A.apply_overrides(load_config(DEFAULT_CONFIG_PATH), o)
    assert cfg.is_futures and cfg.round_trip_cost == pytest.approx(2 * (0.0005 + 0.0005))


def fake(metrics_by_key: dict) -> T6.StageA:
    runs = {k: SimpleNamespace(metrics=m, result=SimpleNamespace(liquidations=0)) for k, m in metrics_by_key.items()}
    return T6.StageA("t", {}, runs, {}, {}, {}, {}, {})


def m(ret=0.2, dd=0.2, pf=1.5, cagr=0.1, rtd=1.0):
    return {"net_profit": ret, "total_return": ret, "max_drawdown": dd, "profit_factor": pf, "cagr": cagr,
            "return_to_drawdown": rtd}


def test_choose_prefers_simpler_within_five_percent():
    keys = {}
    for code, rtd in (("F-C1", 1.0), ("F-R1", 1.04), ("F-R1S", 2.0)):
        for period in ("kesif", "dogrulama"):
            keys[(code, 0.005, "stres", period)] = m(rtd=rtd)
    s = fake(keys)
    s.selections = {("dengeli", code): T6.Selection("dengeli", code, 0.005, {"x": (True, "")})
                    for code in T6.SIMPLICITY}
    assert T6.choose(s, "dengeli") == "F-R1S"                     # açık fark: en iyi seçilir
    s.runs[("F-R1S", 0.005, "stres", "dogrulama")].metrics["return_to_drawdown"] = 1.05
    assert T6.choose(s, "dengeli") == "F-C1"                      # %5'ten az fark: en basit
    s.selections[("dengeli", "F-C1")].checks["x"] = (False, "")
    assert T6.choose(s, "dengeli") == "F-R1"
    for code in T6.SIMPLICITY:
        s.selections[("dengeli", code)].checks["x"] = (False, "")
    assert T6.choose(s, "dengeli") is None


def test_risk_must_pay_and_short_side_must_earn():
    keys = {("F-R1S", r, "stres", p): m(cagr=0.10 if r == 0.005 else 0.08) for r in (0.005, 0.01)
            for p in ("kesif", "dogrulama")}
    s = fake(keys)
    s.f_star = {"F-R1S": 0.02}
    s.side_nets = {("F-R1S", 0.01, p): {1: 100.0, -1: -5.0 if p == "dogrulama" else 50.0} for p in ("kesif", "dogrulama")}
    sel = T6.evaluate(s, T6.CANDIDATES[2], "dengeli")             # f* 2 -> %1
    assert sel.risk == 0.01
    assert sel.checks["4_risk_karsiligi"][0] is False             # daha fazla risk daha az büyüme getirdi
    assert sel.checks["5_short"][0] is False and not sel.passed
    assert sel.checks["6_likidasyon"][0] is None


def test_stage_a_dry_run_on_synthetic_market(tmp_path, monkeypatch):
    """Gerçek veriye dokunmadan bütün kod yolları: koşular, seçim, Monte Carlo, rapor, grafik."""
    monkeypatch.setattr(T6, "RISK_GRID", (0.005, 0.01, 0.02))
    monkeypatch.setattr(T6, "MC_PATHS", 300)
    cfg = A.apply_overrides(load_config(DEFAULT_CONFIG_PATH), {"backtest": {"initial_equity": 10_000.0}})
    symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    n = (1_775_001_600_000 - 1_640_995_200_000) // H1                 # 2022-01-01 -> 2026-04-01, 1 saatlik
    raw = {s: {"1h": make_candles(n, seed=40 + i, start="2022-01-01", tf_ms=H1, vol=0.008)}
           for i, s in enumerate(symbols)}
    market = A.prepare_market(raw, 1_775_001_600_000)
    s = T6.run_stage_a(cfg, market, loose_rules(symbols))
    assert set(s.chosen) == set(T6.PROFILES)
    assert all(code is None or code in T6.SIMPLICITY for code in s.chosen.values())
    assert len(s.runs) == 3 * 3 * 2 * 2
    assert {t.side for (code, *_), run in s.runs.items() if code == "F-R1S" for t in run.result.trades} <= {1, -1}
    text = T6.format_stage_a(s)
    assert "Profil: dengeli" in text and "Profil: agresif" in text and "F-R1S" in text
    summary = T6.stage_a_summary(s)
    trades = [("x", "kesif", s.runs[("F-R1S", 0.01, "stres", "kesif")].result.trades)]
    chart = T6.save_report(tmp_path / "rapor", summary, text, trades, lambda p: T6.plot_stage_a(p, s))
    assert chart.exists() and (tmp_path / "rapor" / "ozet.json").exists()
    assert (tmp_path / "rapor" / "islemler.csv").read_text(encoding="utf-8").startswith("ayar,donem")
