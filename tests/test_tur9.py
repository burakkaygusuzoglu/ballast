"""9. tur: çapraz marjin likidasyonu, büyüme analizi (bootstrap, yarıya indirilmiş avantaj, hedef) ve protokol
kurallarının kodu."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from helpers import make_candles
from test_tur8 import synthetic_data

import arastirma_tur9 as T9
from bot import buyume as B
from bot.config import PROJECT_ROOT
from bot.hedef_agirlik import Costs, run_target_weights
from bot.timeutil import DAY_MS, parse_utc

D0 = parse_utc("2020-01-01")
ZERO = Costs(0.0, 0.0)


def frame(prices, start=D0):
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"ts": start + np.arange(len(p)) * DAY_MS, "open": p, "high": p.copy(), "low": p.copy(),
                         "close": p, "volume": 1.0})


def zero_funding(names, n):
    times = np.array([D0 + j * 8 * 3_600_000 for j in range(1, 3 * n + 4)], dtype="int64")
    return {c: (times, np.zeros(len(times))) for c in names}, {c: D0 for c in names}


def run_one(df, weight, mode="stop", mmr=0.02, futures=True):
    n = len(df)
    funding, fstart = zero_funding(["A"], n)
    return run_target_weights({"A": df}, {"A": np.full(n, weight)}, D0 + DAY_MS, D0 + n * DAY_MS, 1000.0, ZERO,
                              futures, funding=funding, funding_start=fstart, liquidation_mmr=mmr,
                              on_liquidation=mode)


# ---- likidasyon ------------------------------------------------------------------------------------------------
def test_long_liquidated_by_intraday_low_and_path_stops():
    df = frame([100.0] * 6)
    df.loc[3, "low"] = 60.0                                    # 3x long: 1 - 3 x 0,4 < 0 -> likide
    res = run_one(df, 3.0)
    assert res.liquidations == [D0 + 3 * DAY_MS]
    assert len(res.equity) == 3 and res.equity.iloc[-1] == 0.0  # 1., 2. ve 3. gün; yol biter
    assert res.returns.iloc[-1] == -1.0 and (res.returns.iloc[:-1] == 0).all()


def test_liquidation_threshold_uses_maintenance_margin():
    # 3x long, mmr %2: en kötü özsermaye 1 - 3(1 - L) = 0,06 L  ->  L* = 2 / 2,94 = 0,68027
    for low, liquidated in ((68.0, True), (68.1, False), (75.0, False)):
        df = frame([100.0] * 6)
        df.loc[3, "low"] = low
        res = run_one(df, 3.0)
        assert bool(res.liquidations) is liquidated, low
    df = frame([100.0] * 6)
    df.loc[3, "low"] = 68.0
    assert run_one(df, 3.0, mmr=0.0).liquidations == []          # bakım marjini olmasa 0,04 > 0: likidasyon yok


def test_short_liquidated_by_intraday_high():
    df = frame([100.0] * 6)
    df.loc[2, "high"] = 140.0                                   # 3x short: 1 - 3 x 0,4 < 0
    res = run_one(df, -3.0)
    assert res.liquidations == [D0 + 2 * DAY_MS]
    df = frame([100.0] * 6)
    df.loc[2, "low"] = 40.0                                     # short için dip zarar değil
    assert run_one(df, -3.0).liquidations == []


def test_restart_mode_records_minus_one_and_continues():
    df = frame([100.0] * 8)
    df.loc[3, "low"] = 60.0
    res = run_one(df, 3.0, mode="restart")
    assert res.liquidations == [D0 + 3 * DAY_MS]
    assert len(res.equity) == 7 and len(res.returns) == 7       # seri kesilmez
    r = res.returns.to_numpy()
    assert r[2] == -1.0 and np.allclose(np.delete(r, 2), 0.0)
    assert res.equity.iloc[-1] == pytest.approx(1000.0)          # önceki kapanış özsermayesiyle yeniden başlar
    assert res.positions["A"].iloc[2] == 0.0 and res.positions["A"].iloc[-1] == pytest.approx(3.0)


def test_liquidation_is_off_by_default_and_never_on_spot():
    df = frame([100.0] * 6)
    df.loc[3, "low"] = 10.0
    assert run_one(df, 3.0, mmr=None).liquidations == []
    assert run_one(df, 1.0, futures=False).liquidations == []    # spotta kaldıraç ve likidasyon yok
    with pytest.raises(ValueError):
        run_one(df, 1.0, mode="hayir")


def test_returns_series_matches_equity_curve():
    df = make_candles(300, seed=5, start="2020-01-01", tf_ms=DAY_MS, vol=0.03)
    w = np.where(np.arange(300) % 40 < 25, 1.2, -0.6)
    res = run_target_weights({"A": df}, {"A": w}, D0 + DAY_MS, D0 + 300 * DAY_MS, 1000.0, Costs(0.0005, 0.0005), True)
    curve = np.concatenate([[1000.0], res.equity.to_numpy()])
    assert np.allclose(res.returns.to_numpy(), curve[1:] / curve[:-1] - 1)


def test_leverage_multiplier_scales_returns_exactly_without_costs():
    # Açılış = önceki kapanış, her gün hedefe işlem (bant 0), maliyet yok: günlük getiri ağırlıkla doğrusal.
    n = 400
    prices = {"A": make_candles(n, seed=11, start="2020-01-01", tf_ms=DAY_MS, vol=0.02),
              "B": make_candles(n, seed=12, start="2020-01-01", tf_ms=DAY_MS, vol=0.02, base=50.0)}
    rng = np.random.default_rng(3)
    base = {"A": rng.uniform(-0.5, 0.5, n), "B": rng.uniform(-0.5, 0.5, n)}
    out = {}
    for k in (1.0, 2.0, 3.0):
        res = run_target_weights(prices, T9.scaled(base, k), D0 + DAY_MS, D0 + n * DAY_MS, 1000.0, ZERO, True,
                                 band=0.0)
        out[k] = res.returns.to_numpy()
    assert np.allclose(out[2.0], 2 * out[1.0], atol=1e-12) and np.allclose(out[3.0], 3 * out[1.0], atol=1e-12)


def test_codes_and_scaling():
    assert T9.code_of("L", 1.5) == "L-k1,5" and T9.parse_code("LS-k8") == ("LS", 8.0)
    assert all(T9.parse_code(T9.code_of(d, k)) == (d, k) for d in T9.DIRECTIONS for k in T9.GRID)
    w = T9.scaled({"A": np.array([0.2, -0.1])}, 4.0)
    assert np.allclose(w["A"], [0.8, -0.4])


# ---- büyüme analizi --------------------------------------------------------------------------------------------
def test_block_starts_shape_bounds_and_determinism():
    s = B.block_starts(100, 365, 50, 20, seed=1)
    assert s.shape == (50, 19) and s.min() >= 0 and s.max() <= 80
    assert np.array_equal(s, B.block_starts(100, 365, 50, 20, seed=1))
    with pytest.raises(ValueError):
        B.block_starts(10, 365, 5, 20, seed=1)
    assert B.horizon_days(10) == 3653 and B.horizon_days(20) == 7305 and B.horizon_days(5) == 1827


def test_simulate_constant_growth_and_log_growth():
    starts = B.block_starts(50, 400, 30, 20, seed=2)
    st = B.simulate(np.full(50, 0.001), starts, 400, 20)
    assert np.allclose(st.final, 1.001 ** 400)
    assert B.median_log_growth(st.final, 400) == pytest.approx(math.log(1.001) * 365.25)


def test_simulate_contributions_and_first_hit():
    starts = B.block_starts(40, 100, 4, 20, seed=3)
    st = B.simulate(np.zeros(40), starts, 95, 20, initial=10.0, contribution=5.0, every=30, levels=(20.0,),
                    snapshot_days=(60,))
    assert np.allclose(st.final, 25.0) and np.allclose(st.snapshots[60], 20.0)
    assert (st.first_hit[20.0] == 59).all()                     # 2. katkı 60. günün sonunda
    assert B.median_years_to_hit(st.first_hit[20.0]) == pytest.approx(60 / 365.25)


def test_simulate_liquidation_zeroes_path_and_contributions_restart():
    material = np.array([0.0] * 19 + [-1.0])
    starts = np.zeros((3, 2), dtype=np.int64)
    st = B.simulate(material, starts, 40, 20, dd_days=40)
    assert (st.final == 0).all() and (st.ruined_day == 19).all()
    assert B.within(st.ruined_day, 20) == 1.0 and B.within(st.ruined_day, 19) == 0.0
    assert (st.max_dd == 1.0).all()
    assert B.median_log_growth(st.final, 40) == -math.inf
    st = B.simulate(material, starts, 35, 20, contribution=2.0, every=30)
    assert np.allclose(st.final, 2.0)                           # 20. günde sıfırlandı, 30. günde 2 katkı


def test_max_drawdown_tracking_and_haircut():
    starts = np.arange(3, dtype=np.int64)[None, :].repeat(2, axis=0)   # blok 1: yolun i. günü = malzemenin i. günü
    st = B.simulate(np.array([1.0, -0.6, 0.5]), starts, 3, 1, dd_days=3)
    assert np.allclose(st.final, 1.2) and np.allclose(st.max_dd, 0.6)
    assert np.allclose(B.haircut(np.array([0.01, -1.0, -0.5]), 2.0, 0.3), [-0.59, -1.0, -1.0])
    assert B.haircut_step(np.array([0.002, 0.0, 0.001])) == pytest.approx(0.0005)
    assert B.haircut_step(np.array([-0.002, 0.001])) == 0.0         # ortalama negatifse h = 0


def test_drawdown_window_includes_its_last_day():
    starts = np.arange(4, dtype=np.int64)[None, :]
    st = B.simulate(np.array([1.0, 0.0, -0.6, 0.0]), starts, 4, 1, dd_days=3)
    assert st.max_dd[0] == pytest.approx(0.6)                   # düşüş pencerenin son gününde
    st = B.simulate(np.array([1.0, 0.0, 0.0, -0.6]), starts, 4, 1, dd_days=3)
    assert st.max_dd[0] == 0.0                                  # pencere dışı sayılmaz


def test_selection_stats_applies_haircut_scaled_by_k():
    starts = B.block_starts(40, T9.SEL_DAYS, 5, T9.MC_BLOCK, seed=1)
    row = T9.selection_stats(np.full(40, 0.001), 2.0, 0.0005, starts)
    assert row["g_yari"] == pytest.approx(0.0, abs=1e-12)      # 0,001 - 2 x 0,0005 = 0
    assert row["g_tarihsel"] == pytest.approx(math.log(1.001) * 365.25)
    assert row["lik10_yari"] == 0.0 and row["dd50_tarihsel"] == 0.0


def test_median_years_to_hit_lower_median_and_inf():
    assert B.median_years_to_hit(np.array([10, 20, -1, -1])) == pytest.approx(21 / 365.25)   # tam yarısı
    assert math.isinf(B.median_years_to_hit(np.array([10, -1, -1, -1])))


# ---- protokol kuralları ----------------------------------------------------------------------------------------
def test_pick_k_star_prefers_lower_on_ties():
    g = dict(zip(T9.GRID, [0.10, 0.20, 0.20, 0.15, 0.0, -1.0, -math.inf], strict=True))
    assert T9.pick_k_star(g) == 1.5
    assert T9.pick_k_star(dict.fromkeys(T9.GRID, -math.inf)) == 1.0


def test_growth_profile_is_half_kelly_on_grid():
    expected = {8.0: 4.0, 6.0: 3.0, 4.0: 2.0, 3.0: 1.5, 2.0: 1.0, 1.5: 1.0, 1.0: 1.0}
    assert {k: T9.growth_profile(k) for k in expected} == expected


def test_protected_profile_is_uninterrupted_prefix_capped_by_growth():
    ok = {1.0: True, 1.5: True, 2.0: False, 3.0: True, 4.0: True, 6.0: True, 8.0: True}
    assert T9.protected_profile(4.0, ok) == 1.5                 # 2 kaldı: 3 ve 4 sayılmaz
    assert T9.protected_profile(1.0, dict.fromkeys(T9.GRID, True)) == 1.0
    assert T9.protected_profile(3.0, dict.fromkeys(T9.GRID, True)) == 3.0
    assert T9.protected_profile(4.0, {**ok, 1.0: False}) is None


def test_choose_direction():
    assert T9.choose_direction({}) is None
    assert T9.choose_direction({"LS": 0.1}) == "LS"
    assert T9.choose_direction({"L": 0.100, "LS": 0.105}) == "L"     # fark 1 puandan az: daha basit olan
    assert T9.choose_direction({"L": 0.10, "LS": 0.20}) == "LS"
    assert T9.choose_direction({"L": 0.30, "LS": 0.20}) == "L"
    assert T9.choose_direction({"L": -math.inf, "LS": 0.05}) == "LS"


def fake_checks(l_ok=True, ls_ok=False):
    def checks(ok, with_short):
        names = ["1_kar", "2_dusus", "3_al_tut", "4_trend_degeri"] + (["5_short_degeri"] if with_short else [])
        out = {n: (True, "") for n in names}
        out["2_dusus"] = (False, "")                             # 2. ölçüt uygunlukta aranmaz
        if not ok:
            out[names[-1]] = (False, "")
        return out
    return {"T1-25": checks(l_ok, False), "T2-25": checks(ls_ok, True)}


def test_eligibility_uses_edge_gates_only():
    e = T9.eligibility(fake_checks(True, False))
    assert e["L"][0] is True and e["LS"][0] is False and "Short" in e["LS"][1]


def test_validation_gates_use_p3_and_every_span():
    def m(cagr, dd=0.10, ret=0.10, liq=()):
        return {"net_profit": ret * 10_000, "total_return": ret, "cagr": cagr, "max_drawdown": dd,
                "likidasyon": list(liq)}
    s = T9.Study9("test", {}, {"L": (True, ""), "LS": (False, "")})
    for sp in ("P1", "P2", "S", "P3", "TAM"):
        s.metrics[("L-k1", sp)] = m(0.05)
        s.metrics[("L-k2", sp)] = m(0.08 if sp == "P3" else 0.01, dd=0.40 if sp == "TAM" else 0.20)
    v = T9.validate(s, "L", 2.0, "K")
    assert all(ok for ok, _ in v.values()) and set(v) == {"1_p3_kar", "2_likidasyon", "3_kaldirac_degeri",
                                                          "4_p3_dusus"}
    assert set(T9.validate(s, "L", 2.0, "B")) == {"1_p3_kar", "2_likidasyon", "3_kaldirac_degeri"}
    s.metrics[("L-k2", "P3")]["cagr"] = 0.04                    # k = 1'in P3 CAGR'ı 0,05
    assert T9.validate(s, "L", 2.0, "B")["3_kaldirac_degeri"][0] is False
    s.metrics[("L-k2", "P3")]["cagr"] = 0.08
    for sp in ("P1", "P2", "S", "P3"):
        s.metrics[("L-k2", sp)]["likidasyon"] = ["2020-03-12"]
        assert T9.validate(s, "L", 2.0, "B")["2_likidasyon"][0] is False, sp
        s.metrics[("L-k2", sp)]["likidasyon"] = []
    s.metrics[("L-k2", "P3")]["max_drawdown"] = 0.26
    assert T9.validate(s, "L", 2.0, "K")["4_p3_dusus"][0] is False
    assert T9.validate(s, "L", 2.0, "B")["1_p3_kar"][0] is True
    s.metrics[("L-k2", "P3")]["net_profit"] = -1.0
    assert T9.validate(s, "L", 2.0, "B")["1_p3_kar"][0] is False
    s.metrics[("L-k2", "P3")].update(net_profit=1.0, max_drawdown=0.2)
    s.profiles = {("L", "B"): 2.0, ("L", "K"): 1.0, ("LS", "B"): 2.0}
    s.validation = {("L", "B"): T9.validate(s, "L", 2.0, "B"), ("LS", "B"): T9.validate(s, "L", 2.0, "B")}
    assert s.adopted("L", "B") and not s.adopted("L", "K")      # k = 1: kaldıraç önerilmez
    assert not s.adopted("LS", "B")                             # yön uygun değil


def test_protocol_file_is_unchanged_and_listed_in_registry():
    assert T9.protocol_check() == T9.PROTOCOL_SHA256
    registry = (PROJECT_ROOT / "docs" / "research" / "PROTOCOL_REGISTRY.md").read_text(encoding="utf-8")
    proto = T9.PROTOCOL_FILE.read_text(encoding="utf-8")
    assert T9.PROTOCOL_SHA256 in registry and proto.startswith("## 22. 9. araştırma turu protokolü")


# ---- kuru çalıştırma -------------------------------------------------------------------------------------------
def synthetic_with_wicks():
    data = synthetic_data()
    for coin, df in data.prices.items():
        wick = np.arange(len(df)) % 97 == 50                    # arada bir %45'lik gün içi iğne (kapanış aynı)
        df.loc[wick, "low"] = df.loc[wick, "close"] * 0.55
        df.loc[wick, "high"] = df.loc[wick, "close"] * 1.45
        data.prices[coin] = df
    return data


def test_study9_dry_run_on_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.setattr(T9, "MC_PATHS", 200)
    data = synthetic_with_wicks()
    s = T9.run_study9(data, fake_checks(True, False))
    n_s = len(s.material[("L-k1", "S")])
    for d in T9.DIRECTIONS:
        for k in T9.GRID:
            c = T9.code_of(d, k)
            assert all((c, sp) in s.metrics for sp in ("P1", "P2", "S", "P3", "TAM"))
            assert len(s.material[(c, "S")]) == n_s                      # yeniden başlatmalı seri kesilmez
    liquidated = [c for (c, sp), m in s.metrics.items() if sp == "TAM" and m["likidasyon"]]
    assert liquidated and all(not s.metrics[(T9.code_of(d, 1.0), "TAM")]["likidasyon"] for d in T9.DIRECTIONS)
    for c in liquidated:
        assert (s.material[(c, "TAM")] == -1.0).any()
        assert s.runs[(c, "TAM")].equity.iloc[-1] == 0.0
    # Profiller protokoldeki kurallarla yeniden hesaplanınca aynı çıkmalı.
    for d in T9.DIRECTIONS:
        g = {k: s.selection[d][k]["g_yari"] for k in T9.GRID}
        assert s.k_star[d] == T9.pick_k_star(g)
        assert s.profiles[(d, "B")] == T9.growth_profile(s.k_star[d])
        assert s.profiles[(d, "K")] is None or s.profiles[(d, "K")] <= s.profiles[(d, "B")]
    for d in T9.DIRECTIONS:
        ok = {k: (s.metrics[(T9.code_of(d, k), "P1")]["max_drawdown"] <= T9.MAX_DD
                  and s.metrics[(T9.code_of(d, k), "P2")]["max_drawdown"] <= T9.MAX_DD
                  and not s.metrics[(T9.code_of(d, k), "S")]["likidasyon"]) for k in T9.GRID}
        assert s.profiles[(d, "K")] == T9.protected_profile(s.profiles[(d, "B")], ok)
        assert s.h[d] == B.haircut_step(s.material[(T9.code_of(d, 1.0), "S")])
        assert s.h_tam[d] == B.haircut_step(s.material[(T9.code_of(d, 1.0), "TAM")])
    for (c, sp), mat in s.material.items():
        if not s.metrics[(c, sp)]["likidasyon"]:
            assert np.array_equal(mat, s.runs[(c, sp)].returns.to_numpy())
    starts = B.block_starts(n_s, T9.SEL_DAYS, 200, T9.MC_BLOCK, T9.MC_SEED)
    again = T9.selection_stats(s.material[("LS-k2", "S")], 2.0, s.h["LS"], starts)
    assert again == s.selection["LS"][2.0]                               # ortak blok sırası (aynı tohum)
    assert not any(s.adopted("LS", p) for p in T9.PROFILES)               # LS uygun değil
    assert all(c in s.projections and c in s.real for c in ("L-k1", "LS-k1"))
    text = T9.format_study9(s)
    for word in ("Uygunluk", "k\\*", "Profiller", "50.000", "Tahmin değildir", "LİKİDE"):
        assert word in text
    chart = T9.save_report(tmp_path / "rapor", s, text)
    assert chart.exists()
    summary = json.loads((tmp_path / "rapor" / "ozet.json").read_text(encoding="utf-8"))
    assert summary["protokol_sha256"] == T9.PROTOCOL_SHA256 and "L|B" in summary["profiller"]
