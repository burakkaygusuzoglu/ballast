"""8. tur: trend topluluğu sinyalleri, hedef ağırlıklı backtest muhasebesi ve protokol kurallarının kodu."""
import math

import numpy as np
import pandas as pd
import pytest
from helpers import make_candles

import arastirma_tur8 as T8
from bot.hedef_agirlik import (
    SPOT_COSTS,
    Costs,
    SymbolRules,
    buy_and_hold,
    run_target_weights,
    tw_metrics,
)
from bot.timeutil import DAY_MS, parse_utc
from bot.trend_toplulugu import (
    LOOKBACKS,
    annual_volatility,
    channel_states,
    coin_exposure,
    ensemble_signal,
    portfolio_weights,
)

D0 = parse_utc("2020-01-01")


# ---- sinyaller ------------------------------------------------------------------------------------------------
def test_long_entry_on_new_high_and_trailing_midpoint_exit():
    close = np.array([10, 11, 12, 11, 13, 14, 15, 14, 13, 12, 11, 10], dtype=float)
    s = channel_states(close, 3)
    # t=3: dünkü H(10,11,12)=12, kapanış 11 -> giriş yok; t=4: 13 > H(11,12,11)=12 -> long, stop M=(13+11)/2=12
    # t=5: stop max(12, 12.5); t=6: max(12.5, 14)=14; t=7: pencere (14,15,14) M=14.5, kapanış 14 < 14.5 -> çıkış
    assert list(s) == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_short_side_mirrors_long_side():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 600)))
    for lb in (5, 20, 60):
        ls = channel_states(close, lb, allow_short=True)
        mirrored = channel_states(2 * close.max() - close, lb, allow_short=True)   # K - fiyat: tam ayna
        assert (ls == -mirrored).all()
        assert set(np.unique(ls)) <= {-1, 0, 1}
        long_only = channel_states(close, lb)
        assert ((ls == 1) == (long_only == 1)).all()                      # long girişleri aynı zamanda


def test_signals_are_causal():
    df = make_candles(1200, seed=3, tf_ms=DAY_MS, vol=0.03)
    close = df["close"].to_numpy()
    full = coin_exposure(close, 0.25, allow_short=True)
    for cut in (400, 700, 1000):
        part = coin_exposure(close[:cut], 0.25, allow_short=True)
        assert np.allclose(part, full[:cut], equal_nan=True)


def test_ensemble_warmup_and_range():
    close = np.linspace(100, 300, 500)                                    # sürekli yükseliş
    sig = ensemble_signal(close)
    assert sig[4] == 0 and 0 < sig[6] < 1                                 # yalnızca 5 günlük kanal devrede
    assert sig[400] == 1.0                                                # hepsi long
    assert ((sig >= -1) & (sig <= 1)).all()


def test_volatility_and_exposure_cap():
    rng = np.random.default_rng(2)
    rets = rng.normal(0, 0.02, 300)
    close = 100 * np.cumprod(1 + rets)
    vol = annual_volatility(close)
    manual = np.std(pd.Series(close).pct_change().to_numpy()[-90:], ddof=1) * math.sqrt(365)
    assert np.isnan(vol[89]) and vol[-1] == pytest.approx(manual)
    calm = np.linspace(100, 110, 400)                                     # çok düşük volatilite
    e = coin_exposure(calm, 0.25)
    assert e.max() == pytest.approx(2.0)                                  # kaldıraç tavanı


def test_portfolio_weights_split_equally_and_cap_gross():
    e = {"A": np.array([0.0, 1.2, 1.5, -0.8]), "B": np.array([0.0, 0.0, 1.5, 0.6])}
    w = portfolio_weights(e, gross_cap=1.0)
    assert w["A"][1] == pytest.approx(1.0)                                # tek açık coin: 1,2 -> tavan 1,0
    assert w["A"][2] == pytest.approx(0.5) and w["B"][2] == pytest.approx(0.5)   # 0,75 + 0,75 -> 1,0'a ölçek
    assert w["A"][3] == pytest.approx(-0.4) and w["B"][3] == pytest.approx(0.3)
    w2 = portfolio_weights(e, gross_cap=2.0)
    assert w2["A"][2] == pytest.approx(0.75) and w2["B"][2] == pytest.approx(0.75)


# ---- hedef ağırlıklı backtest ---------------------------------------------------------------------------------------
def frame(prices, start=D0):
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"ts": start + np.arange(len(p)) * DAY_MS, "open": p, "high": p, "low": p, "close": p,
                         "volume": 1.0})


def test_flat_prices_cost_only_fees():
    prices = {"A": frame([100.0] * 10)}
    targets = {"A": np.full(10, 0.5)}
    res = run_target_weights(prices, targets, D0 + DAY_MS, D0 + 10 * DAY_MS, 1000.0, SPOT_COSTS, futures=False)
    assert len(res.trades) == 1                                           # hedef değişmiyor: tek işlem
    t = res.trades[0]
    assert t["price"] == pytest.approx(100 * (1 + SPOT_COSTS.slippage))
    assert res.equity.iloc[-1] == pytest.approx(1000 - t["fee"] - t["qty"] * (t["price"] - 100))


def test_band_skips_small_changes_and_trades_big_ones():
    n = 6
    prices = {"A": frame([100.0] * n)}
    targets = {"A": np.array([0.50, 0.55, 0.65, 0.65, 0.0, 0.0])}         # +%10 bant içi, +%30 bant dışı, sonra çıkış
    res = run_target_weights(prices, targets, D0 + DAY_MS, D0 + n * DAY_MS, 1000.0, Costs(0, 0), futures=False)
    days = [(t["ts"] - D0) // DAY_MS for t in res.trades]
    assert days == [1, 3, 5]                                               # açılış, büyük değişiklik, çıkış
    assert res.positions["A"].iloc[-1] == 0


def test_spot_never_shorts_or_borrows_and_futures_short_profits():
    prices = {"A": frame(np.linspace(100, 50, 20))}
    short = {"A": np.full(20, -1.0)}
    spot = run_target_weights(prices, short, D0 + DAY_MS, D0 + 20 * DAY_MS, 1000.0, SPOT_COSTS, futures=False)
    assert spot.trades == [] and spot.equity.iloc[-1] == pytest.approx(1000.0)
    fut = run_target_weights(prices, short, D0 + DAY_MS, D0 + 20 * DAY_MS, 1000.0, Costs(0, 0), futures=True,
                             band=1e9)                                     # bant: yalnızca ilk açılış
    assert fut.equity.iloc[-1] > 1400                                      # fiyat %50 düştü, short kazandı
    lev = {"A": np.full(20, 1.5)}
    spot_lev = run_target_weights(prices, lev, D0 + DAY_MS, D0 + 20 * DAY_MS, 1000.0, SPOT_COSTS, futures=False)
    assert spot_lev.gross_exposure.max() <= 1.0 + 1e-9                     # spotta nakitten fazla alınmaz


def test_funding_charges_longs_and_pays_shorts():
    n = 5
    prices = {"A": frame([100.0] * n)}
    times = np.array([D0 + k * 8 * 3_600_000 for k in range(1, 3 * n + 1)], dtype="int64")
    rates = np.full(len(times), 0.001)
    for w, sign in ((1.0, 1), (-1.0, -1)):
        res = run_target_weights(prices, {"A": np.full(n, w)}, D0 + DAY_MS, D0 + n * DAY_MS, 1000.0, Costs(0, 0),
                                 futures=True, funding={"A": (times, rates)}, funding_start={"A": D0})
        assert res.funding == pytest.approx(sign * 4 * 3 * 0.001 * 1000.0)   # 4 gün x 3 funding
        assert res.equity.iloc[-1] == pytest.approx(1000.0 - res.funding)
    before = run_target_weights(prices, {"A": np.full(n, 1.0)}, D0 + DAY_MS, D0 + n * DAY_MS, 1000.0, Costs(0, 0),
                                futures=True, funding={}, funding_start={})
    assert before.funding == pytest.approx(4 * 3 * 0.0001 * 1000.0)        # kayıt yokken sabit %0,01


def test_real_rules_round_and_skip_small_orders():
    prices = {"A": frame([30_000.0] * 4)}
    rules = {"A": SymbolRules(qty_step=0.001, min_notional=50.0)}
    res = run_target_weights(prices, {"A": np.full(4, 0.5)}, D0 + DAY_MS, D0 + 4 * DAY_MS, 50.0, Costs(0, 0),
                             futures=False, rules=rules)
    assert res.trades == [] and res.skipped_small >= 1                     # 25 USDT < 50 USDT en küçük tutar
    rules = {"A": SymbolRules(qty_step=0.0001, min_notional=5.0)}
    res = run_target_weights(prices, {"A": np.full(4, 0.5)}, D0 + DAY_MS, D0 + 4 * DAY_MS, 50.0, Costs(0, 0),
                             futures=False, rules=rules)
    assert res.trades[0]["qty"] == pytest.approx(0.0008)                   # 0,000833 -> adıma aşağı


def test_buy_and_hold_and_metrics():
    prices = {"A": frame([100, 110, 120, 90, 130]), "B": frame([10, 10, 10, 10, 10])}
    curve = buy_and_hold(prices, D0, D0 + 5 * DAY_MS, 1000.0, Costs(0, 0))
    assert curve.iloc[-1] == pytest.approx(500 * 1.3 + 500)
    m = tw_metrics(curve, 1000.0)
    assert m["max_drawdown"] == pytest.approx(1 - (500 * 0.9 + 500) / (500 * 1.2 + 500))
    assert m["total_return"] == pytest.approx(0.15) and m["sharpe"] > 0


# ---- protokol ------------------------------------------------------------------------------------------------------
def test_protocol_file_is_unchanged():
    assert T8.protocol_check() == T8.PROTOCOL_SHA256


def synthetic_data(end="2026-10-10") -> T8.Data8:
    start = parse_utc("2017-08-17")
    n = (parse_utc(end) - start) // DAY_MS
    prices = {}
    for i, coin in enumerate(T8.COINS):
        df = make_candles(n, seed=80 + i, start="2017-08-17", tf_ms=DAY_MS, vol=0.035, base=1000.0 * (i + 1))
        prices[coin] = df
    fstart = {"BTC/USDT": parse_utc("2019-09-10"), "ETH/USDT": parse_utc("2019-11-27")}
    funding = {}
    for coin, f0 in fstart.items():
        times = np.arange(f0, parse_utc(end), 8 * 3_600_000, dtype="int64")
        funding[coin] = (times, np.full(len(times), 0.0001))
    spot = {c: SymbolRules(qty_step=1e-5, min_notional=5.0) for c in T8.COINS}
    fut = {"BTC/USDT": SymbolRules(qty_step=0.001, min_notional=100.0),
           "ETH/USDT": SymbolRules(qty_step=0.001, min_notional=20.0)}
    return T8.Data8(T8.align(prices), funding, fstart, spot, fut)


def test_short_side_only_after_perpetual_exists():
    data = synthetic_data()
    w = T8.targets_for("T2", 0.25, data)
    early = data.ts < data.funding_start["BTC/USDT"]
    assert (w["BTC/USDT"][early] >= 0).all() and (w["BTC/USDT"][~early] < 0).any()
    t1 = T8.targets_for("T1", 0.25, data)
    assert (t1["ETH/USDT"] >= 0).all() and (np.abs(t1["BTC/USDT"]) + np.abs(t1["ETH/USDT"]) <= 1 + 1e-9).all()


def test_study_dry_run_on_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.setattr(T8, "MC_PATHS", 200)
    data = synthetic_data()
    s = T8.run_study(data)
    assert set(s.checks) == set(T8.CANDIDATES)
    assert "5_short_degeri" in s.checks["T2-25"] and "6_56usdt" in s.checks["T1-15"]
    assert s.chosen_small in (None, "T1-25", "T1-15") and s.chosen_large in (None, "T2-25", "T2-15")
    text = T8.format_study(s)
    assert "Ölçütler" in text and "Al-tut" in text and "50" in text
    chart = T8.save_report(tmp_path / "rapor", s, text)
    assert chart.exists() and (tmp_path / "rapor" / "ozet.json").exists()
    assert all(len(s.runs[(c, "TAM")].equity) > 2500 for c in T8.CANDIDATES)
    assert len(LOOKBACKS) == 9


def test_orders_fill_at_next_day_open_not_close():
    df = frame([100.0] * 5)
    df["open"] = [100.0, 104.0, 108.0, 112.0, 116.0]
    df["close"] = [102.0, 106.0, 110.0, 114.0, 118.0]
    res = run_target_weights({"A": df}, {"A": np.full(5, 0.5)}, D0 + DAY_MS, D0 + 5 * DAY_MS, 1000.0, SPOT_COSTS,
                             futures=False)
    first = res.trades[0]
    assert first["ts"] == D0 + DAY_MS and first["price"] == pytest.approx(104.0 * (1 + SPOT_COSTS.slippage))
    assert first["qty"] == pytest.approx(0.5 * 1000.0 / 102.0)          # karar: önceki kapanış fiyatıyla
