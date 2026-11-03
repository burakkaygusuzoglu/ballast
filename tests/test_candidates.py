"""Aday stratejiler (bot/candidates.py): mum birleştirme, geleceği görmeme ve kural anlamı."""
import numpy as np
import pandas as pd
import pytest

from bot.backtester import iter_signal_bars, run_backtest
from bot.candidates import (
    BreakoutParams,
    MeanRevParams,
    breakout_signals,
    meanrev_signals,
    resample_candles,
)
from bot.timeutil import DAY_MS, parse_utc
from helpers import H1, M15, loose_rules, make_candles, resample, with_overrides

H4 = 4 * H1
RULES = loose_rules(["BTC/USDT"])


def candles_from_closes(closes, half_range: float, start: str = "2024-01-01", tf_ms: int = H1,
                        relative: bool = False) -> pd.DataFrame:
    """Elle kurulmuş seri: açılış = önceki kapanış, en yüksek/en düşük = kapanış ± aralık."""
    close = np.asarray(closes, dtype="float64")
    spread = close * half_range if relative else np.full(len(close), half_range)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "ts": parse_utc(start) + np.arange(len(close), dtype="int64") * tf_ms,
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": 1.0,
    })


# ---------------------------------------------------------------------------
# Mum birleştirme
# ---------------------------------------------------------------------------
def test_resample_matches_reference_on_complete_data():
    m15 = make_candles(2000, seed=21)                       # 500 tam saat
    pd.testing.assert_frame_equal(resample_candles(m15, M15, H1), resample(m15, H1))


def test_resample_aligns_to_utc_and_drops_unfinished_last_period():
    h1 = make_candles(24 * 10 + 6, seed=22, tf_ms=H1)       # 10 gün + 6 saat
    h4 = resample_candles(h1, H1, H4)
    assert (h4["ts"] % H4 == 0).all()
    assert len(h4) == 10 * 6 + 1                            # 06:00'da biten veri: 04-08 dilimi eksik, atılır
    daily = resample_candles(h1, H1, DAY_MS)
    assert (daily["ts"] % DAY_MS == 0).all() and len(daily) == 10
    first = h1.iloc[:24]
    row = daily.iloc[0]
    assert row["open"] == first["open"].iloc[0] and row["close"] == first["close"].iloc[-1]
    assert row["high"] == first["high"].max() and row["low"] == first["low"].min()
    assert row["volume"] == pytest.approx(first["volume"].sum())


def test_resample_builds_period_from_available_bars_during_outage():
    h1 = make_candles(48, seed=23, tf_ms=H1)
    outage = h1.drop(index=[5, 6, 7]).reset_index(drop=True)     # borsa kesintisi: 05:00-08:00 yok
    h4 = resample_candles(outage, H1, H4)
    second = h1.iloc[[4]]                                        # 04-08 diliminde yalnızca 04:00 mumu var
    row = h4.iloc[1]
    assert row["ts"] == h1["ts"].iloc[4]
    assert row["open"] == second["open"].iloc[0] and row["close"] == second["close"].iloc[0]
    assert len(h4) == 12


def test_resample_rejects_invalid_ratio():
    h1 = make_candles(50, seed=24, tf_ms=H1)
    with pytest.raises(ValueError):
        resample_candles(h1, H1, 90 * 60_000)
    with pytest.raises(ValueError):
        resample_candles(h1, H1, M15)


# ---------------------------------------------------------------------------
# Geleceği görmeme: t anındaki sinyal, t sonrası veri yokken hesaplananla aynı olmalı
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "builder, params, columns",
    [
        (breakout_signals, BreakoutParams(), ["atr", "upper", "lower", "cost_r", "entry_signal", "exit_signal"]),
        (meanrev_signals, MeanRevParams(), ["atr", "rsi", "sma_trend", "sma_exit", "cost_r",
                                            "entry_signal", "exit_signal"]),
    ],
)
def test_candidate_signals_have_no_lookahead(cfg, market_15m, builder, params, columns):
    full = builder(market_15m, params, cfg).set_index("ts")
    assert full["entry_signal"].sum() > 0 and full["exit_signal"].sum() > 0, "test anlamlı olsun diye sinyal olmalı"
    rng = np.random.default_rng(5)
    for cut in sorted(rng.integers(650, len(market_15m) - 1, 12)):
        cut_ts = int(market_15m["ts"].iloc[cut])
        part = builder(market_15m.iloc[: cut + 1], params, cfg).set_index("ts")
        for column in columns:
            a, b = full.loc[cut_ts, column], part.loc[cut_ts, column]
            if isinstance(a, (bool, np.bool_)):
                assert bool(a) == bool(b), f"{column} @ {cut}"
            else:
                assert np.isclose(a, b, equal_nan=True, rtol=0, atol=1e-9), f"{column} @ {cut}"


def test_candidates_never_signal_before_warmup(cfg, market_15m):
    frames = (breakout_signals(market_15m, BreakoutParams(), cfg), meanrev_signals(market_15m, MeanRevParams(), cfg))
    for frame in frames:
        assert not frame.loc[~frame["c_ready"], "entry_signal"].any()


# ---------------------------------------------------------------------------
# Kural anlamı: elle kurulmuş serilerde
# ---------------------------------------------------------------------------
def test_breakout_entry_and_exit_rules(cfg):
    closes = np.full(120, 100.0)
    d = candles_from_closes(closes, half_range=1.0)              # en yüksek 101, en düşük 99, ATR = 2
    d.loc[40, ["high", "close"]] = [101.6, 101.5]                 # kırılım ama göstergeler hazır değil
    d.loc[70, ["high", "close"]] = [101.0, 101.0]                 # kanala EŞİT kapanış: kırılım değil
    d.loc[80, ["high", "close"]] = [101.6, 101.5]                 # kırılım
    d.loc[81, ["high", "close"]] = [101.6, 101.5]                 # önceki mumun tepesi kanalı yükseltti
    d.loc[85, ["high", "close"]] = [105.0, 101.7]                 # mumun KENDİ tepesi kanala girmez
    d.loc[100, ["low", "close"]] = [98.4, 98.5]                   # önceki 10 mumun dibinin altında kapanış
    d.loc[101, ["low", "close"]] = [98.4, 98.5]                   # dip artık 98.4: çıkış sinyali yok
    s = breakout_signals(d, BreakoutParams(entry_n=20, exit_n=10, atr_n=20), cfg)
    assert s.index[s["entry_signal"]].tolist() == [80, 85]
    assert not s.loc[40, "entry_signal"] and bool(s.loc[40, "c_breakout"])
    assert s.loc[85, "upper"] == pytest.approx(101.6)
    assert s.index[s["exit_signal"]].tolist() == [100]
    assert not s["regime_ok"].eq(False).any()


def test_breakout_cost_filter_blocks_tight_ranges(cfg):
    d = candles_from_closes(np.full(120, 100.0), half_range=0.1)   # ATR 0.2: stop çok dar, maliyet > 1R'nin %25'i
    d.loc[80, ["high", "close"]] = [101.6, 101.5]
    s = breakout_signals(d, BreakoutParams(), cfg)
    assert bool(s.loc[80, "c_breakout"]) and not bool(s.loc[80, "c_cost"])
    assert s["entry_signal"].sum() == 0


def test_meanrev_buys_dips_only_in_uptrend_and_exits_above_short_average(cfg):
    up = list(100 * 1.001 ** np.arange(700))
    up += [up[-1] * 0.97, up[-1] * 0.97 ** 2, up[-1] * 0.97 ** 2 * 1.05]   # iki sert düşüş, sonra toparlanma
    s = meanrev_signals(candles_from_closes(up, half_range=0.002, relative=True), MeanRevParams(), cfg)
    assert s.loc[700, "rsi"] < 10 and s.loc[700, "close"] > s.loc[700, "sma_trend"]
    assert s.index[s["entry_signal"]].tolist() == [700, 701]
    assert bool(s.loc[702, "exit_signal"]) and not bool(s.loc[701, "exit_signal"])

    down = list(100 * 0.999 ** np.arange(700))
    down += [down[-1] * 0.97, down[-1] * 0.97 ** 2]
    s = meanrev_signals(candles_from_closes(down, half_range=0.002, relative=True), MeanRevParams(), cfg)
    assert s.loc[700, "rsi"] < 10                                  # aşırı satım var ama düşüş eğiliminde
    assert s["entry_signal"].sum() == 0


# ---------------------------------------------------------------------------
# Motor: stratejinin çıkış sinyali
# ---------------------------------------------------------------------------
def forced(candles: pd.DataFrame, entries: list[int], exits: list[int], atr: float) -> pd.DataFrame:
    df = candles.copy()
    df["atr"] = atr
    df["entry_signal"] = df.index.isin(entries)
    df["exit_signal"] = df.index.isin(exits)
    df["regime_ok"] = True
    return df


def test_exit_signal_sells_at_next_open(cfg):
    candles = make_candles(200, seed=31, vol=0.002)
    c = with_overrides(cfg, execution={"entry_order": "market"}, risk={"trail_atr_mult": 0.0},
                       strategy={"exit_on_regime_break": False})
    atr = float(candles["close"].iloc[0]) / 10                   # stop çok aşağıda: hiç tetiklenmez
    sig = forced(candles, entries=[10], exits=[5, 30], atr=atr)  # 5. mumdaki çıkış sinyali: pozisyon yok, etkisiz
    start, end = int(candles["ts"].iloc[0]), int(candles["ts"].iloc[-1]) + M15
    res = run_backtest(c, {"BTC/USDT": sig}, start, end, 10_000, RULES)
    (trade,) = res.trades
    assert trade.entry_ts == candles["ts"].iloc[11]
    assert trade.exit_ts == candles["ts"].iloc[31] and trade.exit_reason == "çıkış sinyali"
    assert trade.exit_price == pytest.approx(candles["open"].iloc[31] * (1 - cfg.costs.slippage))
    assert res.open_positions == []


def test_signal_bars_default_exit_signal_to_false():
    candles = make_candles(20, seed=32)
    df = candles.assign(atr=1.0, entry_signal=False, regime_ok=True)   # exit_signal sütunu yok
    bars = [b["X"] for _, b in iter_signal_bars({"X": df}, int(df["ts"].iloc[0]), int(df["ts"].iloc[-1]) + 1)]
    assert len(bars) == 20 and not any(b.exit_signal for b in bars)


def test_breakout_priority_is_strength_in_atr_units(cfg):
    d = candles_from_closes(np.full(120, 100.0), half_range=1.0)
    d.loc[80, ["high", "close"]] = [103.0, 102.6]
    s = breakout_signals(d, BreakoutParams(), cfg)
    assert s.loc[80, "priority"] == pytest.approx((102.6 - 101.0) / s.loc[80, "atr"])
    assert (s.loc[~s["entry_signal"], "priority"] == 0).all()


def test_engine_gives_scarce_slot_to_strongest_signal(cfg):
    candles = make_candles(60, seed=33, vol=0.002)
    c = with_overrides(cfg, execution={"entry_order": "market"},
                       risk={"max_open_positions": 1, "trail_atr_mult": 0.0},
                       strategy={"exit_on_regime_break": False})
    atr = float(candles["close"].iloc[0]) / 10
    start, end = int(candles["ts"].iloc[0]), int(candles["ts"].iloc[-1]) + M15
    rules = loose_rules(["AAA/USDT", "ZZZ/USDT"])

    def entered(priorities: dict[str, float]) -> str:
        signals = {}
        for symbol in ("AAA/USDT", "ZZZ/USDT"):
            df = forced(candles, entries=[10], exits=[], atr=atr)
            if symbol in priorities:
                df["priority"] = np.where(df["entry_signal"], priorities[symbol], 0.0)
            signals[symbol] = df
        (position,) = run_backtest(c, signals, start, end, 10_000, rules).open_positions
        return position.symbol

    assert entered({}) == "AAA/USDT"                                     # öncelik yoksa sembol adı sırası
    assert entered({"AAA/USDT": 0.5, "ZZZ/USDT": 2.0}) == "ZZZ/USDT"     # güçlü sinyal önce
    assert entered({"AAA/USDT": float("nan"), "ZZZ/USDT": 0.1}) == "ZZZ/USDT"
