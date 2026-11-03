"""3. tur adayları: BTC rejim filtresi, momentum rotasyonu ve piramitleme (docs/ARASTIRMA.md, 9. bölüm)."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from helpers import D1, H4, loose_rules, make_candles, with_overrides

from bot.backtester import iter_signal_bars
from bot.brokers.base import Bar, EntryOrder
from bot.brokers.simulated import SimulatedBroker
from bot.candidates import MomentumParams, apply_regime, btc_regime, momentum_signals
from bot.engine import TradingEngine
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.timeutil import DAY_MS, parse_utc

START = parse_utc("2024-01-01")


def daily_from_closes(closes, start=START) -> pd.DataFrame:
    close = np.asarray(closes, dtype="float64")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"ts": start + np.arange(len(close), dtype="int64") * D1, "open": open_,
                         "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * 0.99,
                         "close": close, "volume": 1.0})


def always_buy_frame(n_bars: int, tf_ms: int) -> pd.DataFrame:
    ts = START + np.arange(n_bars, dtype="int64") * tf_ms
    return pd.DataFrame({"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "atr": 0.1,
                         "c_ready": True, "entry_signal": True, "exit_signal": False, "regime_ok": True,
                         "priority": 1.0})


# ---------------------------------------------------------------------------
# BTC rejim filtresi
# ---------------------------------------------------------------------------
def test_regime_filter_uses_only_daily_bars_closed_before_each_bar():
    btc = daily_from_closes([100.0] * 230 + [130.0] * 10)       # 230. gün kapanışı ortalamanın üstüne çıkar
    regime = btc_regime(btc, 200)
    sig = apply_regime(always_buy_frame(240 * 6, H4), regime, H4)
    by_ts = sig.set_index("ts")
    day230 = START + 230 * DAY_MS
    # 230. günün mumu 231. günün 00:00'ında kapanır: rejim ancak o anda kapanan 4 saatlik mumda görülür.
    assert not by_ts.loc[day230 + 16 * 3_600_000, "entry_signal"]         # 20:00'da kapanan mum: henüz değil
    assert by_ts.loc[day230 + 20 * 3_600_000, "entry_signal"]             # 00:00'da kapanan mum: evet
    assert not by_ts.loc[START + 150 * DAY_MS, "c_ready"]                 # 200 günlük ortalama yokken hazır değil
    assert not sig.loc[~sig["c_ready"], "entry_signal"].any()


def test_regime_break_closes_positions_only_when_requested():
    btc = daily_from_closes([100.0 + i for i in range(220)] + [50.0] * 20)   # yükseliş, sonra sert düşüş
    regime = btc_regime(btc, 200)
    frame = always_buy_frame(240, D1)
    keep = apply_regime(frame, regime, D1)
    close = apply_regime(frame, regime, D1, exit_on_break=True)
    late = START + 230 * DAY_MS
    assert not keep.set_index("ts").loc[late, "exit_signal"]
    assert close.set_index("ts").loc[late, "exit_signal"] and not close.set_index("ts").loc[late, "entry_signal"]


# ---------------------------------------------------------------------------
# Momentum rotasyonu
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def daily_market():
    coins = {f"C{i}/USDT": make_candles(420, seed=60 + i, start="2023-01-01", base=10.0 * (i + 1), tf_ms=D1,
                                         vol=0.03)
             for i in range(5)}
    return coins, coins["C0/USDT"]


def test_momentum_signals_have_no_lookahead(cfg, daily_market):
    coins, btc = daily_market
    params = MomentumParams()
    full = {s: df.set_index("ts") for s, df in momentum_signals(coins, btc, params, cfg).items()}
    assert sum(int(df["entry_signal"].sum()) for df in full.values()) > 0
    assert sum(int(df["exit_signal"].sum()) for df in full.values()) > 0
    rng = np.random.default_rng(3)
    for cut in sorted(rng.integers(250, 419, 8)):
        cut_ts = int(btc["ts"].iloc[cut])
        part_coins = {s: df[df["ts"] <= cut_ts] for s, df in coins.items()}
        part = momentum_signals(part_coins, btc[btc["ts"] <= cut_ts], params, cfg)
        for s in coins:
            a, b = full[s].loc[cut_ts], part[s].set_index("ts").loc[cut_ts]
            for column in ("mom", "rank", "atr", "entry_signal", "exit_signal", "regime_ok", "c_ready"):
                if isinstance(a[column], (bool, np.bool_)):
                    assert bool(a[column]) == bool(b[column]), f"{s} {column} @ {cut}"
                else:
                    assert np.isclose(a[column], b[column], equal_nan=True, atol=1e-12), f"{s} {column} @ {cut}"


def test_momentum_buys_strongest_positive_coins_and_sells_laggards(cfg):
    n = 300
    up = lambda rate: [100.0 * (1 + rate) ** i for i in range(n)]
    coins = {
        "A/USDT": daily_from_closes(up(0.004)),
        "B/USDT": daily_from_closes(up(0.003)),
        "C/USDT": daily_from_closes(up(0.002)),
        "D/USDT": daily_from_closes(up(0.001)),
        "E/USDT": daily_from_closes(up(-0.002)),                 # düşüşte: hiç alınmaz
    }
    sig = momentum_signals(coins, coins["A/USDT"], MomentumParams(), cfg)
    last = {s: df.iloc[-1] for s, df in sig.items()}
    assert [s for s in coins if last[s]["entry_signal"]] == ["A/USDT", "B/USDT", "C/USDT"]
    assert last["D/USDT"]["rank"] == 4 and not last["D/USDT"]["exit_signal"]     # ilk 5'te: tutulur
    assert last["E/USDT"]["exit_signal"]                                          # getiri negatif: satılır


# ---------------------------------------------------------------------------
# Piramitleme
# ---------------------------------------------------------------------------
SYM = "AAA/USDT"


def trend_signals(n=60, start=100.0, step=0.3, entry_bar=5, exit_bar=40, symbol=SYM, atr=4.0) -> pd.DataFrame:
    """Düzenli yükseliş; ATR fiyatın %4'ü (gerçekçi 4 saatlik kripto oynaklığı, birimler %50 sınırına takılmaz)."""
    close = start + step * np.arange(n)
    open_ = np.concatenate([[start], close[:-1]])
    df = pd.DataFrame({"ts": START + np.arange(n, dtype="int64") * H4, "open": open_, "high": close + 0.05,
                       "low": open_ - 0.05, "close": close, "atr": atr, "entry_signal": False,
                       "exit_signal": False, "regime_ok": True, "priority": 1.0})
    df.loc[entry_bar, "entry_signal"] = True
    df.loc[exit_bar, "exit_signal"] = True
    return df


def pyramid_cfg(cfg, **risk):
    base = dict(risk_per_trade=0.005, stop_atr_mult=2.0, trail_atr_mult=0.0, max_position_pct=0.5,
                max_open_positions=5, max_trades_per_day=50, cooldown_bars=0, pyramid_units=4,
                pyramid_step_atr=0.5, max_total_units=10)
    base.update(risk)
    return with_overrides(cfg, timeframes={"entry": "4h", "trend": "1d"}, execution={"entry_order": "market"},
                          costs={"slippage": 0.0}, strategy={"exit_on_regime_break": False}, risk=base)


def run_engine(cfg_p, signals, equity=100_000.0):
    portfolio = Portfolio(equity)
    broker = SimulatedBroker(portfolio, cfg_p.costs.maker_fee, cfg_p.costs.taker_fee, cfg_p.costs.slippage)
    risk = RiskManager(cfg_p, equity)
    engine = TradingEngine(cfg_p, portfolio, broker, risk, loose_rules(list(signals)))
    history = []
    for ts, bars in iter_signal_bars(signals, 0, 2**62):
        engine.on_bar(ts, bars)
        history.append({s: dataclasses.replace(p) for s, p in portfolio.positions.items()})
    return engine, portfolio, history


def test_pyramiding_adds_units_every_half_atr_and_raises_stop(cfg):
    engine, portfolio, history = run_engine(pyramid_cfg(cfg), {SYM: trend_signals()})
    peak = max((h[SYM] for h in history if SYM in h), key=lambda p: p.units)
    assert peak.units == 4 and engine.adds_submitted == 3
    assert peak.stop_price == pytest.approx(peak.last_add_price - 2 * 4.0)    # son birimin 2 ATR altı
    first = next(h[SYM] for h in history if SYM in h)
    unit_prices = sorted({h[SYM].last_add_price for h in history if SYM in h})
    assert len(unit_prices) == 4 and min(np.diff(unit_prices)) >= 0.5 * 4.0 - 1e-9   # birimler en az 0,5 ATR arayla
    assert peak.qty > 3.5 * first.qty and peak.risk_amount > 3.5 * first.risk_amount
    (trade,) = portfolio.trades                                               # tek işlem olarak kapanır
    assert trade.qty == pytest.approx(peak.qty) and trade.exit_reason == "çıkış sinyali"
    assert trade.r_multiple == pytest.approx(trade.net_pnl / peak.risk_amount)
    assert portfolio.cash == pytest.approx(portfolio.initial_cash + trade.net_pnl)   # muhasebe tutarlı


def test_pyramiding_off_keeps_single_unit(cfg):
    engine, _portfolio, history = run_engine(pyramid_cfg(cfg, pyramid_units=1), {SYM: trend_signals()})
    assert engine.adds_submitted == 0 and max(h[SYM].units for h in history if SYM in h) == 1


def test_pyramiding_respects_total_unit_cap(cfg):
    signals = {f"S{i}/USDT": trend_signals(symbol=f"S{i}/USDT") for i in range(3)}
    _, _, history = run_engine(pyramid_cfg(cfg, max_total_units=5), signals)
    assert max(sum(p.units for p in h.values()) for h in history) == 5


def test_pending_add_is_released_when_entries_are_cancelled():
    portfolio = Portfolio(10_000.0)
    broker = SimulatedBroker(portfolio, 0.001, 0.001, 0.0)
    broker.submit_entry(EntryOrder(SYM, 1.0, None, 95.0, 5.0, 0, 1, unit_atr=2.0), ref_price=100.0)
    broker.process_bar(SYM, Bar(H4, 100.0, 101.0, 99.0, 100.5))
    cash = portfolio.cash
    broker.submit_add(EntryOrder(SYM, 1.0, None, 97.0, 5.0, H4, 1, unit_atr=2.0, is_add=True, stop_distance=4.0),
                      ref_price=101.0)
    assert portfolio.cash < cash and broker.pending_add_count() == 1
    broker.cancel_entries()                                                    # şalter / günlük limit
    assert portfolio.cash == pytest.approx(cash) and portfolio.reserved == pytest.approx(0.0)
    restored = SimulatedBroker(portfolio, 0.001, 0.001, 0.0)
    restored.load_state(broker.to_dict())
    assert restored.pending_add_count() == 0


# ---------------------------------------------------------------------------
# 3. tur çalıştırması uçtan uca (arastirma_tur3.py), sentetik veriyle
# ---------------------------------------------------------------------------
T3_COINS = ["BTC/USDT", "AAA/USDT", "BBB/USDT", "CCC/USDT"]


@pytest.fixture(scope="module")
def tur3_run(cfg, tmp_path_factory):
    import arastirma as A
    import arastirma_tur3 as T3
    exam = "2025-03-01"
    c = dataclasses.replace(with_overrides(cfg, data={"exam_start": exam}), symbols=T3_COINS)
    raw = {s: {"1h": make_candles(24 * 460, seed=80 + i, start="2024-01-01", base=50.0 * (i + 1), tf_ms=3_600_000,
                                   vol=0.008)}
           for i, s in enumerate(T3_COINS)}                                     # sınav sonrasına da uzanır
    market = A.prepare_market(raw, parse_utc(exam))
    rules = loose_rules(T3_COINS)
    study = T3.run_tur3(c, market, rules, rules)
    return T3, study, market, parse_utc(exam), tmp_path_factory.mktemp("tur3")


def test_tur3_uses_common_start_after_regime_warmup_and_no_exam_data(tur3_run):
    _T3, study, market, exam, _ = tur3_run
    start, split = study.bounds["kesif"]
    assert start >= parse_utc("2024-01-01") + 200 * DAY_MS and start % DAY_MS == 0
    assert split == parse_utc("2025-01-01") and study.bounds["dogrulama"] == (split, min(
        study.bounds["dogrulama"][1], exam))
    assert study.bounds["dogrulama"][1] <= exam
    for frames in market.values():
        for tf, df in frames.items():
            assert int(df["ts"].iloc[-1]) + {"1h": 3_600_000, "4h": H4, "1d": DAY_MS}[tf] <= exam
    for r in study.reports:
        for name, run in r.portfolio.items():
            assert all(t.entry_ts >= study.bounds[name][0] for t in run.result.trades)
            assert all(t.exit_ts < study.bounds[name][1] for t in run.result.trades)


def test_tur3_applies_protocol_and_reports(tur3_run):
    T3, study, _, _, out = tur3_run
    assert [r.candidate.code for r in study.reports] == ["R0", "R1", "R2", "R3", "R4"]
    ref = study.reports[0]
    assert ref.candidate.reference and not ref.passed and ref.checks["iyilesme"][0] is None
    for r in study.reports[1:]:
        assert set(r.checks) == {"kesif", "dogrulama", "iyilesme", "saglamlik", "islem"}
        assert len(r.robustness) == len(T3.robustness_variants(r.candidate)) > 0
    passed = [r.candidate.code for r in study.reports if r.passed]
    assert set(study.shadow) <= set(passed) and len(study.shadow) <= T3.MAX_SHADOW
    assert all(set(r.ladder) == set(T3.RISK_LADDER) for r in study.reports if r.passed or r.candidate.reference)
    assert not study.reports[3].coins and study.reports[1].coins          # momentum: coin bazında koşu yok
    text = T3.format_tur3(study)
    assert "Ölçütler" in text and "Risk merdiveni" in text and "Şaltersiz tanı" in text
    T3.save_tur3(out, study, text, plot=False)
    summary = json.loads((out / "ozet.json").read_text(encoding="utf-8"))
    assert [c["code"] for c in summary["candidates"]] == ["R0", "R1", "R2", "R3", "R4"]
    assert (out / "sonuc.md").exists() and (out / "islemler.csv").exists()


def test_tur3_improvement_check_compares_with_reference(tur3_run):
    _T3, study, _, _, _ = tur3_run
    ref = study.reports[0]
    for r in study.reports[1:]:
        ok, _ = r.checks["iyilesme"]
        expected = all(r.portfolio[n].metrics["return_to_drawdown"] > ref.portfolio[n].metrics["return_to_drawdown"]
                       for n in ("kesif", "dogrulama"))
        assert ok == expected


# ---------------------------------------------------------------------------
# Gölge botlar: paper trading, backtest ile birebir aynı kararları vermeli
# ---------------------------------------------------------------------------
from helpers import FakeSource  # noqa: E402

from bot.backtester import run_backtest  # noqa: E402
from bot.candidates import resample_candles  # noqa: E402
from bot.paper import PaperTrader  # noqa: E402
from bot.signals import build_market_signals, uses_trend  # noqa: E402
from bot.storage import StateStore  # noqa: E402

SHADOW = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


def shadow_cfg(cfg, **strategy):
    c = with_overrides(
        cfg, timeframes={"entry": "4h", "trend": "1d"}, execution={"entry_order": "market"},
        risk={"risk_per_trade": 0.005, "stop_atr_mult": 2.0, "trail_atr_mult": 0.0, "max_open_positions": 5,
              "max_trades_per_day": 50},
        strategy={"type": "breakout", "exit_on_regime_break": False, **strategy},
        paper={"history_bars": 5000, "initial_equity": 10_000.0, "name": "golge_test"},
    )
    return dataclasses.replace(c, symbols=SHADOW)


def run_shadow_paper(c, source, first_close: int, last_close: int, step: int, tmp_path):
    trader = PaperTrader(c, source, StateStore(tmp_path / "g.db"), loose_rules(SHADOW))
    moment = first_close
    while moment <= last_close:
        trader.sync(moment + 5_000)
        moment += step
    return trader


def test_regime_shadow_paper_matches_backtest(cfg, tmp_path):
    h4 = {s: make_candles(6 * 260, seed=120 + i, start="2023-01-01", base=30.0 * (i + 1), tf_ms=H4, vol=0.012)
          for i, s in enumerate(SHADOW)}
    d1 = {s: resample_candles(df, H4, D1) for s, df in h4.items()}
    c = shadow_cfg(cfg, regime_sma=30)
    assert uses_trend(c)
    source = FakeSource({**{(s, "4h"): df for s, df in h4.items()}, **{(s, "1d"): df for s, df in d1.items()}})
    ts = h4["BTC/USDT"]["ts"]
    start = int(ts.iloc[6 * 120])                     # rejim ve kanal ısındıktan sonra
    trader = run_shadow_paper(c, source, start, int(ts.iloc[-1]), 2 * H4, tmp_path)
    signals = build_market_signals(c, h4, d1)
    bt = run_backtest(c, signals, start, trader.last_ts + H4, 10_000.0, loose_rules(SHADOW))   # ilk işlenen mum: start
    assert len(bt.trades) >= 3, "karşılaştırma anlamlı olsun diye işlem olmalı"
    assert trader.portfolio.trades == bt.trades
    assert sum(df["entry_signal"].sum() for df in signals.values()) < sum(
        df["c_breakout"].sum() for df in signals.values())            # filtre bazı kırılımları gerçekten eledi


def test_momentum_shadow_paper_matches_backtest(cfg, tmp_path):
    d1 = {s: make_candles(320, seed=140 + i, start="2023-01-01", base=20.0 * (i + 1), tf_ms=D1, vol=0.03)
          for i, s in enumerate(SHADOW)}
    c = shadow_cfg(cfg, type="momentum", regime_sma=60, momentum_lookback=28, momentum_top_n=2,
                   momentum_exit_rank=3)
    c = with_overrides(c, timeframes={"entry": "1d", "trend": "1d"})
    assert not uses_trend(c)
    source = FakeSource({(s, "1d"): df for s, df in d1.items()})
    ts = d1["BTC/USDT"]["ts"]
    start = int(ts.iloc[100])
    trader = run_shadow_paper(c, source, start, int(ts.iloc[-1]), D1, tmp_path)
    bt = run_backtest(c, build_market_signals(c, d1), start, trader.last_ts + D1, 10_000.0, loose_rules(SHADOW))
    assert len(bt.trades) >= 3, "karşılaştırma anlamlı olsun diye işlem olmalı"
    assert trader.portfolio.trades == bt.trades


def test_shadow_configs_load_and_match_research_rules():
    from pathlib import Path

    from bot.config import load_config
    from bot.signals import momentum_params
    root = Path(__file__).resolve().parent.parent / "config"
    r1, r3 = load_config(root / "golge_r1.yaml"), load_config(root / "golge_r3.yaml")
    base = load_config(root / "config.yaml")
    assert (r1.paper.name, r3.paper.name, base.paper.name) == ("golge_r1", "golge_r3", "paper")
    assert r1.strategy.regime_sma == 200 and r1.timeframes.entry == "4h" and r1.paper.initial_equity == 50.0
    assert dataclasses.replace(r1.strategy, regime_sma=0) == base.strategy and r1.risk == base.risk  # yalnızca filtre farklı
    assert momentum_params(r3) == MomentumParams()                      # 3. turdaki R3 ile aynı
    assert r3.timeframes.entry == "1d" and r3.paper.initial_equity == 1000 and r3.risk == base.risk
    r3a = load_config(root / "golge_r3a.yaml")
    assert r3a.paper.name == "golge_r3a" and r3a.paper.initial_equity == 50.0 and r3a.strategy == r3.strategy
    assert (r3a.risk.risk_per_trade, r3a.risk.max_position_pct) == (0.02, 1.0)        # 12. bölümdeki ayar
    assert dataclasses.replace(r3a.risk, risk_per_trade=0.005, max_position_pct=0.5, daily_loss_limit=0.03,
                               max_drawdown_halt=0.20) == base.risk
