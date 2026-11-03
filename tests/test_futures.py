"""Vadeli işlem modeli (6. tur): izole marjin muhasebesi, short, funding, likidasyon, sinyaller, kayıt şeması."""
import dataclasses
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
import yaml
from helpers import D1, H4, loose_rules, make_candles

import arastirma as A
from bot.backtester import run_backtest
from bot.brokers.base import Bar, EntryOrder
from bot.brokers.futures import (
    FUNDING_MS,
    SimulatedFuturesBroker,
    make_simulated_broker,
)
from bot.brokers.simulated import InsufficientCash, SimulatedBroker
from bot.candidates import (
    BreakoutParams,
    add_short_side,
    apply_regime_two_sided,
    breakout_signals,
    btc_regime,
)
from bot.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    FuturesConfig,
    MarketRules,
    load_config,
)
from bot.portfolio import Portfolio, Trade
from bot.risk import RiskManager, liquidation_distance, liquidation_price
from bot.storage import StateStore, read_trades

TAKER, SLIP = 0.0005, 0.0005
SYM = "BTC/USDT"
T0 = 1_704_067_200_000            # 2024-01-01 00:00 UTC (bir funding anı)


def fc(**kw) -> FuturesConfig:
    return dataclasses.replace(FuturesConfig(), **kw)


def make_broker(cash=10_000.0, **kw) -> tuple[Portfolio, SimulatedFuturesBroker]:
    portfolio = Portfolio(cash)
    return portfolio, SimulatedFuturesBroker(portfolio, 0.0002, TAKER, SLIP, fc(**kw))


def market(qty=1.0, stop=95.0, side=1) -> EntryOrder:
    return EntryOrder(SYM, qty, None, stop, risk_amount=5.0, placed_ts=T0 - H4, ttl_bars=1, side=side)


def bar(ts, o, h, low, c) -> Bar:
    return Bar(ts, o, h, low, c)


# ---- giriş, çıkış ve muhasebe ----------------------------------------------------------------------------
def test_long_entry_moves_margin_not_notional():
    portfolio, broker = make_broker()
    broker.submit_entry(market(), ref_price=100.0)
    (fill,) = broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.5))
    price = 100.0 * (1 + SLIP)
    assert fill.side == "buy" and fill.entry is True and fill.is_entry
    assert fill.price == pytest.approx(price) and fill.fee == pytest.approx(price * TAKER)
    p = portfolio.positions[SYM]
    assert p.margin == pytest.approx(price / 3) and p.leverage == 3.0 and p.side == 1
    assert portfolio.cash == pytest.approx(10_000.0 - price / 3 - price * TAKER)
    # Özsermaye = nakit + marjin + gerçekleşmemiş kâr; giriş fiyatında yalnızca komisyon kadar eksik.
    assert portfolio.equity({SYM: price}) == pytest.approx(10_000.0 - price * TAKER)
    assert portfolio.equity({SYM: price + 10}) == pytest.approx(10_000.0 - price * TAKER + 10)


def test_short_profits_when_price_falls_and_pays_slippage_both_ways():
    portfolio, broker = make_broker()
    broker.submit_entry(market(stop=105.0, side=-1), ref_price=100.0)
    (entry,) = broker.process_bar(SYM, bar(T0, 100.0, 100.5, 99.0, 99.5))
    assert entry.side == "sell" and entry.entry is True
    assert entry.price == pytest.approx(100.0 * (1 - SLIP))          # short ucuza satar
    assert portfolio.equity({SYM: 90.0}) > portfolio.equity({SYM: 100.0})
    broker.request_exit(SYM, "çıkış sinyali")
    (exit_,) = broker.process_bar(SYM, bar(T0 + H4, 90.0, 91.0, 89.0, 90.5))
    assert exit_.side == "buy" and exit_.entry is False and not exit_.is_entry
    assert exit_.price == pytest.approx(90.0 * (1 + SLIP))           # pahalıya geri alır
    (t,) = portfolio.trades
    gross = entry.price - exit_.price
    assert t.side == -1 and t.gross_pnl == pytest.approx(gross)
    assert t.net_pnl == pytest.approx(gross - entry.fee - exit_.fee)
    assert portfolio.cash == pytest.approx(10_000.0 + t.net_pnl) and portfolio.reserved == 0


def test_short_stop_triggers_on_high_and_gap_fills_at_open():
    _portfolio, broker = make_broker()
    broker.submit_entry(market(stop=105.0, side=-1), ref_price=100.0)
    broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))
    (fill,) = broker.process_bar(SYM, bar(T0 + H4, 100.0, 105.0, 99.0, 104.0))     # tepe stopa değdi
    assert fill.reason == "stop" and fill.price == pytest.approx(105.0 * (1 + SLIP))
    broker.submit_entry(market(stop=105.0, side=-1), ref_price=100.0)
    broker.process_bar(SYM, bar(T0 + 2 * H4, 100.0, 101.0, 99.0, 100.0))
    (gap,) = broker.process_bar(SYM, bar(T0 + 3 * H4, 108.0, 109.0, 107.0, 108.0))   # stopun üstünde açıldı
    assert gap.price == pytest.approx(108.0 * (1 + SLIP))


def test_stop_only_moves_in_profit_direction():
    portfolio, broker = make_broker()
    broker.submit_entry(market(stop=105.0, side=-1), ref_price=100.0)
    broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))
    assert broker.update_stop(SYM, 104.0) and portfolio.positions[SYM].stop_price == 104.0
    assert not broker.update_stop(SYM, 106.0) and portfolio.positions[SYM].stop_price == 104.0


def test_margin_limits_order_size_and_limit_orders_are_rejected():
    _portfolio, broker = make_broker(cash=100.0)
    with pytest.raises(InsufficientCash):
        broker.submit_entry(market(qty=4.0), ref_price=100.0)       # 400 USDT pozisyon, 133 USDT marjin
    broker.submit_entry(market(qty=2.5), ref_price=100.0)           # 250 USDT pozisyon, ~84 USDT marjin: olur
    with pytest.raises(ValueError, match="piyasa"):
        broker.submit_entry(dataclasses.replace(market(), symbol="ETH/USDT", limit_price=100.0), ref_price=100.0)


# ---- funding ---------------------------------------------------------------------------------------------
def test_constant_funding_long_pays_short_receives_at_8h_marks():
    for side, expected_sign in ((1, 1), (-1, -1)):
        portfolio, broker = make_broker()
        stop = 95.0 if side > 0 else 105.0
        broker.submit_entry(market(stop=stop, side=side), ref_price=100.0)
        broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))          # 00:00: giriş, funding yok
        broker.process_bar(SYM, bar(T0 + H4, 100.0, 101.0, 99.0, 100.0))     # 04:00: funding anı değil
        assert portfolio.positions[SYM].funding_paid == 0
        broker.process_bar(SYM, bar(T0 + 2 * H4, 100.0, 101.0, 99.0, 100.0))  # 08:00: bir funding
        p = portfolio.positions[SYM]
        assert p.funding_paid == pytest.approx(expected_sign * 0.0001 * p.qty * 100.0)
        assert portfolio.total_funding == pytest.approx(p.funding_paid)
        broker.request_exit(SYM, "çıkış")
        broker.process_bar(SYM, bar(T0 + 4 * H4, 100.0, 101.0, 99.0, 100.0))  # 16:00: çıkıştan önce bir daha
        (t,) = portfolio.trades
        assert t.funding == pytest.approx(2 * expected_sign * 0.0001 * t.qty * 100.0)
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.entry_fee - t.exit_fee - t.funding)


def test_daily_bars_count_three_funding_events():
    portfolio, broker = make_broker()
    broker.submit_entry(market(), ref_price=100.0)
    broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))
    broker.process_bar(SYM, bar(T0 + D1, 100.0, 101.0, 99.0, 100.0))
    assert portfolio.positions[SYM].funding_paid == pytest.approx(3 * 0.0001 * portfolio.positions[SYM].qty * 100)


def test_historical_funding_uses_real_rates_by_side():
    times = np.array([T0, T0 + FUNDING_MS, T0 + 2 * FUNDING_MS], dtype="int64")
    rates = np.array([0.0005, -0.0002, 0.0003])
    for side in (1, -1):
        portfolio = Portfolio(10_000.0)
        broker = SimulatedFuturesBroker(portfolio, 0.0002, TAKER, SLIP, fc(), {SYM: (times, rates)})
        broker.submit_entry(market(stop=95.0 if side > 0 else 105.0, side=side), ref_price=100.0)
        broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))             # T0 anı girişten önce
        broker.process_bar(SYM, bar(T0 + 4 * H4, 100.0, 101.0, 99.0, 100.0))    # +8 ve +16 saat
        p = portfolio.positions[SYM]
        assert p.funding_paid == pytest.approx(side * (-0.0002 + 0.0003) * p.qty * 100.0)


# ---- likidasyon --------------------------------------------------------------------------------------------
def test_liquidation_price_formula():
    assert liquidation_price(100.0, 1, 3.0, 0.02) == pytest.approx(100 * (2 / 3) / 0.98)
    assert liquidation_price(100.0, -1, 3.0, 0.02) == pytest.approx(100 * (4 / 3) / 1.02)
    assert liquidation_distance(fc(), 1) == pytest.approx(1 - (2 / 3) / 0.98)
    assert liquidation_price(100.0, 1, 1.0, 0.02) == 0.0                    # 1x long likide olmaz


def test_gap_beyond_liquidation_loses_whole_margin():
    portfolio, broker = make_broker(liquidation="open")
    broker.submit_entry(market(stop=95.0), ref_price=100.0)
    broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))
    p = portfolio.positions[SYM]
    margin, cash_before = p.margin, portfolio.cash
    (fill,) = broker.process_bar(SYM, bar(T0 + H4, 60.0, 61.0, 55.0, 58.0))   # likidasyonun altında açıldı
    assert fill.reason == "likidasyon" and broker.liquidations == 1
    (t,) = portfolio.trades
    assert t.gross_pnl == pytest.approx(-margin) and t.exit_fee == 0
    assert t.net_pnl == pytest.approx(-margin - t.entry_fee)
    assert portfolio.cash == pytest.approx(cash_before)                      # nakdin geri kalanı korunur
    assert portfolio.equity({}) == pytest.approx(10_000.0 + t.net_pnl)


def test_touch_model_liquidates_but_open_model_uses_the_stop():
    crash = bar(T0 + H4, 100.0, 100.5, 60.0, 99.0)                           # fitil likidasyona değdi
    for model, reason in (("open", "stop"), ("touch", "likidasyon")):
        _portfolio, broker = make_broker(liquidation=model)
        broker.submit_entry(market(stop=95.0), ref_price=100.0)
        broker.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))
        (fill,) = broker.process_bar(SYM, crash)
        assert fill.reason == reason


def test_state_roundtrip_keeps_side_margin_and_liquidations():
    portfolio, broker = make_broker()
    broker.liquidations = 2
    broker.submit_entry(market(stop=105.0, side=-1), ref_price=100.0)
    snapshot = {"portfolio": portfolio.to_dict(), "broker": broker.to_dict()}
    p2 = Portfolio.from_dict(snapshot["portfolio"])
    b2 = SimulatedFuturesBroker(p2, 0.0002, TAKER, SLIP, fc())
    b2.load_state(snapshot["broker"])
    assert b2.liquidations == 2 and b2.pending_entries[SYM].side == -1
    b2.process_bar(SYM, bar(T0, 100.0, 101.0, 99.0, 100.0))
    p3 = Portfolio.from_dict(p2.to_dict())
    assert p3.positions[SYM].side == -1 and p3.positions[SYM].margin == pytest.approx(p2.positions[SYM].margin)


# ---- risk yöneticisi -----------------------------------------------------------------------------------------
def futures_cfg(**risk):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    return A.apply_overrides(cfg, {"exchange": {"market_type": "futures"},
                                   "costs": {"maker_fee": 0.0002, "taker_fee": TAKER},
                                   "risk": {"max_position_pct": 1.0, **risk}})


def test_short_sizing_and_spot_refuses_short():
    rules = MarketRules(qty_step=1e-6, min_notional=5.0)
    spot = RiskManager(load_config(DEFAULT_CONFIG_PATH), 10_000.0)
    assert spot.size(10_000.0, 10_000.0, 100.0, 105.0, rules, side=-1) == (0.0, 0.0)
    fut = RiskManager(futures_cfg(), 10_000.0)
    assert fut.initial_stop(100.0, 2.0, side=-1) == pytest.approx(104.0)
    qty, risk = fut.size(10_000.0, 10_000.0, 100.0, 104.0, rules, side=-1)
    assert risk == pytest.approx(50.0, rel=1e-3) and qty > 0                 # %0,5 risk
    assert fut.size(10_000.0, 10_000.0, 100.0, 96.0, rules, side=-1) == (0.0, 0.0)   # stop yanlış tarafta


def test_futures_sizing_uses_margin_and_rejects_stop_near_liquidation():
    rules = MarketRules(qty_step=1e-6, min_notional=5.0)
    fut = RiskManager(futures_cfg(risk_per_trade=0.05), 10_000.0)
    qty, _ = fut.size(10_000.0, 1_000.0, 100.0, 99.0, rules)                # nakit az: marjinle sınırlı
    cash_rate = 1 / 3 + TAKER + SLIP
    assert qty * 100.0 == pytest.approx(1_000.0 / cash_rate * 0.98, rel=1e-6)
    assert qty * 100.0 > 1_000.0                                             # kaldıraç: nakitten büyük pozisyon
    assert fut.size(10_000.0, 10_000.0, 100.0, 70.0, rules) == (0.0, 0.0)  # stop %30: likidasyona çok yakın


# ---- sinyaller -----------------------------------------------------------------------------------------------
def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Fiyatın tersi: 1/p. Kırılım aşağı kırılıma dönüşür (tepe <-> dip)."""
    return pd.DataFrame({"ts": df["ts"], "open": 1 / df["open"], "high": 1 / df["low"], "low": 1 / df["high"],
                         "close": 1 / df["close"], "volume": df["volume"]})


def test_short_side_is_exact_mirror_of_long_channel():
    cfg = futures_cfg()
    df = make_candles(1500, seed=3, tf_ms=H4, vol=0.02)
    p = BreakoutParams()
    short = add_short_side(breakout_signals(df, p, cfg), p)
    long_of_mirror = breakout_signals(mirror(df), p, cfg)
    assert (short["c_breakdown"].to_numpy() == long_of_mirror["c_breakout"].to_numpy()).all()
    assert (short["short_exit_signal"].to_numpy() == long_of_mirror["exit_signal"].to_numpy()).all()
    assert short["short_signal"].any() and short["entry_signal"].any()
    assert not (short["short_signal"] & short["entry_signal"]).any()
    assert (short.loc[short["short_signal"], "priority"] > 0).all()


def test_two_sided_regime_longs_only_in_bull_shorts_only_in_bear():
    cfg = futures_cfg()
    df = make_candles(3000, seed=5, tf_ms=H4, vol=0.02)
    daily = df.groupby(df["ts"] // D1 * D1).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                                close=("close", "last"), volume=("volume", "sum"))
    daily = daily.reset_index(names="ts")
    p = BreakoutParams()
    sig = apply_regime_two_sided(add_short_side(breakout_signals(df, p, cfg), p), btc_regime(daily, 50), H4)
    assert sig["entry_signal"].any() and sig["short_signal"].any()
    assert not (sig["entry_signal"] & ~sig["c_regime"]).any()
    assert not (sig["short_signal"] & ~sig["c_short_regime"]).any()
    assert not (sig["c_regime"] & sig["c_short_regime"]).any()


# ---- motor ve backtest -----------------------------------------------------------------------------------------
SYMBOLS = ("AAA/USDT", "BBB/USDT", "CCC/USDT")


def synthetic(seed0=11) -> dict[str, pd.DataFrame]:
    return {s: make_candles(2500, seed=seed0 + i, tf_ms=H4, vol=0.02, start="2023-01-01")
            for i, s in enumerate(SYMBOLS)}


def test_one_x_futures_without_funding_equals_spot():
    """Kaldıraç 1, funding 0, aynı komisyon, yalnızca long: vadeli muhasebe spotla aynı sonucu vermeli."""
    spot = load_config(DEFAULT_CONFIG_PATH)
    fut = A.apply_overrides(spot, {"exchange": {"market_type": "futures"},
                                   "futures": {"leverage": 1.0, "funding_long_8h": 0.0, "funding_short_8h": 0.0}})
    candles = synthetic()
    p = BreakoutParams()
    sig = {s: breakout_signals(df, p, spot) for s, df in candles.items()}
    start, end = A.trade_window(sig, spot.entry_ms)
    rules = loose_rules(SYMBOLS)
    a = run_backtest(spot, sig, start, end, 10_000.0, rules)
    b = run_backtest(fut, sig, start, end, 10_000.0, rules)
    assert len(a.trades) > 20
    assert [(t.symbol, t.entry_ts, t.exit_ts) for t in a.trades] == [(t.symbol, t.entry_ts, t.exit_ts) for t in b.trades]
    assert np.allclose(a.equity.to_numpy(), b.equity.to_numpy(), rtol=1e-9, atol=1e-6)
    assert b.liquidations == 0 and b.total_funding == 0


def test_long_short_futures_backtest_accounting_balances():
    cfg = A.apply_overrides(futures_cfg(), {"strategy": {"allow_short": True},
                                            "futures": {"liquidation": "touch", "funding_long_8h": 0.0003,
                                                        "funding_short_8h": 0.0001}})
    candles = synthetic(seed0=21)
    p = BreakoutParams()
    sig = {s: add_short_side(breakout_signals(df, p, cfg), p) for s, df in candles.items()}
    start, end = A.trade_window(sig, cfg.entry_ms)
    market_ = {s: {"4h": df} for s, df in candles.items()}
    run = A.run_period(cfg, sig, market_, "4h", start, end, loose_rules(SYMBOLS))   # muhasebe öz-denetimi dahil
    sides = {t.side for t in run.result.trades}
    assert sides == {1, -1}
    assert run.result.total_funding > 0
    closed_funding = sum(t.funding for t in run.result.trades)
    open_funding = sum(p.funding_paid for p in run.result.open_positions)
    assert closed_funding + open_funding == pytest.approx(run.result.total_funding)
    assert sum(run.symbol_net.values()) == pytest.approx(run.metrics["net_profit"])


def test_factory_picks_broker_by_market_type():
    spot = load_config(DEFAULT_CONFIG_PATH)
    assert type(make_simulated_broker(spot, Portfolio(100.0))) is SimulatedBroker
    assert isinstance(make_simulated_broker(futures_cfg(), Portfolio(100.0)), SimulatedFuturesBroker)
    with pytest.raises(ValueError):
        make_simulated_broker(spot, Portfolio(100.0), {SYM: (np.array([1]), np.array([0.1]))})


# ---- ayarlar -------------------------------------------------------------------------------------------------------
def write(tmp_path, mutate) -> str:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_futures_config_loads_and_is_validated(tmp_path):
    def ok(d):
        d["exchange"]["market_type"] = "futures"
        d["strategy"]["allow_short"] = True
        d["futures"] = {"leverage": 3.0, "liquidation": "touch"}
    cfg = load_config(write(tmp_path, ok))
    assert cfg.is_futures and cfg.futures.leverage == 3.0 and cfg.futures.liquidation == "touch"
    with pytest.raises(ConfigError, match="allow_short"):
        load_config(write(tmp_path, lambda d: d["strategy"].update(allow_short=True)))       # spotta short yok
    with pytest.raises(ConfigError, match="leverage"):
        load_config(write(tmp_path, lambda d: (d["exchange"].update(market_type="futures"),
                                               d.update(futures={"leverage": 20.0}))))
    with pytest.raises(ConfigError, match="market_type"):
        load_config(write(tmp_path, lambda d: d["exchange"].update(market_type="margin")))


# ---- kayıt şeması (eski veritabanları) -------------------------------------------------------------------------------
OLD_COLUMNS = [f.name for f in dataclasses.fields(Trade) if f.name not in ("side", "funding")]


def old_db(path) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(f"CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(OLD_COLUMNS)})")
        conn.execute("CREATE TABLE equity (ts INTEGER PRIMARY KEY, equity REAL NOT NULL)")
        row = ("BTC/USDT", 1, 2, 0.1, 100.0, 110.0, 0.01, 0.011, 0.0, 1.0, 0.98, 0.5, 1.96, "stop")
        conn.execute(f"INSERT INTO trades({', '.join(OLD_COLUMNS)}) VALUES ({', '.join('?' * len(row))})", row)


def test_old_database_is_read_without_migration_and_migrated_on_open(tmp_path):
    path = tmp_path / "eski.db"
    old_db(path)
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:                    # izleme paneli gibi salt okunur
        (t,) = read_trades(conn)
    assert t.side == 1 and t.funding == 0.0 and t.net_pnl == 0.98
    store = StateStore(path)                                                  # bot açılınca sütunlar eklenir
    (t2,) = store.load_trades()
    assert t2 == t
    store.save({"x": 1}, [dataclasses.replace(t, side=-1, funding=0.25)], [])
    trades = store.load_trades()
    assert [x.side for x in trades] == [1, -1] and trades[1].funding == 0.25
