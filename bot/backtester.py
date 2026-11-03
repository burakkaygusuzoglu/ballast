"""Backtest: önceden hesaplanmış sinyalleri zaman sırasıyla işlem motoruna besler.

Sinyaller tüm geçmiş için vektörel hesaplanır; bu güvenlidir çünkü her gösterge
nedenseldir (testlerle doğrulanır). Motor ve dolum modeli paper trading ile aynıdır.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.brokers.futures import FundingHistory, make_simulated_broker
from bot.config import Config, MarketRules
from bot.engine import (
    SIGNAL_FIELDS,
    SignalBar,
    TradingEngine,
)
from bot.portfolio import Portfolio, Position, Trade
from bot.risk import RiskManager


@dataclass
class BacktestResult:
    equity: pd.Series                 # indeks: mum kapanış zamanı (UTC ms)
    trades: list[Trade]
    open_positions: list[Position]
    total_fees: float
    skipped: Counter
    signals_seen: int
    orders_submitted: int
    entries_cancelled: int
    exposure: float                   # pozisyonda geçen mum oranı
    halted_at: int | None
    halt_reason: str
    initial_equity: float
    start_ts: int
    end_ts: int
    adds_submitted: int = 0           # piramitleme: eklenen birim emri sayısı
    liquidations: int = 0             # vadeli işlem: likidasyon sayısı
    total_funding: float = 0.0        # vadeli işlem: ödenen funding toplamı (negatif = alınan)


def iter_signal_bars(signals: dict[str, pd.DataFrame], start_ts: int, end_ts: int) -> Iterator[tuple[int, dict[str, SignalBar]]]:
    """[start_ts, end_ts) aralığındaki mumları zaman sırasıyla verir: (zaman, {sembol: mum}).

    Backtest ve paper trading mumları bu TEK fonksiyonla üretir; ikisi aynı girdiyi görür.
    """
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for symbol, df in signals.items():
        window = df[(df["ts"] >= start_ts) & (df["ts"] < end_ts)]
        # Çıkış sinyali ya da öncelik üretmeyen stratejilerde bu sütunlar sıfır/yanlış kabul edilir.
        arrays[symbol] = {
            f: window[f].to_numpy() if f in window.columns else np.zeros(len(window), dtype="float64")
            for f in SIGNAL_FIELDS
        }
    if not arrays:
        return
    timeline = np.unique(np.concatenate([a["ts"] for a in arrays.values()]))
    pointers = {symbol: 0 for symbol in arrays}
    for raw_ts in timeline:
        ts = int(raw_ts)
        bars: dict[str, SignalBar] = {}
        for symbol, a in arrays.items():
            i = pointers[symbol]
            if i < len(a["ts"]) and a["ts"][i] == ts:
                bars[symbol] = SignalBar(
                    ts=ts,
                    open=float(a["open"][i]),
                    high=float(a["high"][i]),
                    low=float(a["low"][i]),
                    close=float(a["close"][i]),
                    atr=float(a["atr"][i]),
                    entry_signal=bool(a["entry_signal"][i]),
                    regime_ok=bool(a["regime_ok"][i]),
                    exit_signal=bool(a["exit_signal"][i]),
                    priority=float(a["priority"][i]),
                    short_signal=bool(a["short_signal"][i]),
                    short_exit_signal=bool(a["short_exit_signal"][i]),
                )
                pointers[symbol] = i + 1
        yield ts, bars


def run_backtest(
    cfg: Config,
    signals: dict[str, pd.DataFrame],
    start_ts: int,
    end_ts: int,
    initial_equity: float,
    rules: dict[str, MarketRules],
    logger: logging.Logger | None = None,
    verbose: bool = False,
    funding_history: FundingHistory | None = None,
) -> BacktestResult:
    portfolio = Portfolio(initial_equity)
    broker = make_simulated_broker(cfg, portfolio, funding_history)
    risk = RiskManager(cfg, initial_equity)
    engine = TradingEngine(cfg, portfolio, broker, risk, rules, logger, verbose=verbose)

    for ts, bars in iter_signal_bars(signals, start_ts, end_ts):
        engine.on_bar(ts, bars)

    equity = pd.Series(
        [e for _, e in engine.equity_points],
        index=pd.Index([t for t, _ in engine.equity_points], dtype="int64", name="close_ts"),
        dtype="float64",
        name="equity",
    )
    return BacktestResult(
        equity=equity,
        trades=list(portfolio.trades),
        open_positions=list(portfolio.positions.values()),
        total_fees=portfolio.total_fees,
        skipped=engine.skipped,
        signals_seen=engine.signals_seen,
        orders_submitted=engine.orders_submitted,
        entries_cancelled=broker.cancelled_entries,
        exposure=engine.exposure_bars / engine.bars_processed if engine.bars_processed else 0.0,
        halted_at=engine.halted_at,
        halt_reason=risk.state.halt_reason,
        initial_equity=initial_equity,
        start_ts=start_ts,
        end_ts=end_ts,
        adds_submitted=engine.adds_submitted,
        liquidations=getattr(broker, "liquidations", 0),
        total_funding=portfolio.total_funding,
    )
