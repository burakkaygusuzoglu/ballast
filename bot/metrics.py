"""Performans metrikleri, al-tut karşılaştırması ve karar kapısı."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import GateConfig
from bot.portfolio import Trade
from bot.timeutil import DAY_MS


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def max_drawdown(equity: pd.Series) -> float:
    """Tepeden en büyük düşüş (pozitif oran, 0.15 = %15)."""
    if equity.empty:
        return 0.0
    values = equity.to_numpy(dtype="float64")
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))


def daily_equity(equity: pd.Series) -> pd.Series:
    days = (equity.index.to_numpy() - 1) // DAY_MS
    return equity.groupby(days).last()


def sharpe_ratio(equity: pd.Series) -> float:
    returns = daily_equity(equity).pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(365))


def return_to_drawdown(total_return: float, drawdown: float) -> float:
    if drawdown <= 0:
        return math.inf if total_return > 0 else 0.0
    return total_return / drawdown


def compute_metrics(
    equity: pd.Series,
    trades: list[Trade],
    initial_equity: float,
    total_fees: float,
    exposure: float | None = None,
) -> dict:
    final = float(equity.iloc[-1]) if len(equity) else initial_equity
    start_ts = int(equity.index[0]) if len(equity) else 0
    end_ts = int(equity.index[-1]) if len(equity) else 0
    days = max((end_ts - start_ts) / DAY_MS, 1e-9)
    years = days / 365.25
    total_return = final / initial_equity - 1
    cagr = (final / initial_equity) ** (1 / years) - 1 if years > 0.05 and final > 0 else total_return
    drawdown = max_drawdown(equity)

    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl <= 0]
    gross_profit, gross_loss = sum(wins), -sum(losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else 0.0

    fees_closed = sum(t.entry_fee + t.exit_fee for t in trades)
    slippage_closed = sum(t.slippage_cost for t in trades)
    pre_cost_pnl = sum(t.gross_pnl + t.slippage_cost for t in trades)
    costs_r = [(t.entry_fee + t.exit_fee + t.slippage_cost) / t.risk_amount for t in trades if t.risk_amount > 0]

    return {
        "initial_equity": initial_equity,
        "final_equity": final,
        "net_profit": final - initial_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": drawdown,
        "return_to_drawdown": return_to_drawdown(total_return, drawdown),
        "sharpe": sharpe_ratio(equity) if len(equity) > 1 else 0.0,
        "days": days,
        "n_trades": len(trades),
        "trades_per_week": len(trades) / (days / 7) if days > 0 else 0.0,
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": profit_factor,
        "avg_r": float(np.mean([t.r_multiple for t in trades])) if trades else 0.0,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "best_trade": max((t.net_pnl for t in trades), default=0.0),
        "worst_trade": min((t.net_pnl for t in trades), default=0.0),
        "fees_paid": total_fees,
        "fees_closed": fees_closed,
        "slippage_closed": slippage_closed,
        "pre_cost_pnl": pre_cost_pnl,
        "cost_share": (fees_closed + slippage_closed) / pre_cost_pnl if pre_cost_pnl > 0 else None,
        "avg_cost_r": float(np.mean(costs_r)) if costs_r else 0.0,
        "exposure": exposure,
    }


def buy_and_hold(candles: dict[str, pd.DataFrame], start_ts: int, end_ts: int, initial_equity: float,
                 taker_fee: float, slippage: float, entry_ms: int) -> pd.Series:
    """Eşit ağırlıklı al-tut: dönemin ilk mumunun açılışında piyasa emriyle alır (komisyon + kayma)."""
    per_symbol = initial_equity / len(candles)
    parts = []
    for symbol, df in candles.items():
        window = df[(df["ts"] >= start_ts) & (df["ts"] < end_ts)]
        if window.empty:
            continue
        entry_price = float(window["open"].iloc[0]) * (1 + slippage)
        qty = per_symbol / (entry_price * (1 + taker_fee))
        series = pd.Series(window["close"].to_numpy() * qty, index=window["ts"].to_numpy() + entry_ms)
        parts.append(series.rename(symbol))
    if not parts:
        return pd.Series(dtype="float64")
    combined = pd.concat(parts, axis=1).sort_index().ffill().dropna()
    total = combined.sum(axis=1)
    total.index.name = "close_ts"
    return total.rename("equity")


def evaluate_gate(metrics: dict, benchmark: dict, gate: GateConfig, period: str) -> list[GateCheck]:
    checks = [
        GateCheck(
            f"Kâr faktörü >= {gate.min_profit_factor:.2f}",
            metrics["profit_factor"] >= gate.min_profit_factor,
            _ratio(metrics["profit_factor"]),
        ),
        GateCheck(
            f"Maks. düşüş <= %{gate.max_drawdown * 100:.0f}",
            metrics["max_drawdown"] <= gate.max_drawdown,
            f"%{metrics['max_drawdown'] * 100:.1f}",
        ),
    ]
    if period == "dev":
        checks.append(GateCheck(
            f"En az {gate.min_trades_dev} işlem (istatistiksel anlam)",
            metrics["n_trades"] >= gate.min_trades_dev,
            f"{metrics['n_trades']} işlem",
        ))
    if period == "exam" and gate.require_positive_exam:
        checks.append(GateCheck(
            "Sınavda net kâr pozitif",
            metrics["net_profit"] > 0,
            f"{metrics['net_profit']:+,.2f} USDT",
        ))
    if gate.require_beat_benchmark:
        mine, theirs = metrics["return_to_drawdown"], benchmark["return_to_drawdown"]
        checks.append(GateCheck(
            "Getiri/düşüş oranı al-tut'tan iyi",
            mine > theirs,
            f"{_ratio(mine)} vs {_ratio(theirs)}",
        ))
    return checks


def _ratio(value: float) -> str:
    return "sonsuz" if math.isinf(value) else f"{value:.2f}"
