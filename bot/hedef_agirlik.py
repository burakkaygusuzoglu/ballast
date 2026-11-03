"""Hedef ağırlıklı günlük portföy backtest'i (8. tur).

Her gün kapanışta her coin için hedef ağırlık (özsermayenin katı; vadelide negatif = short) verilir; işlemler sonraki
günün açılışında yapılır. Muhasebe: özsermaye = nakit + Σ miktar x fiyat. Vadelide miktar negatif olabilir (short)
ve nakit negatife düşebilir (kaldıraç); doğrusal USDT vadeli sözleşmenin kâr/zararı bu gösterimle birebir aynıdır.

- İşlem bandı: hedef 0'a inerse kapat, 0'dan açılırsa aç, yön değişirse çevir; aksi hâlde yalnızca
  |hedef - mevcut| >= bant x |hedef| ise işlem yapılır.
- Maliyet: işlem tutarı x (komisyon); fiyat kayma kadar kötü (alış pahalı, satış ucuz).
- Funding (vadeli): pozisyon, gün içindeki funding anlarının (açılıştan sonraki 24 saat) toplam oranı x açılış fiyatı
  x miktar kadar öder; oran pozitifse long öder, short alır.
- Spotta nakit negatife düşemez: önce satışlar, sonra alışlar; nakit yetmezse alış küçültülür.
- Gerçek sermaye: miktar adımı ve en küçük işlem tutarı verilirse, işlemler adıma aşağı yuvarlanır ve küçük
  işlemler atlanır.
- Likidasyon (isteğe bağlı, vadeli, çapraz marjin; 9. tur): her gün işlemlerden ve funding'den sonra long pozisyonlar
  günün en düşük, short pozisyonlar en yüksek fiyatıyla aynı anda değerlenir. Bu en kötü özsermaye bakım marjinini
  (o fiyatlarla pozisyon büyüklüğü x oran) aşmıyorsa hesap likide olur ve özsermayenin tamamı kaybedilir.
  - "stop": yol biter (son gün özsermaye 0).
  - "restart": o günün getirisi -1 kaydedilir; seri önceki kapanış özsermayesiyle, pozisyonsuz devam eder
    (bootstrap malzemesi için).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.timeutil import DAY_MS

FUNDING_DEFAULT_8H = 0.0001
EVENTS_PER_DAY = 3


@dataclass(frozen=True)
class Costs:
    fee: float
    slippage: float


SPOT_COSTS = Costs(fee=0.001, slippage=0.0005)
FUTURES_COSTS = Costs(fee=0.0005, slippage=0.0005)


@dataclass(frozen=True)
class SymbolRules:
    qty_step: float = 0.0          # 0 = yuvarlama yok
    min_notional: float = 0.0      # 0 = sınır yok


@dataclass
class TWResult:
    equity: pd.Series                       # indeks: gün kapanış zamanı (UTC ms)
    trades: list[dict]
    fees: float
    funding: float
    skipped_small: int
    gross_exposure: pd.Series               # gün sonu brüt ağırlık
    positions: dict[str, pd.Series] = field(default_factory=dict)   # gün sonu ağırlıklar
    returns: pd.Series | None = None        # günlük getiri (önceki gün sonuna göre); likidasyon günü -1
    liquidations: list[int] = field(default_factory=list)            # likidasyon günlerinin açılış zamanı (UTC ms)


def _floor_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    steps = math.floor(abs(qty) / step + 1e-9)
    return math.copysign(round(steps * step, 12), qty)


def run_target_weights(
    prices: dict[str, pd.DataFrame],
    targets: dict[str, np.ndarray],
    start_ts: int,
    end_ts: int,
    initial_equity: float,
    costs: Costs,
    futures: bool,
    band: float = 0.20,
    funding: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    funding_start: dict[str, int] | None = None,
    rules: dict[str, SymbolRules] | None = None,
    liquidation_mmr: float | None = None,
    on_liquidation: str = "stop",
) -> TWResult:
    """prices: coin -> günlük mumlar (ts, open, high, low, close), aynı zaman çizgisinde. targets: coin -> hedef
    ağırlık dizisi, prices ile aynı satırlarda (t günü kapanışında verilen karar). Dönem [start_ts, end_ts): ilk gün
    açılışında, bir önceki günün kapanış kararıyla pozisyon açılır. liquidation_mmr verilirse (yalnızca vadeli)
    likidasyon kontrol edilir; on_liquidation "stop" ya da "restart" (modül açıklamasına bakın)."""
    if on_liquidation not in ("stop", "restart"):
        raise ValueError(f"Bilinmeyen likidasyon davranışı: {on_liquidation}")
    names = list(prices)
    ts = prices[names[0]]["ts"].to_numpy(dtype="int64")
    for n in names:
        if not np.array_equal(prices[n]["ts"].to_numpy(dtype="int64"), ts):
            raise ValueError("Bütün coinlerin günlük mumları aynı zaman çizgisinde olmalı.")
    opens = {n: prices[n]["open"].to_numpy(dtype="float64") for n in names}
    closes = {n: prices[n]["close"].to_numpy(dtype="float64") for n in names}
    check_liq = futures and liquidation_mmr is not None
    if check_liq:
        highs = {n: prices[n]["high"].to_numpy(dtype="float64") for n in names}
        lows = {n: prices[n]["low"].to_numpy(dtype="float64") for n in names}
    rules = rules or {}
    funding = funding or {}
    funding_start = funding_start or {}

    days = np.flatnonzero((ts >= start_ts) & (ts < end_ts))
    if len(days) == 0 or days[0] == 0:
        raise ValueError("Dönem, verinin içinde olmalı ve öncesinde en az bir gün (karar günü) bulunmalı.")
    cash = float(initial_equity)
    qty = {n: 0.0 for n in names}
    fees = fund_total = 0.0
    skipped = 0
    trades: list[dict] = []
    eq_points, gross_points = [], []
    pos_points = {n: [] for n in names}
    prev_eq = float(initial_equity)
    day_returns: list[float] = []
    liquidations: list[int] = []

    def equity_at(i: int) -> float:
        return cash + sum(qty[n] * closes[n][i] for n in names)

    for k, i in enumerate(days):
        decision = i - 1
        # 1) Önceki günün kapanış kararı: hedef miktarlar (kapanış fiyatı ve kapanış özsermayesiyle).
        eq_dec = equity_at(decision) if k else float(initial_equity)
        orders = {}
        for n in names:
            w_target = float(targets[n][decision])
            if not futures:
                w_target = max(w_target, 0.0)
            px_dec = closes[n][decision]
            w_now = qty[n] * px_dec / eq_dec if eq_dec > 0 else 0.0
            if w_target == 0.0 and qty[n] == 0.0:
                continue
            trade = (w_target == 0.0 or qty[n] == 0.0 or math.copysign(1, w_target) != math.copysign(1, w_now)
                     or abs(w_target - w_now) >= band * abs(w_target))
            if not trade:
                continue
            target_qty = w_target * eq_dec / px_dec if eq_dec > 0 else 0.0
            delta = target_qty - qty[n]
            if w_target == 0.0:
                delta = -qty[n]
            orders[n] = delta
        # 2) Açılışta işlemler: önce satışlar (nakit açar), sonra alışlar.
        for n in sorted(orders, key=lambda x: orders[x] > 0):
            requested = orders[n]
            r = rules.get(n, SymbolRules())
            full_exit = abs(requested + qty[n]) < 1e-15
            delta = requested if full_exit else _floor_step(requested, r.qty_step)
            price = opens[n][i] * (1 + costs.slippage if delta > 0 else 1 - costs.slippage)
            if delta == 0 or abs(delta) * price < r.min_notional:
                if requested != 0:                     # adıma yuvarlanınca sıfırlanan ya da en küçük tutarın altı
                    skipped += 1
                continue
            if not futures and delta > 0:                  # spotta nakit yetmezse alış küçülür
                affordable = cash / (price * (1 + costs.fee))
                if delta > affordable:
                    delta = _floor_step(affordable * 0.999999, r.qty_step)
                    if delta <= 0 or delta * price < r.min_notional:
                        skipped += 1
                        continue
            notional = abs(delta) * price
            fee = notional * costs.fee
            cash -= delta * price + fee
            qty[n] += delta
            if abs(qty[n]) < 1e-15:
                qty[n] = 0.0
            fees += fee
            trades.append({"ts": int(ts[i]), "coin": n, "qty": delta, "price": price, "fee": fee})
        # 3) Funding (vadeli): açılıştan sonraki 24 saatteki funding anları, bugünkü pozisyonla.
        if futures:
            for n in names:
                if qty[n] == 0.0:
                    continue
                start_f = funding_start.get(n)
                if n in funding and start_f is not None and ts[i] >= start_f:
                    times, rates = funding[n]
                    lo, hi = np.searchsorted(times, ts[i], side="right"), np.searchsorted(times, ts[i] + DAY_MS,
                                                                                          side="right")
                    rate = float(rates[lo:hi].sum())
                else:
                    rate = EVENTS_PER_DAY * FUNDING_DEFAULT_8H
                amount = qty[n] * opens[n][i] * rate
                cash -= amount
                fund_total += amount
        # 4) Likidasyon (çapraz marjin, kötümser): long günün dibinde, short günün tepesinde, aynı anda.
        if check_liq and any(qty[n] != 0.0 for n in names):
            worst = {n: (lows[n][i] if qty[n] > 0 else highs[n][i]) for n in names if qty[n] != 0.0}
            worst_eq = cash + sum(qty[n] * worst[n] for n in worst)
            maintenance = liquidation_mmr * sum(abs(qty[n]) * worst[n] for n in worst)
            if worst_eq <= maintenance:
                liquidations.append(int(ts[i]))
                day_returns.append(-1.0)
                for n in names:
                    qty[n] = 0.0
                    pos_points[n].append(0.0)
                stop = on_liquidation == "stop"
                cash = 0.0 if stop else prev_eq          # restart: önceki kapanış özsermayesiyle, pozisyonsuz
                eq_points.append((int(ts[i]) + DAY_MS, cash))
                gross_points.append(0.0)
                if stop:
                    break
                continue
        # 5) Gün sonu.
        eq = equity_at(i)
        day_returns.append(eq / prev_eq - 1 if prev_eq > 0 else 0.0)
        prev_eq = eq
        eq_points.append((int(ts[i]) + DAY_MS, eq))
        gross = sum(abs(qty[n] * closes[n][i]) for n in names) / eq if eq > 0 else 0.0
        gross_points.append(gross)
        for n in names:
            pos_points[n].append(qty[n] * closes[n][i] / eq if eq > 0 else 0.0)
        if not futures and cash < -1e-6:
            raise AssertionError(f"Spotta nakit negatife düştü: {cash}")
    index = pd.Index([t for t, _ in eq_points], dtype="int64", name="close_ts")
    return TWResult(
        equity=pd.Series([e for _, e in eq_points], index=index, dtype="float64", name="equity"),
        trades=trades, fees=fees, funding=fund_total, skipped_small=skipped,
        gross_exposure=pd.Series(gross_points, index=index, dtype="float64"),
        positions={n: pd.Series(v, index=index, dtype="float64") for n, v in pos_points.items()},
        returns=pd.Series(day_returns, index=index, dtype="float64", name="return"),
        liquidations=liquidations,
    )


def buy_and_hold(prices: dict[str, pd.DataFrame], start_ts: int, end_ts: int, initial_equity: float,
                 costs: Costs = SPOT_COSTS) -> pd.Series:
    """Başlangıç günü açılışında sermaye coinlere eşit bölünür ve tutulur (gün sonu özsermaye)."""
    names = list(prices)
    ts = prices[names[0]]["ts"].to_numpy(dtype="int64")
    days = np.flatnonzero((ts >= start_ts) & (ts < end_ts))
    first = days[0]
    units = {}
    for n in names:
        price = prices[n]["open"].to_numpy()[first] * (1 + costs.slippage)
        units[n] = initial_equity / len(names) / (price * (1 + costs.fee))
    values = [sum(units[n] * prices[n]["close"].to_numpy()[i] for n in names) for i in days]
    return pd.Series(values, index=pd.Index(ts[days] + DAY_MS, dtype="int64", name="close_ts"), dtype="float64")


def tw_metrics(equity: pd.Series, initial: float, trades: list[dict] | None = None, fees: float = 0.0,
               funding: float = 0.0, gross: pd.Series | None = None) -> dict:
    final = float(equity.iloc[-1])
    days = (int(equity.index[-1]) - int(equity.index[0])) / DAY_MS + 1
    years = days / 365.25
    total = final / initial - 1
    cagr = (final / initial) ** (1 / years) - 1 if years > 0.05 and final > 0 else total
    curve = np.concatenate([[initial], equity.to_numpy()])
    dd = float(np.max(1 - curve / np.maximum.accumulate(curve)))
    rets = np.diff(curve) / curve[:-1]
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(365)) if len(rets) > 2 and np.std(rets) > 0 else 0.0
    return {"initial": initial, "final": final, "net_profit": final - initial, "total_return": total, "cagr": cagr,
            "max_drawdown": dd, "return_to_drawdown": total / dd if dd > 0 else math.inf, "sharpe": sharpe,
            "days": days, "n_trades": len(trades or []), "fees": fees, "funding": funding,
            "avg_gross": float(gross.mean()) if gross is not None and len(gross) else float("nan")}
