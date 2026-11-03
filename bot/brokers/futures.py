"""Vadeli işlem simülasyon broker'ı: USDⓈ-M sürekli sözleşme, izole marjin, tek yönlü mod (6. tur).

Spot simülasyon broker'ıyla AYNI kötümser dolum modeli; farklar:
- Giriş: marjin (= pozisyon büyüklüğü / kaldıraç) ve komisyon nakitten düşülür. Long da short da açılabilir.
- Özsermaye: marjin + gerçekleşmemiş kâr/zarar (Position.value).
- Funding: UTC 00:00, 08:00, 16:00'da o an açık pozisyonlardan, mumun açılış fiyatıyla. Sabit oranlar
  (config.futures) ya da gerçek funding geçmişi (sembol -> zaman ve oran dizileri).
- Likidasyon: izole marjinin tamamı kaybedilir, nakdin geri kalanı etkilenmez.
  * "open": mum likidasyon fiyatının ötesinde açılırsa (iki mum arasında boşluk).
  * "touch" (stres): mumun ucu likidasyon fiyatına değerse; borsadaki stopun çalışmadığı kabul edilir.
- Piramitleme ve limit emirle giriş yok.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from bot.brokers.base import Bar, EntryOrder, Fill
from bot.brokers.simulated import (
    MARKET_RESERVE_BUFFER,
    InsufficientCash,
    SimulatedBroker,
)
from bot.config import Config, FuturesConfig
from bot.portfolio import Portfolio, Position, Trade
from bot.risk import liquidation_price

FUNDING_MS = 8 * 3_600_000            # Binance USDⓈ-M: UTC 00:00, 08:00, 16:00

FundingHistory = dict[str, tuple[np.ndarray, np.ndarray]]   # sembol -> (funding anları UTC ms, oranlar), sıralı


class SimulatedFuturesBroker(SimulatedBroker):
    supports_adds = False

    def __init__(self, portfolio: Portfolio, maker_fee: float, taker_fee: float, slippage: float,
                 fc: FuturesConfig, funding_history: FundingHistory | None = None):
        super().__init__(portfolio, maker_fee, taker_fee, slippage)
        self.fc = fc
        self.funding_history = funding_history or {}
        self.liquidations = 0

    # ---- emir yönetimi -------------------------------------------------------
    def margin_rate(self) -> float:
        """Pozisyonun her 1 USDT'si için girişte gereken nakit: marjin + taker komisyonu."""
        return 1 / self.fc.leverage + self.taker_fee

    def submit_entry(self, order: EntryOrder, ref_price: float) -> None:
        if order.symbol in self.portfolio.positions or order.symbol in self.pending_entries:
            raise ValueError(f"{order.symbol}: zaten açık pozisyon ya da bekleyen emir var.")
        if order.qty <= 0:
            raise ValueError("Emir miktarı pozitif olmalı.")
        if order.limit_price is not None:
            raise ValueError("Vadeli işlemde giriş yalnızca piyasa emriyle yapılır.")
        reserve = order.qty * ref_price * (1 + self.slippage) * self.margin_rate() * MARKET_RESERVE_BUFFER
        if reserve > self.portfolio.cash + 1e-9:
            raise InsufficientCash(f"{order.symbol}: emir için yeterli serbest nakit (marjin) yok.")
        self.portfolio.cash -= reserve
        self.portfolio.reserved += reserve
        order.reserved = reserve
        self.pending_entries[order.symbol] = order

    def submit_add(self, order: EntryOrder, ref_price: float) -> None:
        raise ValueError("Vadeli işlemde piramitleme yok.")

    def update_stop(self, symbol: str, new_stop: float) -> bool:
        """Stop yalnızca kâr yönüne taşınır: long'da yukarı, short'ta aşağı."""
        position = self.portfolio.positions.get(symbol)
        if position is None:
            return False
        better = new_stop < position.stop_price if position.side < 0 else new_stop > position.stop_price
        if better:
            position.stop_price = new_stop
        return better

    # ---- mum işleme ------------------------------------------------------------
    def process_bar(self, symbol: str, bar: Bar) -> list[Fill]:
        fills: list[Fill] = []
        position = self.portfolio.positions.get(symbol)

        # 0) Funding: bu mumun açılışına kadar geçen funding anlarında pozisyon açıktı.
        if position is not None:
            self._charge_funding(position, bar)

        # 1) Önceki kapanışta kararlaştırılan çıkış: bu mumun açılışında (açılış likidasyonun ötesindeyse
        #    pozisyon iki mum arasında likide olmuştur).
        exit_reason = self.pending_exits.pop(symbol, None)
        if exit_reason is not None and position is not None:
            if _beyond(position.liq_price, bar.open, position.side):
                fills.append(self._liquidate(symbol, bar.ts))
            else:
                fills.append(self._exit(symbol, bar.open, bar.ts, exit_reason))
            position = None

        # 2) Bekleyen giriş emri (piyasa, açılışta).
        order = self.pending_entries.get(symbol)
        if order is not None:
            fill = self._try_entry(order, bar)
            if fill is not None:
                fills.append(fill)
                position = self.portfolio.positions[symbol]

        # 3) Likidasyon ve stop.
        if position is not None:
            fill = self._check_position(symbol, position, bar)
            if fill is not None:
                fills.append(fill)
        return fills

    def _check_position(self, symbol: str, position: Position, bar: Bar) -> Fill | None:
        side, entered_now = position.side, position.entry_ts == bar.ts
        if not entered_now and _beyond(position.liq_price, bar.open, side):
            return self._liquidate(symbol, bar.ts)
        if self.fc.liquidation == "touch" and _beyond(position.liq_price, bar.low if side > 0 else bar.high, side):
            return self._liquidate(symbol, bar.ts)
        gap_possible = not entered_now or position.entered_at_open
        if side > 0 and bar.low <= position.stop_price:
            raw = min(bar.open, position.stop_price) if gap_possible else position.stop_price
            return self._exit(symbol, raw, bar.ts, "stop")
        if side < 0 and bar.high >= position.stop_price:
            raw = max(bar.open, position.stop_price) if gap_possible else position.stop_price
            return self._exit(symbol, raw, bar.ts, "stop")
        return None

    def _charge_funding(self, position: Position, bar: Bar) -> None:
        start, end = position.funding_ts, bar.ts
        if end <= start:
            return
        history = self.funding_history.get(position.symbol)
        if history is None:
            events = end // FUNDING_MS - start // FUNDING_MS
            rate = self.fc.funding_long_8h if position.side > 0 else self.fc.funding_short_8h
            amount = events * rate * position.qty * bar.open
        else:
            times, rates = history
            lo, hi = np.searchsorted(times, start, side="right"), np.searchsorted(times, end, side="right")
            # Gerçek oran r > 0 iken long öder, short alır.
            amount = position.side * float(rates[lo:hi].sum()) * position.qty * bar.open
        position.funding_ts = end
        if amount:
            position.funding_paid += amount
            self.portfolio.cash -= amount
            self.portfolio.total_funding += amount

    def _try_entry(self, order: EntryOrder, bar: Bar) -> Fill | None:
        del self.pending_entries[order.symbol]
        self._release(order)
        side, lev = order.side, self.fc.leverage
        raw = bar.open
        price = raw * (1 + side * self.slippage)          # long pahalıya alır, short ucuza satar
        qty = order.qty
        if qty * price * self.margin_rate() > self.portfolio.cash + 1e-9:
            qty = self.portfolio.cash / (price * self.margin_rate()) * 0.999999
        if qty <= 0:
            self.cancelled_entries += 1
            return None
        notional = qty * price
        margin = notional / lev
        fee = notional * self.taker_fee
        self.portfolio.cash -= margin + fee
        self.portfolio.total_fees += fee
        self.portfolio.positions[order.symbol] = Position(
            symbol=order.symbol,
            qty=qty,
            entry_price=price,
            entry_ts=bar.ts,
            entry_fee=fee,
            entry_slippage=qty * raw * self.slippage,
            stop_price=order.stop_price,
            initial_stop=order.stop_price,
            highest=price,
            risk_amount=order.risk_amount * qty / order.qty,
            entered_at_open=True,
            last_add_price=price,
            unit_atr=order.unit_atr,
            side=side,
            leverage=lev,
            margin=margin,
            liq_price=liquidation_price(price, side, lev, self.fc.maintenance_margin),
            lowest=price,
            funding_ts=bar.ts,                             # bu mumun funding anı girişten önce geçti
        )
        label = "giriş (piyasa)" if side > 0 else "short giriş (piyasa)"
        return Fill(order.symbol, "buy" if side > 0 else "sell", qty, price, fee, bar.ts, label, entry=True)

    def _exit(self, symbol: str, raw_price: float, ts: int, reason: str) -> Fill:
        position = self.portfolio.positions.pop(symbol)
        side = position.side
        price = raw_price * (1 - side * self.slippage)    # long ucuza satar, short pahalıya geri alır
        pnl = side * position.qty * (price - position.entry_price)
        fee = position.qty * price * self.taker_fee
        self.portfolio.cash += position.margin + pnl - fee
        self.portfolio.total_fees += fee
        net = pnl - position.entry_fee - fee - position.funding_paid
        self._record(position, ts, price, fee, position.entry_slippage + position.qty * raw_price * self.slippage,
                     pnl, net, reason)
        return Fill(symbol, "sell" if side > 0 else "buy", position.qty, price, fee, ts, reason, entry=False)

    def _liquidate(self, symbol: str, ts: int) -> Fill:
        """İzole marjinin tamamı kaybedilir (kalan, borsanın sigorta fonuna gider)."""
        position = self.portfolio.positions.pop(symbol)
        self.liquidations += 1
        loss = position.margin
        net = -loss - position.entry_fee - position.funding_paid
        self._record(position, ts, position.liq_price, 0.0, position.entry_slippage, -loss, net, "likidasyon")
        side = "sell" if position.side > 0 else "buy"
        return Fill(symbol, side, position.qty, position.liq_price, 0.0, ts, "likidasyon", entry=False)

    def _record(self, position: Position, ts: int, price: float, fee: float, slippage_cost: float,
                gross: float, net: float, reason: str) -> None:
        self.portfolio.trades.append(
            Trade(
                symbol=position.symbol,
                entry_ts=position.entry_ts,
                exit_ts=ts,
                qty=position.qty,
                entry_price=position.entry_price,
                exit_price=price,
                entry_fee=position.entry_fee,
                exit_fee=fee,
                slippage_cost=slippage_cost,
                gross_pnl=gross,
                net_pnl=net,
                risk_amount=position.risk_amount,
                r_multiple=net / position.risk_amount if position.risk_amount > 0 else 0.0,
                exit_reason=reason,
                side=position.side,
                funding=position.funding_paid,
            )
        )

    # ---- kalıcılık -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "liquidations": self.liquidations}

    def load_state(self, data: dict[str, Any]) -> None:
        super().load_state(data)
        self.liquidations = int(data.get("liquidations", 0))


def _beyond(liq_price: float, price: float, side: int) -> bool:
    """Fiyat likidasyon fiyatına değdi ya da geçti mi?"""
    return price <= liq_price if side > 0 else price >= liq_price


def make_simulated_broker(cfg: Config, portfolio: Portfolio,
                          funding_history: FundingHistory | None = None) -> SimulatedBroker:
    """Backtest ve paper için doğru simülasyon broker'ı: spot ya da vadeli (exchange.market_type)."""
    c = cfg.costs
    if cfg.is_futures:
        return SimulatedFuturesBroker(portfolio, c.maker_fee, c.taker_fee, c.slippage, cfg.futures, funding_history)
    if funding_history:
        raise ValueError("Funding geçmişi yalnızca vadeli işlemde kullanılır.")
    return SimulatedBroker(portfolio, c.maker_fee, c.taker_fee, c.slippage)
