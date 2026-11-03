"""Simülasyon broker'ı - backtest ve paper trading için ORTAK dolum modeli.

Bilinçli olarak kötümserdir:
- Limit alış, fiyat limitin ALTINA inerse dolar (sadece dokunması yetmez). Maker komisyonu.
- Piyasa emri bir sonraki mumun açılışından + kayma ile dolar. Taker komisyonu.
- Stop, mumun dibi stop'a değerse stop fiyatından; mum zaten stop'un altında açıldıysa
  (boşluk) açılış fiyatından dolar. Her ikisinde de kayma + taker komisyonu düşülür.
- Aynı mumda giriş dolup stop seviyesi de görüldüyse, fiyat limitten stop'a inerken
  ikisinden de geçmiş demektir: önce giriş, sonra stop.
- Bekleyen limit emir için nakit ayrılır (gerçek borsadaki gibi), iptalde geri verilir.
- Piramitleme (isteğe bağlı): açık pozisyona eklenen birim bir sonraki mumun açılışında piyasa emriyle
  dolar ve pozisyonla birleşir (ortalama maliyet); stop, son birimin altına yalnızca yukarı taşınır.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bot.brokers.base import Bar, Broker, EntryOrder, Fill
from bot.portfolio import Portfolio, Position, Trade

MARKET_RESERVE_BUFFER = 1.01   # piyasa emrinde açılış fiyatı belirsiz; %1 pay ayrılır


class InsufficientCash(ValueError):
    pass


class SimulatedBroker(Broker):
    supports_adds = True

    def __init__(self, portfolio: Portfolio, maker_fee: float, taker_fee: float, slippage: float):
        self.portfolio = portfolio
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.pending_entries: dict[str, EntryOrder] = {}
        self.pending_exits: dict[str, str] = {}
        self.pending_adds: dict[str, EntryOrder] = {}
        self.cancelled_entries = 0

    # ---- emir yönetimi -------------------------------------------------------
    def has_pending_entry(self, symbol: str) -> bool:
        return symbol in self.pending_entries

    def pending_count(self) -> int:
        return len(self.pending_entries)

    def submit_entry(self, order: EntryOrder, ref_price: float) -> None:
        if order.symbol in self.portfolio.positions or order.symbol in self.pending_entries:
            raise ValueError(f"{order.symbol}: zaten açık pozisyon ya da bekleyen emir var.")
        if order.qty <= 0:
            raise ValueError("Emir miktarı pozitif olmalı.")
        if order.limit_price is not None:
            reserve = order.qty * order.limit_price * (1 + self.maker_fee)
        else:
            reserve = order.qty * ref_price * (1 + self.slippage) * (1 + self.taker_fee) * MARKET_RESERVE_BUFFER
        if reserve > self.portfolio.cash + 1e-9:
            raise InsufficientCash(f"{order.symbol}: emir için yeterli serbest nakit yok.")
        self.portfolio.cash -= reserve
        self.portfolio.reserved += reserve
        order.reserved = reserve
        self.pending_entries[order.symbol] = order

    def submit_add(self, order: EntryOrder, ref_price: float) -> None:
        """Açık pozisyona birim ekleme (piyasa emri, bir sonraki mumun açılışında)."""
        if order.symbol not in self.portfolio.positions or order.symbol in self.pending_adds:
            raise ValueError(f"{order.symbol}: ekleme için açık pozisyon yok ya da zaten bekleyen ekleme var.")
        if order.qty <= 0:
            raise ValueError("Emir miktarı pozitif olmalı.")
        reserve = order.qty * ref_price * (1 + self.slippage) * (1 + self.taker_fee) * MARKET_RESERVE_BUFFER
        if reserve > self.portfolio.cash + 1e-9:
            raise InsufficientCash(f"{order.symbol}: ekleme için yeterli serbest nakit yok.")
        self.portfolio.cash -= reserve
        self.portfolio.reserved += reserve
        order.reserved = reserve
        self.pending_adds[order.symbol] = order

    def pending_add_count(self) -> int:
        return len(self.pending_adds)

    def has_pending_exit(self, symbol: str) -> bool:
        return symbol in self.pending_exits

    def _release(self, order: EntryOrder) -> None:
        self.portfolio.cash += order.reserved
        self.portfolio.reserved = max(0.0, self.portfolio.reserved - order.reserved)
        order.reserved = 0.0

    def cancel_entries(self) -> list[str]:
        symbols = list(self.pending_entries)
        for symbol in symbols:
            self._release(self.pending_entries.pop(symbol))
            self.cancelled_entries += 1
        for symbol in list(self.pending_adds):           # şalter/günlük limit: eklemeler de iptal
            self._release(self.pending_adds.pop(symbol))
        return symbols

    def request_exit(self, symbol: str, reason: str) -> None:
        if symbol in self.portfolio.positions:
            self.pending_exits.setdefault(symbol, reason)

    def update_stop(self, symbol: str, new_stop: float) -> bool:
        position = self.portfolio.positions.get(symbol)
        if position is not None and new_stop > position.stop_price:
            position.stop_price = new_stop
            return True
        return False

    # ---- mum işleme ------------------------------------------------------------
    def process_bar(self, symbol: str, bar: Bar) -> list[Fill]:
        """Bekleyen emirleri ve stopu bu mumun açılış/yüksek/düşük değerlerine karşı işler."""
        fills: list[Fill] = []
        position = self.portfolio.positions.get(symbol)

        # 1) Önceki mum kapanışında kararlaştırılan çıkış: bu mumun açılışında.
        exit_reason = self.pending_exits.pop(symbol, None)
        if exit_reason is not None and position is not None:
            fills.append(self._exit(symbol, bar.open, bar.ts, exit_reason))
            position = None

        # 1b) Bekleyen birim ekleme: pozisyon hâlâ açıksa bu mumun açılışında dolar.
        add = self.pending_adds.pop(symbol, None)
        if add is not None:
            self._release(add)
            if position is not None:
                fills.append(self._fill_add(add, bar))

        # 2) Bekleyen giriş emri.
        order = self.pending_entries.get(symbol)
        if order is not None:
            fill = self._try_entry(order, bar)
            if fill is not None:
                fills.append(fill)
                position = self.portfolio.positions[symbol]

        # 3) Stop kontrolü.
        if position is not None and bar.low <= position.stop_price:
            gap_possible = position.entry_ts != bar.ts or position.entered_at_open
            raw = min(bar.open, position.stop_price) if gap_possible else position.stop_price
            fills.append(self._exit(symbol, raw, bar.ts, "stop"))
        return fills

    def _try_entry(self, order: EntryOrder, bar: Bar) -> Fill | None:
        if order.limit_price is None:
            raw, rate, slip, label = bar.open, self.taker_fee, self.slippage, "giriş (piyasa)"
        elif bar.low < order.limit_price:
            raw, rate, slip, label = order.limit_price, self.maker_fee, 0.0, "giriş (limit)"
        else:
            order.bars_waited += 1
            if order.bars_waited >= order.ttl_bars:
                del self.pending_entries[order.symbol]
                self._release(order)
                self.cancelled_entries += 1
            return None

        del self.pending_entries[order.symbol]
        self._release(order)
        price = raw * (1 + slip)
        qty = order.qty
        if qty * price * (1 + rate) > self.portfolio.cash + 1e-9:
            qty = self.portfolio.cash / (price * (1 + rate)) * 0.999999
        cost = qty * price
        fee = cost * rate
        self.portfolio.cash -= cost + fee
        self.portfolio.total_fees += fee
        self.portfolio.positions[order.symbol] = Position(
            symbol=order.symbol,
            qty=qty,
            entry_price=price,
            entry_ts=bar.ts,
            entry_fee=fee,
            entry_slippage=qty * raw * slip,
            stop_price=order.stop_price,
            initial_stop=order.stop_price,
            highest=price,
            risk_amount=order.risk_amount * qty / order.qty,
            entered_at_open=order.limit_price is None,
            last_add_price=price,
            unit_atr=order.unit_atr,
        )
        return Fill(order.symbol, "buy", qty, price, fee, bar.ts, label)

    def _fill_add(self, order: EntryOrder, bar: Bar) -> Fill:
        position = self.portfolio.positions[order.symbol]
        raw = bar.open
        price = raw * (1 + self.slippage)
        qty = order.qty
        if qty * price * (1 + self.taker_fee) > self.portfolio.cash + 1e-9:
            qty = self.portfolio.cash / (price * (1 + self.taker_fee)) * 0.999999
        cost = qty * price
        fee = cost * self.taker_fee
        self.portfolio.cash -= cost + fee
        self.portfolio.total_fees += fee
        total = position.qty + qty
        position.entry_price = (position.qty * position.entry_price + qty * price) / total
        position.qty = total
        position.entry_fee += fee
        position.entry_slippage += qty * raw * self.slippage
        position.risk_amount += order.risk_amount * qty / order.qty
        position.units += 1
        position.last_add_price = price
        position.highest = max(position.highest, price)
        if order.stop_distance > 0:
            position.stop_price = max(position.stop_price, price - order.stop_distance)
        return Fill(order.symbol, "buy", qty, price, fee, bar.ts, f"birim ekleme ({position.units}. birim)")

    def _exit(self, symbol: str, raw_price: float, ts: int, reason: str) -> Fill:
        position = self.portfolio.positions.pop(symbol)
        price = raw_price * (1 - self.slippage)
        proceeds = position.qty * price
        fee = proceeds * self.taker_fee
        self.portfolio.cash += proceeds - fee
        self.portfolio.total_fees += fee
        gross = position.qty * (price - position.entry_price)
        net = gross - position.entry_fee - fee
        self.portfolio.trades.append(
            Trade(
                symbol=symbol,
                entry_ts=position.entry_ts,
                exit_ts=ts,
                qty=position.qty,
                entry_price=position.entry_price,
                exit_price=price,
                entry_fee=position.entry_fee,
                exit_fee=fee,
                slippage_cost=position.entry_slippage + position.qty * raw_price * self.slippage,
                gross_pnl=gross,
                net_pnl=net,
                risk_amount=position.risk_amount,
                r_multiple=net / position.risk_amount if position.risk_amount > 0 else 0.0,
                exit_reason=reason,
            )
        )
        return Fill(symbol, "sell", position.qty, price, fee, ts, reason)

    # ---- kalıcılık -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "pending_entries": [asdict(o) for o in self.pending_entries.values()],
            "pending_exits": dict(self.pending_exits),
            "pending_adds": [asdict(o) for o in self.pending_adds.values()],
            "cancelled_entries": self.cancelled_entries,
        }

    def load_state(self, data: dict[str, Any]) -> None:
        self.pending_entries = {o["symbol"]: EntryOrder(**o) for o in data.get("pending_entries", [])}
        self.pending_exits = dict(data.get("pending_exits", {}))
        self.pending_adds = {o["symbol"]: EntryOrder(**o) for o in data.get("pending_adds", [])}
        self.cancelled_entries = int(data.get("cancelled_entries", 0))
