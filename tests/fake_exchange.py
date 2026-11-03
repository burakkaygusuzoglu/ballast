"""Sahte Binance spot borsası (ccxt arayüzünün botun kullandığı kısmı). Testler internete bağlanmaz.

- Piyasa emirleri o anki fiyattan dolar; alışta komisyon coin, satışta USDT olarak kesilir (%0,1).
- Stop emirleri borsada bekler ve satılacak coini kilitler; test, mumun en düşüğünü vererek tetikler.
- İstenirse ağ hatası (emir yine de borsaya ulaşmış hâlde) ya da emir reddi üretir.
"""
from __future__ import annotations

import math
from typing import Any

import ccxt

FEE = 0.001


class FakeExchange:
    def __init__(self, symbols, balances: dict[str, float] | None = None, step: float = 1e-5,
                 stop_market: bool = True):
        order_types = ["LIMIT", "MARKET", "STOP_LOSS_LIMIT"] + (["STOP_LOSS"] if stop_market else [])
        self.markets = {s: {"symbol": s, "active": True, "spot": True, "info": {"orderTypes": order_types}}
                        for s in symbols}
        self.step = step
        self.free = {"USDT": 0.0, **(balances or {})}
        self.prices: dict[str, float] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.by_client_id: dict[str, str] = {}
        self.next_id = 1
        self.fail_next: list[Exception] = []          # sıradaki create_order çağrılarında atılacak hatalar
        self.fail_calls: dict[str, list[Exception]] = {}   # yöntem adı -> sıradaki çağrılarda atılacak hatalar
        self.network_error_after_create = False       # emir borsaya ulaşır ama cevap kaybolur
        self.reject_stops = False
        self.created: list[dict[str, Any]] = []

    # ---- ccxt yardımcıları ---------------------------------------------------
    def market(self, symbol):
        return self.markets[symbol]

    def amount_to_precision(self, symbol, qty):
        steps = math.floor(qty / self.step + 1e-9)
        if steps <= 0:
            raise ccxt.InvalidOrder("amount must be greater than minimum amount precision")
        return f"{steps * self.step:.8f}"

    def price_to_precision(self, symbol, price):
        return f"{round(price, 2):.2f}"

    def _maybe_fail(self, method):
        queue = self.fail_calls.get(method)
        if queue:
            raise queue.pop(0)

    def fetch_balance(self):
        self._maybe_fail("fetch_balance")
        return {"free": dict(self.free)}

    def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}

    # ---- emirler -------------------------------------------------------------
    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        params = params or {}
        if self.fail_next:
            raise self.fail_next.pop(0)
        base = symbol.split("/")[0]
        order_id = str(self.next_id)
        self.next_id += 1
        order: dict[str, Any] = {"id": order_id, "symbol": symbol, "type": type_, "side": side, "amount": amount,
                                 "price": price, "params": params, "status": "open", "filled": 0.0, "cost": 0.0,
                                 "average": None, "fees": []}
        if type_ == "market":
            px = self.prices[symbol]
            cost = amount * px
            if side == "buy":
                if cost > self.free["USDT"] + 1e-9:
                    raise ccxt.InsufficientFunds("insufficient balance")
                self.free["USDT"] -= cost
                self.free[base] = self.free.get(base, 0.0) + amount * (1 - FEE)
                fees = [{"cost": amount * FEE, "currency": base}]
            else:
                if amount > self.free.get(base, 0.0) + 1e-9:
                    raise ccxt.InsufficientFunds("insufficient balance")
                self.free[base] -= amount
                self.free["USDT"] += cost * (1 - FEE)
                fees = [{"cost": cost * FEE, "currency": "USDT"}]
            order.update(status="closed", filled=amount, cost=cost, average=px, fees=fees)
        elif type_ in ("STOP_LOSS", "STOP_LOSS_LIMIT"):
            if self.reject_stops:
                raise ccxt.InvalidOrder("stop price would trigger immediately")
            if amount > self.free.get(base, 0.0) + 1e-9:
                raise ccxt.InsufficientFunds("insufficient balance for stop")
            self.free[base] -= amount                       # satılacak coin kilitlenir
        else:
            raise ccxt.InvalidOrder(f"desteklenmeyen tür {type_}")
        self.orders[order_id] = order
        if "newClientOrderId" in params:
            self.by_client_id[params["newClientOrderId"]] = order_id
        self.created.append(order)
        if self.network_error_after_create:
            self.network_error_after_create = False
            raise ccxt.NetworkError("bağlantı koptu (emir borsaya ulaştı)")
        return dict(order)

    def fetch_order(self, order_id, symbol, params=None):
        params = params or {}
        self._maybe_fail("fetch_order")
        if order_id is None and "origClientOrderId" in params:
            if params["origClientOrderId"] not in self.by_client_id:
                raise ccxt.OrderNotFound("unknown order")
            order_id = self.by_client_id[params["origClientOrderId"]]
        if order_id not in self.orders:
            raise ccxt.OrderNotFound("unknown order")
        return dict(self.orders[order_id])

    def cancel_order(self, order_id, symbol):
        self._maybe_fail("cancel_order")
        order = self.orders.get(order_id)
        if order is None or order["status"] != "open":
            raise ccxt.OrderNotFound("Unknown order sent.")
        order["status"] = "canceled"
        base = symbol.split("/")[0]
        self.free[base] = self.free.get(base, 0.0) + order["amount"]
        return dict(order)

    # ---- test yardımcıları ---------------------------------------------------
    def open_stops(self, symbol=None):
        return [o for o in self.orders.values() if o["type"].startswith("STOP") and o["status"] == "open"
                and (symbol is None or o["symbol"] == symbol)]

    def trigger(self, symbol, low, fill=True):
        """Mumun en düşüğü stopa değdiyse stop emri dolar (fill=False: tetiklendi ama dolmadı)."""
        for order in self.open_stops(symbol):
            stop = float(order["params"]["stopPrice"])
            if low <= stop and fill:
                cost = order["amount"] * stop
                self.free["USDT"] += cost * (1 - FEE)
                order.update(status="closed", filled=order["amount"], cost=cost, average=stop,
                             fees=[{"cost": cost * FEE, "currency": "USDT"}])


class FakeLiveClient:
    """ExchangeClient yerine: sahte borsa + sahte mum kaynağı + test saati."""

    def __init__(self, exchange: FakeExchange, source=None, now_ms: int = 0,
                 restrictions: dict[str, Any] | None = None):
        self.ex = exchange
        self.source = source
        self.now = now_ms
        self.id = "binance"
        self.time_offset_ms = 0
        self.restrictions = restrictions or {"enableWithdrawals": False, "enableSpotAndMarginTrading": True,
                                             "ipRestrict": True}

    def _call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def milliseconds(self):
        return self.now

    def server_now(self):
        return self.now

    def refresh_time_offset(self):
        return 0

    def api_restrictions(self):
        return dict(self.restrictions)

    def markets(self):
        return self.ex.markets

    def market_rules(self, symbols, default):
        from bot.config import MarketRules
        return {s: MarketRules(qty_step=self.ex.step, min_notional=5.0) for s in symbols}

    def fetch_ohlcv_range(self, symbol, timeframe, since_ms, until_ms, label=""):
        return self.source.fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms, label)
