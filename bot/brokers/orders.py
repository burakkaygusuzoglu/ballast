"""Binance spot emirleri için ince ve test edilebilir katman (ccxt üzerinde).

Canlı broker borsayla YALNIZCA bu sınıf üzerinden konuşur; testlerde yerine sahte bir sınıf konur.

Güvenlik:
- Her emir kendine özgü bir müşteri kimliğiyle (clientOrderId) gönderilir. Ağ hatasında emir
  körlemesine tekrar gönderilmez: önce o kimlikle borsada emir var mı diye bakılır. Emir ancak borsada
  KESİNLİKLE olmadığı anlaşıldığında (yoldaki bir emrin artık işleme alınamayacağı süre, recvWindow,
  geçtikten sonra) yeniden gönderilir. Emin olunamazsa hiç gönderilmez (OrderStatusUnknown). Böylece
  bağlantı koparsa aynı alım iki kez yapılmaz.
- Miktarlar borsanın adımına AŞAĞI yuvarlanır (ccxt amount_to_precision).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import ccxt

QUOTE = "USDT"


@dataclass(frozen=True)
class OrderResult:
    id: str
    status: str          # "open" | "closed" | "canceled" | "expired" | "rejected"
    filled: float        # dolan miktar (coin)
    average: float       # ortalama dolum fiyatı (USDT)
    cost: float          # dolan tutar (USDT)
    fee_base: float      # coin cinsinden ödenen komisyon (alışlarda Binance varsayılanı)
    fee_quote: float     # USDT cinsinden ödenen komisyon (satışlarda Binance varsayılanı)
    fee_other: float     # başka varlıkla (ör. BNB) ödenen komisyon miktarı


class OrderStatusUnknown(ccxt.NetworkError):
    """Emir gönderildi ama borsaya ulaşıp ulaşmadığı doğrulanamadı; güvenlik için tekrar gönderilmedi."""


def base_asset(symbol: str) -> str:
    return symbol.split("/")[0]


def parse_order(order: dict[str, Any], symbol: str) -> OrderResult:
    base = base_asset(symbol)
    fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
    fee_base = fee_quote = fee_other = 0.0
    for fee in fees:
        if not fee or fee.get("cost") is None:
            continue
        cost, currency = float(fee["cost"]), fee.get("currency")
        if currency == base:
            fee_base += cost
        elif currency == QUOTE:
            fee_quote += cost
        else:
            fee_other += cost
    filled = float(order.get("filled") or 0.0)
    cost = float(order.get("cost") or 0.0)
    average = order.get("average")
    average = float(average) if average else (cost / filled if filled > 0 else 0.0)
    return OrderResult(str(order.get("id")), str(order.get("status") or "open"), filled, average, cost,
                       fee_base, fee_quote, fee_other)


class SpotOrderAPI:
    STOP_LIMIT_GAP = 0.01      # yalnızca stop-limit kullanılabilen paritelerde: limit = stop x (1 - %1)
    ORDER_ATTEMPTS = 3
    RECV_WINDOW_S = 10.0       # ccxt'nin Binance recvWindow'u: bundan eski bir emir borsada işleme alınmaz
    LOOKUP_DELAYS = (2.0, 4.0, 5.0, 5.0, 10.0, 10.0, 15.0, 15.0, 30.0, 30.0)   # toplam ~2 dk

    def __init__(self, client: Any):
        self.client = client
        self.ex = client.ex

    # ---- yuvarlama ve bakiye -------------------------------------------------
    def amount(self, symbol: str, qty: float) -> float:
        try:
            return float(self.ex.amount_to_precision(symbol, qty))
        except ccxt.InvalidOrder:      # adımdan küçük miktar
            return 0.0

    def price(self, symbol: str, price: float) -> float:
        return float(self.ex.price_to_precision(symbol, price))

    def free_balance(self, asset: str) -> float:
        balance = self.client._call(self.ex.fetch_balance)
        return float((balance.get("free") or {}).get(asset) or 0.0)

    # ---- emirler -------------------------------------------------------------
    def _create(self, symbol: str, type_: str, side: str, qty: float, price: float | None = None,
                params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Emri gönderir. Ağ hatasında emir borsaya ulaşmış olabilir: aynı müşteri kimliğiyle borsada aranır;
        bulunursa o emir döner, kesinlikle yoksa yeniden gönderilir, emin olunamazsa OrderStatusUnknown."""
        cid = f"kb{uuid.uuid4().hex[:24]}"
        full = {**(params or {}), "newClientOrderId": cid}
        last_error: Exception | None = None
        for _ in range(self.ORDER_ATTEMPTS):
            try:
                return self.ex.create_order(symbol, type_, side, qty, price, full)
            except ccxt.NetworkError as exc:
                last_error = exc
            found = self._find_sent_order(symbol, cid)
            if found is not None:
                return found
        raise ccxt.NetworkError(f"{symbol}: emir {self.ORDER_ATTEMPTS} denemede gönderilemedi: {last_error}")

    def _find_sent_order(self, symbol: str, cid: str) -> dict[str, Any] | None:
        """Ağ hatasıyla biten bir gönderimden sonra: emir borsaya ulaştıysa onu, ulaşmadığı KESİNSE None döndürür.

        "Bulunamadı" cevabı ancak recvWindow süresi geçtikten sonra kesin sayılır (daha önce, yoldaki bir emir
        hâlâ işleme alınabilir). Borsaya hiç ulaşılamazsa emir yeniden gönderilmez: OrderStatusUnknown atılır.
        """
        waited = 0.0
        for delay in self.LOOKUP_DELAYS:
            time.sleep(delay)
            waited += delay
            try:
                return self.ex.fetch_order(None, symbol, {"origClientOrderId": cid})
            except ccxt.OrderNotFound:
                if waited >= self.RECV_WINDOW_S:
                    return None
            except ccxt.NetworkError:
                continue
        raise OrderStatusUnknown(
            f"{symbol}: emir gönderildi ama borsaya ulaşıp ulaşmadığı doğrulanamadı (müşteri kimliği {cid}). "
            "Güvenlik için tekrar gönderilmedi; Binance -> Emirler -> Spot geçmişinden kontrol edin."
        )

    def _settle(self, order: dict[str, Any], symbol: str) -> OrderResult:
        """Piyasa emri genelde anında dolar; dolmadıysa kısa aralıklarla durumunu sorgular."""
        result = parse_order(order, symbol)
        for _ in range(10):
            if result.status in ("closed", "canceled", "expired", "rejected"):
                break
            time.sleep(0.5)
            result = self.fetch(symbol, result.id)
        return result

    def market_buy(self, symbol: str, qty: float) -> OrderResult:
        return self._settle(self._create(symbol, "market", "buy", qty), symbol)

    def market_sell(self, symbol: str, qty: float) -> OrderResult:
        return self._settle(self._create(symbol, "market", "sell", qty), symbol)

    def place_stop(self, symbol: str, qty: float, stop_price: float) -> str:
        """Borsa tarafında koruyucu satış stopu. Pariteye göre STOP_LOSS (piyasa) ya da STOP_LOSS_LIMIT."""
        stop = self.price(symbol, stop_price)
        order_types = (self.ex.market(symbol).get("info") or {}).get("orderTypes") or []
        if "STOP_LOSS" in order_types:
            order = self._create(symbol, "STOP_LOSS", "sell", qty, None, {"stopPrice": stop})
        else:
            limit = self.price(symbol, stop_price * (1 - self.STOP_LIMIT_GAP))
            order = self._create(symbol, "STOP_LOSS_LIMIT", "sell", qty, limit,
                                 {"stopPrice": stop, "timeInForce": "GTC"})
        return str(order["id"])

    def cancel(self, symbol: str, order_id: str) -> None:
        self.client._call(self.ex.cancel_order, order_id, symbol)

    def fetch(self, symbol: str, order_id: str) -> OrderResult:
        return parse_order(self.client._call(self.ex.fetch_order, order_id, symbol), symbol)
