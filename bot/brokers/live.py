"""Canlı broker: gerçek (ya da Binance testnet) emirleri. Motorla AYNI arayüz.

Backtest/paper'daki simülasyon broker'ından farkları:
- Emirler anında gönderilir: sinyal mum kapanışında verilir, piyasa emri hemen (yani bir sonraki mumun
  açılışında) dolar. Dolumlar motora hemen döner; motor riski aynı anda günceller.
- Koruyucu stop BORSADA bekler (STOP_LOSS ya da STOP_LOSS_LIMIT). Bot kapalıyken ya da internet
  kesildiğinde bile pozisyon korunur.
- Her mumda stop emirlerinin durumu sorgulanır: dolmuşsa işlem kaydedilir; fiyat stopun altına inip
  stop-limit dolmamışsa pozisyon piyasa emriyle kapatılır; stop kaybolmuşsa yeniden konur.
- Alışta Binance komisyonu coin olarak keser; pozisyon miktarı ELE GEÇEN coin miktarıdır.
- Her emirden hemen sonra durum diske yazılır (on_change). Program çökse bile yeniden başlatmada aynı
  alım ikinci kez yapılmaz.
- Küçük pozisyonlar: komisyon ve yuvarlamadan sonra bile stop fiyatında Binance'in minimum işlem tutarının
  üstünde kalacak miktar alınır (gerekirse miktar en fazla %50 artırılır, olmuyorsa sinyal atlanır).
  Böylece satılamayan "toz" pozisyon oluşmaz.
- Satış yapılamazsa (ağ, erişim/IP, borsa bakımı ya da stop iptal edilemedi) pozisyon korunur: stop
  borsada kalır ya da yeniden konur ve satış bir sonraki mumda tekrar denenir. Yalnızca borsa satışı
  kalıcı olarak reddederse (ör. minimum tutarın altında kalan artık) coinler hesapta bırakılır ve
  kullanıcı uyarılır.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import ccxt

from bot.brokers.base import Bar, Broker, EntryOrder, Fill
from bot.brokers.orders import OrderResult, OrderStatusUnknown, base_asset
from bot.config import MarketRules
from bot.portfolio import Portfolio, Position, Trade

CASH_BUFFER = 0.98          # piyasa alışında fiyat oynayabilir: serbest nakdin en fazla %98'i
BALANCE_TOLERANCE = 0.98    # pozisyon miktarının en az %98'i hesapta görünmeli (komisyon/yuvarlama payı)
MIN_NOTIONAL_MARGIN = 1.02  # stop fiyatında bile minimum işlem tutarının %2 üstünde kalmalı
MAX_SIZE_BUMP = 1.5         # minimum tutar için miktar en fazla %50 artırılabilir
DEFAULT_MIN_NOTIONAL = 5.0


class LiveBroker(Broker):
    def __init__(self, portfolio: Portfolio, api: Any, bar_ms: int, taker_fee: float,
                 logger: logging.Logger | None = None, on_change: Callable[[], None] | None = None,
                 rules: dict[str, MarketRules] | None = None):
        self.portfolio = portfolio
        self.api = api
        self.bar_ms = bar_ms
        self.taker_fee = taker_fee
        self.rules = rules or {}
        self.log = logger or logging.getLogger("bot.live")
        self.on_change = on_change or (lambda: None)
        self.stops: dict[str, dict[str, Any]] = {}     # sembol -> {"id", "price", "qty"}
        self.pending_entries: dict[str, EntryOrder] = {}   # canlıda bekleyen giriş yok (emirler anında)
        self.pending_exits: dict[str, str] = {}
        self.cancelled_entries = 0
        self.clock_ts = 0                               # şu an dolacak emirlerin mumu (açılış zamanı)

    # ---- motor arayüzü -------------------------------------------------------
    def has_pending_entry(self, symbol: str) -> bool:
        return False

    def pending_count(self) -> int:
        return 0

    def cancel_entries(self) -> list[str]:
        return []

    def submit_entry(self, order: EntryOrder, ref_price: float) -> list[Fill]:
        symbol = order.symbol
        if symbol in self.portfolio.positions:
            return []
        affordable = self.portfolio.cash * CASH_BUFFER / (ref_price * (1 + self.taker_fee))
        qty = self.api.amount(symbol, min(order.qty, affordable))
        if qty <= 0:
            self.log.warning(f"{symbol}: alış miktarı borsa adımının altında kaldı; işlem atlandı.")
            return []
        qty = self._sellable_size(symbol, qty, order.stop_price)
        if qty is None or qty > affordable:
            self.log.warning(f"{symbol}: pozisyon, komisyon ve yuvarlamadan sonra Binance'in minimum işlem "
                             f"tutarının altında kalacaktı (satılamazdı); işlem atlandı.")
            return []
        try:
            result = self.api.market_buy(symbol, qty)
        except OrderStatusUnknown as exc:  # çok nadir: uzun bağlantı kesintisi; kullanıcı elle kontrol etmeli
            self.log.error(str(exc))
            return []
        except Exception as exc:          # borsa reddi (bakiye, minimum tutar vb.): bu sinyal atlanır
            self.log.error(f"{symbol}: alış emri reddedildi: {exc}")
            return []
        if result.filled <= 0:
            self.log.error(f"{symbol}: alış emri dolmadı (durum: {result.status}).")
            return []
        received = result.filled - result.fee_base
        fee = result.fee_base * result.average + result.fee_quote
        if result.fee_other:
            fee += result.cost * self.taker_fee      # BNB ile ödendiyse USDT karşılığı tahmin edilir
        self.portfolio.cash -= result.cost + result.fee_quote
        self.portfolio.total_fees += fee
        position = Position(
            symbol=symbol, qty=received, entry_price=result.average, entry_ts=self.clock_ts, entry_fee=fee,
            entry_slippage=max(0.0, (result.average - ref_price) * received), stop_price=order.stop_price,
            initial_stop=order.stop_price, highest=result.average,
            risk_amount=order.risk_amount * received / order.qty, entered_at_open=True,
        )
        self.portfolio.positions[symbol] = position
        self.on_change()
        fills = [Fill(symbol, "buy", received, result.average, fee, self.clock_ts, "giriş (piyasa)")]
        if not self._place_stop(symbol, position.stop_price):
            self.log.error(f"{symbol}: koruyucu stop konamadı; pozisyon güvenlik için hemen kapatılıyor.")
            fills += self._sell(symbol, "stop konamadı")
        return fills

    def request_exit(self, symbol: str, reason: str) -> list[Fill]:
        if symbol not in self.portfolio.positions:
            return []
        cancelled = self._cancel_stop(symbol)
        if cancelled is None:             # stop hâlâ borsada (coinler kilitli): pozisyon korunuyor, sonra tekrar
            return self._retry_exit_later(symbol, reason)
        if cancelled:                     # stop bu arada dolmuş
            return cancelled
        return self._sell(symbol, reason)

    def update_stop(self, symbol: str, new_stop: float) -> bool:
        position = self.portfolio.positions.get(symbol)
        if position is None or new_stop <= position.stop_price:
            return False
        old = self.stops.get(symbol)
        if old is not None:
            try:
                self.api.cancel(symbol, old["id"])
            except Exception as exc:
                self.log.warning(f"{symbol}: eski stop iptal edilemedi ({exc}); stop güncellenmedi.")
                return False
        if self._place_stop(symbol, new_stop):
            position.stop_price = new_stop
            return True
        self._place_stop(symbol, position.stop_price)      # yenisi konamadıysa eskisi geri konur
        return False

    def process_bar(self, symbol: str, bar: Bar) -> list[Fill]:
        """Mum kapanışında: borsadaki stopun durumu. Dolmuşsa işlem kaydı; fiyat stopu geçtiyse koruma."""
        self.clock_ts = bar.ts + self.bar_ms
        position = self.portfolio.positions.get(symbol)
        if position is None:
            self.pending_exits.pop(symbol, None)
            return []
        done = self._stop_fill_if_done(symbol, bar.ts)
        if done:
            return done
        if symbol in self.pending_exits:                  # önceki mumda ağ hatasıyla yarım kalan satış
            return self.request_exit(symbol, self.pending_exits[symbol])
        stop = self.stops.get(symbol)
        if stop is None or self._stop_status(symbol) in ("canceled", "expired", "rejected"):
            self.stops.pop(symbol, None)
            if bar.low <= position.stop_price:
                return self._sell(symbol, "stop (borsa stopu yoktu)")
            self.log.warning(f"{symbol}: borsada stop bulunamadı; yeniden konuyor.")
            if not self._place_stop(symbol, position.stop_price):
                return self._sell(symbol, "stop konamadı")
            return []
        if bar.low <= position.stop_price:
            # Stop-limit tetiklendi ama dolmadı (sert düşüş): pozisyonu piyasa emriyle kapat.
            return self.request_exit(symbol, "stop (piyasa)")
        return []

    # ---- yardımcılar ---------------------------------------------------------
    def _stop_status(self, symbol: str) -> str | None:
        stop = self.stops.get(symbol)
        if stop is None:
            return None
        try:
            return self.api.fetch(symbol, stop["id"]).status
        except Exception as exc:
            self.log.warning(f"{symbol}: stop durumu sorgulanamadı ({exc}).")
            return "open"

    def _stop_fill_if_done(self, symbol: str, ts: int) -> list[Fill]:
        stop = self.stops.get(symbol)
        if stop is None:
            return []
        try:
            result = self.api.fetch(symbol, stop["id"])
        except Exception as exc:
            self.log.warning(f"{symbol}: stop durumu sorgulanamadı ({exc}).")
            return []
        if result.status != "closed" or result.filled <= 0:
            return []
        self.stops.pop(symbol, None)
        return [self._record_exit(symbol, result, ts, "stop")]

    def _cancel_stop(self, symbol: str) -> list[Fill] | None:
        """Satıştan önce borsadaki stopu kaldırır. Dönüş: stop bu arada dolduysa çıkış dolumu; iptal edildiyse
        ya da stop yoksa []; iptal edilemedi ve hâlâ açık olabilirse None (coinler kilitli, satılamaz)."""
        stop = self.stops.get(symbol)
        if stop is None:
            return []
        done = self._stop_fill_if_done(symbol, self.clock_ts)
        if done:
            return done
        try:
            self.api.cancel(symbol, stop["id"])
        except Exception as exc:          # bu arada dolmuş ya da zaten iptal edilmiş olabilir
            done = self._stop_fill_if_done(symbol, self.clock_ts)
            if done:
                return done
            if self._stop_status(symbol) == "open":
                self.log.warning(f"{symbol}: stop iptal edilemedi ({exc}); stop borsada kalıyor, satış bir "
                                 f"sonraki mumda tekrar denenecek.")
                return None
        self.stops.pop(symbol, None)
        return []

    def _retry_exit_later(self, symbol: str, reason: str) -> list[Fill]:
        """Satış şimdi yapılamadı: pozisyon korunur (stop yoksa yeniden konur), bir sonraki mumda tekrar denenir."""
        self.pending_exits[symbol] = reason
        if symbol not in self.stops:
            self._place_stop(symbol, self.portfolio.positions[symbol].stop_price)
        self.on_change()
        return []

    def _place_stop(self, symbol: str, stop_price: float) -> bool:
        position = self.portfolio.positions[symbol]
        qty = self.api.amount(symbol, position.qty)
        if qty <= 0:
            return False
        try:
            order_id = self.api.place_stop(symbol, qty, stop_price)
        except Exception as exc:
            self.log.error(f"{symbol}: stop emri reddedildi: {exc}")
            return False
        self.stops[symbol] = {"id": order_id, "price": stop_price, "qty": qty}
        self.on_change()
        return True

    def _sellable_size(self, symbol: str, qty: float, stop_price: float) -> float | None:
        """Alış komisyonu coin olarak kesilip miktar borsa adımına yuvarlandıktan sonra bile, stop fiyatında
        minimum işlem tutarının üstünde kalacak alış miktarı. Gerekirse artırır; olmuyorsa None."""
        rule = self.rules.get(symbol)
        min_notional = rule.min_notional if rule else DEFAULT_MIN_NOTIONAL
        need = min_notional * MIN_NOTIONAL_MARGIN / stop_price

        def sellable(bought: float) -> float:
            return self.api.amount(symbol, bought * (1 - self.taker_fee))

        if sellable(qty) >= need:
            return qty
        step = rule.qty_step if rule else 0.0
        candidate = self.api.amount(symbol, need / (1 - self.taker_fee))
        for _ in range(3):
            if sellable(candidate) >= need:
                break
            candidate = self.api.amount(symbol, candidate + step)
        if sellable(candidate) < need or candidate > qty * MAX_SIZE_BUMP:
            return None
        return candidate

    def _sell(self, symbol: str, reason: str) -> list[Fill]:
        """Pozisyonu piyasa emriyle satar. Borsadaki stop önceden kaldırılmış olmalıdır."""
        position = self.portfolio.positions[symbol]
        try:
            held = self.api.free_balance(base_asset(symbol))
        except Exception as exc:
            self.log.warning(f"{symbol}: bakiye sorgulanamadı ({exc}); satış bir sonraki mumda tekrar denenecek.")
            return self._retry_exit_later(symbol, reason)
        qty = self.api.amount(symbol, min(position.qty, held))
        if qty <= 0:
            self.log.error(f"{symbol}: satılacak bakiye yok (hesapta {held}); pozisyon kayıtlardan düşülüyor.")
            self._forget(symbol)
            return []
        try:
            result = self.api.market_sell(symbol, qty)
        except (ccxt.InvalidOrder, ccxt.InsufficientFunds) as exc:
            # Kalıcı ret (ör. minimum tutarın altında kalan artık): tekrar denemek işe yaramaz.
            self.log.error(f"{symbol}: satış borsa tarafından reddedildi ({exc}). Coinler Binance hesabında "
                           f"kaldı; elle kontrol edin. Pozisyon bot kayıtlarından düşülüyor.")
            self._forget(symbol)
            return []
        except Exception as exc:
            # Geçici sorun (ağ, erişim/IP, borsa bakımı): pozisyon korunur, satış bir sonraki mumda denenir.
            self.log.warning(f"{symbol}: satış gönderilemedi ({exc}); bir sonraki mumda tekrar denenecek.")
            return self._retry_exit_later(symbol, reason)
        self.pending_exits.pop(symbol, None)
        return [self._record_exit(symbol, result, self.clock_ts, reason)]

    def _forget(self, symbol: str) -> None:
        del self.portfolio.positions[symbol]
        self.stops.pop(symbol, None)
        self.pending_exits.pop(symbol, None)
        self.on_change()

    def _record_exit(self, symbol: str, result: OrderResult, ts: int, reason: str) -> Fill:
        position = self.portfolio.positions.pop(symbol)
        fee = result.fee_quote + result.fee_base * result.average
        if result.fee_other:
            fee += result.cost * self.taker_fee
        proceeds = result.cost - result.fee_quote
        self.portfolio.cash += proceeds
        self.portfolio.total_fees += fee
        qty = result.filled
        gross = qty * (result.average - position.entry_price)
        entry_fee = position.entry_fee * min(1.0, qty / position.qty) if position.qty > 0 else position.entry_fee
        net = gross - entry_fee - fee
        self.portfolio.trades.append(Trade(
            symbol=symbol, entry_ts=position.entry_ts, exit_ts=ts, qty=qty, entry_price=position.entry_price,
            exit_price=result.average, entry_fee=entry_fee, exit_fee=fee, slippage_cost=position.entry_slippage,
            gross_pnl=gross, net_pnl=net, risk_amount=position.risk_amount,
            r_multiple=net / position.risk_amount if position.risk_amount > 0 else 0.0, exit_reason=reason,
        ))
        self.on_change()
        return Fill(symbol, "sell", qty, result.average, fee, ts, reason)

    # ---- kalıcılık ve mutabakat ---------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"stops": self.stops, "pending_exits": self.pending_exits, "cancelled_entries": self.cancelled_entries}

    def load_state(self, data: dict[str, Any]) -> None:
        self.stops = {s: dict(v) for s, v in (data.get("stops") or {}).items()}
        self.pending_exits = dict(data.get("pending_exits") or {})
        self.cancelled_entries = int(data.get("cancelled_entries", 0))

    def reconcile(self) -> list[str]:
        """Yeniden başlatmada kayıtlarla borsayı karşılaştırır; uyarıları döndürür."""
        notes = []
        for symbol, position in list(self.portfolio.positions.items()):
            if self._stop_fill_if_done(symbol, self.clock_ts):
                notes.append(f"{symbol}: bot kapalıyken stop çalışmış; işlem kaydedildi.")
                continue
            status = self._stop_status(symbol)
            # Açık stop emri coinleri kilitler: serbest bakiyeye yalnızca gerçekten açık stopun miktarı eklenir.
            locked = (self.stops.get(symbol) or {}).get("qty", 0.0) if status == "open" else 0.0
            held = self.api.free_balance(base_asset(symbol)) + locked
            if held < position.qty * BALANCE_TOLERANCE:
                notes.append(f"{symbol}: hesapta pozisyon kadar coin yok ({held} < {position.qty}); kayıttan düşüldü.")
                del self.portfolio.positions[symbol]
                self.stops.pop(symbol, None)
                continue
            if status != "open":
                self.stops.pop(symbol, None)
                if self._place_stop(symbol, position.stop_price):
                    notes.append(f"{symbol}: borsadaki stop yeniden kondu.")
                else:
                    notes.append(f"{symbol}: stop konamadı; pozisyon bir sonraki mumda korunarak kapatılacak.")
        self.on_change()
        return notes
