"""Broker arayüzü: motor emirleri bu sözleşme üzerinden verir.

Simülasyon (backtest + paper) ve canlı broker aynı arayüzü uygular; motor hangisiyle
çalıştığını bilmez. Böylece test edilen kod ile çalışan kod aynı kalır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Bar:
    ts: int          # mumun açılış zamanı (UTC ms)
    open: float
    high: float
    low: float
    close: float


@dataclass
class EntryOrder:
    symbol: str
    qty: float
    limit_price: float | None   # None = piyasa emri
    stop_price: float
    risk_amount: float
    placed_ts: int              # sinyalin geldiği mum
    ttl_bars: int
    reserved: float = 0.0
    bars_waited: int = 0
    unit_atr: float = 0.0       # sinyal anındaki ATR (piramitleme için pozisyona yazılır)
    is_add: bool = False        # piramitleme: açık pozisyona birim ekleme emri
    stop_distance: float = 0.0  # ekleme dolunca stop = dolum fiyatı - bu mesafe (yalnızca yukarı)
    side: int = 1               # 1 = long (alış), -1 = short (vadeli işlemde açığa satış)


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str        # "buy" | "sell"
    qty: float
    price: float
    fee: float
    ts: int          # dolumun gerçekleştiği mum
    reason: str
    entry: bool | None = None   # pozisyon açan dolum mu? None: spotta alış = giriş, satış = çıkış

    @property
    def is_entry(self) -> bool:
        return self.side == "buy" if self.entry is None else self.entry


class Broker(ABC):
    """submit_entry ve request_exit, emri anında dolduran broker'larda (canlı) dolumları döndürür;
    simülasyon broker'ı None döndürür ve dolumları bir sonraki mumun process_bar çağrısında bildirir."""

    @abstractmethod
    def submit_entry(self, order: EntryOrder, ref_price: float) -> list[Fill] | None: ...

    @abstractmethod
    def request_exit(self, symbol: str, reason: str) -> list[Fill] | None: ...

    @abstractmethod
    def update_stop(self, symbol: str, new_stop: float) -> bool: ...

    @abstractmethod
    def cancel_entries(self) -> list[str]: ...

    @abstractmethod
    def has_pending_entry(self, symbol: str) -> bool: ...

    @abstractmethod
    def pending_count(self) -> int: ...

    @abstractmethod
    def process_bar(self, symbol: str, bar: Bar) -> list[Fill]: ...
