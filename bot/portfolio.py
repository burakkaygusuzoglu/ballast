"""Portföy muhasebesi: serbest nakit, bekleyen emirler için ayrılmış nakit, açık pozisyonlar
ve kapanmış işlemler. Özsermaye = nakit + ayrılmış nakit + pozisyonların değeri.

Spot pozisyonun değeri = miktar x fiyat. Vadeli (izole marjin) pozisyonun değeri = marjin + gerçekleşmemiş
kâr/zarar; long'da miktar x (fiyat - giriş), short'ta miktar x (giriş - fiyat)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float        # kayma dahil gerçekleşen fiyat
    entry_ts: int             # girişin dolduğu mumun açılış zamanı
    entry_fee: float
    entry_slippage: float     # girişte kaymanın USDT maliyeti
    stop_price: float         # güncel stop (yalnızca yukarı taşınır)
    initial_stop: float
    highest: float            # iz süren stop referansı
    risk_amount: float        # planlanan risk (USDT) - R hesabı için
    entered_at_open: bool = False
    units: int = 1            # piramitleme: pozisyondaki birim sayısı
    last_add_price: float = 0.0   # son birimin dolum fiyatı (0 = giriş fiyatı)
    unit_atr: float = 0.0     # ilk alımdaki ATR: birim ekleme adımı ve stop mesafesi bununla ölçülür
    # Vadeli işlem (spotta varsayılanlar kalır):
    side: int = 1             # 1 = long, -1 = short
    leverage: float = 0.0     # 0 = spot
    margin: float = 0.0       # izole marjin (USDT)
    liq_price: float = 0.0    # likidasyon fiyatı
    lowest: float = 0.0       # short'ta iz süren stop referansı
    funding_paid: float = 0.0  # ödenen funding toplamı (negatif = alınan)
    funding_ts: int = 0       # funding'in en son hesaplandığı an (UTC ms)

    @property
    def is_futures(self) -> bool:
        return self.leverage > 0

    def unrealized(self, price: float) -> float:
        """Gerçekleşmemiş kâr/zarar (komisyon ve funding hariç)."""
        if self.is_futures:
            return self.side * self.qty * (price - self.entry_price)
        return self.qty * (price - self.entry_price)

    def value(self, price: float) -> float:
        """Pozisyonun özsermayeye katkısı."""
        if self.is_futures:
            return self.margin + self.side * self.qty * (price - self.entry_price)
        return self.qty * price


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_ts: int
    exit_ts: int
    qty: float
    entry_price: float
    exit_price: float
    entry_fee: float
    exit_fee: float
    slippage_cost: float      # giriş + çıkış kaymasının USDT maliyeti (fiyatlara dahildir)
    gross_pnl: float          # komisyon öncesi, kayma dahil fiyatlarla
    net_pnl: float            # komisyon ve kayma sonrası
    risk_amount: float        # planlanan risk (1R, USDT)
    r_multiple: float         # net kâr / planlanan risk
    exit_reason: str
    side: int = 1             # 1 = long, -1 = short (vadeli işlem)
    funding: float = 0.0      # işlem boyunca ödenen funding (net kâra dahil; negatif = alınan)


class Portfolio:
    def __init__(self, initial_cash: float):
        if initial_cash <= 0:
            raise ValueError("Başlangıç bakiyesi pozitif olmalı.")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.reserved = 0.0
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.total_fees = 0.0
        self.total_funding = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash + self.reserved
        for symbol, position in self.positions.items():
            value += position.value(prices.get(symbol, position.entry_price))
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "reserved": self.reserved,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "positions": [asdict(p) for p in self.positions.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], trades: list[Trade] | None = None) -> Portfolio:
        portfolio = cls(data["initial_cash"])
        portfolio.cash = float(data["cash"])
        portfolio.reserved = float(data["reserved"])
        portfolio.total_fees = float(data["total_fees"])
        portfolio.total_funding = float(data.get("total_funding", 0.0))
        portfolio.positions = {p["symbol"]: Position(**p) for p in data["positions"]}
        portfolio.trades = list(trades or [])
        return portfolio
