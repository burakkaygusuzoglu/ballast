"""Risk yöneticisi - stratejinin her önerisi üzerinde veto yetkisine sahiptir.

- Pozisyon büyüklüğü: stop çalışırsa kayıp (giriş + çıkış komisyonu ve kayma DAHİL)
  özsermayenin risk_per_trade kadarı olur; tek pozisyon max_position_pct ile sınırlıdır.
  risk.dd_scaling > 0 ise risk tepeden düşüşle doğrusal azalır (7. tur, Grossman-Zhou).
  risk.sizing = "allocation" ise her pozisyon doğrudan özsermayenin max_position_pct kadarıdır (L1).
- Stoplar: ilk stop = giriş - k x ATR; iz süren stop = en yüksek - m x ATR, yalnızca yukarı.
  Vadeli işlemde short için aynası: giriş + k x ATR; en düşük + m x ATR, yalnızca aşağı.
- Vadeli işlem (izole marjin): pozisyon başına marjin = büyüklük / kaldıraç; marjin ve komisyon serbest
  nakitten karşılanır. Stop, likidasyon fiyatına çok yakınsa işlem açılmaz.
- Şalterler: günlük zarar limiti (UTC gün sonuna kadar yeni işlem yok) ve tepeden
  maksimum düşüş (bot tamamen durur, yalnızca elle sıfırlanır).
- Ayrıca: günlük işlem sınırı, aynı anda açık pozisyon sınırı, çıkış sonrası bekleme.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

from bot.config import Config, MarketRules
from bot.timeutil import fmt_utc, utc_day

LIMIT_CASH_BUFFER = 0.999
MARKET_CASH_BUFFER = 0.98


def liquidation_price(entry: float, side: int, leverage: float, maintenance_margin: float) -> float:
    """İzole marjinde tek pozisyonun likidasyon fiyatı (Binance formülünün 1. kademe sadeleştirmesi).

    Long: marjin + miktar x (LP - giriş) = bakım x miktar x LP  ->  LP = giriş x (1 - 1/k) / (1 - bakım)
    Short: aynası                                               ->  LP = giriş x (1 + 1/k) / (1 + bakım)
    """
    if side < 0:
        return entry * (1 + 1 / leverage) / (1 + maintenance_margin)
    return entry * (1 - 1 / leverage) / (1 - maintenance_margin)


def liquidation_distance(fc, side: int) -> float:
    """Girişten likidasyon fiyatına oransal uzaklık (3x ve %2 bakım marjininde long ~%32, short ~%31)."""
    return abs(liquidation_price(1.0, side, fc.leverage, fc.maintenance_margin) - 1.0)


def floor_to_step(qty: float, step: float) -> float:
    """Miktarı borsanın adım büyüklüğüne AŞAĞI yuvarlar (asla fazla almayız)."""
    if step <= 0:
        return qty
    steps = math.floor(qty / step + 1e-9)
    decimals = max(0, -Decimal(str(step)).normalize().as_tuple().exponent)
    return round(steps * step, decimals)


@dataclass
class RiskState:
    peak_equity: float
    last_equity: float
    day: int | None = None
    day_start_equity: float = 0.0
    trades_today: int = 0
    daily_blocked: bool = False
    halted: bool = False
    halt_reason: str = ""
    last_exit_ts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskState":
        return cls(**data)


class RiskManager:
    def __init__(self, cfg: Config, initial_equity: float, state: RiskState | None = None):
        self.cfg = cfg
        self.rc = cfg.risk
        self.tf_ms = cfg.entry_ms
        self.entry_rate = cfg.entry_fee_rate
        self.exit_rate = cfg.exit_fee_rate
        self.cash_buffer = LIMIT_CASH_BUFFER if cfg.execution.entry_order == "limit" else MARKET_CASH_BUFFER
        self.futures = cfg.futures if cfg.is_futures else None
        self.state = state or RiskState(peak_equity=initial_equity, last_equity=initial_equity)

    # ---- özsermaye takibi ve şalterler -------------------------------------
    def on_bar_close(self, close_ts: int, equity: float) -> list[str]:
        """Her mum kapanışında çağrılır; tetiklenen olayları döndürür."""
        st, events = self.state, []
        day = utc_day(close_ts - 1)   # 00:00'da kapanan mum önceki güne aittir
        if st.day is None:
            st.day, st.day_start_equity = day, st.last_equity
        elif day != st.day:
            st.day = day
            st.day_start_equity = st.last_equity
            st.trades_today = 0
            st.daily_blocked = False
            events.append("new_day")
        st.last_equity = equity
        st.peak_equity = max(st.peak_equity, equity)

        if not st.halted and equity <= st.peak_equity * (1 - self.rc.max_drawdown_halt):
            st.halted = True
            st.halt_reason = (
                f"{fmt_utc(close_ts)}: özsermaye tepeden %{self.rc.max_drawdown_halt * 100:.0f} düştü "
                f"({st.peak_equity:,.2f} -> {equity:,.2f})"
            )
            events.append("halt")
        if (not st.daily_blocked and st.day_start_equity > 0
                and equity <= st.day_start_equity * (1 - self.rc.daily_loss_limit)):
            st.daily_blocked = True
            events.append("daily_block")
        return events

    def can_open(self, symbol: str, bar_ts: int, open_count: int, pending_count: int) -> tuple[bool, str]:
        st, rc = self.state, self.rc
        if st.halted:
            return False, "bot durdu (maks. düşüş şalteri)"
        if st.daily_blocked:
            return False, "günlük zarar limiti"
        if st.trades_today + pending_count >= rc.max_trades_per_day:
            return False, "günlük işlem sınırı"
        if open_count + pending_count >= rc.max_open_positions:
            return False, "maks. açık pozisyon"
        last_exit = st.last_exit_ts.get(symbol)
        if last_exit is not None and bar_ts - last_exit < rc.cooldown_bars * self.tf_ms:
            return False, "çıkış sonrası bekleme"
        return True, ""

    def record_entry(self) -> None:
        self.state.trades_today += 1

    def record_exit(self, symbol: str, bar_ts: int) -> None:
        self.state.last_exit_ts[symbol] = bar_ts

    def reset_halt(self) -> None:
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.peak_equity = self.state.last_equity

    # ---- boyut ve stoplar ----------------------------------------------------
    def initial_stop(self, entry_price: float, atr: float, side: int = 1) -> float:
        if side < 0:
            return entry_price + self.rc.stop_atr_mult * atr
        return entry_price - self.rc.stop_atr_mult * atr

    def trailing_stop(self, current_stop: float, extreme: float, atr: float, side: int = 1) -> float:
        """extreme: long'da girişten beri en yüksek, short'ta en düşük fiyat."""
        if self.rc.trail_atr_mult <= 0:   # 0 = iz süren stop kapalı (çıkışı strateji yönetir)
            return current_stop
        if side < 0:
            return min(current_stop, extreme + self.rc.trail_atr_mult * atr)
        return max(current_stop, extreme - self.rc.trail_atr_mult * atr)

    def effective_risk(self, equity: float) -> float:
        """İşlem başına risk. dd_scaling > 0 ise: taban x max(0, 1 - tepeden düşüş / dd_scaling).

        Tepe, mum kapanışlarında güncellenen en yüksek özsermayedir (şalterle aynı). Bot zirvedeyken taban riski,
        dd_scaling'in yarısı kadar düşüşte yarısını kullanır; sınırda risk sıfırdır.
        """
        base, limit = self.rc.risk_per_trade, self.rc.dd_scaling
        if limit <= 0:
            return base
        peak = self.state.peak_equity
        drawdown = max(0.0, 1 - equity / peak) if peak > 0 else 0.0
        return base * max(0.0, 1 - drawdown / limit)

    def size(self, equity: float, free_cash: float, entry: float, stop: float, rules: MarketRules,
             side: int = 1) -> tuple[float, float]:
        """(miktar, planlanan risk USDT). Uygun boyut yoksa (0, 0)."""
        if side < 0:
            if self.futures is None or equity <= 0 or math.isnan(stop) or not (0 < entry < stop):
                return 0.0, 0.0
        elif equity <= 0 or math.isnan(stop) or not (0 < stop < entry):
            return 0.0, 0.0
        loss_per_unit = abs(entry - stop) + entry * self.entry_rate + stop * self.exit_rate
        if self.futures is None:
            max_notional = min(equity * self.rc.max_position_pct, free_cash / (1 + self.entry_rate) * self.cash_buffer)
        else:
            fc = self.futures
            if abs(entry - stop) / entry > fc.max_stop_to_liq * liquidation_distance(fc, side):
                return 0.0, 0.0
            margin_rate = 1 / fc.leverage + self.entry_rate      # pozisyonun her 1 USDT'si için nakit ihtiyacı
            max_notional = min(equity * self.rc.max_position_pct, free_cash / margin_rate * self.cash_buffer)
        if self.rc.sizing == "allocation":       # sabit pay: özsermayenin max_position_pct kadarı (L1)
            qty = max_notional / entry
        else:
            qty = min(equity * self.effective_risk(equity) / loss_per_unit, max_notional / entry)
        qty = floor_to_step(qty, rules.qty_step)
        if qty <= 0 or qty * entry < rules.min_notional:
            return 0.0, 0.0
        return qty, qty * loss_per_unit
