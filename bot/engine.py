"""İşlem motoru: her KAPANAN mumda sırasıyla
  1) bekleyen emirler ve stoplar bu mumun fiyat hareketine karşı işlenir,
  2) açık pozisyonların iz süren stopu, rejim çıkışı ve stratejinin çıkış sinyali işlenir,
  3) özsermaye piyasa değeriyle hesaplanır, risk şalterleri kontrol edilir,
  4) yeni sinyaller risk yöneticisinin onayıyla emre dönüşür (bir sonraki mumda işlenir).

Backtest ve paper trading bu motoru AYNEN kullanır; fark yalnızca mumların kaynağıdır.
Vadeli işlemde (6. tur) motor short pozisyonları da yönetir: short girişi short_signal ile, çıkışı
short_exit_signal ile; stoplar ve iz süren stop aynalanır. Spotta bu sütunlar yoktur (hep yanlış).
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass

from bot.brokers.base import Bar, EntryOrder, Fill
from bot.brokers.simulated import SimulatedBroker
from bot.config import Config, MarketRules
from bot.portfolio import Portfolio
from bot.risk import RiskManager, floor_to_step
from bot.timeutil import fmt_utc

SIGNAL_FIELDS = ("ts", "open", "high", "low", "close", "atr", "entry_signal", "regime_ok", "exit_signal", "priority",
                 "short_signal", "short_exit_signal")


@dataclass(frozen=True)
class SignalBar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    atr: float
    entry_signal: bool
    regime_ok: bool
    exit_signal: bool = False   # strateji "kapanışta çık" diyorsa: sonraki mumun açılışında satılır
    priority: float = 0.0       # boş yerden fazla sinyal varsa büyük olan önce (eşitlikte sembol adı)
    short_signal: bool = False        # vadeli işlem: short girişi (kapanışta karar, sonraki açılışta dolum)
    short_exit_signal: bool = False   # vadeli işlem: short çıkışı

    def bar(self) -> Bar:
        return Bar(self.ts, self.open, self.high, self.low, self.close)


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


class TradingEngine:
    def __init__(
        self,
        cfg: Config,
        portfolio: Portfolio,
        broker: SimulatedBroker,
        risk: RiskManager,
        rules: dict[str, MarketRules],
        logger: logging.Logger | None = None,
        verbose: bool = False,
    ):
        self.cfg = cfg
        self.portfolio = portfolio
        self.broker = broker
        self.risk = risk
        self.rules = rules
        self.log = logger or logging.getLogger("bot.engine")
        self.verbose = verbose
        self.equity_points: list[tuple[int, float]] = []
        self.last_prices: dict[str, float] = {}
        self.skipped: Counter[str] = Counter()
        self.signals_seen = 0
        self.orders_submitted = 0
        self.exposure_bars = 0
        self.bars_processed = 0
        self.halted_at: int | None = None
        self.events: list[tuple[int, str]] = []
        self.adds_submitted = 0

    def on_bar(self, ts: int, bars: dict[str, SignalBar]) -> list[Fill]:
        close_ts = ts + self.cfg.entry_ms
        fills_all: list[Fill] = []

        # 1) Dolumlar ve stoplar.
        for symbol in sorted(bars):
            sb = bars[symbol]
            for fill in self.broker.process_bar(symbol, sb.bar()):
                if fill.is_entry:
                    self.risk.record_entry()
                else:
                    self.risk.record_exit(symbol, ts)
                self._log_fill(fill)
                fills_all.append(fill)
            self.last_prices[symbol] = sb.close

        # 2) Açık pozisyonlar: iz süren stop ve rejim çıkışı (karar kapanışta, uygulama sonraki mumda).
        for symbol, position in list(self.portfolio.positions.items()):
            sb = bars.get(symbol)
            if sb is None:
                continue
            # Giriş mumunun tepesi dolumdan önce oluşmuş olabilir; o mumda kapanışı kullanırız.
            if position.side < 0:                      # short (vadeli): aynası, en düşük fiyat izlenir
                reference = sb.close if position.entry_ts == ts else sb.low
                position.lowest = min(position.lowest, reference)
                if not math.isnan(sb.atr):
                    self.broker.update_stop(
                        symbol, self.risk.trailing_stop(position.stop_price, position.lowest, sb.atr, side=-1))
                if sb.short_exit_signal and symbol in self.portfolio.positions:
                    fills_all += self._apply(self.broker.request_exit(symbol, "çıkış sinyali"))
                continue
            reference = sb.close if position.entry_ts == ts else sb.high
            position.highest = max(position.highest, reference)
            if not math.isnan(sb.atr):
                self.broker.update_stop(symbol, self.risk.trailing_stop(position.stop_price, position.highest, sb.atr))
            if self.cfg.strategy.exit_on_regime_break and not sb.regime_ok:
                fills_all += self._apply(self.broker.request_exit(symbol, "rejim bozuldu"))
            if sb.exit_signal and symbol in self.portfolio.positions:
                fills_all += self._apply(self.broker.request_exit(symbol, "çıkış sinyali"))
            self._maybe_add_unit(symbol, sb)

        # 3) Özsermaye ve şalterler.
        equity = self.portfolio.equity(self.last_prices)
        self.equity_points.append((close_ts, equity))
        self.bars_processed += 1
        if self.portfolio.positions:
            self.exposure_bars += 1
        for event in self.risk.on_bar_close(close_ts, equity):
            fills_all += self._handle_risk_event(event, close_ts)

        # 4) Yeni girişler (bir sonraki mumda işlenecek emirler). Öncelik: güçlü sinyal önce.
        for symbol in sorted(bars, key=lambda s: (-_finite(bars[s].priority), s)):
            sb = bars[symbol]
            if not (sb.entry_signal or sb.short_signal):
                continue
            self.signals_seen += 1
            if symbol in self.portfolio.positions or self.broker.has_pending_entry(symbol):
                self.skipped["zaten pozisyonda/emirde"] += 1
                continue
            allowed, reason = self.risk.can_open(symbol, ts, len(self.portfolio.positions), self.broker.pending_count())
            if not allowed:
                self.skipped[reason] += 1
                continue
            fills_all += self._submit_entry(symbol, sb, equity, side=1 if sb.entry_signal else -1)
        return fills_all

    # ---- yardımcılar -------------------------------------------------------
    def _apply(self, fills: list[Fill] | None) -> list[Fill]:
        """Canlı broker emirleri anında doldurur: dolumlar risk yöneticisine hemen işlenir.
        (Simülasyon broker'ı burada hiçbir şey döndürmez; dolumları bir sonraki mumda bildirir.)"""
        for fill in fills or []:
            if fill.is_entry:
                self.risk.record_entry()
            else:
                self.risk.record_exit(fill.symbol, fill.ts)
            self._log_fill(fill)
        return list(fills or [])

    def _submit_entry(self, symbol: str, sb: SignalBar, equity: float, side: int = 1) -> list[Fill]:
        if math.isnan(sb.atr) or sb.atr <= 0:
            self.skipped["ATR geçersiz"] += 1
            return []
        use_limit = self.cfg.execution.entry_order == "limit"
        entry_price = sb.close
        stop = self.risk.initial_stop(entry_price, sb.atr, side)
        rules = self.rules.get(symbol, self.cfg.market_rules_default)
        qty, risk_amount = self.risk.size(equity, self.portfolio.cash, entry_price, stop, rules, side)
        if qty <= 0:
            self.skipped["boyut çok küçük (min. tutar)"] += 1
            return []
        order = EntryOrder(
            symbol=symbol,
            qty=qty,
            limit_price=entry_price if use_limit else None,
            stop_price=stop,
            risk_amount=risk_amount,
            placed_ts=sb.ts,
            ttl_bars=self.cfg.execution.entry_ttl_bars,
            unit_atr=sb.atr,
            side=side,
        )
        if self.verbose:
            kind = f"limit {entry_price:,.2f}" if use_limit else "piyasa"
            action = "alış" if side > 0 else "short"
            self.log.info(
                f"EMİR   {fmt_utc(sb.ts + self.cfg.entry_ms)} | {symbol} {action} {qty:.6f} ({kind}) | "
                f"stop {stop:,.2f} | risk {risk_amount:.2f} USDT"
            )
        fills = self.broker.submit_entry(order, ref_price=entry_price)
        self.orders_submitted += 1
        return self._apply(fills)

    def _maybe_add_unit(self, symbol: str, sb: SignalBar) -> None:
        """Piramitleme (Kaplumbağa): fiyat son birimden pyramid_step_atr x ATR yükseldiyse bir birim daha.

        Karar mum kapanışında, dolum bir sonraki mumun açılışında (giriş emirleriyle aynı zamanlama). Her birim
        kendi stopuna göre risk_per_trade kadar risk taşır; eklemeden sonra stop son birimin stop_atr_mult x ATR
        altına çekilir. ATR, ilk alımdaki değerdir.
        """
        rc = self.cfg.risk
        if rc.pyramid_units <= 1 or not getattr(self.broker, "supports_adds", False):
            return
        position = self.portfolio.positions.get(symbol)
        if position is None or self.broker.has_pending_exit(symbol) or symbol in self.broker.pending_adds:
            return
        if position.units >= rc.pyramid_units or position.unit_atr <= 0:
            return
        st = self.risk.state
        if st.halted or st.daily_blocked or st.trades_today >= rc.max_trades_per_day:
            return
        total_units = sum(p.units for p in self.portfolio.positions.values()) + self.broker.pending_add_count()
        if total_units >= rc.max_total_units:
            return
        last = position.last_add_price or position.entry_price
        if sb.close < last + rc.pyramid_step_atr * position.unit_atr:
            return
        distance = rc.stop_atr_mult * position.unit_atr
        unit_stop = sb.close - distance
        equity = self.portfolio.equity(self.last_prices)
        rules = self.rules.get(symbol, self.cfg.market_rules_default)
        qty, risk_amount = self.risk.size(equity, self.portfolio.cash, sb.close, unit_stop, rules)
        room = equity * rc.max_position_pct - position.qty * sb.close    # tek pozisyon sınırı (birimler dahil)
        if qty > 0 and qty * sb.close > room:
            scaled = floor_to_step(max(room, 0.0) / sb.close, rules.qty_step)
            risk_amount *= scaled / qty
            qty = scaled
        if qty <= 0 or qty * sb.close < rules.min_notional:
            self.skipped["birim ekleme: boyut çok küçük"] += 1
            return
        order = EntryOrder(symbol=symbol, qty=qty, limit_price=None, stop_price=unit_stop, risk_amount=risk_amount,
                           placed_ts=sb.ts, ttl_bars=1, unit_atr=position.unit_atr, is_add=True,
                           stop_distance=distance)
        try:
            self.broker.submit_add(order, ref_price=sb.close)
        except ValueError:                       # nakit yetersiz: ekleme yapılmaz
            self.skipped["birim ekleme: nakit yetersiz"] += 1
            return
        self.adds_submitted += 1
        if self.verbose:
            self.log.info(f"EKLEME {fmt_utc(sb.ts + self.cfg.entry_ms)} | {symbol} {position.units + 1}. birim "
                          f"{qty:.6f} | yeni stop ~{unit_stop:,.2f}")

    def _handle_risk_event(self, event: str, close_ts: int) -> list[Fill]:
        self.events.append((close_ts, event))
        fills: list[Fill] = []
        if event == "halt":
            self.halted_at = close_ts
            self.broker.cancel_entries()
            for symbol in list(self.portfolio.positions):
                fills += self._apply(self.broker.request_exit(symbol, "maks. düşüş şalteri"))
            self.log.warning(f"ŞALTER  {self.risk.state.halt_reason}")
        elif event == "daily_block":
            self.broker.cancel_entries()
            if self.verbose:
                self.log.warning(f"GÜNLÜK LİMİT  {fmt_utc(close_ts)}: yeni işlem UTC gün sonuna kadar kapalı.")
        return fills

    def _log_fill(self, fill: Fill) -> None:
        if not self.verbose:
            return
        side = "ALIŞ " if fill.side == "buy" else "SATIŞ"
        if fill.entry is not None:              # vadeli işlem: açılış/kapanış yönüyle
            side = f"{side} ({'açılış' if fill.entry else 'kapanış'})"
        self.log.info(
            f"DOLUM  {fmt_utc(fill.ts)} | {fill.symbol} {side} {fill.qty:.6f} @ {fill.price:,.2f} | "
            f"komisyon {fill.fee:.2f} | {fill.reason}"
        )
