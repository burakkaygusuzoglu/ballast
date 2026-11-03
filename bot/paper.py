"""Paper trading: canlı piyasa verisi + sanal bakiye.

Backtest ile AYNI sinyal fonksiyonunu, AYNI mum üreticisini, AYNI motoru ve AYNI dolum
modelini kullanır; fark yalnızca mumların gerçek zamanlı gelmesidir. Bu yüzden paper
sonuçları aynı dönemin backtest'iyle karşılaştırılabilir.

Veri bütünlüğü kuralları:
- Bir mum ancak TÜM semboller için kapanmış verisi geldiyse işlenir.
- Trend zaman dilimi kullanan stratejide giriş mumu, ihtiyaç duyduğu trend mumu (o ana kadar
  kapanmış son trend mumu) gelmeden işlenmez. Borsa verisi birkaç saniye gecikirse bot kısa
  aralıklarla tekrar dener.
- İlk başlatmada geçmiş yalnızca göstergelerin ısınması için kullanılır; işlem, başlatma
  anından SONRA kapanan mumlarla başlar.
- Bot kapalıyken kaçırılan mumlar, yeniden başlatıldığında sırayla işlenir.
- Her senkronizasyondan sonra durum SQLite'a atomik olarak yazılır.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from bot.backtester import iter_signal_bars
from bot.brokers.futures import make_simulated_broker
from bot.config import Config, MarketRules
from bot.data import candles_from_rows, drop_unclosed, merge_candles
from bot.engine import TradingEngine
from bot.portfolio import Portfolio
from bot.risk import RiskManager, RiskState
from bot.signals import build_market_signals, uses_trend
from bot.storage import StateStore
from bot.timeutil import fmt_utc

SNAPSHOT_VERSION = 1


class CandleSource(Protocol):
    def fetch_ohlcv_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, label: str = "") -> list[list]: ...


class PaperStateMismatch(RuntimeError):
    pass


BrokerFactory = Callable[[Portfolio, Callable[[], None]], Any]


class PaperTrader:
    """Mum akışını motora besleyen döngü. Varsayılan: paper (simülasyon broker'ı).

    Canlı modda aynı sınıf, broker_factory ile verilen canlı broker'la çalışır (catch_up=False):
    bot kapalıyken kaçırılan mumlar için geriye dönük emir verilmez, yalnızca son mum işlenir.
    """

    def __init__(self, cfg: Config, source: CandleSource, store: StateStore, rules: dict[str, MarketRules],
                 logger: logging.Logger | None = None, broker_factory: BrokerFactory | None = None,
                 catch_up: bool = True, initial_equity: float | None = None, label: str = "Paper trading"):
        self.cfg = cfg
        self.source = source
        self.store = store
        self.rules = rules
        self.log = logger or logging.getLogger("bot.paper")
        self.windows: dict[tuple[str, str], pd.DataFrame] = {}
        self.lagging = False
        self.broker_factory = broker_factory
        self.catch_up = catch_up
        self.label = label
        self.skipped_bars = 0
        self._saved_trades = 0
        self._saved_equity = 0
        start_equity = cfg.paper.initial_equity if initial_equity is None else initial_equity

        snapshot = store.load_snapshot()
        if snapshot is None:
            self.portfolio = Portfolio(start_equity)
            self.broker = self._new_broker()
            self.risk = RiskManager(cfg, start_equity)
            self.last_ts: int | None = None
            self.resumed = False
        else:
            if snapshot.get("version") != SNAPSHOT_VERSION:
                raise PaperStateMismatch("Kayıtlı paper durumu bu sürümle uyumsuz; '--reset-state' ile sıfırlayın.")
            if snapshot.get("setup") != self._setup_fingerprint():
                raise PaperStateMismatch(
                    "Kayıtlı paper durumu farklı sembol/zaman dilimi ayarlarıyla başlatılmış. Eski ayarlara "
                    "dönün ya da '--reset-state' ile sıfırdan başlayın."
                )
            self.portfolio = Portfolio.from_dict(snapshot["portfolio"], trades=store.load_trades())
            self._saved_trades = len(self.portfolio.trades)
            self.broker = self._new_broker()
            self.broker.load_state(snapshot["broker"])
            self.risk = RiskManager(cfg, self.portfolio.initial_cash, RiskState.from_dict(snapshot["risk"]))
            self.last_ts = snapshot["last_ts"]
            self.resumed = True
        self.engine = TradingEngine(cfg, self.portfolio, self.broker, self.risk, rules, self.log, verbose=True)
        if snapshot is not None:
            # Açık pozisyonlar yeniden başlatmada son bilinen fiyatla değerlenir.
            self.engine.last_prices = {k: float(v) for k, v in snapshot.get("last_prices", {}).items()}

    def _new_broker(self):
        if self.broker_factory is not None:
            return self.broker_factory(self.portfolio, self.persist)
        return make_simulated_broker(self.cfg, self.portfolio)

    def persist(self) -> None:
        """Durumu, yeni işlemleri ve özsermaye noktalarını diske yazar (canlıda her emirden sonra)."""
        if not hasattr(self, "engine"):
            return
        trades = self.portfolio.trades[self._saved_trades:]
        points = self.engine.equity_points[self._saved_equity:]
        self.store.save(self.snapshot(), trades, points)
        self._saved_trades += len(trades)
        self._saved_equity += len(points)

    def _setup_fingerprint(self) -> dict[str, Any]:
        s = self.cfg.strategy
        out: dict[str, Any] = {"symbols": list(self.cfg.symbols), "entry": self.cfg.timeframes.entry,
                               "trend": self.cfg.timeframes.trend, "strategy": s.type}
        if s.regime_sma:                          # yalnızca 3. tur ayarları kullanılıyorsa (C1 kayıtları uyumlu kalır)
            out["regime_sma"] = s.regime_sma
        if s.type == "momentum":
            out["momentum"] = [s.momentum_lookback, s.momentum_top_n, s.momentum_exit_rank]
        return out

    # ---- durum -------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "version": SNAPSHOT_VERSION,
            "setup": self._setup_fingerprint(),
            "last_ts": self.last_ts,
            "last_prices": self.engine.last_prices,
            "portfolio": self.portfolio.to_dict(),
            "broker": self.broker.to_dict(),
            "risk": self.risk.state.to_dict(),
        }

    def equity(self) -> float:
        return self.portfolio.equity(self.engine.last_prices)

    # ---- veri ----------------------------------------------------------------
    def _refresh(self, symbol: str, timeframe: str, tf_ms: int, now_ms: int, anchor_ms: int) -> None:
        key = (symbol, timeframe)
        window = self.windows.get(key)
        history = self.cfg.paper.history_bars
        if window is None or window.empty:
            since = anchor_ms - history * tf_ms
        else:
            since = int(window["ts"].iloc[-1]) - 2 * tf_ms   # küçük örtüşme: düzeltilmiş mumları yakala
        rows = self.source.fetch_ohlcv_range(symbol, timeframe, since, now_ms)
        merged = merge_candles(window, drop_unclosed(candles_from_rows(rows), tf_ms, now_ms))
        keep_from = min(anchor_ms, now_ms) - history * tf_ms
        self.windows[key] = merged[merged["ts"] >= keep_from].reset_index(drop=True)

    def _processable_until(self) -> int | None:
        """Tüm semboller için hem giriş hem trend verisi tam olan son giriş mumu (açılış zamanı)."""
        cfg = self.cfg
        limits = []
        for symbol in cfg.symbols:
            w_entry = self.windows.get((symbol, cfg.timeframes.entry))
            if w_entry is None or w_entry.empty:
                return None
            last_entry = int(w_entry["ts"].iloc[-1])
            if not uses_trend(cfg):              # yalnızca giriş zaman dilimini kullanan strateji
                limits.append(last_entry)
                continue
            w_trend = self.windows.get((symbol, cfg.timeframes.trend))
            if w_trend is None or w_trend.empty:
                return None
            last_trend_close = int(w_trend["ts"].iloc[-1]) + cfg.trend_ms
            # Kapanışı t+E olan giriş mumu, kapanışı floor((t+E)/H)*H olan trend mumuna ihtiyaç duyar.
            limits.append(min(last_entry, last_trend_close + cfg.trend_ms - 2 * cfg.entry_ms))
        return min(limits)

    def sync(self, now_ms: int) -> list[int]:
        """Yeni kapanan mumları işler; işlenen mumların zamanlarını döndürür."""
        cfg = self.cfg
        anchor = self.last_ts if self.last_ts is not None else now_ms
        for symbol in cfg.symbols:
            self._refresh(symbol, cfg.timeframes.entry, cfg.entry_ms, now_ms, anchor)
            if uses_trend(cfg):
                self._refresh(symbol, cfg.timeframes.trend, cfg.trend_ms, now_ms, anchor)

        limit = self._processable_until()
        expected_last = (now_ms // cfg.entry_ms) * cfg.entry_ms - cfg.entry_ms   # son kapanmış mum
        self.lagging = limit is None or limit < expected_last
        if limit is None:
            return []

        if self.last_ts is None:
            self.last_ts = limit
            self.store.save(self.snapshot(), [], [])
            self.log.info(f"{self.label} başladı. İşlem, {fmt_utc(limit + cfg.entry_ms)} sonrası kapanan mumlarla başlar.")
            return []
        if limit <= self.last_ts:
            return []

        if not self.catch_up and limit > self.last_ts + cfg.entry_ms:
            missed = (limit - self.last_ts) // cfg.entry_ms - 1
            self.skipped_bars += missed
            self.log.warning(f"Bot kapalıyken {missed} mum geçti; canlı modda geriye dönük emir verilmez, "
                             f"yalnızca son mum işlenir.")
            self.last_ts = limit - cfg.entry_ms
        entry = {symbol: self.windows[(symbol, cfg.timeframes.entry)] for symbol in cfg.symbols}
        trend = ({symbol: self.windows[(symbol, cfg.timeframes.trend)] for symbol in cfg.symbols}
                 if uses_trend(cfg) else None)
        signals = build_market_signals(cfg, entry, trend)
        processed: list[int] = []
        for ts, bars in iter_signal_bars(signals, self.last_ts + 1, limit + 1):
            self.engine.on_bar(ts, bars)
            self.last_ts = ts
            processed.append(ts)
        if processed:
            self.persist()
        return processed

    def next_wakeup_ms(self, now_ms: int) -> int:
        step = self.cfg.entry_ms
        return (now_ms // step + 1) * step + self.cfg.paper.poll_buffer_seconds * 1000
