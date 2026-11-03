"""Strateji seçimi: config.yaml'daki strategy.type'a göre doğru sinyal fonksiyonunu çağırır.

Backtest, paper trading ve araştırma araçları sinyalleri yalnızca buradan alır; böylece
hangi strateji seçilirse seçilsin hepsi birebir aynı kodu kullanır.

- breakout: kanal kırılımı (trend takibi, Kaplumbağa mantığı) - yalnızca giriş zaman dilimi.
  strategy.regime_sma > 0 ise yeni alım yalnızca BTC günlük kapanışı SMA'nın üstündeyken (3. turdaki R1).
  strategy.allow_short (yalnızca vadeli işlem): aynalı short tarafı; rejim filtresi açıksa short yalnızca BTC
  günlük kapanışı SMA'nın altındayken (6. turdaki F-R1S).
- momentum: günlük momentum rotasyonu, coinler arası sıralama + BTC rejim filtresi (3. turdaki R3).
- pullback: 3 katmanlı geri çekilme stratejisi - giriş + trend zaman dilimi
"""
from __future__ import annotations

import pandas as pd

from bot.candidates import (
    BreakoutParams,
    MomentumParams,
    add_short_side,
    apply_regime,
    apply_regime_two_sided,
    breakout_signals,
    btc_regime,
    momentum_signals,
)
from bot.config import REGIME_SYMBOL, Config
from bot.strategy import CONDITIONS as PULLBACK_CONDITIONS
from bot.strategy import compute_signals

BREAKOUT_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("c_ready", "Göstergeler hazır"),
    ("c_breakout", "Kanal kırılımı"),
    ("c_cost", "Maliyet filtresi"),
)
REGIME_CONDITION = ("c_regime", "BTC rejimi olumlu")
MOMENTUM_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("c_ready", "Göstergeler hazır"),
    ("c_top", "En güçlü coinler arasında"),
    ("c_positive", "Getiri pozitif"),
    REGIME_CONDITION,
    ("c_cost", "Maliyet filtresi"),
)


def uses_trend(cfg: Config) -> bool:
    """Strateji trend zaman dilimi verisine ihtiyaç duyuyor mu? (Rejim filtresi günlük BTC mumu kullanır.)"""
    return cfg.strategy.type == "pullback" or (cfg.strategy.type == "breakout" and cfg.strategy.regime_sma > 0)


def breakout_params(cfg: Config) -> BreakoutParams:
    s = cfg.strategy
    return BreakoutParams(entry_n=s.channel_entry, exit_n=s.channel_exit, atr_n=s.channel_atr)


def momentum_params(cfg: Config) -> MomentumParams:
    s = cfg.strategy
    return MomentumParams(lookback=s.momentum_lookback, top_n=s.momentum_top_n, exit_rank=s.momentum_exit_rank,
                          atr_n=s.channel_atr, sma_n=s.regime_sma)


def build_signals(cfg: Config, entry: pd.DataFrame, trend: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tek sembol için sinyaller (rejim filtresi ve momentum coinler arası olduğu için build_market_signals)."""
    s = cfg.strategy
    if s.type == "momentum" or (s.type == "breakout" and s.regime_sma):
        raise ValueError("Rejim filtresi ve momentum tüm coinlerin verisine ihtiyaç duyar: build_market_signals.")
    if s.type == "breakout":
        sig = breakout_signals(entry, breakout_params(cfg), cfg)
        return add_short_side(sig, breakout_params(cfg)) if s.allow_short else sig
    if trend is None:
        raise ValueError("pullback stratejisi trend zaman dilimi verisine ihtiyaç duyar.")
    return compute_signals(entry, trend, cfg)


def build_market_signals(cfg: Config, entry: dict[str, pd.DataFrame],
                         trend: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Tüm semboller için sinyaller. entry/trend: sembol -> mumlar (giriş ve trend zaman dilimi)."""
    s = cfg.strategy
    if s.type == "momentum":
        return momentum_signals(entry, entry[REGIME_SYMBOL], momentum_params(cfg), cfg)
    if s.type == "breakout" and s.regime_sma:
        if trend is None or REGIME_SYMBOL not in trend:
            raise ValueError(f"Rejim filtresi {REGIME_SYMBOL} günlük mumlarına ihtiyaç duyar.")
        regime = btc_regime(trend[REGIME_SYMBOL], s.regime_sma)
        bp = breakout_params(cfg)
        if s.allow_short:
            return {sym: apply_regime_two_sided(add_short_side(breakout_signals(df, bp, cfg), bp), regime,
                                                cfg.entry_ms) for sym, df in entry.items()}
        return {sym: apply_regime(breakout_signals(df, bp, cfg), regime, cfg.entry_ms) for sym, df in entry.items()}
    return {sym: build_signals(cfg, df, (trend or {}).get(sym)) for sym, df in entry.items()}


def conditions(cfg: Config) -> tuple[tuple[str, str], ...]:
    s = cfg.strategy
    if s.type == "momentum":
        return MOMENTUM_CONDITIONS
    if s.type == "breakout":
        return BREAKOUT_CONDITIONS + ((REGIME_CONDITION,) if s.regime_sma else ())
    return PULLBACK_CONDITIONS


def describe(cfg: Config) -> str:
    s = cfg.strategy
    if s.type == "momentum":
        return (f"Momentum rotasyonu ({s.momentum_lookback} gün, ilk {s.momentum_top_n}, çıkış {s.momentum_exit_rank}. "
                f"sıra), BTC > SMA{s.regime_sma}, ilk stop {cfg.risk.stop_atr_mult:g}×ATR({s.channel_atr})")
    if s.type == "breakout":
        regime = f", BTC > SMA{s.regime_sma}" if s.regime_sma else ""
        if s.allow_short:
            regime = (f", long BTC > SMA{s.regime_sma} / short BTC < SMA{s.regime_sma}" if s.regime_sma
                      else ", long ve short")
        market = f", vadeli {cfg.futures.leverage:g}x izole" if cfg.is_futures else ""
        return (f"Kanal kırılımı {s.channel_entry}/{s.channel_exit} ({cfg.timeframes.entry}){regime}, "
                f"ilk stop {cfg.risk.stop_atr_mult:g}×ATR({s.channel_atr}){market}")
    return f"3 katmanlı geri çekilme ({cfg.timeframes.entry}, trend {cfg.timeframes.trend})"
