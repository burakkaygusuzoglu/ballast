"""Araştırma turu: aday stratejileri GELİŞTİRME döneminde, önceden yazılmış protokole göre sınar.

Protokol: docs/ARASTIRMA.md, 4. bölüm. Adaylar, parametreler ve geçme ölçütleri sonuçlar
görülmeden önce yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey
değiştirmek, testi geçersiz kılar.

Kullanım:
    python arastirma.py              # S0 (referans) + S1-S4: keşif, doğrulama, sağlamlık, Monte Carlo, portföy
    python arastirma.py --no-plot    # grafik çizmeden

GÜVENLİK: Sınav dönemi (data.exam_start sonrası) burada ASLA kullanılmaz. Veri yüklenir
yüklenmez sınav tarihinden sonra kapanan mumlar atılır.

Çıktı: reports/<zaman>_arastirma/ altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bot.backtester import BacktestResult, run_backtest
from bot.candidates import BreakoutParams, MeanRevParams, breakout_signals, meanrev_signals, resample_candles
from bot.cli import load_config_or_exit, open_file, setup_console
from bot.config import Config, MarketRules
from bot.data import candle_path, load_candles, load_market_rules
from bot.metrics import buy_and_hold, compute_metrics, daily_equity
from bot.portfolio import Trade
from bot.strategy import compute_signals, first_ready_ts
from bot.timeutil import DAY_MS, TIMEFRAME_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 4. bölüm) ------------------------------------
VALIDATION_START = "2025-01-01"   # keşif: ısınma sonrası -> 2024-12-31 | doğrulama: 2025-01-01 -> sınav öncesi
EXPLORE_MIN_PF = 1.2              # ölçüt 1
VALIDATE_MIN_PF = 1.0             # ölçüt 2
MAX_DRAWDOWN = 0.20               # ölçüt 1 ve 2
ROBUST_MIN_PF = 1.0               # ölçüt 4
MC_RUNS = 1000
MC_SEED = 20261008
SOURCE_TIMEFRAMES = ("15m", "1h")  # indirilen veriler; 4h ve 1d, 1h mumlarından UTC'ye hizalı türetilir
PERIODS = (("kesif", "Keşif"), ("dogrulama", "Doğrulama"))

Market = dict[str, dict[str, pd.DataFrame]]   # sembol -> zaman dilimi -> mumlar
Progress = Callable[[str], None]


# ---------------------------------------------------------------------------
# Adaylar
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    kind: str                      # "pullback" (mevcut 3 katmanlı strateji), "breakout", "meanrev"
    entry_tf: str
    trend_tf: str
    overrides: dict = field(default_factory=dict)   # config bölümlerine uygulanan değişiklikler
    params: BreakoutParams | MeanRevParams | None = None
    min_trades: int = 100          # ölçüt 3: keşif + doğrulama toplam işlem sayısı
    reference: bool = False        # S0: yalnızca karşılaştırma için, aday değil


def protocol_candidates() -> list[Candidate]:
    """Protokoldeki adaylar, ders kitabı parametreleriyle (optimizasyon yok)."""
    strategy_exit = {"risk": {"stop_atr_mult": 2.0, "trail_atr_mult": 0.0},
                     "strategy": {"exit_on_regime_break": False}}
    return [
        Candidate("S0", "Referans: mevcut strateji, 15 dk", "pullback", "15m", "1h", reference=True),
        Candidate("S1", "Geri çekilme, 1 saat", "pullback", "1h", "4h",
                  {"timeframes": {"entry": "1h", "trend": "4h"}}),
        Candidate("S2", "Kırılım 20/10, 4 saat", "breakout", "4h", "1d",
                  {"timeframes": {"entry": "4h", "trend": "1d"}, "execution": {"entry_order": "market"},
                   **strategy_exit},
                  BreakoutParams(entry_n=20, exit_n=10, atr_n=20), min_trades=30),
        Candidate("S3", "Kırılım 20/10, günlük (Kaplumbağa)", "breakout", "1d", "1d",
                  {"timeframes": {"entry": "1d", "trend": "1d"}, "execution": {"entry_order": "market"},
                   **strategy_exit},
                  BreakoutParams(entry_n=20, exit_n=10, atr_n=20), min_trades=30),
        Candidate("S4", "Ortalamaya dönüş RSI-2, 1 saat", "meanrev", "1h", "4h",
                  {"timeframes": {"entry": "1h", "trend": "4h"},
                   "risk": {"stop_atr_mult": 3.0, "trail_atr_mult": 0.0},
                   "strategy": {"exit_on_regime_break": False}},
                  MeanRevParams(rsi_n=2, rsi_entry=10.0, trend_n=200, exit_n=5, atr_n=14), min_trades=100),
    ]


def robustness_variants(c: Candidate) -> list[tuple[str, Candidate]]:
    """Ölçüt 4: ana parametreler TEK TEK ±%25 (tam sayılar en yakın değere yuvarlanır)."""
    def variant(label: str, overrides: dict | None = None, params=None) -> tuple[str, Candidate]:
        return label, dataclasses.replace(c, overrides=_merge(c.overrides, overrides or {}),
                                          params=params if params is not None else c.params)

    if c.reference:
        return []
    if c.kind == "pullback":
        return [
            variant("RSI geri çekilme seviyesi 30", {"strategy": {"rsi_pullback_level": 30.0}}),
            variant("RSI geri çekilme seviyesi 50", {"strategy": {"rsi_pullback_level": 50.0}}),
            variant("stop 1,5 ATR", {"risk": {"stop_atr_mult": 1.5}}),
            variant("stop 2,5 ATR", {"risk": {"stop_atr_mult": 2.5}}),
            variant("iz süren stop 2,25 ATR", {"risk": {"trail_atr_mult": 2.25}}),
            variant("iz süren stop 3,75 ATR", {"risk": {"trail_atr_mult": 3.75}}),
        ]
    if c.kind == "breakout":
        p = c.params
        return [
            variant("giriş kanalı 15", params=dataclasses.replace(p, entry_n=15)),
            variant("giriş kanalı 25", params=dataclasses.replace(p, entry_n=25)),
            variant("çıkış kanalı 8", params=dataclasses.replace(p, exit_n=8)),
            variant("çıkış kanalı 12", params=dataclasses.replace(p, exit_n=12)),
            variant("stop 1,5 ATR", {"risk": {"stop_atr_mult": 1.5}}),
            variant("stop 2,5 ATR", {"risk": {"stop_atr_mult": 2.5}}),
        ]
    if c.kind == "meanrev":
        p = c.params
        return [
            variant("RSI eşiği 7,5", params=dataclasses.replace(p, rsi_entry=7.5)),
            variant("RSI eşiği 12,5", params=dataclasses.replace(p, rsi_entry=12.5)),
            variant("trend SMA 150", params=dataclasses.replace(p, trend_n=150)),
            variant("trend SMA 250", params=dataclasses.replace(p, trend_n=250)),
            variant("çıkış SMA 4", params=dataclasses.replace(p, exit_n=4)),
            variant("çıkış SMA 6", params=dataclasses.replace(p, exit_n=6)),
            variant("stop 2,25 ATR", {"risk": {"stop_atr_mult": 2.25}}),
            variant("stop 3,75 ATR", {"risk": {"stop_atr_mult": 3.75}}),
        ]
    raise ValueError(f"Bilinmeyen aday türü: {c.kind}")


def _merge(base: dict, extra: dict) -> dict:
    out = {section: dict(values) for section, values in base.items()}
    for section, values in extra.items():
        out.setdefault(section, {}).update(values)
    return out


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    changes = {name: dataclasses.replace(getattr(cfg, name), **values) for name, values in overrides.items()}
    return dataclasses.replace(cfg, **changes)


def build_signals(c: Candidate, cfg_c: Config, market: Market) -> dict[str, pd.DataFrame]:
    signals = {}
    for symbol, frames in market.items():
        entry = frames[c.entry_tf]
        if c.kind == "pullback":
            signals[symbol] = compute_signals(entry, frames[c.trend_tf], cfg_c)
        elif c.kind == "breakout":
            signals[symbol] = breakout_signals(entry, c.params, cfg_c)
        elif c.kind == "meanrev":
            signals[symbol] = meanrev_signals(entry, c.params, cfg_c)
        else:
            raise ValueError(f"Bilinmeyen aday türü: {c.kind}")
    return signals


# ---------------------------------------------------------------------------
# Veri: sınav dönemi asla yüklenmez
# ---------------------------------------------------------------------------
def before_exam(df: pd.DataFrame, tf_ms: int, exam_ms: int) -> pd.DataFrame:
    """Yalnızca sınav başlangıcından ÖNCE (ya da tam o anda) kapanmış mumlar."""
    return df[df["ts"] + tf_ms <= exam_ms].reset_index(drop=True)


def prepare_market(raw: dict[str, dict[str, pd.DataFrame]], exam_ms: int) -> Market:
    """Sınav sonrası mumları atar; 4h ve 1d mumlarını 1h'ten türetir."""
    h1 = TIMEFRAME_MS["1h"]
    market: Market = {}
    for symbol, frames in raw.items():
        out = {tf: before_exam(df, TIMEFRAME_MS[tf], exam_ms) for tf, df in frames.items()}
        out["4h"] = resample_candles(out["1h"], h1, TIMEFRAME_MS["4h"])
        out["1d"] = resample_candles(out["1h"], h1, DAY_MS)
        for tf, df in out.items():
            if len(df) and int(df["ts"].iloc[-1]) + TIMEFRAME_MS[tf] > exam_ms:
                raise AssertionError(f"{symbol} {tf}: sınav dönemi verisi sızdı.")
        market[symbol] = out
    return market


def load_research_market(cfg: Config) -> Market:
    raw: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in cfg.symbols:
        raw[symbol] = {}
        for tf in SOURCE_TIMEFRAMES:
            path = candle_path(cfg.data_dir, cfg.exchange.id, symbol, tf)
            if not path.exists():
                raise FileNotFoundError(path.name)
            raw[symbol][tf] = load_candles(path)
    return prepare_market(raw, parse_utc(cfg.data.exam_start))


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------
@dataclass
class PeriodRun:
    start: int
    end: int
    result: BacktestResult
    metrics: dict
    symbol_net: dict[str, float]   # sembol başına net kâr (USDT): kapanan işlemler + dönem sonu açık pozisyon
    benchmark: dict                # aynı dönemde al-tut metrikleri


@dataclass
class CandidateReport:
    candidate: Candidate
    runs: dict[str, PeriodRun]
    robustness: list[tuple[str, dict]]
    monte_carlo: dict
    checks: dict[str, tuple[bool | None, str]]

    @property
    def passed(self) -> bool:
        return not self.candidate.reference and all(ok for ok, _ in self.checks.values())


@dataclass
class PortfolioRun:
    members: list[str]
    start: int
    end: int
    equity: pd.Series
    metrics: dict
    benchmark: dict
    correlation: pd.DataFrame


@dataclass
class Study:
    created: str
    symbols: list[str]
    initial_equity: float
    data_range: dict[str, tuple[str, str]]
    reports: list[CandidateReport]
    portfolios: dict[str, dict[str, PortfolioRun]]   # {"gecenler"/"tum": {"kesif"/"dogrulama": ...}}
    bench_curves: dict[str, pd.Series]


def trade_window(signals: dict[str, pd.DataFrame], entry_ms: int) -> tuple[int, int]:
    """(göstergeler hazır olduktan sonraki ilk mum, verinin bittiği an)."""
    data_start = max(int(df["ts"].iloc[0]) for df in signals.values())
    data_end = min(int(df["ts"].iloc[-1]) for df in signals.values()) + entry_ms
    ready = [first_ready_ts(df) for df in signals.values()]
    if any(r is None for r in ready):
        raise ValueError("Göstergelerin ısınması için veri yetersiz; daha uzun geçmiş gerekli.")
    return max([data_start, *[int(r) for r in ready if r is not None]]), data_end


def symbol_net(result: BacktestResult, signals: dict[str, pd.DataFrame]) -> dict[str, float]:
    net = {symbol: 0.0 for symbol in signals}
    for t in result.trades:
        net[t.symbol] += t.net_pnl
    for p in result.open_positions:
        df = signals[p.symbol]
        last_close = float(df.loc[df["ts"] < result.end_ts, "close"].iloc[-1])
        net[p.symbol] += p.unrealized(last_close) - p.entry_fee - p.funding_paid   # vadelide funding dahil
    return net


def run_period(cfg_c: Config, signals: dict[str, pd.DataFrame], market: Market, entry_tf: str,
               start: int, end: int, rules: dict[str, MarketRules]) -> PeriodRun:
    initial = cfg_c.backtest.initial_equity
    result = run_backtest(cfg_c, signals, start, end, initial, rules)
    metrics = compute_metrics(result.equity, result.trades, initial, result.total_fees, result.exposure)
    metrics["pre_cost_profit_factor"] = pre_cost_profit_factor(result.trades)
    nets = symbol_net(result, signals)
    # Muhasebe öz-denetimi: sembollerin net kârlarının toplamı özsermaye değişimine eşit olmalı.
    if abs(sum(nets.values()) - metrics["net_profit"]) > 1e-6 * initial:
        raise AssertionError(f"Muhasebe tutarsızlığı: {sum(nets.values())} != {metrics['net_profit']}")
    bench_curve = buy_and_hold({s: f[entry_tf] for s, f in market.items()}, start, end, initial,
                               cfg_c.costs.taker_fee, cfg_c.costs.slippage, TIMEFRAME_MS[entry_tf])
    bench = compute_metrics(bench_curve, [], initial, 0.0)
    return PeriodRun(start, end, result, metrics, nets, bench)


def pre_cost_profit_factor(trades: list[Trade]) -> float:
    """Komisyon ve kayma hiç olmasaydı kâr faktörü: sinyalin kendi avantajını gösterir."""
    pnl = np.array([t.gross_pnl + t.slippage_cost for t in trades], dtype="float64")
    gain, loss = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    if loss > 0:
        return float(gain / loss)
    return math.inf if gain > 0 else 0.0


def path_drawdown(r_multiples: np.ndarray, risk: float) -> float:
    path = np.concatenate([[1.0], np.cumprod(1.0 + risk * r_multiples)])
    return float(np.max(1.0 - path / np.maximum.accumulate(path)))


def monte_carlo(r_multiples: list[float], risk: float, runs: int = MC_RUNS, seed: int = MC_SEED) -> dict:
    """İşlem sırası karıştırılır: aynı işlemlerle hangi düşüşlerin yaşanabileceğini gösterir."""
    r = np.asarray(r_multiples, dtype="float64")
    if len(r) < 2:
        return {"n": int(len(r)), "avg_r": float(r.mean()) if len(r) else 0.0,
                "actual_dd": None, "median_dd": None, "p95_dd": None}
    rng = np.random.default_rng(seed)
    drawdowns = np.array([path_drawdown(rng.permutation(r), risk) for _ in range(runs)])
    return {
        "n": int(len(r)),
        "avg_r": float(r.mean()),
        "actual_dd": path_drawdown(r, risk),
        "median_dd": float(np.median(drawdowns)),
        "p95_dd": float(np.quantile(drawdowns, 0.95)),
    }


def evaluate(c: Candidate, runs: dict[str, PeriodRun], robustness: list[tuple[str, dict]]
             ) -> dict[str, tuple[bool | None, str]]:
    """Protokoldeki dört ölçüt: (geçti mi, kısa açıklama)."""
    k, d = runs["kesif"].metrics, runs["dogrulama"].metrics
    problems = []
    if not k["profit_factor"] >= EXPLORE_MIN_PF:
        problems.append(f"PF {_ratio(k['profit_factor'])} < {EXPLORE_MIN_PF}")
    if not k["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not k["max_drawdown"] <= MAX_DRAWDOWN:
        problems.append(f"düşüş %{k['max_drawdown'] * 100:.1f}")
    losers = [s.split("/")[0] for s, v in runs["kesif"].symbol_net.items() if not v > 0]
    if losers:
        problems.append(f"{', '.join(losers)} net ≤ 0")
    checks: dict[str, tuple[bool | None, str]] = {"kesif": (not problems, "; ".join(problems) or "tümü sağlandı")}

    problems = []
    if not d["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not d["profit_factor"] >= VALIDATE_MIN_PF:
        problems.append(f"PF {_ratio(d['profit_factor'])} < {VALIDATE_MIN_PF}")
    if not d["max_drawdown"] <= MAX_DRAWDOWN:
        problems.append(f"düşüş %{d['max_drawdown'] * 100:.1f}")
    checks["dogrulama"] = (not problems, "; ".join(problems) or "tümü sağlandı")

    n = k["n_trades"] + d["n_trades"]
    checks["islem"] = (n >= c.min_trades, f"{n} işlem (en az {c.min_trades})")

    if c.reference or not robustness:
        checks["saglamlik"] = (None, "uygulanmadı")
    else:
        weak = [label for label, m in robustness if not m["profit_factor"] >= ROBUST_MIN_PF]
        detail = f"PF < {ROBUST_MIN_PF}: {', '.join(weak)}" if weak else "tüm varyantlarda PF ≥ 1"
        checks["saglamlik"] = (not weak, detail)
    return checks


def run_candidate(c: Candidate, cfg: Config, market: Market, rules: dict[str, MarketRules],
                  progress: Progress) -> CandidateReport:
    cfg_c = apply_overrides(cfg, c.overrides)
    signals = build_signals(c, cfg_c, market)
    trade_start, data_end = trade_window(signals, cfg_c.entry_ms)
    split = parse_utc(VALIDATION_START)
    if not trade_start < split < data_end:
        raise ValueError(f"{c.code}: veri keşif ve doğrulama dönemlerini kapsamıyor.")
    bounds = {"kesif": (trade_start, split), "dogrulama": (split, min(data_end, parse_utc(cfg.data.exam_start)))}
    runs = {}
    for name, label in PERIODS:
        t0 = time.perf_counter()
        runs[name] = run_period(cfg_c, signals, market, c.entry_tf, *bounds[name], rules)
        progress(f"  {c.code} {label:<10} {runs[name].metrics['n_trades']:>4} işlem  "
                 f"{_pct(runs[name].metrics['total_return']):>8}  ({time.perf_counter() - t0:.1f} sn)")

    robustness = []
    for label, variant in robustness_variants(c):
        cfg_v = apply_overrides(cfg, variant.overrides)
        signals_v = build_signals(variant, cfg_v, market)
        start = max(trade_window(signals_v, cfg_v.entry_ms)[0], trade_start)
        result = run_backtest(cfg_v, signals_v, start, split, cfg.backtest.initial_equity, rules)
        robustness.append((label, compute_metrics(result.equity, result.trades, cfg.backtest.initial_equity,
                                                  result.total_fees, result.exposure)))
    if robustness:
        progress(f"  {c.code} sağlamlık: {len(robustness)} varyant")

    trades = runs["kesif"].result.trades + runs["dogrulama"].result.trades
    mc = monte_carlo([t.r_multiple for t in trades], cfg_c.risk.risk_per_trade)
    return CandidateReport(c, runs, robustness, mc, evaluate(c, runs, robustness))


def combine_equal(curves: list[pd.Series], initial: float, start_ts: int) -> pd.Series:
    """Eşit sermayeli birleşim (yeniden dengeleme yok), günlük kapanışlarla."""
    daily = [daily_equity(curve) / initial for curve in curves]
    frame = pd.concat(daily, axis=1).sort_index().ffill().fillna(1.0)
    combined = frame.mean(axis=1) * initial
    index = (combined.index.to_numpy(dtype="int64") + 1) * DAY_MS          # günün kapanış anı
    series = pd.Series(combined.to_numpy(), index=pd.Index(index, dtype="int64"))
    head = pd.Series([initial], index=pd.Index([start_ts], dtype="int64"))
    return pd.concat([head, series]).rename("equity")


def run_portfolio(members: list[CandidateReport], cfg: Config, market: Market,
                  rules: dict[str, MarketRules], period: str) -> PortfolioRun:
    initial = cfg.backtest.initial_equity
    start = max(r.runs[period].start for r in members)
    end = min(r.runs[period].end for r in members)
    runs = []
    for r in members:
        run = r.runs[period]
        if (run.start, run.end) != (start, end):   # ortak pencere: geç ısınan adaya hizalanır
            cfg_c = apply_overrides(cfg, r.candidate.overrides)
            run = run_period(cfg_c, build_signals(r.candidate, cfg_c, market), market, r.candidate.entry_tf,
                             start, end, rules)
        runs.append(run)
    equity = combine_equal([run.result.equity for run in runs], initial, start)
    trades: list[Trade] = [t for run in runs for t in run.result.trades]
    metrics = compute_metrics(equity, trades, initial, sum(run.result.total_fees for run in runs) / len(runs))
    returns = pd.concat([daily_equity(run.result.equity).pct_change() for run in runs], axis=1)
    returns.columns = [r.candidate.code for r in members]
    bench_curve = buy_and_hold({s: f["1h"] for s, f in market.items()}, start, end, initial,
                               cfg.costs.taker_fee, cfg.costs.slippage, TIMEFRAME_MS["1h"])
    return PortfolioRun([r.candidate.code for r in members], start, end, equity, metrics,
                        compute_metrics(bench_curve, [], initial, 0.0), returns.corr())


def run_study(cfg: Config, market: Market, rules: dict[str, MarketRules],
              candidates: list[Candidate] | None = None, progress: Progress | None = None) -> Study:
    progress = progress or (lambda _msg: None)
    reports = []
    for c in candidates or protocol_candidates():
        progress(f"{c.code} · {c.name}")
        reports.append(run_candidate(c, cfg, market, rules, progress))

    portfolios: dict[str, dict[str, PortfolioRun]] = {}
    groups = {"gecenler": [r for r in reports if r.passed],
              "tum": [r for r in reports if not r.candidate.reference]}
    for key, members in groups.items():
        if key == "tum" and len(members) < 2:
            continue
        if members:
            progress(f"Portföy ({'geçenler' if key == 'gecenler' else 'tüm adaylar'}): "
                     f"{', '.join(r.candidate.code for r in members)}")
            portfolios[key] = {name: run_portfolio(members, cfg, market, rules, name) for name, _ in PERIODS}

    bench_curves = {}
    for name, _ in PERIODS:
        start = min(r.runs[name].start for r in reports)
        end = max(r.runs[name].end for r in reports)
        bench_curves[name] = buy_and_hold({s: f["1h"] for s, f in market.items()}, start, end,
                                          cfg.backtest.initial_equity, cfg.costs.taker_fee, cfg.costs.slippage,
                                          TIMEFRAME_MS["1h"])
    data_range = {s: (fmt_utc(int(f["1h"]["ts"].iloc[0]), False),
                      fmt_utc(int(f["1h"]["ts"].iloc[-1]) + TIMEFRAME_MS["1h"], False)) for s, f in market.items()}
    return Study(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), list(market), cfg.backtest.initial_equity,
                 data_range, reports, portfolios, bench_curves)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
def _pct(x: float | None, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _ratio(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return "sonsuz" if math.isinf(x) else f"{x:.2f}"


def _mark(ok: bool | None) -> str:
    return "—" if ok is None else ("GEÇTİ" if ok else "KALDI")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def format_markdown(study: Study) -> str:
    bases = [s.split("/")[0] for s in study.symbols]
    out = [
        f"### Sonuçlar ({study.created})",
        "",
        f"Veri (1 saatlik): {', '.join(f'{s} {a} → {b}' for s, (a, b) in study.data_range.items())}. "
        f"Sınav dönemi kullanılmadı. Başlangıç özsermayesi her dönemde {study.initial_equity:,.0f} USDT; "
        "komisyon ve kayma dahil. Sembol sütunları başlangıç özsermayesine oranla net kârdır.",
        "",
    ]
    for name, label in PERIODS:
        rows = []
        for r in study.reports:
            run = r.runs[name]
            m, b = run.metrics, run.benchmark
            rows.append([
                f"{r.candidate.code} {r.candidate.name}",
                f"{fmt_utc(run.start, False)} → {fmt_utc(run.end, False)}",
                _pct(m["total_return"]), _pct(m["max_drawdown"], False), _ratio(m["profit_factor"]),
                _ratio(m["pre_cost_profit_factor"]),
                _pct((m["fees_closed"] + m["slippage_closed"]) / m["initial_equity"], False),
                str(m["n_trades"]), _pct(m["win_rate"], False), f"{m['avg_r']:+.2f}",
                *[_pct(run.symbol_net[s] / study.initial_equity) for s in study.symbols],
                f"{_pct(b['total_return'])} / {_pct(b['max_drawdown'], False)}",
                fmt_utc(run.result.halted_at, False) if run.result.halted_at else "-",
            ])
        out += [f"**{label} dönemi**", ""]
        out += _table(["Aday", "Dönem", "Net getiri", "Maks. düşüş", "Kâr faktörü", "Maliyetsiz KF", "Maliyet",
                       "İşlem", "Kazanma", "Ort. R", *[f"{b} net" for b in bases], "Al-tut getiri / düşüş",
                       "Şalter"], rows)
        out.append("")
    out += ["Maliyetsiz KF: komisyon ve kayma olmasaydı kâr faktörü. Maliyet: kapanan işlemlerde ödenen komisyon ve "
            "kaymanın başlangıç özsermayesine oranı. Şalter: %20 düşüş şalterinin devreye girdiği gün (sonrasında "
            "dönem sonuna kadar işlem yok).", ""]

    rows = []
    for r in study.reports:
        cells = [f"{_mark(ok)} ({text})" for ok, text in r.checks.values()]
        verdict = "referans" if r.candidate.reference else ("**GEÇTİ**" if r.passed else "KALDI")
        rows.append([r.candidate.code, *cells, verdict])
    out += ["**Ölçütler**", ""]
    out += _table(["Aday", "1. Keşif", "2. Doğrulama", "3. İşlem sayısı", "4. Sağlamlık", "Sonuç"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        base = r.runs["kesif"].metrics
        rows.append([r.candidate.code, "ana parametreler", _ratio(base["profit_factor"]),
                     _pct(base["total_return"]), str(base["n_trades"])])
        for label, m in r.robustness:
            rows.append(["", label, _ratio(m["profit_factor"]), _pct(m["total_return"]), str(m["n_trades"])])
    out += ["**Sağlamlık (keşif dönemi, parametreler tek tek ±%25)**", ""]
    out += _table(["Aday", "Değişiklik", "Kâr faktörü", "Net getiri", "İşlem"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        mc = r.monte_carlo
        rows.append([r.candidate.code, str(mc["n"]), f"{mc['avg_r']:+.2f}", _pct(mc["actual_dd"], False),
                     _pct(mc["median_dd"], False), _pct(mc["p95_dd"], False)])
    out += [f"**Monte Carlo (keşif + doğrulama işlemleri, sıra {MC_RUNS} kez karıştırıldı, işlem başına %1 risk)**", ""]
    out += _table(["Aday", "İşlem", "Ort. R", "Gerçek sırayla maks. düşüş", "Medyan", "En kötü %5"], rows)
    out.append("")

    if "gecenler" not in study.portfolios:
        out += ["Ölçütleri geçen aday olmadığından \"geçenler\" portföyü oluşmadı.", ""]
    if study.portfolios:
        rows = []
        for key, title in (("gecenler", "Geçen adaylar"), ("tum", "Tüm adaylar")):
            for name, label in PERIODS:
                p = study.portfolios.get(key, {}).get(name)
                if p is None:
                    continue
                m, b = p.metrics, p.benchmark
                rows.append([f"{title} ({', '.join(p.members)})", label, _pct(m["total_return"]),
                             _pct(m["max_drawdown"], False), _ratio(m["return_to_drawdown"]), f"{m['sharpe']:.2f}",
                             str(m["n_trades"]), f"{_pct(b['total_return'])} / {_pct(b['max_drawdown'], False)}"])
        out += ["**Eşit sermayeli portföy (günlük kapanışlarla)**", ""]
        out += _table(["Portföy", "Dönem", "Net getiri", "Maks. düşüş", "Getiri/düşüş", "Sharpe", "İşlem",
                       "Al-tut getiri / düşüş"], rows)
        out.append("")
        if "tum" in study.portfolios:
            corr = study.portfolios["tum"]["dogrulama"].correlation
            rows = [[code, *[f"{corr.loc[code, other]:+.2f}" for other in corr.columns]] for code in corr.index]
            out += ["**Adayların günlük getiri korelasyonu (doğrulama)**", ""]
            out += _table(["", *corr.columns], rows)
            out.append("")
    return "\n".join(out)


def _clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def study_summary(study: Study) -> dict:
    return _clean({
        "created": study.created,
        "symbols": study.symbols,
        "initial_equity": study.initial_equity,
        "data_range": study.data_range,
        "protocol": {"validation_start": VALIDATION_START, "explore_min_pf": EXPLORE_MIN_PF,
                     "validate_min_pf": VALIDATE_MIN_PF, "max_drawdown": MAX_DRAWDOWN,
                     "robust_min_pf": ROBUST_MIN_PF, "mc_runs": MC_RUNS, "mc_seed": MC_SEED},
        "candidates": [{
            "code": r.candidate.code,
            "name": r.candidate.name,
            "reference": r.candidate.reference,
            "params": asdict(r.candidate.params) if r.candidate.params else None,
            "overrides": r.candidate.overrides,
            "passed": r.passed,
            "checks": {k: {"passed": ok, "detail": text} for k, (ok, text) in r.checks.items()},
            "periods": {name: {"start": fmt_utc(run.start, False), "end": fmt_utc(run.end, False),
                               "metrics": run.metrics, "symbol_net": run.symbol_net, "buy_and_hold": run.benchmark,
                               "halt": run.result.halt_reason or None}
                        for name, run in r.runs.items()},
            "robustness": [{"variant": label, "metrics": m} for label, m in r.robustness],
            "monte_carlo": r.monte_carlo,
        } for r in study.reports],
        "portfolios": {key: {name: {"members": p.members, "start": fmt_utc(p.start, False),
                                    "end": fmt_utc(p.end, False), "metrics": p.metrics, "buy_and_hold": p.benchmark,
                                    "correlation": p.correlation.round(4).to_dict()}
                             for name, p in group.items()}
                       for key, group in study.portfolios.items()},
    })


def save_study(out_dir: Path, study: Study, text: str, plot: bool = True) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(study_summary(study), indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    rows = []
    for r in study.reports:
        for name, run in r.runs.items():
            for t in run.result.trades:
                rows.append({"aday": r.candidate.code, "donem": name, **asdict(t),
                             "entry_utc": fmt_utc(t.entry_ts), "exit_utc": fmt_utc(t.exit_ts)})
    pd.DataFrame(rows).to_csv(out_dir / "islemler.csv", index=False, encoding="utf-8-sig")
    if not plot:
        return None
    chart = out_dir / "grafik.png"
    plot_study(chart, study)
    return chart


def plot_study(path: Path, study: Study) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"S0": "#9E9E9E", "S1": "#4C72B0", "S2": "#55A868", "S3": "#C44E52", "S4": "#8172B2"}
    fig, axes = plt.subplots(len(PERIODS), 1, figsize=(12, 9))
    for ax, (name, label) in zip(axes, PERIODS, strict=True):
        for r in study.reports:
            curve = daily_equity(r.runs[name].result.equity) / study.initial_equity * 100
            dates = pd.to_datetime((curve.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True)
            ax.plot(dates, curve.to_numpy(), linewidth=1.4, color=colors.get(r.candidate.code),
                    linestyle=":" if r.candidate.reference else "-", label=f"{r.candidate.code} {r.candidate.name}")
        bench = daily_equity(study.bench_curves[name]) / study.initial_equity * 100
        ax.plot(pd.to_datetime((bench.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True), bench.to_numpy(),
                color="#DD8452", linewidth=1.1, linestyle="--", label="Al-tut (BTC+ETH eşit)")
        ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_title(f"{label} dönemi · özsermaye (başlangıç = 100) · komisyon ve kayma dahil", loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Aday stratejileri geliştirme döneminde protokole göre sınar.")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    try:
        market = load_research_market(cfg)
    except FileNotFoundError as exc:
        print(f"[HATA] Veri bulunamadı: {exc}\nÖnce 2_veri_indir.bat çalıştırın.")
        return 2
    print(f"Veri yüklendi; {cfg.data.exam_start} (sınav) sonrası mumlar atıldı.")
    rules = load_market_rules(cfg.data_dir, cfg.symbols, cfg.market_rules_default)
    try:
        study = run_study(cfg, market, rules, progress=print)
    except ValueError as exc:
        print(f"[HATA] {exc}")
        return 2
    text = format_markdown(study)
    print("\n" + text)
    out_dir = cfg.reports_dir / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_arastirma"
    chart = save_study(out_dir, study, text, plot=not args.no_plot)
    print(f"Rapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
