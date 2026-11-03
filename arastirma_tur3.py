"""3. araştırma turu: C1'i geliştirme adayları (R0-R4), daha yüksek kazanç ve makul risk.

Protokol: docs/ARASTIRMA.md, 9. bölüm. Adaylar, parametreler ve ölçütler sonuçlar görülmeden önce
yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey değiştirmek testi geçersiz kılar.

GÜVENLİK: Sınav dönemi (data.exam_start sonrası) burada ASLA kullanılmaz; veri yüklenirken atılır.

Kullanım:
    python arastirma_tur3.py
    python arastirma_tur3.py --no-plot

Çıktı: reports/<zaman>_arastirma_tur3/ altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import arastirma as A
from bot.backtester import run_backtest
from bot.candidates import (
    BreakoutParams,
    MomentumParams,
    apply_regime,
    breakout_signals,
    btc_regime,
    momentum_signals,
)
from bot.cli import load_config_or_exit, open_file, setup_console
from bot.config import Config, MarketRules
from bot.data import candle_path, load_candles, load_market_rules
from bot.metrics import buy_and_hold, compute_metrics, daily_equity
from bot.timeutil import DAY_MS, TIMEFRAME_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 9. bölüm) ------------------------------------
BTC = "BTC/USDT"
MIN_TRADES = 100                  # ölçüt 5: keşif + doğrulama
FEASIBILITY_EQUITY = 50.0        # bilgi: küçük hesap ve gerçek minimum işlem tutarlarıyla
RISK_LADDER = (0.0075, 0.01)      # bilgi: işlem başına risk
NO_HALT = 0.999                   # şaltersiz tanı koşusu (yalnızca %99,9 düşüşte durur)
MAX_SHADOW = 2                    # gölge teste en fazla iki aday
MIN_NOTIONAL_SKIP = "boyut çok küçük (min. tutar)"

C1_RULES = {
    "timeframes": {"entry": "4h", "trend": "1d"},
    "execution": {"entry_order": "market"},
    "risk": {"risk_per_trade": 0.005, "stop_atr_mult": 2.0, "trail_atr_mult": 0.0, "max_open_positions": 5},
    "strategy": {"exit_on_regime_break": False},
}
PYRAMID = {"risk": {"pyramid_units": 4, "pyramid_step_atr": 0.5, "max_total_units": 10}}


@dataclass(frozen=True)
class T3Candidate:
    code: str
    name: str
    kind: str                                  # "breakout" | "momentum"
    entry_tf: str
    breakout: BreakoutParams | None = None
    momentum: MomentumParams | None = None
    regime_sma: int | None = None              # kırılım + BTC rejim filtresi
    overrides: dict = field(default_factory=dict)
    reference: bool = False


def tur3_candidates() -> list[T3Candidate]:
    c1 = BreakoutParams(entry_n=20, exit_n=10, atr_n=20)
    return [
        T3Candidate("R0", "Referans: C1", "breakout", "4h", breakout=c1, overrides=C1_RULES, reference=True),
        T3Candidate("R1", "C1 + BTC rejim filtresi", "breakout", "4h", breakout=c1, regime_sma=200,
                    overrides=C1_RULES),
        T3Candidate("R2", "R1 + piramitleme", "breakout", "4h", breakout=c1, regime_sma=200,
                    overrides=A._merge(C1_RULES, PYRAMID)),
        T3Candidate("R3", "Momentum rotasyonu (günlük)", "momentum", "1d", momentum=MomentumParams(),
                    overrides=A._merge(C1_RULES, {"timeframes": {"entry": "1d", "trend": "1d"}})),
        T3Candidate("R4", "Gün içi kırılım (1 saat)", "breakout", "1h", breakout=c1,
                    overrides=A._merge(C1_RULES, {"timeframes": {"entry": "1h", "trend": "4h"}})),
    ]


def robustness_variants(c: T3Candidate) -> list[tuple[str, T3Candidate]]:
    """Ölçüt 4: adayın YENİ parametreleri tek tek ±%25."""
    def v(label: str, **changes) -> tuple[str, T3Candidate]:
        return label, dataclasses.replace(c, **changes)

    if c.reference:
        return []
    if c.code == "R1":
        return [v("rejim ortalaması 150 gün", regime_sma=150), v("rejim ortalaması 250 gün", regime_sma=250)]
    if c.code == "R2":
        risk = c.overrides["risk"]
        return [v(label, overrides=A._merge(c.overrides, {"risk": {**risk, key: value}}))
                for label, key, value in (("ekleme adımı 0,375 ATR", "pyramid_step_atr", 0.375),
                                          ("ekleme adımı 0,625 ATR", "pyramid_step_atr", 0.625),
                                          ("en fazla 3 birim", "pyramid_units", 3),
                                          ("en fazla 5 birim", "pyramid_units", 5))]
    if c.code == "R3":
        m = c.momentum
        return [v("getiri penceresi 21 gün", momentum=dataclasses.replace(m, lookback=21)),
                v("getiri penceresi 35 gün", momentum=dataclasses.replace(m, lookback=35)),
                v("ilk 2 coin (çıkış 4)", momentum=dataclasses.replace(m, top_n=2, exit_rank=4)),
                v("ilk 4 coin (çıkış 6)", momentum=dataclasses.replace(m, top_n=4, exit_rank=6))]
    if c.code == "R4":
        p = c.breakout
        stop = c.overrides["risk"]
        return [v("giriş kanalı 15", breakout=dataclasses.replace(p, entry_n=15)),
                v("giriş kanalı 25", breakout=dataclasses.replace(p, entry_n=25)),
                v("çıkış kanalı 8", breakout=dataclasses.replace(p, exit_n=8)),
                v("çıkış kanalı 12", breakout=dataclasses.replace(p, exit_n=12)),
                v("stop 1,5 ATR", overrides=A._merge(c.overrides, {"risk": {**stop, "stop_atr_mult": 1.5}})),
                v("stop 2,5 ATR", overrides=A._merge(c.overrides, {"risk": {**stop, "stop_atr_mult": 2.5}}))]
    raise ValueError(f"Bilinmeyen aday: {c.code}")


def build_signals(c: T3Candidate, cfg_c: Config, market: A.Market) -> dict[str, pd.DataFrame]:
    btc_daily = market[BTC]["1d"]
    if c.kind == "momentum":
        return momentum_signals({s: f["1d"] for s, f in market.items()}, btc_daily, c.momentum, cfg_c)
    signals = {s: breakout_signals(f[c.entry_tf], c.breakout, cfg_c) for s, f in market.items()}
    if c.regime_sma:
        regime = btc_regime(btc_daily, c.regime_sma)
        signals = {s: apply_regime(df, regime, TIMEFRAME_MS[c.entry_tf]) for s, df in signals.items()}
    return signals


def load_tur3_market(cfg: Config) -> A.Market:
    raw = {}
    for symbol in cfg.symbols:
        path = candle_path(cfg.data_dir, cfg.exchange.id, symbol, "1h")
        if not path.exists():
            raise FileNotFoundError(path.name)
        raw[symbol] = {"1h": load_candles(path)}
    if BTC not in raw:
        raise ValueError("BTC rejim filtresi için BTC/USDT verisi gerekli.")
    return A.prepare_market(raw, parse_utc(cfg.data.exam_start))


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------
@dataclass
class T3Report:
    candidate: T3Candidate
    portfolio: dict[str, A.PeriodRun]
    coins: dict[str, dict[str, A.PeriodRun]]
    robustness: list[tuple[str, dict]]
    no_halt: dict[str, dict]
    monte_carlo: dict
    feasibility: dict[str, dict]
    adds: dict[str, int]
    checks: dict[str, tuple[bool | None, str]] = field(default_factory=dict)
    ladder: dict[float, dict[str, dict]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.candidate.reference and all(ok for ok, _ in self.checks.values())


@dataclass
class T3Study:
    created: str
    symbols: list[str]
    initial_equity: float
    bounds: dict[str, tuple[int, int]]
    reports: list[T3Report]
    shadow: list[str]
    bench_curves: dict[str, pd.Series]


def _metrics(result, initial: float) -> dict:
    m = compute_metrics(result.equity, result.trades, initial, result.total_fees, result.exposure)
    m["halted_at"] = fmt_utc(result.halted_at, False) if result.halted_at else None
    return m


def run_candidate(c: T3Candidate, cfg: Config, market: A.Market, bounds: dict[str, tuple[int, int]],
                  rules: dict[str, MarketRules], real_rules: dict[str, MarketRules], progress: A.Progress
                  ) -> T3Report:
    cfg_c = A.apply_overrides(cfg, c.overrides)
    signals = build_signals(c, cfg_c, market)
    portfolio, adds = {}, {}
    for name, label in A.PERIODS:
        t0 = time.perf_counter()
        run = A.run_period(cfg_c, signals, market, c.entry_tf, *bounds[name], rules)
        portfolio[name] = run
        adds[name] = run.result.adds_submitted
        m = run.metrics
        progress(f"  {c.code} {label:<10} {m['n_trades']:>4} işlem  {A._pct(m['total_return']):>8}  "
                 f"düşüş {A._pct(m['max_drawdown'], False):>6}  ({time.perf_counter() - t0:.1f} sn)")

    coins = {}
    if c.kind == "breakout":            # momentum coinler arası sıralamadır; tek coinlik hesap anlamsız
        for symbol in signals:
            coins[symbol] = {name: A.run_period(cfg_c, {symbol: signals[symbol]}, {symbol: market[symbol]},
                                                c.entry_tf, *bounds[name], rules) for name, _ in A.PERIODS}

    initial = cfg.backtest.initial_equity
    robustness = []
    for label, variant in robustness_variants(c):
        cfg_v = A.apply_overrides(cfg, variant.overrides)
        signals_v = build_signals(variant, cfg_v, market)
        start = max(A.trade_window(signals_v, cfg_v.entry_ms)[0], bounds["kesif"][0])
        result = run_backtest(cfg_v, signals_v, start, bounds["kesif"][1], initial, rules)
        robustness.append((label, _metrics(result, initial)))

    cfg_nh = A.apply_overrides(cfg_c, {"risk": {"max_drawdown_halt": NO_HALT}})
    no_halt = {name: _metrics(run_backtest(cfg_nh, signals, *bounds[name], initial, rules), initial)
               for name, _ in A.PERIODS}

    trades = portfolio["kesif"].result.trades + portfolio["dogrulama"].result.trades
    mc = A.monte_carlo([t.r_multiple for t in trades], cfg_c.risk.risk_per_trade)

    feasibility = {}
    for name, _ in A.PERIODS:
        result = run_backtest(cfg_c, signals, *bounds[name], FEASIBILITY_EQUITY, real_rules)
        feasibility[name] = {"metrics": _metrics(result, FEASIBILITY_EQUITY), "signals": result.signals_seen,
                             "skipped_min_notional": int(result.skipped.get(MIN_NOTIONAL_SKIP, 0))}
    progress(f"  {c.code} sağlamlık {len(robustness)} varyant, şaltersiz tanı, Monte Carlo, 50 USDT tamam")
    return T3Report(c, portfolio, coins, robustness, no_halt, mc, feasibility, adds)


def evaluate(r: T3Report, ref: T3Report) -> dict[str, tuple[bool | None, str]]:
    k, d = r.portfolio["kesif"].metrics, r.portfolio["dogrulama"].metrics
    checks: dict[str, tuple[bool | None, str]] = {}
    problems = []
    if not k["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not k["profit_factor"] >= A.EXPLORE_MIN_PF:
        problems.append(f"KF {A._ratio(k['profit_factor'])} < {A.EXPLORE_MIN_PF}")
    if not k["max_drawdown"] <= A.MAX_DRAWDOWN:
        problems.append(f"düşüş %{k['max_drawdown'] * 100:.1f}")
    checks["kesif"] = (not problems, "; ".join(problems) or "tümü sağlandı")

    problems = []
    if not d["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not d["profit_factor"] >= A.VALIDATE_MIN_PF:
        problems.append(f"KF {A._ratio(d['profit_factor'])} < {A.VALIDATE_MIN_PF}")
    if not d["max_drawdown"] <= A.MAX_DRAWDOWN:
        problems.append(f"düşüş %{d['max_drawdown'] * 100:.1f}")
    checks["dogrulama"] = (not problems, "; ".join(problems) or "tümü sağlandı")

    if r.candidate.reference:
        checks["iyilesme"] = (None, "referans")
    else:
        better = []
        for name, label in A.PERIODS:
            mine = r.portfolio[name].metrics["return_to_drawdown"]
            base = ref.portfolio[name].metrics["return_to_drawdown"]
            better.append((mine > base, f"{label.lower()} {mine:+.2f} / R0 {base:+.2f}"))
        checks["iyilesme"] = (all(ok for ok, _ in better), "; ".join(text for _, text in better))

    if r.candidate.reference:
        checks["saglamlik"] = (None, "referans")
    else:
        weak = [label for label, m in r.robustness if not m["profit_factor"] >= A.ROBUST_MIN_PF]
        checks["saglamlik"] = (bool(r.robustness) and not weak,
                               f"KF < {A.ROBUST_MIN_PF}: {', '.join(weak)}" if weak else "tüm varyantlarda KF ≥ 1")

    n = k["n_trades"] + d["n_trades"]
    checks["islem"] = (n >= MIN_TRADES, f"{n} işlem (en az {MIN_TRADES})")
    return checks


def run_ladder(r: T3Report, cfg: Config, market: A.Market, bounds, rules) -> dict[float, dict[str, dict]]:
    out = {}
    for risk in RISK_LADDER:
        c = r.candidate
        cfg_r = A.apply_overrides(cfg, A._merge(c.overrides, {"risk": {**c.overrides.get("risk", {}),
                                                                       "risk_per_trade": risk}}))
        signals = build_signals(c, cfg_r, market)
        initial = cfg.backtest.initial_equity
        out[risk] = {name: _metrics(run_backtest(cfg_r, signals, *bounds[name], initial, rules), initial)
                     for name, _ in A.PERIODS}
    return out


def common_bounds(cfg: Config, market: A.Market, candidates: list[T3Candidate]) -> dict[str, tuple[int, int]]:
    """Tüm adaylar aynı anda başlar: en yavaş göstergenin hazır olduğu an (protokol)."""
    starts, ends = [], []
    for c in candidates:
        cfg_c = A.apply_overrides(cfg, c.overrides)
        start, end = A.trade_window(build_signals(c, cfg_c, market), cfg_c.entry_ms)
        starts.append(start)
        ends.append(end)
    start = max(starts)
    start = -(-start // DAY_MS) * DAY_MS               # bir sonraki UTC gün başına yuvarla (tüm dilimler hizalı)
    split, exam = parse_utc(A.VALIDATION_START), parse_utc(cfg.data.exam_start)
    end = min(*ends, exam)
    if not start < split < end:
        raise ValueError("Veri keşif ve doğrulama dönemlerini kapsamıyor.")
    return {"kesif": (start, split), "dogrulama": (split, end)}


def run_tur3(cfg: Config, market: A.Market, rules: dict[str, MarketRules], real_rules: dict[str, MarketRules],
             progress: A.Progress | None = None) -> T3Study:
    progress = progress or (lambda _msg: None)
    candidates = tur3_candidates()
    bounds = common_bounds(cfg, market, candidates)
    progress(f"Ortak dönemler: keşif {fmt_utc(bounds['kesif'][0], False)} → {fmt_utc(bounds['kesif'][1], False)}, "
             f"doğrulama {fmt_utc(bounds['dogrulama'][0], False)} → {fmt_utc(bounds['dogrulama'][1], False)}")
    reports = []
    for c in candidates:
        progress(f"{c.code} · {c.name}")
        reports.append(run_candidate(c, cfg, market, bounds, rules, real_rules, progress))
    ref = reports[0]
    for r in reports:
        r.checks = evaluate(r, ref)
    passed = sorted((r for r in reports if r.passed),
                    key=lambda r: r.portfolio["dogrulama"].metrics["return_to_drawdown"], reverse=True)
    shadow = [r.candidate.code for r in passed[:MAX_SHADOW]]
    for r in reports:
        if r.candidate.reference or r.passed:
            r.ladder = run_ladder(r, cfg, market, bounds, rules)
            progress(f"  {r.candidate.code} risk merdiveni tamam")
    bench_curves = {}
    for name, _ in A.PERIODS:
        bench_curves[name] = buy_and_hold({s: f["4h"] for s, f in market.items()}, *bounds[name],
                                          cfg.backtest.initial_equity, cfg.costs.taker_fee, cfg.costs.slippage,
                                          TIMEFRAME_MS["4h"])
    return T3Study(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), list(market), cfg.backtest.initial_equity,
                   bounds, reports, shadow, bench_curves)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
_pct, _ratio, _mark, _table = A._pct, A._ratio, A._mark, A._table


def _halt(m: dict) -> str:
    return m.get("halted_at") or "-"


def format_tur3(study: T3Study) -> str:
    b = study.bounds
    out = [
        f"### 3. tur sonuçları ({study.created})",
        "",
        f"Coinler ({len(study.symbols)}): {', '.join(s.split('/')[0] for s in study.symbols)}. "
        f"Keşif {fmt_utc(b['kesif'][0], False)} → {fmt_utc(b['kesif'][1], False)}, doğrulama "
        f"{fmt_utc(b['dogrulama'][0], False)} → {fmt_utc(b['dogrulama'][1], False)}; sınav dönemi kullanılmadı. "
        f"Her dönem {study.initial_equity:,.0f} USDT ile ayrı başlar; komisyon ve kayma dahil.",
        "",
    ]
    for name, label in A.PERIODS:
        rows = []
        for r in study.reports:
            run = r.portfolio[name]
            m, bench = run.metrics, run.benchmark
            rows.append([f"{r.candidate.code} {r.candidate.name}", _pct(m["total_return"]),
                         _pct(m["max_drawdown"], False), f"{m['return_to_drawdown']:+.2f}",
                         _ratio(m["profit_factor"]), str(m["n_trades"]), _pct(m["win_rate"], False),
                         f"{m['avg_r']:+.2f}", str(r.adds.get(name, 0)),
                         f"{_pct(bench['total_return'])} / {_pct(bench['max_drawdown'], False)}",
                         fmt_utc(run.result.halted_at, False) if run.result.halted_at else "-"])
        out += [f"**{label} dönemi (tek hesap, en fazla 5 pozisyon)**", ""]
        out += _table(["Aday", "Net getiri", "Maks. düşüş", "Getiri/düşüş", "Kâr faktörü", "İşlem", "Kazanma",
                       "Ort. R", "Birim ekleme", "Al-tut getiri / düşüş", "Şalter"], rows)
        out.append("")

    rows = []
    for r in study.reports:
        cells = [f"{_mark(ok)} ({text})" for ok, text in r.checks.values()]
        verdict = "referans" if r.candidate.reference else ("**GEÇTİ**" if r.passed else "KALDI")
        rows.append([r.candidate.code, *cells, verdict])
    out += ["**Ölçütler**", ""]
    out += _table(["Aday", "1. Keşif", "2. Doğrulama", "3. İyileşme (R0'a göre)", "4. Sağlamlık",
                   "5. İşlem sayısı", "Sonuç"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        for name, label in A.PERIODS:
            m = r.no_halt[name]
            rows.append([r.candidate.code, label, _pct(m["total_return"]), _pct(m["max_drawdown"], False),
                         _ratio(m["profit_factor"]), str(m["n_trades"])])
    out += ["**Şaltersiz tanı (bilgi: %20 şalteri kapalı; şalterden sonraki davranışı gösterir)**", ""]
    out += _table(["Aday", "Dönem", "Net getiri", "Maks. düşüş", "Kâr faktörü", "İşlem"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        if not r.ladder:
            continue
        for name, label in A.PERIODS:
            base = r.portfolio[name].metrics
            cells = [f"{_pct(base['total_return'])} / {_pct(base['max_drawdown'], False)} / {_halt_run(r, name)}"]
            for risk in RISK_LADDER:
                m = r.ladder[risk][name]
                cells.append(f"{_pct(m['total_return'])} / {_pct(m['max_drawdown'], False)} / {_halt(m)}")
            rows.append([r.candidate.code, label, *cells])
    out += ["**Risk merdiveni (bilgi): net getiri / maks. düşüş / şalter**", ""]
    out += _table(["Aday", "Dönem", "%0,5 risk", *[f"%{x * 100:.2g} risk".replace(".", ",") for x in RISK_LADDER]],
                  rows)
    out.append("")

    rows = []
    for r in study.reports:
        base = r.portfolio["kesif"].metrics
        rows.append([r.candidate.code, "ana parametreler", _ratio(base["profit_factor"]), _pct(base["total_return"]),
                     str(base["n_trades"])])
        for label, m in r.robustness:
            rows.append(["", label, _ratio(m["profit_factor"]), _pct(m["total_return"]), str(m["n_trades"])])
    out += ["**Sağlamlık (keşif, yeni parametreler tek tek ±%25)**", ""]
    out += _table(["Aday", "Değişiklik", "Kâr faktörü", "Net getiri", "İşlem"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        mc = r.monte_carlo
        rows.append([r.candidate.code, str(mc["n"]), f"{mc['avg_r']:+.2f}", _pct(mc["actual_dd"], False),
                     _pct(mc["median_dd"], False), _pct(mc["p95_dd"], False)])
    out += [f"**Monte Carlo (işlem sırası {A.MC_RUNS} kez karıştırıldı, adayın kendi riskiyle)**", ""]
    out += _table(["Aday", "İşlem", "Ort. R", "Gerçek sırayla maks. düşüş", "Medyan", "En kötü %5"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        for name, label in A.PERIODS:
            f = r.feasibility[name]
            m = f["metrics"]
            rows.append([r.candidate.code, label, _pct(m["total_return"]), str(m["n_trades"]), str(f["signals"]),
                         str(f["skipped_min_notional"])])
    out += [f"**Uygulanabilirlik ({FEASIBILITY_EQUITY:.2f} USDT, Binance'in gerçek minimum işlem tutarlarıyla)**", ""]
    out += _table(["Aday", "Dönem", "Net getiri", "İşlem", "Sinyal", "Minimum tutar yüzünden atlanan"], rows)
    out.append("")

    for r in study.reports:
        if not r.coins:
            continue
        rows = []
        for symbol, runs in r.coins.items():
            k, d = runs["kesif"].metrics, runs["dogrulama"].metrics
            rows.append([symbol.split("/")[0], _pct(k["total_return"]), _ratio(k["profit_factor"]), str(k["n_trades"]),
                         _pct(d["total_return"]), _ratio(d["profit_factor"]), str(d["n_trades"])])
        out += [f"**{r.candidate.code}: coin bazında (her coin ayrı hesapta)**", ""]
        out += _table(["Coin", "Keşif net", "Keşif KF", "Keşif işlem", "Doğrulama net", "Doğrulama KF",
                       "Doğrulama işlem"], rows)
        out.append("")

    if study.shadow:
        out.append(f"**Karar: {', '.join(study.shadow)} ölçütlerin tamamını geçti.** Protokol gereği C1'in yanında "
                   "gölge (sanal) bot olarak en az 4 hafta / 30 işlem ileriye doğru sınanacak.")
    else:
        out.append("**Karar: hiçbir aday ölçütlerin tamamını geçemedi.** Protokol gereği C1 değişmeden devam eder.")
    return "\n".join(out)


def _halt_run(r: T3Report, name: str) -> str:
    run = r.portfolio[name]
    return fmt_utc(run.result.halted_at, False) if run.result.halted_at else "-"


def tur3_summary(study: T3Study) -> dict:
    def period(run: A.PeriodRun) -> dict:
        return {"start": fmt_utc(run.start, False), "end": fmt_utc(run.end, False), "metrics": run.metrics,
                "symbol_net": run.symbol_net, "buy_and_hold": run.benchmark, "halt": run.result.halt_reason or None}

    return A._clean({
        "created": study.created,
        "symbols": study.symbols,
        "bounds": {k: [fmt_utc(a, False), fmt_utc(b, False)] for k, (a, b) in study.bounds.items()},
        "shadow": study.shadow,
        "candidates": [{
            "code": r.candidate.code,
            "name": r.candidate.name,
            "reference": r.candidate.reference,
            "passed": r.passed,
            "checks": {k: {"passed": ok, "detail": text} for k, (ok, text) in r.checks.items()},
            "portfolio": {name: period(run) for name, run in r.portfolio.items()},
            "adds": r.adds,
            "no_halt": r.no_halt,
            "robustness": [{"variant": label, "metrics": m} for label, m in r.robustness],
            "monte_carlo": r.monte_carlo,
            "feasibility": r.feasibility,
            "ladder": {str(k): v for k, v in r.ladder.items()},
            "coins": {s: {name: period(run) for name, run in runs.items()} for s, runs in r.coins.items()},
        } for r in study.reports],
    })


def save_tur3(out_dir: Path, study: T3Study, text: str, plot: bool = True) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(tur3_summary(study), indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    rows = []
    for r in study.reports:
        for name, run in r.portfolio.items():
            for t in run.result.trades:
                rows.append({"aday": r.candidate.code, "donem": name, **asdict(t),
                             "entry_utc": fmt_utc(t.entry_ts), "exit_utc": fmt_utc(t.exit_ts)})
    pd.DataFrame(rows).to_csv(out_dir / "islemler.csv", index=False, encoding="utf-8-sig")
    if not plot:
        return None
    chart = out_dir / "grafik.png"
    plot_tur3(chart, study)
    return chart


def plot_tur3(path: Path, study: T3Study) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"R0": "#4C72B0", "R1": "#55A868", "R2": "#8172B3", "R3": "#C44E52", "R4": "#937860"}
    fig, axes = plt.subplots(len(A.PERIODS), 1, figsize=(12, 9))
    for ax, (name, label) in zip(axes, A.PERIODS, strict=True):
        for r in study.reports:
            curve = daily_equity(r.portfolio[name].result.equity) / study.initial_equity * 100
            dates = pd.to_datetime((curve.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True)
            ax.plot(dates, curve.to_numpy(), linewidth=1.5, color=colors.get(r.candidate.code),
                    label=f"{r.candidate.code} {r.candidate.name}")
        bench = daily_equity(study.bench_curves[name]) / study.initial_equity * 100
        ax.plot(pd.to_datetime((bench.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True), bench.to_numpy(),
                color="#DD8452", linewidth=1.0, linestyle="--", label="Al-tut (12 coin eşit)")
        ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_title(f"{label} dönemi (başlangıç = 100) · komisyon ve kayma dahil", loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="3. araştırma turu: C1'i geliştirme adayları.")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    try:
        market = load_tur3_market(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[HATA] {exc}")
        return 2
    print(f"Veri yüklendi; {cfg.data.exam_start} (sınav) sonrası mumlar atıldı.")
    rules = load_market_rules(cfg.data_dir, list(market), cfg.market_rules_default)   # 1. ve 2. turla aynı
    real_rules = rules
    try:
        study = run_tur3(cfg, market, rules, real_rules, progress=print)
    except ValueError as exc:
        print(f"[HATA] {exc}")
        return 2
    text = format_tur3(study)
    print("\n" + text)
    out_dir = cfg.reports_dir / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_arastirma_tur3"
    chart = save_tur3(out_dir, study, text, plot=not args.no_plot)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
