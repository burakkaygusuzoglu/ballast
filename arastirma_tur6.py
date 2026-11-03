"""6. araştırma turu: vadeli işlemler (futures), kaldıraç ve short.

Protokol: docs/ARASTIRMA.md, 17. bölüm (aynı metin: docs/tur6_protokol.md). Adaylar, model, ölçütler ve seçim
kuralı sonuçlar görülmeden önce yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey değiştirmek
testi geçersiz kılar.

- F-C1:  C1 vadelide, yalnızca long
- F-R1:  R1 (C1 + BTC rejim filtresi) vadelide, yalnızca long
- F-R1S: F-R1 + aynalı short (yalnızca BTC günlük kapanışı SMA200 altındayken)
- Her aday 7 risk seviyesinde (%0,5 → %2,5), 3x izole marjin, kararlar STRES maliyet modeliyle.

    python arastirma_tur6.py           # Aşama A: geliştirme dönemleri (sınav verisi YÜKLENMEZ)
    python arastirma_tur6.py --sinav   # Aşama A + Aşama B (sınav dönemi, yalnızca bilgi; bir kez)

Çıktı: reports/<zaman>_arastirma_tur6_A (ya da _B) altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import arastirma as A
import arastirma_tur3 as T3
import arastirma_tur4 as T4
import arastirma_tur5 as T5
from bot.candidates import (
    BreakoutParams,
    add_short_side,
    apply_regime,
    apply_regime_two_sided,
    breakout_signals,
    btc_regime,
)
from bot.cli import disable_quick_edit, load_config_or_exit, open_file, setup_console
from bot.config import PROJECT_ROOT, Config, MarketRules
from bot.data import load_market_rules
from bot.metrics import daily_equity
from bot.timeutil import DAY_MS, TIMEFRAME_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 17. bölüm) ------------------------------------
PROTOCOL_FILE = PROJECT_ROOT / "docs" / "tur6_protokol.md"
PROTOCOL_SHA256 = "8deb024e2bf70914db46d8d093dd07d0bdf6e9583d9055d104143f3c365c1708"
BTC = T3.BTC
SYMBOLS = T4.C1_SYMBOLS
PERIOD_BOUNDS = T4.PROTOCOL_BOUNDS                 # keşif 2022-07-20 → 2024-12-31, doğrulama 2025 → 2026-03-31
RISK_GRID = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025)
BASE_RISK = 0.005
PROFILES = {"dengeli": (0.5, 0.30), "agresif": (0.75, 0.45)}   # (Kelly kesri, iki dönemde en fazla düşüş)
VALIDATION_MIN_PF = 1.1
TIE_TOLERANCE = 0.05
SIMPLICITY = ("F-C1", "F-R1", "F-R1S")
LEVERAGE, MAINTENANCE, MAX_STOP_TO_LIQ = 3.0, 0.02, 0.8
FUTURES_COSTS = {"maker_fee": 0.0002, "taker_fee": 0.0005, "slippage": 0.0005}
MODELS = {"stres": {"funding_long_8h": 0.0003, "funding_short_8h": 0.0001, "liquidation": "touch"},
          "temel": {"funding_long_8h": 0.0001, "funding_short_8h": -0.0001, "liquidation": "open"}}
FUTURES_POSITION_PCT = 1.0
DAILY_LIMIT_PER_RISK, HALT_PER_RISK, HALT_CAP = 6, 40, 0.60
NO_HALT = T3.NO_HALT
MC_BLOCK_DAYS, MC_PATHS, MC_SEED, MC_DAYS = 20, 10_000, 20261015, 365
EXAM_START = T5.EXAM_START
C1_PARAMS = BreakoutParams(entry_n=20, exit_n=10, atr_n=20)
H4 = TIMEFRAME_MS["4h"]


@dataclass(frozen=True)
class FCandidate:
    code: str
    name: str
    regime_sma: int | None
    allow_short: bool


CANDIDATES = (
    FCandidate("F-C1", "C1 vadelide, yalnızca long", None, False),
    FCandidate("F-R1", "R1 vadelide (BTC > SMA200), yalnızca long", 200, False),
    FCandidate("F-R1S", "R1 + aynalı short (BTC < SMA200 iken)", 200, True),
)
REFERENCES = {"C1": "C1 spot (canlı bot)", "R1": "R1 spot (gölge bot)"}


def futures_overrides(risk: float, model: str, allow_short: bool, halt: bool = True) -> dict:
    """C1 kurallarının üstüne vadeli işlem modeli ve risk seviyesine göre ölçeklenen şalterler."""
    overrides = A._merge(T3.C1_RULES, {
        "exchange": {"market_type": "futures"},
        "costs": dict(FUTURES_COSTS),
        "futures": {"leverage": LEVERAGE, "maintenance_margin": MAINTENANCE, "max_stop_to_liq": MAX_STOP_TO_LIQ,
                    **MODELS[model]},
        "risk": {"risk_per_trade": risk, "max_position_pct": FUTURES_POSITION_PCT,
                 "daily_loss_limit": DAILY_LIMIT_PER_RISK * risk,
                 "max_drawdown_halt": min(HALT_PER_RISK * risk, HALT_CAP) if halt else NO_HALT},
        "strategy": {"allow_short": allow_short},
    })
    return overrides


def build_signals(c: FCandidate, cfg_c: Config, market: A.Market) -> dict[str, pd.DataFrame]:
    """Sinyaller maliyet filtresi yüzünden komisyona bağlıdır (vadeli komisyonlarla); riske ve modele bağlı değildir."""
    signals = {s: breakout_signals(f["4h"], C1_PARAMS, cfg_c) for s, f in market.items()}
    if c.allow_short:
        signals = {s: add_short_side(df, C1_PARAMS) for s, df in signals.items()}
    if c.regime_sma:
        regime = btc_regime(market[BTC]["1d"], c.regime_sma)
        apply = apply_regime_two_sided if c.allow_short else apply_regime
        signals = {s: apply(df, regime, H4) for s, df in signals.items()}
    return signals


def side_net(run: A.PeriodRun, side: int, closes: dict[str, float]) -> float:
    """Bir yönün net kârı: kapanan işlemler + dönem sonundaki açık pozisyonlar (funding ve komisyon dahil)."""
    total = sum(t.net_pnl for t in run.result.trades if t.side == side)
    total += sum(p.unrealized(closes[p.symbol]) - p.entry_fee - p.funding_paid
                 for p in run.result.open_positions if p.side == side)
    return total


def last_closes(market: A.Market, end: int) -> dict[str, float]:
    return {s: float(f["4h"].loc[f["4h"]["ts"] < end, "close"].iloc[-1]) for s, f in market.items()}


# ---------------------------------------------------------------------------
# Aşama A
# ---------------------------------------------------------------------------
@dataclass
class Selection:
    profile: str
    candidate: str
    risk: float
    checks: dict[str, tuple[bool | None, str]]

    @property
    def passed(self) -> bool:
        return all(ok is not False for ok, _ in self.checks.values())


@dataclass
class StageA:
    created: str
    bounds: dict[str, tuple[int, int]]
    runs: dict[tuple[str, float, str, str], A.PeriodRun]      # (aday, risk, model, dönem)
    references: dict[tuple[str, str], A.PeriodRun]             # (C1/R1, dönem), spot %0,5
    f_star: dict[str, float]
    selections: dict[tuple[str, str], Selection]               # (profil, aday)
    chosen: dict[str, str | None]                              # profil -> aday
    side_nets: dict[tuple[str, float, str], dict[int, float]]  # (aday, risk, dönem) -> {1: long, -1: short}
    no_halt: dict[tuple[str, float, str], A.PeriodRun] = field(default_factory=dict)
    monte_carlo: dict[str, dict] = field(default_factory=dict)


def protocol_check() -> str:
    digest = hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise AssertionError(f"Protokol dosyası değişmiş: {digest} != {PROTOCOL_SHA256}")
    return digest


def period_bounds(cfg: Config, market: A.Market) -> dict[str, tuple[int, int]]:
    bounds = {name: (parse_utc(a), parse_utc(b)) for name, (a, b) in PERIOD_BOUNDS.items()}
    if bounds["dogrulama"][1] > parse_utc(cfg.data.exam_start):
        raise ValueError("Doğrulama dönemi sınav dönemine taşıyor.")
    for c in CANDIDATES:
        cfg_c = A.apply_overrides(cfg, futures_overrides(BASE_RISK, "stres", c.allow_short))
        start, end = A.trade_window(build_signals(c, cfg_c, market), cfg_c.entry_ms)
        if start > bounds["kesif"][0] or end < bounds["dogrulama"][1]:
            raise ValueError(f"{c.code}: veri protokol dönemlerini kapsamıyor "
                             f"({fmt_utc(start, False)} → {fmt_utc(end, False)}).")
    return bounds


def pick_risk(f_star: float, fraction: float) -> float:
    eligible = [r for r in RISK_GRID if r <= fraction * f_star + 1e-12]
    return max(eligible) if eligible else BASE_RISK


def evaluate(s: StageA, c: FCandidate, profile: str) -> Selection:
    fraction, dd_limit = PROFILES[profile]
    risk = pick_risk(s.f_star[c.code], fraction)
    k, d = (s.runs[(c.code, risk, "stres", "kesif")].metrics, s.runs[(c.code, risk, "stres", "dogrulama")].metrics)
    checks: dict[str, tuple[bool | None, str]] = {}
    checks["1_kar"] = (k["net_profit"] > 0 and d["net_profit"] > 0,
                       f"keşif {A._pct(k['total_return'])}, doğrulama {A._pct(d['total_return'])}")
    checks["2_dusus"] = (k["max_drawdown"] <= dd_limit and d["max_drawdown"] <= dd_limit,
                         (f"keşif {A._pct(k['max_drawdown'], False)}, doğrulama {A._pct(d['max_drawdown'], False)} "
                         f"(sınır %{dd_limit * 100:.0f})"))
    checks["3_kf"] = (d["profit_factor"] >= VALIDATION_MIN_PF, f"doğrulama KF {A._ratio(d['profit_factor'])}")
    if risk > BASE_RISK:
        base = s.runs[(c.code, BASE_RISK, "stres", "dogrulama")].metrics["cagr"]
        checks["4_risk_karsiligi"] = (d["cagr"] > base, f"doğrulama CAGR {A._pct(d['cagr'])} / %0,5'te {A._pct(base)}")
    else:
        checks["4_risk_karsiligi"] = (None, "seçilen seviye %0,5: uygulanmaz")
    if c.allow_short:
        nets = [s.side_nets[(c.code, risk, period)][-1] for period in ("kesif", "dogrulama")]
        checks["5_short"] = (all(n > 0 for n in nets), (f"short net kâr: keşif {nets[0]:+,.0f}, doğrulama "
                                                         f"{nets[1]:+,.0f} USDT"))
    liq = [s.runs[(c.code, risk, "stres", p)].result.liquidations for p in ("kesif", "dogrulama")]
    checks["6_likidasyon"] = (None, f"stres likidasyonu: keşif {liq[0]}, doğrulama {liq[1]} (kayıplar dahil)")
    return Selection(profile, c.code, risk, checks)


def choose(s: StageA, profile: str) -> str | None:
    passed = [code for code in SIMPLICITY if s.selections[(profile, code)].passed]
    if not passed:
        return None
    rtd = {code: s.runs[(code, s.selections[(profile, code)].risk, "stres", "dogrulama")].metrics["return_to_drawdown"]
           for code in passed}
    best = max(rtd.values())
    close = [code for code in passed if best - rtd[code] < TIE_TOLERANCE * abs(best)]
    return close[0]                                   # SIMPLICITY sırasıyla: en basit olan


def run_stage_a(cfg: Config, market: A.Market, rules: dict[str, MarketRules],
                progress: Callable[[str], None] | None = None) -> StageA:
    progress = progress or (lambda _msg: None)
    bounds = period_bounds(cfg, market)
    closes = {name: last_closes(market, end) for name, (_, end) in bounds.items()}
    runs, side_nets = {}, {}
    for c in CANDIDATES:
        sig = build_signals(c, A.apply_overrides(cfg, futures_overrides(BASE_RISK, "stres", c.allow_short)), market)
        for risk in RISK_GRID:
            for model in MODELS:
                cfg_c = A.apply_overrides(cfg, futures_overrides(risk, model, c.allow_short))
                for name, (start, end) in bounds.items():
                    run = A.run_period(cfg_c, sig, market, "4h", start, end, rules)
                    runs[(c.code, risk, model, name)] = run
                    if model == "stres":
                        side_nets[(c.code, risk, name)] = {1: side_net(run, 1, closes[name]),
                                                           -1: side_net(run, -1, closes[name])}
            k, d = runs[(c.code, risk, "stres", "kesif")].metrics, runs[(c.code, risk, "stres", "dogrulama")].metrics
            progress(f"  {c.code:<6} risk %{risk * 100:<5.2f} keşif CAGR {A._pct(k['cagr']):>7} düşüş "
                     f"{A._pct(k['max_drawdown'], False):>6} | doğrulama CAGR {A._pct(d['cagr']):>7} düşüş "
                     f"{A._pct(d['max_drawdown'], False):>6}")
    references = {}
    t3 = {x.code: x for x in T3.tur3_candidates()}
    for code, t3code in (("C1", "R0"), ("R1", "R1")):
        cfg_c = A.apply_overrides(cfg, t3[t3code].overrides)
        sig = T3.build_signals(t3[t3code], cfg_c, market)
        for name, (start, end) in bounds.items():
            references[(code, name)] = A.run_period(cfg_c, sig, market, "4h", start, end, rules)
    f_star = {}
    for c in CANDIDATES:
        cagr = {r: runs[(c.code, r, "stres", "kesif")].metrics["cagr"] for r in RISK_GRID}
        f_star[c.code] = max(RISK_GRID, key=lambda r, cagr=cagr: (cagr[r], -r))
    s = StageA(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), bounds, runs, references, f_star, {}, {}, side_nets)
    for profile in PROFILES:
        for c in CANDIDATES:
            s.selections[(profile, c.code)] = evaluate(s, c, profile)
        s.chosen[profile] = choose(s, profile)
    # Bilgi: seçilen seviyelerde şaltersiz koşu ve Monte Carlo.
    for (profile, code), sel in s.selections.items():
        c = next(x for x in CANDIDATES if x.code == code)
        sig = build_signals(c, A.apply_overrides(cfg, futures_overrides(BASE_RISK, "stres", c.allow_short)), market)
        cfg_c = A.apply_overrides(cfg, futures_overrides(sel.risk, "stres", c.allow_short, halt=False))
        for name, (start, end) in bounds.items():
            if (code, sel.risk, name) not in s.no_halt:
                s.no_halt[(code, sel.risk, name)] = A.run_period(cfg_c, sig, market, "4h", start, end, rules)
    for profile, code in s.chosen.items():
        if code is not None:
            risk = s.selections[(profile, code)].risk
            s.monte_carlo[profile] = block_bootstrap([s.runs[(code, risk, "stres", p)] for p in ("kesif", "dogrulama")])
    return s


def block_bootstrap(runs: list[A.PeriodRun], days: int = MC_DAYS, paths: int = MC_PATHS,
                    block: int = MC_BLOCK_DAYS, seed: int = MC_SEED) -> dict:
    """Günlük getirilerin blok yeniden örneklemesi: 1 yıllık sonuç dağılımı (bilgi amaçlı)."""
    rets = np.concatenate([daily_equity(r.result.equity).pct_change().dropna().to_numpy() for r in runs])
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(days / block)
    starts = rng.integers(0, len(rets) - block + 1, size=(paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(paths, -1)[:, :days]
    curves = np.cumprod(1 + rets[idx], axis=1)
    finals = curves[:, -1]
    peaks = np.maximum.accumulate(np.concatenate([np.ones((paths, 1)), curves], axis=1), axis=1)[:, 1:]
    dd = np.max(1 - curves / peaks, axis=1)
    return {"medyan": float(np.median(finals)), "p10": float(np.percentile(finals, 10)),
            "p90": float(np.percentile(finals, 90)), "dusus30": float(np.mean(dd >= 0.30)),
            "dusus50": float(np.mean(dd >= 0.50)), "zarar": float(np.mean(finals < 1.0)),
            "medyan_dusus": float(np.median(dd)), "gun": len(rets)}


# ---------------------------------------------------------------------------
# Aşama B: sınav dönemi (yalnızca bilgi)
# ---------------------------------------------------------------------------
@dataclass
class StageB:
    start: int
    end: int
    runs: dict[str, A.PeriodRun]                              # etiket -> koşu


def run_stage_b(cfg: Config, market: A.Market, start: int, end: int, rules: dict[str, MarketRules],
                stage_a: StageA) -> StageB:
    if not (parse_utc(EXAM_START) == start < end):
        raise ValueError("Sınav dönemi 2026-04-01'de başlamalı ve en az bir tam gün içermeli.")
    runs = {}
    t3 = {x.code: x for x in T3.tur3_candidates()}
    for code, t3code in (("C1", "R0"), ("R1", "R1")):
        cfg_c = A.apply_overrides(cfg, t3[t3code].overrides)
        runs[f"{code} spot %0,5"] = A.run_period(cfg_c, T3.build_signals(t3[t3code], cfg_c, market), market, "4h",
                                                 start, end, rules)
    for profile, code in stage_a.chosen.items():
        if code is None:
            continue
        c = next(x for x in CANDIDATES if x.code == code)
        risk = stage_a.selections[(profile, code)].risk
        label = f"{code} %{risk * 100:g} ({profile})"
        if label in runs:
            continue
        cfg_c = A.apply_overrides(cfg, futures_overrides(risk, "stres", c.allow_short))
        runs[label] = A.run_period(cfg_c, build_signals(c, cfg_c, market), market, "4h", start, end, rules)
    return StageB(start, end, runs)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
def _pct_risk(r: float) -> str:
    return f"%{r * 100:g}".replace(".", ",")


def _status(ok: bool | None) -> str:
    return "-" if ok is None else ("GEÇTİ" if ok else "KALDI")


def format_stage_a(s: StageA) -> str:
    out = [f"### 6. tur, Aşama A: geliştirme dönemleri ({s.created})", "",
           ("Keşif 2022-07-20 → 2024-12-31, doğrulama 2025-01-01 → 2026-03-31; sınav dönemi yüklenmedi. Her dönem "
           "10.000 USDT ile ayrı başlar. Vadeli: 3x izole marjin, taker %0,05 + %0,05 kayma. Kararlar stres "
           "modeliyle: long 8 saatte %0,03 funding öder, short %0,01 öder; mum ucu likidasyona değerse likidasyon."),
           f"Protokol parmak izi: {PROTOCOL_SHA256[:16]}… (docs/tur6_protokol.md)", ""]
    out += ["**Referans (spot, %0,5 risk)**", ""]
    rows = []
    for code, label in REFERENCES.items():
        for name, plabel in A.PERIODS:
            m = s.references[(code, name)].metrics
            rows.append([label, plabel, A._pct(m["total_return"]), A._pct(m["cagr"]), A._pct(m["max_drawdown"], False),
                         A._ratio(m["profit_factor"]), str(m["n_trades"])])
    out += A._table(["Bot", "Dönem", "Net getiri", "CAGR", "Maks. düşüş", "KF", "İşlem"], rows) + [""]
    for c in CANDIDATES:
        out += [f"**{c.code}: {c.name}** (büyüme tepesi f\\* = {_pct_risk(s.f_star[c.code])})", ""]
        rows = []
        for risk in RISK_GRID:
            k = s.runs[(c.code, risk, "stres", "kesif")]
            d = s.runs[(c.code, risk, "stres", "dogrulama")]
            tk = s.runs[(c.code, risk, "temel", "kesif")].metrics["cagr"]
            td = s.runs[(c.code, risk, "temel", "dogrulama")].metrics["cagr"]
            row = [_pct_risk(risk),
                   A._pct(k.metrics["cagr"]), A._pct(k.metrics["max_drawdown"], False), A._ratio(k.metrics["profit_factor"]),
                   A._pct(d.metrics["cagr"]), A._pct(d.metrics["max_drawdown"], False), A._ratio(d.metrics["profit_factor"]),
                   f"{k.metrics['n_trades']}/{d.metrics['n_trades']}",
                   f"{k.result.liquidations}/{d.result.liquidations}", f"{A._pct(tk)} / {A._pct(td)}"]
            if c.allow_short:
                row.append(" / ".join(f"{s.side_nets[(c.code, risk, p)][-1]:+,.0f}" for p in ("kesif", "dogrulama")))
            rows.append(row)
        header = ["Risk", "Keşif CAGR", "Keşif düşüş", "Keşif KF", "Doğr. CAGR", "Doğr. düşüş", "Doğr. KF",
                  "İşlem (K/D)", "Likidasyon (K/D)", "Temel model CAGR (K / D)"]
        if c.allow_short:
            header.append("Short net USDT (K / D)")
        out += A._table(header, rows) + [""]
    labels = {"1_kar": "1. Kâr", "2_dusus": "2. Düşüş", "3_kf": "3. KF", "4_risk_karsiligi": "4. Risk karşılığı",
              "5_short": "5. Short", "6_likidasyon": "6. Likidasyon"}
    for profile, (fraction, dd_limit) in PROFILES.items():
        out += [f"**Profil: {profile}** (f\\* × {fraction:g}, düşüş sınırı %{dd_limit * 100:.0f})".replace(".", ","), ""]
        rows = []
        for c in CANDIDATES:
            sel = s.selections[(profile, c.code)]
            cells = [f"{_status(ok)} ({detail})" for key, (ok, detail) in sel.checks.items()]
            rows.append([c.code, _pct_risk(sel.risk), " · ".join(f"{labels[k]}: {v}" for k, v in
                                                                zip(sel.checks, cells, strict=True)),
                         "GEÇTİ" if sel.passed else "KALDI"])
        out += A._table(["Aday", "Seçilen risk", "Ölçütler", "Sonuç"], rows) + [""]
        chosen = s.chosen[profile]
        if chosen is None:
            out += [f"**{profile} kararı: hiçbir aday geçmedi.** Bu profil için vadeli bot yapılmaz.", ""]
        else:
            sel = s.selections[(profile, chosen)]
            d = s.runs[(chosen, sel.risk, "stres", "dogrulama")].metrics
            out += [(f"**{profile} kararı: {chosen}, {_pct_risk(sel.risk)} risk** (doğrulama getiri/düşüş "
                    f"{A._ratio(d['return_to_drawdown'])})."), ""]
    out += ["**Bilgi: seçilen seviyeler şaltersiz (stres modeli)**", ""]
    rows = []
    order = {name: i for i, (name, _) in enumerate(A.PERIODS)}
    for (code, risk, name), run in sorted(s.no_halt.items(), key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]])):
        m = run.metrics
        rows.append([code, _pct_risk(risk), dict(A.PERIODS)[name], A._pct(m["total_return"]),
                     A._pct(m["max_drawdown"], False), str(run.result.liquidations)])
    out += A._table(["Aday", "Risk", "Dönem", "Net getiri", "Maks. düşüş", "Likidasyon"], rows) + [""]
    if s.monte_carlo:
        paths = f"{MC_PATHS:,}".replace(",", ".")
        out += [f"**Bilgi: Monte Carlo, 1 yıl** (keşif + doğrulama günlük getirileri, 20 günlük bloklar, {paths} yol)",
                ""]
        rows = []
        for profile, mc in s.monte_carlo.items():
            code = s.chosen[profile]
            risk = s.selections[(profile, code)].risk
            rows.append([profile, f"{code} {_pct_risk(risk)}", f"{mc['medyan']:.2f}x", f"{mc['p10']:.2f}x",
                         f"{mc['p90']:.2f}x", A._pct(mc["medyan_dusus"], False), A._pct(mc["dusus30"], False),
                         A._pct(mc["dusus50"], False), A._pct(mc["zarar"], False),
                         f"{50.0 * (mc['medyan'] - 1):+.1f}"])
        out += A._table(["Profil", "Ayar", "Medyan", "%10 dilim", "%90 dilim", "Medyan düşüş", "Düşüş ≥ %30",
                         "Düşüş ≥ %50", "Yılı zararla bitirme", "50 USDT ile medyan kâr"], rows) + [""]
    return "\n".join(out)


def format_stage_b(b: StageB) -> str:
    out = [(f"### 6. tur, Aşama B: sınav dönemi ({fmt_utc(b.start, False)} → {fmt_utc(b.end - DAY_MS, False)}), "
           "yalnızca bilgi"), "", "Bu dönem 5. turda kullanıldı; sonuçlar karar değiştirmez.", ""]
    rows = []
    for label, run in b.runs.items():
        m = run.metrics
        rows.append([label, A._pct(m["total_return"]), A._pct(m["max_drawdown"], False), A._ratio(m["profit_factor"]),
                     str(m["n_trades"]), str(run.result.liquidations)])
    out += A._table(["Ayar", "Net getiri", "Maks. düşüş", "KF", "İşlem", "Likidasyon"], rows)
    return "\n".join(out)


def stage_a_summary(s: StageA) -> dict:
    grid = {f"{code}|{risk}|{model}|{name}": A._clean(run.metrics) | {"likidasyon": run.result.liquidations,
                                                                    "funding": run.result.total_funding}
            for (code, risk, model, name), run in s.runs.items()}
    return {"olusturuldu": s.created, "protokol_sha256": PROTOCOL_SHA256,
            "f_yildiz": s.f_star, "secim": {p: c for p, c in s.chosen.items()},
            "profiller": {f"{p}|{c}": {"risk": sel.risk, "gecti": sel.passed,
                                       "olcutler": {k: [ok, d] for k, (ok, d) in sel.checks.items()}}
                          for (p, c), sel in s.selections.items()},
            "monte_carlo": s.monte_carlo, "izgara": grid,
            "referans": {f"{c}|{n}": A._clean(r.metrics) for (c, n), r in s.references.items()}}


def save_report(out_dir: Path, summary: dict, text: str, trades: list[tuple[str, str, list]],
                plot: Callable[[Path], None] | None) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "islemler.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ayar", "donem", "sembol", "yon", "giris", "cikis", "giris_fiyat", "cikis_fiyat", "net_kar",
                    "funding", "r", "neden"])
        for label, period, items in trades:
            for t in items:
                w.writerow([label, period, t.symbol, "long" if t.side > 0 else "short", fmt_utc(t.entry_ts),
                            fmt_utc(t.exit_ts), f"{t.entry_price:.8g}", f"{t.exit_price:.8g}", f"{t.net_pnl:.4f}",
                            f"{t.funding:.4f}", f"{t.r_multiple:.3f}", t.exit_reason])
    if plot is None:
        return None
    chart = out_dir / "grafik.png"
    plot(chart)
    return chart


def plot_stage_a(path: Path, s: StageA) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"F-C1": "#4C72B0", "F-R1": "#64B5CD", "F-R1S": "#C44E52"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (name, label) in zip(axes[0], A.PERIODS, strict=True):
        for c in CANDIDATES:
            y = [s.runs[(c.code, r, "stres", name)].metrics["cagr"] * 100 for r in RISK_GRID]
            ax.plot([r * 100 for r in RISK_GRID], y, marker="o", color=colors[c.code], label=c.code)
        ax.axhline(s.references[("C1", name)].metrics["cagr"] * 100, color="gray", linestyle="--",
                   label="C1 spot %0,5")
        ax.set_title(f"{label}: işlem başına risk ve yıllık büyüme (stres modeli)")
        ax.set_xlabel("İşlem başına risk (%)")
        ax.set_ylabel("CAGR (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax, (name, label) in zip(axes[1], A.PERIODS, strict=True):
        initial = s.references[("C1", name)].metrics["initial_equity"]
        x, y = T5._curve(s.references[("C1", name)].result.equity, initial)
        ax.plot(x, y.to_numpy(), color="gray", linestyle="--", label="C1 spot %0,5")
        for profile, code in s.chosen.items():
            if code is None:
                continue
            risk = s.selections[(profile, code)].risk
            x, y = T5._curve(s.runs[(code, risk, "stres", name)].result.equity, initial)
            ax.plot(x, y.to_numpy(), color=colors[code], linestyle="-" if profile == "agresif" else ":",
                    label=f"{profile}: {code} {_pct_risk(risk)}")
        ax.set_title(f"{label}: özsermaye (başlangıç = 100)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()
    parser = argparse.ArgumentParser(description="6. araştırma turu: vadeli işlemler, kaldıraç ve short.")
    parser.add_argument("--sinav", action="store_true", help="Aşama B: sınav dönemi (yalnızca bilgi, bir kez)")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)
    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    protocol_check()
    try:
        raw = T5.load_raw(cfg)
    except FileNotFoundError as exc:
        print(f"[HATA] Veri yok: {exc}. Önce V1_tur4_veri_indir.bat çalıştırılmalı.")
        return 2
    rules = load_market_rules(cfg.data_dir, list(SYMBOLS), cfg.market_rules_default)
    market_dev = A.prepare_market(raw, parse_utc(cfg.data.exam_start))
    print(f"Geliştirme verisi yüklendi; {cfg.data.exam_start} (sınav) sonrası mumlar atıldı.")
    stage_a = run_stage_a(cfg, market_dev, rules, progress=print)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    trades = [(f"{code} %{risk * 100:g}", name, run.result.trades)
              for (code, risk, model, name), run in stage_a.runs.items() if model == "stres"
              and any(sel.candidate == code and sel.risk == risk for sel in stage_a.selections.values())]
    text = format_stage_a(stage_a)
    summary = {"asama_a": stage_a_summary(stage_a)}
    if args.sinav:
        end = T5.last_full_day(raw)
        market_full = A.prepare_market(raw, end)
        print(f"Sınav verisi yüklendi: {EXAM_START} → {fmt_utc(end - DAY_MS, False)} (son tam gün).")
        stage_b = run_stage_b(cfg, market_full, parse_utc(EXAM_START), end, rules, stage_a)
        text += "\n\n" + format_stage_b(stage_b)
        trades += [(label, "sinav", run.result.trades) for label, run in stage_b.runs.items()]
        summary["asama_b"] = {label: A._clean(run.metrics) | {"likidasyon": run.result.liquidations}
                              for label, run in stage_b.runs.items()}
    out_dir = cfg.reports_dir / f"{stamp}_arastirma_tur6_{'B' if args.sinav else 'A'}"
    chart = save_report(out_dir, summary, text, trades, None if args.no_plot else (lambda p: plot_stage_a(p, stage_a)))
    print("\n" + text)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
