"""5. araştırma turu: kısa ve uzun vadeli botlar (L1, P2) ve sınav testi.

Protokol: docs/ARASTIRMA.md, 15. bölüm. Adaylar, kurallar, ölçütler ve kararlar sonuçlar görülmeden önce
yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey değiştirmek testi geçersiz kılar.

- C1  (kısa vade, gerçek bot): 4 saatlik kanal kırılımı 20/10, 12 coin, %0,5 risk
- R1, R3, R3A: gölge botlardaki adaylar (3. ve 4. tur kurallarıyla birebir)
- L1  (uzun vade, YENİ): BTC ve ETH, günlük kapanış kendi 200 günlük ortalamasının üstündeyse tut
- P2  (YENİ): sermayenin yarısı C1, yarısı L1 (ayrı hesaplar gibi, yeniden dengeleme yok)

İki aşama:
    python arastirma_tur5.py           # Aşama A: geliştirme dönemleri, L1 ve P2 (sınav verisi YÜKLENMEZ)
    python arastirma_tur5.py --sinav   # Aşama B: sınav dönemi (2026-04-01 -> son tam gün), BİR KEZ

Çıktı: reports/<zaman>_arastirma_tur5_A (ya da _B) altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import arastirma as A
import arastirma_tur3 as T3
import arastirma_tur4 as T4
from bot.candidates import TrendParams, trend_signals
from bot.cli import disable_quick_edit, load_config_or_exit, open_file, setup_console
from bot.config import Config, MarketRules
from bot.data import candle_path, load_candles, load_market_rules
from bot.metrics import buy_and_hold, compute_metrics, daily_equity
from bot.timeutil import DAY_MS, TIMEFRAME_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 15. bölüm) ------------------------------------
BTC, ETH = "BTC/USDT", "ETH/USDT"
C1_SYMBOLS = T4.C1_SYMBOLS
L1_SYMBOLS = (BTC, ETH)                              # 2 Ocak 2022 sıralamasının ilk ikisi
L1_PARAMS = TrendParams(sma_n=200, atr_n=20)
L1_ROBUST_SMA = (150, 250)                           # sağlamlık: ±%25
L1_RULES = {
    "timeframes": {"entry": "1d", "trend": "1d"},
    "execution": {"entry_order": "market"},
    "risk": {"sizing": "allocation", "max_position_pct": 0.5, "stop_atr_mult": 10.0, "trail_atr_mult": 0.0,
             "max_open_positions": 2, "cooldown_bars": 0, "daily_loss_limit": 0.99, "max_drawdown_halt": 0.40},
    "strategy": {"exit_on_regime_break": False},
}
L1_MAX_DD = 0.40
L1_MIN_TRADES = 6                                    # keşif + doğrulama, kapanmış işlem
P2_MAX_DD = 0.30
EXAM_START = "2026-04-01"
EXAM_MIN_PF = 1.2
EXAM_MAX_DD = {"C1": 0.20, "R1": 0.20, "R3": 0.20, "P2": 0.30, "R3A": 0.40, "L1": 0.40}
EXAM_PF_EXEMPT = ("L1", "P2")                         # pozisyonu aylarca tutar: kapanmış işlem az
EXAM_MIN_TRADES = 10                                  # R1, R3, R3A bundan az işlem yaparsa sonuçsuz
L1_MIN_EXPOSURE = 0.10                                # L1 zamanın %10'undan azında pozisyondaysa sonuçsuz
SHADOW_CAPITAL = {"C1": 50.0, "R1": 50.0, "R3": 1000.0, "R3A": 50.0, "L1": 50.0, "P2": 50.0}
EXAM_ORDER = ("C1", "R1", "R3", "R3A", "L1", "P2")


@dataclass(frozen=True)
class Member:
    """Tek başına çalışan bir bot: kuralları, coinleri ve sinyal üreticisi."""
    code: str
    name: str
    entry_tf: str
    symbols: tuple[str, ...]
    overrides: dict
    build: Callable[[Config, A.Market], dict[str, pd.DataFrame]]


def l1_builder(params: TrendParams) -> Callable[[Config, A.Market], dict[str, pd.DataFrame]]:
    return lambda cfg_c, market: {s: trend_signals(f["1d"], params, cfg_c) for s, f in market.items()}


def members(l1_params: TrendParams = L1_PARAMS) -> dict[str, Member]:
    t3 = {c.code: c for c in T3.tur3_candidates()}
    w0a = next(c for c in T4.tur4_candidates() if c.code == "W0A")

    def t3_builder(c: T3.T3Candidate) -> Callable[[Config, A.Market], dict[str, pd.DataFrame]]:
        return lambda cfg_c, market: T3.build_signals(c, cfg_c, market)

    return {
        "C1": Member("C1", "Kısa vade: 4 saatlik kırılım (gerçek bot)", "4h", C1_SYMBOLS, t3["R0"].overrides,
                     t3_builder(t3["R0"])),
        "R1": Member("R1", "C1 + BTC rejim filtresi", "4h", C1_SYMBOLS, t3["R1"].overrides, t3_builder(t3["R1"])),
        "R3": Member("R3", "Orta vade: momentum rotasyonu", "1d", C1_SYMBOLS, t3["R3"].overrides,
                     t3_builder(t3["R3"])),
        "R3A": Member("R3A", "R3, %2 risk", "1d", C1_SYMBOLS, w0a.overrides,
                      lambda cfg_c, market: T4.build_signals(w0a, cfg_c, market)),
        "L1": Member("L1", f"Uzun vade: BTC+ETH {l1_params.sma_n} günlük ortalama", "1d", L1_SYMBOLS, L1_RULES,
                     l1_builder(l1_params)),
    }


def submarket(market: A.Market, symbols) -> A.Market:
    return {s: market[s] for s in symbols}


def run_member(m: Member, cfg: Config, market: A.Market, start: int, end: int, rules: dict[str, MarketRules],
               initial: float | None = None) -> A.PeriodRun:
    cfg_c = A.apply_overrides(cfg, m.overrides)
    if initial is not None:
        cfg_c = A.apply_overrides(cfg_c, {"backtest": {"initial_equity": initial}})
    sub = submarket(market, m.symbols)
    return A.run_period(cfg_c, m.build(cfg_c, sub), sub, m.entry_tf, start, end, rules)


@dataclass
class Combined:
    """P2: iki botun eşit sermayeli birleşimi (yeniden dengeleme yok)."""
    equity: pd.Series
    metrics: dict
    trades: list


def combine(runs: list[A.PeriodRun], start: int) -> Combined:
    initial = runs[0].metrics["initial_equity"]
    equity = A.combine_equal([r.result.equity for r in runs], initial, start)
    trades = [t for r in runs for t in r.result.trades]
    metrics = compute_metrics(equity, trades, initial, sum(r.result.total_fees for r in runs) / len(runs))
    return Combined(equity, metrics, trades)


def daily_benchmark(market: A.Market, symbols, start: int, end: int, cfg: Config) -> tuple[pd.Series, dict]:
    """Eşit ağırlıklı al-tut, günlük kapanışlarla."""
    initial = cfg.backtest.initial_equity
    curve = buy_and_hold({s: market[s]["1d"] for s in symbols}, start, end, initial, cfg.costs.taker_fee,
                         cfg.costs.slippage, DAY_MS)
    return curve, compute_metrics(curve, [], initial, 0.0)


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
def load_raw(cfg: Config) -> dict[str, dict[str, pd.DataFrame]]:
    raw = {}
    for symbol in C1_SYMBOLS:
        path = candle_path(cfg.data_dir, cfg.exchange.id, symbol, "1h")
        if not path.exists():
            raise FileNotFoundError(path.name)
        raw[symbol] = {"1h": load_candles(path)}
    return raw


def last_full_day(raw: dict[str, dict[str, pd.DataFrame]]) -> int:
    """Tüm coinlerde verinin kapsadığı son tam UTC gününün bitişi."""
    end = min(int(f["1h"]["ts"].iloc[-1]) + TIMEFRAME_MS["1h"] for f in raw.values())
    return end // DAY_MS * DAY_MS


# ---------------------------------------------------------------------------
# Aşama A: geliştirme dönemleri (sınav verisi yüklenmez)
# ---------------------------------------------------------------------------
@dataclass
class StageA:
    created: str
    bounds: dict[str, tuple[int, int]]
    c1: dict[str, A.PeriodRun]
    l1: dict[str, A.PeriodRun]
    p2: dict[str, Combined]
    robustness: list[tuple[str, A.PeriodRun]]
    feasibility: dict[str, dict[str, dict]]
    bench_curves: dict[str, pd.Series]               # dönem -> BTC+ETH eşit al-tut (günlük)
    l1_checks: dict[str, tuple[bool, str]] = field(default_factory=dict)
    p2_checks: dict[str, tuple[bool, str]] = field(default_factory=dict)

    @property
    def l1_passed(self) -> bool:
        return all(ok for ok, _ in self.l1_checks.values())

    @property
    def p2_passed(self) -> bool:
        return all(ok for ok, _ in self.p2_checks.values())


def _l1_period_check(run: A.PeriodRun) -> tuple[bool, str]:
    m, b = run.metrics, run.benchmark
    problems = []
    if not m["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not m["max_drawdown"] <= L1_MAX_DD:
        problems.append(f"düşüş %{m['max_drawdown'] * 100:.1f} > %{L1_MAX_DD * 100:.0f}")
    if not m["return_to_drawdown"] > b["return_to_drawdown"]:
        problems.append(f"getiri/düşüş {m['return_to_drawdown']:+.2f} ≤ al-tut {b['return_to_drawdown']:+.2f}")
    detail = f"getiri/düşüş {m['return_to_drawdown']:+.2f}, al-tut {b['return_to_drawdown']:+.2f}"
    return not problems, "; ".join(problems) or f"tümü sağlandı ({detail})"


def evaluate_l1(study: StageA) -> dict[str, tuple[bool, str]]:
    checks = {name: _l1_period_check(study.l1[name]) for name, _ in A.PERIODS}
    weak = [label for label, run in study.robustness
            if not (run.metrics["net_profit"] > 0
                    and run.metrics["return_to_drawdown"] > run.benchmark["return_to_drawdown"])]
    checks["saglamlik"] = (bool(study.robustness) and not weak,
                           f"zayıf: {', '.join(weak)}" if weak else "150 ve 250 günde de net > 0 ve al-tut'tan iyi")
    n = sum(study.l1[name].metrics["n_trades"] for name, _ in A.PERIODS)
    checks["islem"] = (n >= L1_MIN_TRADES, f"{n} kapanmış işlem (en az {L1_MIN_TRADES})")
    return checks


def evaluate_p2(study: StageA) -> dict[str, tuple[bool, str]]:
    checks = {}
    for name, _ in A.PERIODS:
        m = study.p2[name].metrics
        problems = []
        if not m["net_profit"] > 0:
            problems.append("net getiri ≤ 0")
        if not m["max_drawdown"] <= P2_MAX_DD:
            problems.append(f"düşüş %{m['max_drawdown'] * 100:.1f} > %{P2_MAX_DD * 100:.0f}")
        checks[name] = (not problems, "; ".join(problems) or "tümü sağlandı")
    better = []
    for name, label in A.PERIODS:
        mine, base = study.p2[name].metrics["return_to_drawdown"], study.c1[name].metrics["return_to_drawdown"]
        better.append((mine > base, f"{label.lower()} {mine:+.2f} / C1 {base:+.2f}"))
    checks["iyilesme"] = (all(ok for ok, _ in better), "; ".join(text for _, text in better))
    checks["l1"] = (study.l1_passed, "L1 geçti" if study.l1_passed else "L1 kaldı")
    return checks


def run_stage_a(cfg: Config, market: A.Market, rules: dict[str, MarketRules], real_rules: dict[str, MarketRules],
                progress: A.Progress | None = None) -> StageA:
    progress = progress or (lambda _msg: None)
    exam = parse_utc(cfg.data.exam_start)
    for frames in market.values():                      # güvenlik: sınav verisi bu aşamaya giremez
        for tf, df in frames.items():
            if len(df) and int(df["ts"].iloc[-1]) + TIMEFRAME_MS[tf] > exam:
                raise AssertionError("Aşama A'ya sınav dönemi verisi verildi.")
    bounds = {name: (parse_utc(a), parse_utc(b)) for name, (a, b) in T4.PROTOCOL_BOUNDS.items()}
    team = members()
    c1, l1, p2 = {}, {}, {}
    for name, label in A.PERIODS:
        c1[name] = run_member(team["C1"], cfg, market, *bounds[name], rules)
        l1[name] = run_member(team["L1"], cfg, market, *bounds[name], rules)
        p2[name] = combine([c1[name], l1[name]], bounds[name][0])
        for code, m in (("C1", c1[name].metrics), ("L1", l1[name].metrics), ("P2", p2[name].metrics)):
            progress(f"  {code:<3} {label:<10} {m['n_trades']:>4} işlem  {A._pct(m['total_return']):>8}  "
                     f"düşüş {A._pct(m['max_drawdown'], False):>6}  getiri/düşüş {m['return_to_drawdown']:+.2f}")
    robustness = []
    for sma_n in L1_ROBUST_SMA:
        m = members(TrendParams(sma_n=sma_n, atr_n=L1_PARAMS.atr_n))["L1"]
        cfg_v = A.apply_overrides(cfg, m.overrides)
        sub = submarket(market, m.symbols)
        start = max(A.trade_window(m.build(cfg_v, sub), cfg_v.entry_ms)[0], bounds["kesif"][0])
        robustness.append((f"{sma_n} günlük ortalama", run_member(m, cfg, market, start, bounds["kesif"][1], rules)))
    feasibility = {}
    for code in ("L1", "P2"):
        feasibility[code] = {}
        for name, _ in A.PERIODS:
            if code == "L1":
                runs = [run_member(team["L1"], cfg, market, *bounds[name], real_rules, SHADOW_CAPITAL["L1"])]
                metrics = runs[0].metrics
            else:
                half = SHADOW_CAPITAL["P2"] / 2
                runs = [run_member(team[c], cfg, market, *bounds[name], real_rules, half) for c in ("C1", "L1")]
                metrics = combine(runs, bounds[name][0]).metrics
            feasibility[code][name] = {"metrics": metrics, "trades": metrics["n_trades"],
                                       "skipped_min_notional": sum(int(r.result.skipped.get(T3.MIN_NOTIONAL_SKIP, 0))
                                                                   for r in runs)}
    bench_curves = {name: daily_benchmark(market, L1_SYMBOLS, *bounds[name], cfg)[0] for name, _ in A.PERIODS}
    study = StageA(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), bounds, c1, l1, p2, robustness, feasibility,
                   bench_curves)
    study.l1_checks = evaluate_l1(study)
    study.p2_checks = evaluate_p2(study)
    return study


# ---------------------------------------------------------------------------
# Aşama B: sınav dönemi (bir kez)
# ---------------------------------------------------------------------------
@dataclass
class ExamResult:
    code: str
    metrics: dict
    benchmark: dict
    exposure: float
    status: str                          # "GEÇTİ" | "KALDI" | "sonuçsuz"
    detail: str
    decision: str
    shadow: dict                         # gölge botun sermayesiyle (bilgi)


@dataclass
class StageB:
    created: str
    start: int
    end: int
    stage_a: StageA
    results: dict[str, ExamResult]
    runs: dict[str, A.PeriodRun]
    p2: Combined
    bench: dict[str, tuple[pd.Series, dict]]


def exam_status(code: str, m: dict, bench_rd: float, exposure: float, c1_rd: float | None = None
                ) -> tuple[str, str]:
    if code in ("R1", "R3", "R3A") and m["n_trades"] < EXAM_MIN_TRADES:
        return "sonuçsuz", f"{m['n_trades']} işlem (en az {EXAM_MIN_TRADES} gerekir)"
    if code == "L1" and exposure < L1_MIN_EXPOSURE:
        return "sonuçsuz", f"zamanın %{exposure * 100:.0f}'inde pozisyonda (en az %{L1_MIN_EXPOSURE * 100:.0f})"
    problems = []
    if not m["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not m["max_drawdown"] <= EXAM_MAX_DD[code]:
        problems.append(f"düşüş %{m['max_drawdown'] * 100:.1f} > %{EXAM_MAX_DD[code] * 100:.0f}")
    if code not in EXAM_PF_EXEMPT and not m["profit_factor"] >= EXAM_MIN_PF:
        problems.append(f"KF {A._ratio(m['profit_factor'])} < {EXAM_MIN_PF}")
    if not m["return_to_drawdown"] > bench_rd:
        problems.append(f"getiri/düşüş {m['return_to_drawdown']:+.2f} ≤ al-tut {bench_rd:+.2f}")
    if c1_rd is not None and not m["return_to_drawdown"] > c1_rd:
        problems.append(f"getiri/düşüş {m['return_to_drawdown']:+.2f} ≤ C1 {c1_rd:+.2f}")
    return ("KALDI", "; ".join(problems)) if problems else ("GEÇTİ", "tümü sağlandı")


def decide(code: str, status: str, stage_a: StageA) -> str:
    if code == "C1":
        return "değişmez (gerçek bot)"
    if code == "L1":
        if stage_a.l1_passed and status == "GEÇTİ":
            return "gölge bot olarak eklenir (50 USDT sanal)"
        return "eklenmez" + ("" if stage_a.l1_passed else " (Aşama A'da kaldı)")
    if code == "P2":
        ok = stage_a.p2_passed and status == "GEÇTİ"
        return "geçti: sermaye planı için bilgi" if ok else "kaldı: bilgi"
    if status == "KALDI":
        return "gölge bot durdurulur"
    return "gölge test sürer" + (" (sınav sonuçsuz)" if status == "sonuçsuz" else "")


def run_stage_b(cfg: Config, market: A.Market, start: int, end: int, rules: dict[str, MarketRules],
                real_rules: dict[str, MarketRules], stage_a: StageA, progress: A.Progress | None = None
                ) -> StageB:
    progress = progress or (lambda _msg: None)
    if not (parse_utc(EXAM_START) == start < end):
        raise ValueError("Sınav dönemi 2026-04-01'de başlamalı ve en az bir tam gün içermeli.")
    team = members()
    runs = {code: run_member(m, cfg, market, start, end, rules) for code, m in team.items()}
    p2 = combine([runs["C1"], runs["L1"]], start)
    bench = {"c1": daily_benchmark(market, C1_SYMBOLS, start, end, cfg),
             "l1": daily_benchmark(market, L1_SYMBOLS, start, end, cfg)}
    results = {}
    for code in EXAM_ORDER:
        if code == "P2":
            m, exposure = p2.metrics, float("nan")
            b = bench["c1"][1]
            status, detail = exam_status(code, m, b["return_to_drawdown"], 1.0,
                                         runs["C1"].metrics["return_to_drawdown"])
        else:
            run = runs[code]
            m, b, exposure = run.metrics, run.benchmark, float(run.result.exposure)
            status, detail = exam_status(code, m, b["return_to_drawdown"], exposure)
        if code == "P2":
            half = SHADOW_CAPITAL["P2"] / 2
            shadow = combine([run_member(team[c], cfg, market, start, end, real_rules, half) for c in ("C1", "L1")],
                             start).metrics
        else:
            shadow = run_member(team[code], cfg, market, start, end, real_rules, SHADOW_CAPITAL[code]).metrics
        results[code] = ExamResult(code, m, b, exposure, status, detail, decide(code, status, stage_a), shadow)
        progress(f"  {code:<3} {m['n_trades']:>4} işlem  {A._pct(m['total_return']):>8}  "
                 f"düşüş {A._pct(m['max_drawdown'], False):>6}  KF {A._ratio(m['profit_factor']):>6}  -> {status}")
    return StageB(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), start, end, stage_a, results, runs, p2, bench)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
_pct, _ratio, _mark, _table = A._pct, A._ratio, A._mark, A._table


def _rd(m: dict) -> str:
    return f"{m['return_to_drawdown']:+.2f}"


def format_stage_a(s: StageA) -> str:
    b = s.bounds
    out = [
        f"### 5. tur, Aşama A: geliştirme dönemleri ({s.created})",
        "",
        (f"Keşif {fmt_utc(b['kesif'][0], False)} → {fmt_utc(b['kesif'][1] - DAY_MS, False)}, doğrulama "
         f"{fmt_utc(b['dogrulama'][0], False)} → {fmt_utc(b['dogrulama'][1] - DAY_MS, False)}; sınav dönemi "
         "yüklenmedi. Her dönem 10,000 USDT ile ayrı başlar; komisyon ve kayma dahil."),
        "",
    ]
    rows = []
    for name, label in A.PERIODS:
        l1, c1, p2 = s.l1[name], s.c1[name], s.p2[name]
        bench = l1.benchmark
        for code, m in (("L1 uzun vade (BTC+ETH, 200 gün)", l1.metrics), ("C1 kısa vade (4 saat)", c1.metrics),
                        ("P2 = yarı C1 + yarı L1", p2.metrics)):
            rows.append([label, code, _pct(m["total_return"]), _pct(m["max_drawdown"], False), _rd(m),
                         _ratio(m["profit_factor"]), str(m["n_trades"]),
                         "-" if m.get("exposure") is None else _pct(m["exposure"], False)])
        rows.append([label, "Al-tut BTC+ETH (eşit)", _pct(bench["total_return"]), _pct(bench["max_drawdown"], False),
                     _rd(bench), "-", "-", "-"])
    out += _table(["Dönem", "Bot", "Net getiri", "Maks. düşüş", "Getiri/düşüş", "Kâr faktörü", "İşlem",
                   "Piyasada"], rows)
    out.append("")

    rows = [[label, _pct(run.metrics["total_return"]), _pct(run.metrics["max_drawdown"], False), _rd(run.metrics),
             _rd(run.benchmark), str(run.metrics["n_trades"]), fmt_utc(run.start, False)]
            for label, run in s.robustness]
    out += ["**L1 sağlamlık (keşif)**", ""]
    out += _table(["Ortalama", "Net getiri", "Maks. düşüş", "Getiri/düşüş", "Al-tut getiri/düşüş", "İşlem",
                   "Başlangıç"], rows)
    out.append("")

    out += ["**Ölçütler**", ""]
    rows = [["L1", *[f"{_mark(ok)} ({text})" for ok, text in s.l1_checks.values()],
             "**GEÇTİ**" if s.l1_passed else "KALDI"]]
    out += _table(["Aday", "1. Keşif", "2. Doğrulama", "3. Sağlamlık", "4. İşlem", "Sonuç"], rows)
    out.append("")
    rows = [["P2", *[f"{_mark(ok)} ({text})" for ok, text in s.p2_checks.values()],
             "**GEÇTİ**" if s.p2_passed else "KALDI"]]
    out += _table(["Aday", "Keşif", "Doğrulama", "C1'e göre iyileşme", "L1 geçti mi", "Sonuç"], rows)
    out.append("")

    rows = []
    for code in ("L1", "P2"):
        for name, label in A.PERIODS:
            f = s.feasibility[code][name]
            m = f["metrics"]
            rows.append([code, label, _pct(m["total_return"]), _pct(m["max_drawdown"], False), str(f["trades"]),
                         str(f["skipped_min_notional"])])
    out += [(f"**Uygulanabilirlik ({SHADOW_CAPITAL['L1']:.2f} USDT, gerçek minimum işlem tutarlarıyla; "
             "P2'de her bot yarısıyla)**"), ""]
    out += _table(["Bot", "Dönem", "Net getiri", "Maks. düşüş", "İşlem", "Minimum tutar yüzünden atlanan"], rows)
    out.append("")
    verdict = f"L1 {'geçti' if s.l1_passed else 'kaldı'}, P2 {'geçti' if s.p2_passed else 'kaldı'}"
    out.append(f"**Aşama A kararı: {verdict}.** Protokol gereği Aşama B (sınav) tüm adaylarla bir kez çalıştırılır; "
               "L1'in eklenmesi için iki aşamayı da geçmesi gerekir.")
    return "\n".join(out)


def format_stage_b(s: StageB) -> str:
    out = [
        f"### 5. tur, Aşama B: sınav dönemi ({s.created})",
        "",
        (f"Sınav dönemi {fmt_utc(s.start, False)} → {fmt_utc(s.end - DAY_MS, False)} "
         f"({(s.end - s.start) // DAY_MS} gün). 10,000 USDT, komisyon ve kayma dahil. Kurallar önceden kilitlendi "
         "(docs/ARASTIRMA.md, 15. bölüm)."),
        "",
    ]
    rows = []
    for code in EXAM_ORDER:
        r = s.results[code]
        m = r.metrics
        name = "P2 = yarı C1 + yarı L1" if code == "P2" else f"{code} {members()[code].name}"
        rows.append([name, _pct(m["total_return"]), _pct(m["max_drawdown"], False), _rd(m),
                     _ratio(m["profit_factor"]), str(m["n_trades"]),
                     "-" if code == "P2" else _pct(r.exposure, False), _rd(r.benchmark),
                     f"%{EXAM_MAX_DD[code] * 100:.0f}", f"**{r.status}** ({r.detail})", r.decision])
    out += _table(["Aday", "Net getiri", "Maks. düşüş", "Getiri/düşüş", "Kâr faktörü", "İşlem", "Piyasada",
                   "Al-tut getiri/düşüş", "Düşüş sınırı", "Sınav", "Karar"], rows)
    out.append("")
    for key, label in (("c1", "12 coin eşit"), ("l1", "BTC+ETH eşit")):
        m = s.bench[key][1]
        out.append(f"Al-tut ({label}, günlük): {_pct(m['total_return'])}, düşüş {_pct(m['max_drawdown'], False)}, "
                   f"getiri/düşüş {_rd(m)}.")
    out.append("")
    rows = []
    for code in EXAM_ORDER:
        m = s.results[code].shadow
        rows.append([code, f"{SHADOW_CAPITAL[code]:,.2f}", f"{SHADOW_CAPITAL[code] * (1 + m['total_return']):,.2f}",
                     _pct(m["total_return"]), _pct(m["max_drawdown"], False), str(m["n_trades"])])
    out += ["**Gölge botun sermayesiyle (bilgi; gerçek minimum işlem tutarlarıyla)**", ""]
    out += _table(["Bot", "Başlangıç USDT", "Son USDT", "Net getiri", "Maks. düşüş", "İşlem"], rows)
    out.append("")
    a = s.stage_a
    out.append(f"Aşama A: L1 {'geçti' if a.l1_passed else 'kaldı'}, P2 {'geçti' if a.p2_passed else 'kaldı'}.")
    return "\n".join(out)


def stage_a_summary(s: StageA) -> dict:
    return A._clean({
        "created": s.created,
        "bounds": {k: [fmt_utc(x, False), fmt_utc(y, False)] for k, (x, y) in s.bounds.items()},
        "l1_passed": s.l1_passed, "p2_passed": s.p2_passed,
        "l1_checks": {k: {"passed": ok, "detail": t} for k, (ok, t) in s.l1_checks.items()},
        "p2_checks": {k: {"passed": ok, "detail": t} for k, (ok, t) in s.p2_checks.items()},
        "c1": {n: r.metrics for n, r in s.c1.items()},
        "l1": {n: {"metrics": r.metrics, "buy_and_hold": r.benchmark, "symbol_net": r.symbol_net}
               for n, r in s.l1.items()},
        "p2": {n: c.metrics for n, c in s.p2.items()},
        "robustness": [{"variant": label, "metrics": r.metrics, "buy_and_hold": r.benchmark}
                       for label, r in s.robustness],
        "feasibility": s.feasibility,
    })


def stage_b_summary(s: StageB) -> dict:
    return A._clean({
        "created": s.created,
        "exam": [fmt_utc(s.start, False), fmt_utc(s.end, False)],
        "stage_a": {"l1_passed": s.stage_a.l1_passed, "p2_passed": s.stage_a.p2_passed},
        "results": {code: {"status": r.status, "detail": r.detail, "decision": r.decision, "metrics": r.metrics,
                           "buy_and_hold": r.benchmark, "exposure": r.exposure, "shadow_capital": r.shadow}
                    for code, r in s.results.items()},
        "buy_and_hold_daily": {k: v[1] for k, v in s.bench.items()},
    })


def _trades_csv(path: Path, groups: list[tuple[str, str, list]]) -> None:
    rows = [{"bot": code, "donem": period, **asdict(t), "entry_utc": fmt_utc(t.entry_ts), "exit_utc": fmt_utc(t.exit_ts)}
            for code, period, trades in groups for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_stage(out_dir: Path, summary: dict, text: str, trades: list[tuple[str, str, list]],
               plot: Callable[[Path], None] | None) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _trades_csv(out_dir / "islemler.csv", trades)
    if plot is None:
        return None
    chart = out_dir / "grafik.png"
    plot(chart)
    return chart


def _curve(equity: pd.Series, initial: float) -> tuple[pd.DatetimeIndex, pd.Series]:
    daily = daily_equity(equity) / initial * 100
    return pd.to_datetime((daily.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True), daily


def plot_stage_a(path: Path, s: StageA) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(A.PERIODS), 1, figsize=(12, 9))
    for ax, (name, label) in zip(axes, A.PERIODS, strict=True):
        initial = s.l1[name].metrics["initial_equity"]
        for series, text, color, style in (
                (s.l1[name].result.equity, "L1 uzun vade (BTC+ETH, 200 gün)", "#55A868", "-"),
                (s.c1[name].result.equity, "C1 kısa vade (4 saat)", "#4C72B0", "-"),
                (s.p2[name].equity, "P2 = yarı C1 + yarı L1", "#8172B3", "--")):
            x, y = _curve(series, initial)
            ax.plot(x, y.to_numpy(), color=color, linestyle=style, linewidth=1.5, label=text)
        x, y = _curve(s.bench_curves[name], initial)
        ax.plot(x, y.to_numpy(), color="#DD8452", linestyle=":", linewidth=1.0, label="Al-tut BTC+ETH (eşit)")
        ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_title(f"{label} dönemi (başlangıç = 100) · komisyon ve kayma dahil", loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_stage_b(path: Path, s: StageB) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"C1": "#4C72B0", "R1": "#64B5CD", "R3": "#C44E52", "R3A": "#C44E52", "L1": "#55A868", "P2": "#8172B3"}
    styles = {"R3A": "--", "P2": "--"}
    fig, ax = plt.subplots(figsize=(12, 6))
    initial = s.runs["C1"].metrics["initial_equity"]
    for code in EXAM_ORDER:
        equity = s.p2.equity if code == "P2" else s.runs[code].result.equity
        x, y = _curve(equity, initial)
        ax.plot(x, y.to_numpy(), color=colors[code], linestyle=styles.get(code, "-"), linewidth=1.5,
                label=f"{code} ({s.results[code].status})")
    for key, label, style in (("c1", "Al-tut 12 coin", ":"), ("l1", "Al-tut BTC+ETH", "-.")):
        x, y = _curve(s.bench[key][0], initial)
        ax.plot(x, y.to_numpy(), color="#DD8452", linestyle=style, linewidth=1.0, label=label)
    ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_title("Sınav dönemi (başlangıç = 100) · komisyon ve kayma dahil", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()                 # pencereye tıklamak çalışmayı dondurmasın
    parser = argparse.ArgumentParser(description="5. araştırma turu: kısa ve uzun vadeli botlar, sınav testi.")
    parser.add_argument("--sinav", action="store_true", help="Aşama B: sınav dönemi (bir kez)")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    try:
        raw = load_raw(cfg)
    except FileNotFoundError as exc:
        print(f"[HATA] Veri yok: {exc}. Önce V1_tur4_veri_indir.bat çalıştırılmalı.")
        return 2
    rules = load_market_rules(cfg.data_dir, list(C1_SYMBOLS), cfg.market_rules_default)
    exam_ms = parse_utc(cfg.data.exam_start)
    market_dev = A.prepare_market(raw, exam_ms)
    print(f"Geliştirme verisi yüklendi; {cfg.data.exam_start} (sınav) sonrası mumlar atıldı.")
    stage_a = run_stage_a(cfg, market_dev, rules, rules, progress=print)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if not args.sinav:
        text = format_stage_a(stage_a)
        trades = [(code, name, run.result.trades) for code, runs in (("C1", stage_a.c1), ("L1", stage_a.l1))
                  for name, run in runs.items()]
        out_dir = cfg.reports_dir / f"{stamp}_arastirma_tur5_A"
        chart = save_stage(out_dir, stage_a_summary(stage_a), text, trades,
                           None if args.no_plot else (lambda p: plot_stage_a(p, stage_a)))
    else:
        end = last_full_day(raw)
        market_full = A.prepare_market(raw, end)
        print(f"Sınav verisi yüklendi: {EXAM_START} → {fmt_utc(end - DAY_MS, False)} (son tam gün).")
        try:
            stage_b = run_stage_b(cfg, market_full, parse_utc(EXAM_START), end, rules, rules, stage_a, progress=print)
        except ValueError as exc:
            print(f"[HATA] {exc}")
            return 2
        text = format_stage_a(stage_a) + "\n\n" + format_stage_b(stage_b)
        trades = [(code, "sinav", run.result.trades) for code, run in stage_b.runs.items()]
        out_dir = cfg.reports_dir / f"{stamp}_arastirma_tur5_B"
        summary = {"asama_a": stage_a_summary(stage_a), "asama_b": stage_b_summary(stage_b)}
        chart = save_stage(out_dir, summary, text, trades,
                           None if args.no_plot else (lambda p: plot_stage_b(p, stage_b)))
    print("\n" + text)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
