"""4. araştırma turu: momentum rotasyonu (R3) için geniş coin evreni (W0, W0A, W1, W2).

Protokol: docs/ARASTIRMA.md, 13. bölüm. Adaylar, evren kuralı, dönemler ve ölçütler sonuçlar görülmeden önce
yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey değiştirmek testi geçersiz kılar.

- W0  (referans): R3, C1'in 12 coini, %0,5 risk (3. turdaki R3'ün aynısı)
- W0A (referans): R3, C1'in 12 coini, %2 risk (12. bölümdeki agresif ayar, G-R3A ile aynı)
- W1: R3, geniş evren, %0,5 risk  -> W0 ile karşılaştırılır
- W2: R3, geniş evren, %2 risk    -> W0A ile karşılaştırılır

GÜVENLİK: Sınav dönemi (data.exam_start sonrası) burada ASLA kullanılmaz; veri yüklenirken atılır.

Önce: python tur4_veri.py (geniş evrenin 1 saatlik verisi ve data/evren_tur4.json)

Kullanım:
    python arastirma_tur4.py
    python arastirma_tur4.py --no-plot

Çıktı: reports/<zaman>_arastirma_tur4/ altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
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
import arastirma_tur3 as T3
import tur4_veri
from bot.backtester import run_backtest
from bot.candidates import MomentumParams, momentum_signals
from bot.cli import disable_quick_edit, load_config_or_exit, open_file, setup_console
from bot.config import Config, MarketRules
from bot.data import candle_path, load_candles, load_market_rules
from bot.metrics import buy_and_hold, daily_equity
from bot.timeutil import DAY_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 13. bölüm) ------------------------------------
BTC = T3.BTC
C1_SYMBOLS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT", "DOT/USDT", "AVAX/USDT",
              "DOGE/USDT", "SHIB/USDT", "UNI/USDT", "ALGO/USDT")              # W0: C1'in 12 coini
PROTOCOL_BOUNDS = {"kesif": ("2022-07-20", "2025-01-01"),                     # 3. turla aynı dönemler
                   "dogrulama": ("2025-01-01", "2026-04-01")}                 # (bitiş tarihi hariç)
MAX_DRAWDOWN = A.MAX_DRAWDOWN                 # %0,5 risk: ölçüt 1 ve 2'de en fazla %20 düşüş
AGGRESSIVE_MAX_DRAWDOWN = 0.40                # %2 risk (W2): en fazla %40 düşüş
MIN_TRADES = T3.MIN_TRADES                    # ölçüt 5: keşif + doğrulama en az 100 işlem
FEASIBILITY_EQUITY = T3.FEASIBILITY_EQUITY    # bilgi: 50 USDT, gerçek minimum işlem tutarlarıyla
NO_HALT = T3.NO_HALT                          # şaltersiz tanı
MAX_SHADOW = 2
R3_RULES = A._merge(T3.C1_RULES, {"timeframes": {"entry": "1d", "trend": "1d"}})   # 3. turdaki R3'ün kuralları
AGGRESSIVE_RULES = A._merge(R3_RULES, {"risk": {"risk_per_trade": 0.02, "max_position_pct": 1.0,
                                                "max_drawdown_halt": 0.999, "daily_loss_limit": 0.99}})
REFERENCE_OF = {"W1": "W0", "W2": "W0A"}       # ölçüt 3: her aday aynı riskteki W0 ile karşılaştırılır
UNIVERSE_LABELS = {"c1": "C1'in 12 coini", "genis": "geniş evren"}


@dataclass(frozen=True)
class T4Candidate:
    code: str
    name: str
    universe: str                                # "c1" | "genis"
    risk_label: str
    overrides: dict
    max_drawdown: float
    momentum: MomentumParams = field(default_factory=MomentumParams)
    reference: bool = False


def tur4_candidates() -> list[T4Candidate]:
    return [
        T4Candidate("W0", "R3, 12 coin, %0,5 risk (referans)", "c1", "%0,5", R3_RULES, MAX_DRAWDOWN, reference=True),
        T4Candidate("W0A", "R3, 12 coin, %2 risk (referans)", "c1", "%2", AGGRESSIVE_RULES, AGGRESSIVE_MAX_DRAWDOWN,
                    reference=True),
        T4Candidate("W1", "R3, geniş evren, %0,5 risk", "genis", "%0,5", R3_RULES, MAX_DRAWDOWN),
        T4Candidate("W2", "R3, geniş evren, %2 risk", "genis", "%2", AGGRESSIVE_RULES, AGGRESSIVE_MAX_DRAWDOWN),
    ]


def robustness_variants(c: T4Candidate) -> list[tuple[str, T4Candidate]]:
    """Ölçüt 4: 3. turdaki R3 varyantlarının aynısı (getiri penceresi 21/35 gün, ilk 2/4 coin, çıkış eşiği +2)."""
    if c.reference:
        return []
    m = c.momentum
    return [("getiri penceresi 21 gün", dataclasses.replace(c, momentum=dataclasses.replace(m, lookback=21))),
            ("getiri penceresi 35 gün", dataclasses.replace(c, momentum=dataclasses.replace(m, lookback=35))),
            ("ilk 2 coin (çıkış 4)", dataclasses.replace(c, momentum=dataclasses.replace(m, top_n=2, exit_rank=4))),
            ("ilk 4 coin (çıkış 6)", dataclasses.replace(c, momentum=dataclasses.replace(m, top_n=4, exit_rank=6)))]


def build_signals(c: T4Candidate, cfg_c: Config, market: A.Market) -> dict[str, pd.DataFrame]:
    """3. turdaki R3 ile aynı kod: günlük mumlar, coinler arası sıralama, BTC rejim filtresi."""
    return momentum_signals({s: f["1d"] for s, f in market.items()}, market[BTC]["1d"], c.momentum, cfg_c)


# ---------------------------------------------------------------------------
# Veri: geniş evren data/evren_tur4.json dosyasından; sınav dönemi asla yüklenmez
# ---------------------------------------------------------------------------
def load_universes(cfg: Config) -> dict[str, list[str]]:
    path = cfg.data_dir / tur4_veri.UNIVERSE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path.name} yok: önce tur4_veri.py (V1_tur4_veri_indir.bat) çalıştırılmalı.")
    selected = json.loads(path.read_text(encoding="utf-8"))["secilen"]
    if len(selected) < tur4_veri.MIN_SIZE:
        raise ValueError(f"Geniş evrende yalnızca {len(selected)} coin var (en az {tur4_veri.MIN_SIZE}).")
    wide = list(dict.fromkeys([*tur4_veri.ALWAYS, *selected]))
    return {"c1": list(C1_SYMBOLS), "genis": wide}


def load_tur4_market(cfg: Config, universes: dict[str, list[str]]) -> A.Market:
    raw = {}
    for symbol in dict.fromkeys(s for symbols in universes.values() for s in symbols):
        path = candle_path(cfg.data_dir, cfg.exchange.id, symbol, "1h")
        if not path.exists():
            raise FileNotFoundError(path.name)
        raw[symbol] = {"1h": load_candles(path)}
    if BTC not in raw:
        raise ValueError("BTC rejim filtresi için BTC/USDT verisi gerekli.")
    return A.prepare_market(raw, parse_utc(cfg.data.exam_start))


def submarket(market: A.Market, symbols: list[str]) -> A.Market:
    return {s: market[s] for s in symbols}


def protocol_bounds(cfg: Config, market: A.Market, universes: dict[str, list[str]],
                    candidates: list[T4Candidate]) -> dict[str, tuple[int, int]]:
    """3. turla aynı dönemler. Veri her adayın göstergelerini dönem başından önce hazırlamıyorsa durur."""
    bounds = {name: (parse_utc(a), parse_utc(b)) for name, (a, b) in PROTOCOL_BOUNDS.items()}
    if bounds["dogrulama"][1] > parse_utc(cfg.data.exam_start):
        raise ValueError("Doğrulama dönemi sınav dönemine taşıyor.")
    for c in candidates:
        for _, variant in [(c.code, c), *robustness_variants(c)]:
            cfg_c = A.apply_overrides(cfg, variant.overrides)
            start, end = A.trade_window(build_signals(variant, cfg_c, submarket(market, universes[c.universe])),
                                        cfg_c.entry_ms)
            if start > bounds["kesif"][0] or end < bounds["dogrulama"][1]:
                raise ValueError(f"{c.code}: veri protokol dönemlerini kapsamıyor "
                                 f"({fmt_utc(start, False)} → {fmt_utc(end, False)}).")
    return bounds


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------
@dataclass
class T4Report:
    candidate: T4Candidate
    symbols: list[str]
    portfolio: dict[str, A.PeriodRun]
    robustness: list[tuple[str, dict]]
    no_halt: dict[str, dict]
    monte_carlo: dict
    feasibility: dict[str, dict]
    checks: dict[str, tuple[bool | None, str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.candidate.reference and all(ok for ok, _ in self.checks.values())


@dataclass
class T4Study:
    created: str
    universes: dict[str, list[str]]
    initial_equity: float
    bounds: dict[str, tuple[int, int]]
    reports: list[T4Report]
    shadow: list[str]
    bench_curves: dict[str, dict[str, pd.Series]]      # evren -> dönem -> al-tut eğrisi


def run_candidate(c: T4Candidate, cfg: Config, market: A.Market, bounds: dict[str, tuple[int, int]],
                  rules: dict[str, MarketRules], real_rules: dict[str, MarketRules], progress: A.Progress
                  ) -> T4Report:
    cfg_c = A.apply_overrides(cfg, c.overrides)
    signals = build_signals(c, cfg_c, market)
    portfolio = {}
    for name, label in A.PERIODS:
        t0 = time.perf_counter()
        run = A.run_period(cfg_c, signals, market, "1d", *bounds[name], rules)
        portfolio[name] = run
        m = run.metrics
        progress(f"  {c.code:<3} {label:<10} {m['n_trades']:>4} işlem  {A._pct(m['total_return']):>8}  "
                 f"düşüş {A._pct(m['max_drawdown'], False):>6}  KF {A._ratio(m['profit_factor']):>5}  "
                 f"({time.perf_counter() - t0:.1f} sn)")

    initial = cfg.backtest.initial_equity
    robustness = []
    for label, variant in robustness_variants(c):
        cfg_v = A.apply_overrides(cfg, variant.overrides)
        signals_v = build_signals(variant, cfg_v, market)
        start = max(A.trade_window(signals_v, cfg_v.entry_ms)[0], bounds["kesif"][0])
        result = run_backtest(cfg_v, signals_v, start, bounds["kesif"][1], initial, rules)
        robustness.append((label, T3._metrics(result, initial)))

    cfg_nh = A.apply_overrides(cfg_c, {"risk": {"max_drawdown_halt": NO_HALT}})
    no_halt = {name: T3._metrics(run_backtest(cfg_nh, signals, *bounds[name], initial, rules), initial)
               for name, _ in A.PERIODS}

    trades = portfolio["kesif"].result.trades + portfolio["dogrulama"].result.trades
    mc = A.monte_carlo([t.r_multiple for t in trades], cfg_c.risk.risk_per_trade)

    feasibility = {}
    for name, _ in A.PERIODS:
        result = run_backtest(cfg_c, signals, *bounds[name], FEASIBILITY_EQUITY, real_rules)
        feasibility[name] = {"metrics": T3._metrics(result, FEASIBILITY_EQUITY), "signals": result.signals_seen,
                             "skipped_min_notional": int(result.skipped.get(T3.MIN_NOTIONAL_SKIP, 0))}
    progress(f"  {c.code:<3} sağlamlık {len(robustness)} varyant, şaltersiz tanı, Monte Carlo, "
             f"{FEASIBILITY_EQUITY:.2f} USDT tamam")
    return T4Report(c, list(market), portfolio, robustness, no_halt, mc, feasibility)


def _period_check(m: dict, min_pf: float, max_dd: float) -> tuple[bool, str]:
    problems = []
    if not m["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not m["profit_factor"] >= min_pf:
        problems.append(f"KF {A._ratio(m['profit_factor'])} < {min_pf}")
    if not m["max_drawdown"] <= max_dd:
        problems.append(f"düşüş %{m['max_drawdown'] * 100:.1f} > %{max_dd * 100:.0f}")
    return not problems, "; ".join(problems) or "tümü sağlandı"


def evaluate(r: T4Report, ref: T4Report | None) -> dict[str, tuple[bool | None, str]]:
    """13. bölümdeki beş ölçüt. Referanslar (W0, W0A) için yalnızca bilgi."""
    k, d = r.portfolio["kesif"].metrics, r.portfolio["dogrulama"].metrics
    max_dd = r.candidate.max_drawdown
    checks: dict[str, tuple[bool | None, str]] = {
        "kesif": _period_check(k, A.EXPLORE_MIN_PF, max_dd),
        "dogrulama": _period_check(d, A.VALIDATE_MIN_PF, max_dd),
    }
    if r.candidate.reference or ref is None:
        checks["iyilesme"] = (None, "referans")
        checks["saglamlik"] = (None, "referans")
    else:
        better = []
        for name, label in A.PERIODS:
            mine = r.portfolio[name].metrics["return_to_drawdown"]
            base = ref.portfolio[name].metrics["return_to_drawdown"]
            better.append((mine > base, f"{label.lower()} {mine:+.2f} / {ref.candidate.code} {base:+.2f}"))
        checks["iyilesme"] = (all(ok for ok, _ in better), "; ".join(text for _, text in better))
        weak = [label for label, m in r.robustness if not m["profit_factor"] >= A.ROBUST_MIN_PF]
        checks["saglamlik"] = (bool(r.robustness) and not weak,
                               f"KF < {A.ROBUST_MIN_PF}: {', '.join(weak)}" if weak else "tüm varyantlarda KF ≥ 1")
    n = k["n_trades"] + d["n_trades"]
    checks["islem"] = (n >= MIN_TRADES, f"{n} işlem (en az {MIN_TRADES})")
    return checks


def run_tur4(cfg: Config, market: A.Market, universes: dict[str, list[str]], rules: dict[str, MarketRules],
             real_rules: dict[str, MarketRules], progress: A.Progress | None = None) -> T4Study:
    progress = progress or (lambda _msg: None)
    candidates = tur4_candidates()
    bounds = protocol_bounds(cfg, market, universes, candidates)
    progress(f"Dönemler (3. turla aynı): keşif {fmt_utc(bounds['kesif'][0], False)} → "
             f"{fmt_utc(bounds['kesif'][1], False)}, doğrulama {fmt_utc(bounds['dogrulama'][0], False)} → "
             f"{fmt_utc(bounds['dogrulama'][1], False)} (bitiş hariç)")
    reports = []
    for c in candidates:
        progress(f"{c.code} · {c.name} · {len(universes[c.universe])} coin")
        reports.append(run_candidate(c, cfg, submarket(market, universes[c.universe]), bounds, rules, real_rules,
                                     progress))
    by_code = {r.candidate.code: r for r in reports}
    for r in reports:
        r.checks = evaluate(r, by_code.get(REFERENCE_OF.get(r.candidate.code, "")))
    passed = sorted((r for r in reports if r.passed),
                    key=lambda r: r.portfolio["dogrulama"].metrics["return_to_drawdown"], reverse=True)
    bench_curves = {}
    for key, symbols in universes.items():
        sub = submarket(market, symbols)
        bench_curves[key] = {name: buy_and_hold({s: f["1d"] for s, f in sub.items()}, *bounds[name],
                                                cfg.backtest.initial_equity, cfg.costs.taker_fee,
                                                cfg.costs.slippage, DAY_MS)
                             for name, _ in A.PERIODS}
    return T4Study(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), universes, cfg.backtest.initial_equity,
                   bounds, reports, [r.candidate.code for r in passed[:MAX_SHADOW]], bench_curves)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
_pct, _ratio, _mark, _table = A._pct, A._ratio, A._mark, A._table


def _coin(symbol: str) -> str:
    return symbol.split("/")[0]


def _halt(run: A.PeriodRun) -> str:
    return fmt_utc(run.result.halted_at, False) if run.result.halted_at else "-"


def coin_contribution(r: T4Report, name: str) -> list[tuple[str, float, int]]:
    """(sembol, net kâr USDT, kapanan işlem) - net kâra göre büyükten küçüğe."""
    run = r.portfolio[name]
    counts: dict[str, int] = {}
    for t in run.result.trades:
        counts[t.symbol] = counts.get(t.symbol, 0) + 1
    rows = [(s, float(run.symbol_net.get(s, 0.0)), counts.get(s, 0)) for s in r.symbols]
    return sorted(rows, key=lambda x: (-x[1], x[0]))


def new_coin_share(r: T4Report, name: str, c1_coins: list[str]) -> tuple[float, float]:
    """(C1'in coinlerinden gelen net kâr, geniş evrene yeni eklenen coinlerden gelen net kâr) USDT."""
    rows = coin_contribution(r, name)
    return sum(v for s, v, _ in rows if s in c1_coins), sum(v for s, v, _ in rows if s not in c1_coins)


def format_tur4(study: T4Study) -> str:
    b = study.bounds
    wide, c1_coins = study.universes["genis"], study.universes["c1"]
    added = [_coin(s) for s in wide if s not in c1_coins]
    out = [
        f"### 4. tur sonuçları ({study.created})",
        "",
        (f"Geniş evren ({len(wide)} coin): {', '.join(_coin(s) for s in wide)}. C1'in {len(c1_coins)} coinine "
         f"eklenenler ({len(added)}): {', '.join(added) or '-'}."),
        (f"Keşif {fmt_utc(b['kesif'][0], False)} → {fmt_utc(b['kesif'][1] - DAY_MS, False)}, doğrulama "
         f"{fmt_utc(b['dogrulama'][0], False)} → {fmt_utc(b['dogrulama'][1] - DAY_MS, False)}; sınav dönemi "
         f"kullanılmadı. Her dönem {study.initial_equity:,.0f} USDT ile ayrı başlar; komisyon ve kayma dahil."),
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
                         f"{m['avg_r']:+.2f}", f"{_pct(bench['total_return'])} / {_pct(bench['max_drawdown'], False)}",
                         _halt(run)])
        out += [f"**{label} dönemi (tek hesap, en fazla 5 pozisyon)**", ""]
        out += _table(["Aday", "Net getiri", "Maks. düşüş", "Getiri/düşüş", "Kâr faktörü", "İşlem", "Kazanma",
                       "Ort. R", "Al-tut (aynı evren) getiri / düşüş", "Şalter"], rows)
        out.append("")

    rows = []
    for r in study.reports:
        cells = [f"{_mark(ok)} ({text})" for ok, text in r.checks.values()]
        verdict = "referans" if r.candidate.reference else ("**GEÇTİ**" if r.passed else "KALDI")
        rows.append([r.candidate.code, f"%{r.candidate.max_drawdown * 100:.0f}", *cells, verdict])
    out += ["**Ölçütler (W1 W0 ile, W2 W0A ile karşılaştırılır)**", ""]
    out += _table(["Aday", "Düşüş sınırı", "1. Keşif", "2. Doğrulama", "3. İyileşme", "4. Sağlamlık",
                   "5. İşlem sayısı", "Sonuç"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        main = r.portfolio["kesif"].metrics
        rows.append([r.candidate.code, "ana parametreler", _ratio(main["profit_factor"]), _pct(main["total_return"]),
                     _pct(main["max_drawdown"], False), str(main["n_trades"])])
        for label, m in r.robustness:
            rows.append(["", label, _ratio(m["profit_factor"]), _pct(m["total_return"]),
                         _pct(m["max_drawdown"], False), str(m["n_trades"])])
    out += ["**Sağlamlık (keşif dönemi)**", ""]
    out += _table(["Aday", "Değişiklik", "Kâr faktörü", "Net getiri", "Maks. düşüş", "İşlem"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        for name, label in A.PERIODS:
            m = r.no_halt[name]
            rows.append([r.candidate.code, label, _pct(m["total_return"]), _pct(m["max_drawdown"], False),
                         _ratio(m["profit_factor"]), str(m["n_trades"])])
    out += ["**Şaltersiz tanı (bilgi: %20 şalteri kapalı; %2 riskli adaylarda şalter zaten kapalı)**", ""]
    out += _table(["Aday", "Dönem", "Net getiri", "Maks. düşüş", "Kâr faktörü", "İşlem"], rows)
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
            rows.append([r.candidate.code, label, _pct(m["total_return"]), _pct(m["max_drawdown"], False),
                         str(m["n_trades"]), str(f["signals"]), str(f["skipped_min_notional"])])
    out += [f"**Uygulanabilirlik ({FEASIBILITY_EQUITY:.2f} USDT, Binance'in gerçek minimum işlem tutarlarıyla)**", ""]
    out += _table(["Aday", "Dönem", "Net getiri", "Maks. düşüş", "İşlem", "Sinyal", "Minimum tutar yüzünden atlanan"],
                  rows)
    out.append("")

    for r in study.reports:
        if r.candidate.universe != "genis":
            continue
        contrib = {name: {s: (v, n) for s, v, n in coin_contribution(r, name)} for name, _ in A.PERIODS}
        order = [s for s, _, _ in coin_contribution(r, "kesif")]
        rows = []
        for s in order:
            (kv, kn), (dv, dn) = contrib["kesif"][s], contrib["dogrulama"][s]
            rows.append([_coin(s) + ("" if s in c1_coins else " (yeni)"), f"{kv:+,.0f}", str(kn), f"{dv:+,.0f}",
                         str(dn)])
        shares = [new_coin_share(r, name, c1_coins) for name, _ in A.PERIODS]
        rows.append([f"**C1'in {len(c1_coins)} coini toplam**", f"{shares[0][0]:+,.0f}", "",
                     f"{shares[1][0]:+,.0f}", ""])
        rows.append(["**Yeni coinler toplam**", f"{shares[0][1]:+,.0f}", "", f"{shares[1][1]:+,.0f}", ""])
        out += [f"**{r.candidate.code}: coin başına katkı (net kâr USDT, aynı portföy hesabının içinde)**", ""]
        out += _table(["Coin", "Keşif net", "Keşif işlem", "Doğrulama net", "Doğrulama işlem"], rows)
        out.append("")

    if study.shadow:
        out.append(f"**Karar: {', '.join(study.shadow)} ölçütlerin tamamını geçti.** Protokol gereği yalnızca gölge "
                   "(sanal) bot olarak ileriye doğru sınanacak; gerçek paraya geçiş Burak'ın onayıyla.")
    else:
        out.append("**Karar: W1 ve W2 ölçütlerin tamamını geçemedi.** Protokol gereği geniş evren eklenmez; "
                   "C1 ve gölge botlar değişmeden devam eder.")
    return "\n".join(out)


def tur4_summary(study: T4Study) -> dict:
    def period(run: A.PeriodRun) -> dict:
        return {"start": fmt_utc(run.start, False), "end": fmt_utc(run.end, False), "metrics": run.metrics,
                "symbol_net": run.symbol_net, "buy_and_hold": run.benchmark, "halt": run.result.halt_reason or None}

    return A._clean({
        "created": study.created,
        "universes": study.universes,
        "bounds": {k: [fmt_utc(a, False), fmt_utc(b, False)] for k, (a, b) in study.bounds.items()},
        "shadow": study.shadow,
        "candidates": [{
            "code": r.candidate.code,
            "name": r.candidate.name,
            "universe": r.candidate.universe,
            "risk": r.candidate.risk_label,
            "max_drawdown_limit": r.candidate.max_drawdown,
            "reference": r.candidate.reference,
            "passed": r.passed,
            "checks": {k: {"passed": ok, "detail": text} for k, (ok, text) in r.checks.items()},
            "portfolio": {name: period(run) for name, run in r.portfolio.items()},
            "no_halt": r.no_halt,
            "robustness": [{"variant": label, "metrics": m} for label, m in r.robustness],
            "monte_carlo": r.monte_carlo,
            "feasibility": r.feasibility,
        } for r in study.reports],
    })


def save_tur4(out_dir: Path, study: T4Study, text: str, plot: bool = True) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(tur4_summary(study), indent=2, ensure_ascii=False),
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
    plot_tur4(chart, study)
    return chart


def plot_tur4(path: Path, study: T4Study) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"W0": ("#4C72B0", "-"), "W0A": ("#4C72B0", "--"), "W1": ("#C44E52", "-"), "W2": ("#C44E52", "--")}
    fig, axes = plt.subplots(len(A.PERIODS), 1, figsize=(12, 9))
    for ax, (name, label) in zip(axes, A.PERIODS, strict=True):
        for r in study.reports:
            curve = daily_equity(r.portfolio[name].result.equity) / study.initial_equity * 100
            color, style = styles.get(r.candidate.code, ("#555555", "-"))
            ax.plot(pd.to_datetime((curve.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True), curve.to_numpy(),
                    linewidth=1.5, color=color, linestyle=style, label=f"{r.candidate.code} {r.candidate.name}")
        bench = daily_equity(study.bench_curves["genis"][name]) / study.initial_equity * 100
        ax.plot(pd.to_datetime((bench.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True), bench.to_numpy(),
                color="#DD8452", linewidth=1.0, linestyle=":",
                label=f"Al-tut (geniş evren, {len(study.universes['genis'])} coin eşit)")
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
    disable_quick_edit()                 # pencereye tıklamak çalışmayı dondurmasın
    parser = argparse.ArgumentParser(description="4. araştırma turu: momentum için geniş coin evreni.")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    try:
        universes = load_universes(cfg)
        market = load_tur4_market(cfg, universes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[HATA] {exc}")
        return 2
    print(f"Veri yüklendi ({len(market)} coin); {cfg.data.exam_start} (sınav) sonrası mumlar atıldı.")
    rules = load_market_rules(cfg.data_dir, list(market), cfg.market_rules_default)   # önceki turlarla aynı
    try:
        study = run_tur4(cfg, market, universes, rules, rules, progress=print)
    except ValueError as exc:
        print(f"[HATA] {exc}")
        return 2
    text = format_tur4(study)
    print("\n" + text)
    out_dir = cfg.reports_dir / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_arastirma_tur4"
    chart = save_tur4(out_dir, study, text, plot=not args.no_plot)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
