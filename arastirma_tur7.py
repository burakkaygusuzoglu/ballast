"""7. araştırma turu: korumalı büyüme sistemi (R1 + düşüşe göre ölçeklenen risk, spot).

Protokol: docs/ARASTIRMA.md, 19. bölüm (aynı metin: docs/tur7_protokol.md). Adaylar, ölçütler ve seçim kuralı
sonuçlar görülmeden önce yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey değiştirmek testi
geçersiz kılar.

- R1-K10 / R1-K15 / R1-K20: R1 kuralları, taban risk %1,0 / %1,5 / %2,0; risk tepeden düşüşle doğrusal azalır
  (%25'te sıfır), şalter %25. Spot, kaldıraç yok.
- Referans: R1 sabit %0,5 (gölge bot) ve C1 sabit %0,5 (gerçek bot). Bilgi: R1 sabit %1,0, ölçeklemesiz.

    python arastirma_tur7.py           # Aşama A: geliştirme dönemleri (sınav verisi YÜKLENMEZ)
    python arastirma_tur7.py --sinav   # Aşama A + Aşama B (sınav dönemi, yalnızca bilgi; bir kez)

Çıktı: reports/<zaman>_arastirma_tur7_A (ya da _B) altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
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
from bot.cli import disable_quick_edit, load_config_or_exit, open_file, setup_console
from bot.config import PROJECT_ROOT, Config, MarketRules
from bot.data import load_market_rules
from bot.metrics import daily_equity
from bot.timeutil import DAY_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 19. bölüm) ------------------------------------
PROTOCOL_FILE = PROJECT_ROOT / "docs" / "tur7_protokol.md"
PROTOCOL_SHA256 = "cac5c2b1311201194acfe3e9b9b669257b060c2096a4540b4a45bd7e30de40bc"
SYMBOLS = T4.C1_SYMBOLS
PERIOD_BOUNDS = T4.PROTOCOL_BOUNDS                 # keşif 2022-07-20 → 2024-12-31, doğrulama 2025 → 2026-03-31
DD_LIMIT = 0.25                                    # Burak'ın kayıp sınırı: ölçek sınırı ve şalter
BASE_RISKS = {"R1-K10": 0.010, "R1-K15": 0.015, "R1-K20": 0.020}
DAILY_LIMIT_PER_RISK = 6
VALIDATION_MIN_PF = 1.1
MAX_SKIP_RATIO = 0.50                              # ölçüt 5: 50 USDT ile en fazla %50 sinyal atlanabilir
REAL_EQUITY = 50.0
MIN_NOTIONAL_SKIP = T3.MIN_NOTIONAL_SKIP
MC_BLOCK_DAYS, MC_PATHS, MC_SEED, MC_DAYS = 20, 10_000, 20261016, 365
EXAM_START = T5.EXAM_START


@dataclass(frozen=True)
class Setup:
    code: str
    name: str
    t3code: str                                    # sinyaller: 3. turdaki R0 (C1) ya da R1
    overrides: dict
    candidate: bool = False


def setups() -> dict[str, Setup]:
    t3 = {c.code: c for c in T3.tur3_candidates()}
    out = {}
    for code, base in BASE_RISKS.items():
        out[code] = Setup(code, f"R1, taban risk %{base * 100:g}, düşüşe göre azalan".replace(".", ","), "R1",
                          A._merge(t3["R1"].overrides, {"risk": {
                              "risk_per_trade": base, "dd_scaling": DD_LIMIT, "max_drawdown_halt": DD_LIMIT,
                              "daily_loss_limit": DAILY_LIMIT_PER_RISK * base}}), candidate=True)
    out["R1"] = Setup("R1", "R1 sabit %0,5 (gölge bot)", "R1", t3["R1"].overrides)
    out["C1"] = Setup("C1", "C1 sabit %0,5 (gerçek bot)", "R0", t3["R0"].overrides)
    out["R1-S10"] = Setup("R1-S10", "R1 sabit %1, ölçeklemesiz (bilgi)", "R1",
                          A._merge(t3["R1"].overrides, {"risk": {"risk_per_trade": 0.01, "max_drawdown_halt": DD_LIMIT,
                                                                 "daily_loss_limit": DAILY_LIMIT_PER_RISK * 0.01}}))
    return out


def signals_for(s: Setup, cfg_c: Config, market: A.Market) -> dict[str, pd.DataFrame]:
    t3 = {c.code: c for c in T3.tur3_candidates()}
    return T3.build_signals(t3[s.t3code], cfg_c, market)


def skip_ratio(run: A.PeriodRun) -> float:
    skipped = run.result.skipped.get(MIN_NOTIONAL_SKIP, 0)
    total = skipped + run.result.orders_submitted
    return skipped / total if total else 0.0


def mean_trade_risk(run: A.PeriodRun) -> float:
    """İşlemlerin ortalama riski (planlanan risk / girişten önceki özsermaye)."""
    eq = run.result.equity
    risks = []
    for t in run.result.trades:
        before = eq[eq.index <= t.entry_ts]
        if len(before) and before.iloc[-1] > 0:
            risks.append(t.risk_amount / float(before.iloc[-1]))
    return float(np.mean(risks)) if risks else float("nan")


# ---------------------------------------------------------------------------
# Aşama A
# ---------------------------------------------------------------------------
@dataclass
class StageA:
    created: str
    bounds: dict[str, tuple[int, int]]
    runs: dict[tuple[str, str], A.PeriodRun]                  # (kurulum, dönem), 10.000 USDT
    real: dict[tuple[str, str], A.PeriodRun]                  # (kurulum, dönem), 50 USDT, gerçek kurallar
    checks: dict[str, dict[str, tuple[bool, str]]] = field(default_factory=dict)
    chosen: str | None = None
    monte_carlo: dict = field(default_factory=dict)

    def passed(self, code: str) -> bool:
        return all(ok for ok, _ in self.checks[code].values())


def protocol_check() -> str:
    digest = hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise AssertionError(f"Protokol dosyası değişmiş: {digest} != {PROTOCOL_SHA256}")
    return digest


def period_bounds(cfg: Config, market: A.Market) -> dict[str, tuple[int, int]]:
    bounds = {name: (parse_utc(a), parse_utc(b)) for name, (a, b) in PERIOD_BOUNDS.items()}
    if bounds["dogrulama"][1] > parse_utc(cfg.data.exam_start):
        raise ValueError("Doğrulama dönemi sınav dönemine taşıyor.")
    for s in setups().values():
        cfg_c = A.apply_overrides(cfg, s.overrides)
        start, end = A.trade_window(signals_for(s, cfg_c, market), cfg_c.entry_ms)
        if start > bounds["kesif"][0] or end < bounds["dogrulama"][1]:
            raise ValueError(f"{s.code}: veri protokol dönemlerini kapsamıyor.")
    return bounds


def evaluate(s: StageA, code: str) -> dict[str, tuple[bool, str]]:
    k, d = s.runs[(code, "kesif")], s.runs[(code, "dogrulama")]
    rk, rd = s.runs[("R1", "kesif")].metrics, s.runs[("R1", "dogrulama")].metrics
    km, dm = k.metrics, d.metrics
    halted = [p for p, run in (("keşif", k), ("doğrulama", d)) if run.result.halted_at is not None]
    ratio = skip_ratio(s.real[(code, "dogrulama")])
    return {
        "1_kar": (km["net_profit"] > 0 and dm["net_profit"] > 0,
                  f"keşif {A._pct(km['total_return'])}, doğrulama {A._pct(dm['total_return'])}"),
        "2_salter": (not halted and km["max_drawdown"] < DD_LIMIT and dm["max_drawdown"] < DD_LIMIT,
                     f"düşüş keşif {A._pct(km['max_drawdown'], False)}, doğrulama {A._pct(dm['max_drawdown'], False)}"
                     + (f"; şalter: {', '.join(halted)}" if halted else "")),
        "3_kf": (dm["profit_factor"] >= VALIDATION_MIN_PF, f"doğrulama KF {A._ratio(dm['profit_factor'])}"),
        "4_getiri": (km["total_return"] > rk["total_return"] and dm["total_return"] > rd["total_return"],
                     f"R1 %0,5: keşif {A._pct(rk['total_return'])}, doğrulama {A._pct(rd['total_return'])}"),
        "5_56usdt": (ratio <= MAX_SKIP_RATIO, f"doğrulamada atlanan sinyal %{ratio * 100:.0f}"),
    }


def run_stage_a(cfg: Config, market: A.Market, rules: dict[str, MarketRules],
                progress: Callable[[str], None] | None = None) -> StageA:
    progress = progress or (lambda _msg: None)
    bounds = period_bounds(cfg, market)
    s = StageA(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), bounds, {}, {})
    for setup in setups().values():
        cfg_c = A.apply_overrides(cfg, setup.overrides)
        sig = signals_for(setup, cfg_c, market)
        cfg_real = A.apply_overrides(cfg_c, {"backtest": {"initial_equity": REAL_EQUITY}})
        for name, (start, end) in bounds.items():
            s.runs[(setup.code, name)] = A.run_period(cfg_c, sig, market, "4h", start, end, rules)
            s.real[(setup.code, name)] = A.run_period(cfg_real, sig, market, "4h", start, end, rules)
        k, d = s.runs[(setup.code, "kesif")].metrics, s.runs[(setup.code, "dogrulama")].metrics
        progress(f"  {setup.code:<7} keşif {A._pct(k['total_return']):>8} düşüş {A._pct(k['max_drawdown'], False):>6}"
                 f" | doğrulama {A._pct(d['total_return']):>8} düşüş {A._pct(d['max_drawdown'], False):>6}")
    for code in BASE_RISKS:
        s.checks[code] = evaluate(s, code)
    passing = [code for code in BASE_RISKS if s.passed(code)]
    s.chosen = max(passing, key=lambda c: BASE_RISKS[c]) if passing else None
    if s.chosen is not None:
        s.monte_carlo = block_bootstrap([s.runs[(s.chosen, p)] for p in ("kesif", "dogrulama")])
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
            "p90": float(np.percentile(finals, 90)), "dusus25": float(np.mean(dd >= DD_LIMIT)),
            "zarar": float(np.mean(finals < 1.0)), "medyan_dusus": float(np.median(dd)), "gun": len(rets)}


# ---------------------------------------------------------------------------
# Aşama B: sınav dönemi (yalnızca bilgi)
# ---------------------------------------------------------------------------
@dataclass
class StageB:
    start: int
    end: int
    runs: dict[str, A.PeriodRun]
    real: dict[str, A.PeriodRun]


def run_stage_b(cfg: Config, market: A.Market, start: int, end: int, rules: dict[str, MarketRules],
                stage_a: StageA) -> StageB:
    if not (parse_utc(EXAM_START) == start < end):
        raise ValueError("Sınav dönemi 2026-04-01'de başlamalı ve en az bir tam gün içermeli.")
    codes = [c for c in (stage_a.chosen, "R1", "C1") if c is not None]
    b = StageB(start, end, {}, {})
    for code in codes:
        setup = setups()[code]
        cfg_c = A.apply_overrides(cfg, setup.overrides)
        sig = signals_for(setup, cfg_c, market)
        b.runs[code] = A.run_period(cfg_c, sig, market, "4h", start, end, rules)
        cfg_real = A.apply_overrides(cfg_c, {"backtest": {"initial_equity": REAL_EQUITY}})
        b.real[code] = A.run_period(cfg_real, sig, market, "4h", start, end, rules)
    return b


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
LABELS = {"1_kar": "1. Kâr", "2_salter": "2. Şalter", "3_kf": "3. KF", "4_getiri": "4. Daha yüksek getiri",
          "5_56usdt": "5. 50 USDT"}


def _status(ok: bool) -> str:
    return "GEÇTİ" if ok else "KALDI"


def format_stage_a(s: StageA) -> str:
    out = [f"### 7. tur, Aşama A: geliştirme dönemleri ({s.created})", "",
           ("Keşif 2022-07-20 → 2024-12-31, doğrulama 2025-01-01 → 2026-03-31; sınav dönemi yüklenmedi. Her dönem "
            "10.000 USDT ile ayrı başlar; spot, komisyon ve kayma dahil. Adaylarda risk tepeden düşüşle doğrusal "
            "azalır (%25'te sıfır), şalter %25."),
           f"Protokol parmak izi: {PROTOCOL_SHA256[:16]}… (docs/tur7_protokol.md)", ""]
    rows = []
    for code, setup in setups().items():
        for name, plabel in A.PERIODS:
            run = s.runs[(code, name)]
            m = run.metrics
            rows.append([f"{code}: {setup.name}", plabel, A._pct(m["total_return"]), A._pct(m["cagr"]),
                         A._pct(m["max_drawdown"], False), A._ratio(m["return_to_drawdown"]),
                         A._ratio(m["profit_factor"]), str(m["n_trades"]), A._pct(mean_trade_risk(run), False),
                         "evet" if run.result.halted_at is not None else "-"])
    out += A._table(["Kurulum", "Dönem", "Net getiri", "CAGR", "Maks. düşüş", "Getiri/düşüş", "KF", "İşlem",
                     "Ort. işlem riski", "Şalter"], rows) + [""]
    out += ["**50 USDT ile (gerçek minimum işlem tutarları)**", ""]
    rows = []
    for code in setups():
        for name, plabel in A.PERIODS:
            run = s.real[(code, name)]
            m = run.metrics
            rows.append([code, plabel, A._pct(m["total_return"]), f"{m['final_equity']:.2f}",
                         A._pct(m["max_drawdown"], False), str(m["n_trades"]),
                         str(run.result.skipped.get(MIN_NOTIONAL_SKIP, 0)), f"%{skip_ratio(run) * 100:.0f}"])
    out += A._table(["Kurulum", "Dönem", "Net getiri", "Son bakiye (USDT)", "Maks. düşüş", "İşlem",
                     "Min. tutar yüzünden atlanan", "Atlanan oranı"], rows) + [""]
    out += ["**Ölçütler**", ""]
    rows = []
    for code in BASE_RISKS:
        cells = [f"{LABELS[k]}: {_status(ok)} ({detail})" for k, (ok, detail) in s.checks[code].items()]
        rows.append([code, " · ".join(cells), _status(s.passed(code))])
    out += A._table(["Aday", "Ölçütler", "Sonuç"], rows) + [""]
    if s.chosen is None:
        out += ["**Karar: hiçbir aday geçmedi.** Sistem benimsenmez; R1 sabit %0,5 gölge testine devam eder.", ""]
    else:
        out += [f"**Karar: {s.chosen} seçildi** ({setups()[s.chosen].name}). Sıradaki adım: en az 4 hafta sanal test.",
                ""]
        mc = s.monte_carlo
        paths = f"{MC_PATHS:,}".replace(",", ".")
        out += [(f"**Bilgi: Monte Carlo, 1 yıl** ({s.chosen}; keşif + doğrulama günlük getirileri, 20 günlük bloklar, "
                f"{paths} yol)"), ""]
        out += A._table(["Medyan", "%10 dilim", "%90 dilim", "Medyan düşüş", "Düşüş ≥ %25", "Yılı zararla bitirme",
                         "50 USDT ile medyan kâr"],
                        [[f"{mc['medyan']:.2f}x", f"{mc['p10']:.2f}x", f"{mc['p90']:.2f}x",
                          A._pct(mc["medyan_dusus"], False), A._pct(mc["dusus25"], False), A._pct(mc["zarar"], False),
                          f"{REAL_EQUITY * (mc['medyan'] - 1):+.1f}"]]) + [""]
        out += [f"**Bilgi: {s.chosen} coin bazında net kâr (10.000 USDT, USDT)**", ""]
        nets = {name: s.runs[(s.chosen, name)].symbol_net for name in ("kesif", "dogrulama")}
        rows = [[sym.replace("/USDT", ""), f"{nets['kesif'][sym]:+,.0f}", f"{nets['dogrulama'][sym]:+,.0f}"]
                for sym in nets["kesif"]]
        out += A._table(["Coin", "Keşif", "Doğrulama"], rows) + [""]
    return "\n".join(out)


def format_stage_b(b: StageB) -> str:
    out = [(f"### 7. tur, Aşama B: sınav dönemi ({fmt_utc(b.start, False)} → {fmt_utc(b.end - DAY_MS, False)}), "
           "yalnızca bilgi"), "", "Bu dönem 5. turda kullanıldı; sonuçlar karar değiştirmez.", ""]
    rows = []
    for code, run in b.runs.items():
        m, r = run.metrics, b.real[code].metrics
        rows.append([f"{code}: {setups()[code].name}", A._pct(m["total_return"]), A._pct(m["max_drawdown"], False),
                     A._ratio(m["profit_factor"]), str(m["n_trades"]), f"{r['final_equity']:.2f}"])
    out += A._table(["Kurulum", "Net getiri", "Maks. düşüş", "KF", "İşlem", "50 USDT → son bakiye"], rows)
    return "\n".join(out)


def stage_a_summary(s: StageA) -> dict:
    def pack(run: A.PeriodRun) -> dict:
        return A._clean(run.metrics) | {"salter": run.result.halted_at, "atlanan_oran": skip_ratio(run),
                                        "ort_islem_riski": A._clean(mean_trade_risk(run))}
    return {"olusturuldu": s.created, "protokol_sha256": PROTOCOL_SHA256, "secilen": s.chosen,
            "olcutler": {c: {k: [ok, d] for k, (ok, d) in v.items()} for c, v in s.checks.items()},
            "monte_carlo": s.monte_carlo,
            "kosular": {f"{c}|{n}": pack(r) for (c, n), r in s.runs.items()},
            "gercek_bakiye": {f"{c}|{n}": pack(r) for (c, n), r in s.real.items()}}


def save_report(out_dir: Path, summary: dict, text: str, trades: list[tuple[str, str, list]],
                plot: Callable[[Path], None] | None) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "islemler.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kurulum", "donem", "sembol", "giris", "cikis", "giris_fiyat", "cikis_fiyat", "net_kar",
                    "planlanan_risk", "r", "neden"])
        for label, period, items in trades:
            for t in items:
                w.writerow([label, period, t.symbol, fmt_utc(t.entry_ts), fmt_utc(t.exit_ts), f"{t.entry_price:.8g}",
                            f"{t.exit_price:.8g}", f"{t.net_pnl:.4f}", f"{t.risk_amount:.4f}", f"{t.r_multiple:.3f}",
                            t.exit_reason])
    if plot is None:
        return None
    chart = out_dir / "grafik.png"
    plot(chart)
    return chart


def plot_stage(path: Path, s: StageA, b: StageB | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    styles = {"C1": ("gray", "--"), "R1": ("#4C72B0", "-"), "R1-K10": ("#55A868", "-"),
              "R1-K15": ("#DD8452", "-"), "R1-K20": ("#C44E52", "-"), "R1-S10": ("#8172B3", ":")}
    panels = [(name, label, s.runs) for name, label in A.PERIODS]
    if b is not None:
        panels.append(("sinav", "Sınav (bilgi)", {(c, "sinav"): r for c, r in b.runs.items()}))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5))
    for ax, (name, label, runs) in zip(np.atleast_1d(axes), panels, strict=True):
        for code in setups():
            run = runs.get((code, name))
            if run is None:
                continue
            x, y = T5._curve(run.result.equity, run.metrics["initial_equity"])
            color, style = styles[code]
            ax.plot(x, y.to_numpy(), color=color, linestyle=style, linewidth=1.3,
                    label=f"{code}{' ✓' if code == s.chosen else ''}")
        ax.axhline(100 * (1 - DD_LIMIT), color="black", linewidth=0.6, linestyle=":")
        ax.set_title(f"{label}: özsermaye (başlangıç 100)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()
    parser = argparse.ArgumentParser(description="7. araştırma turu: korumalı büyüme sistemi.")
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
    text = format_stage_a(stage_a)
    summary = {"asama_a": stage_a_summary(stage_a)}
    trades = [(code, name, run.result.trades) for (code, name), run in stage_a.runs.items()
              if code in (stage_a.chosen, "R1")]
    stage_b = None
    if args.sinav:
        end = T5.last_full_day(raw)
        market_full = A.prepare_market(raw, end)
        print(f"Sınav verisi yüklendi: {EXAM_START} → {fmt_utc(end - DAY_MS, False)} (son tam gün).")
        stage_b = run_stage_b(cfg, market_full, parse_utc(EXAM_START), end, rules, stage_a)
        text += "\n\n" + format_stage_b(stage_b)
        summary["asama_b"] = {c: A._clean(r.metrics) | {"son_bakiye_56": stage_b.real[c].metrics["final_equity"]}
                              for c, r in stage_b.runs.items()}
        trades += [(code, "sinav", run.result.trades) for code, run in stage_b.runs.items()]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.reports_dir / f"{stamp}_arastirma_tur7_{'B' if args.sinav else 'A'}"
    chart = save_report(out_dir, summary, text, trades,
                        None if args.no_plot else (lambda p: plot_stage(p, stage_a, stage_b)))
    print("\n" + text)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
