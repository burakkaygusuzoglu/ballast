"""2. araştırma turu: trend takibi (T2, T3) geniş coin evreninde, tek hesapta.

Protokol: docs/ARASTIRMA.md, 6. bölüm. Coin listesi, adaylar ve ölçütler sonuçlar görülmeden önce
yazıldı; bu dosya onları aynen uygular.

Önce 7_tur2_veri_indir.bat (arastirma_verisi.py) evreni seçip verisini indirir. Bu betik:
  - portföyü (10 coin, tek hesap, en fazla 5 açık pozisyon) keşif ve doğrulama dönemlerinde sınar,
  - her coini ayrıca tek başına sınar (genişlik ölçütü için),
  - sağlamlık, Monte Carlo ve 50 USDT ile uygulanabilirlik ölçümlerini yapar,
  - protokoldeki beş ölçütü uygular, kazananı (varsa) seçer.

GÜVENLİK: Sınav dönemi (data.exam_start sonrası) burada ASLA kullanılmaz.

Kullanım:
    python arastirma_tur2.py
    python arastirma_tur2.py --no-plot

Çıktı: reports/<zaman>_arastirma_tur2/ altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import arastirma as A
from arastirma_verisi import TIMEFRAME, UNIVERSE_SIZE, load_universe
from bot.backtester import run_backtest
from bot.cli import load_config_or_exit, open_file, setup_console
from bot.config import Config, MarketRules
from bot.data import candle_path, load_candles, load_market_rules
from bot.metrics import buy_and_hold, compute_metrics, daily_equity
from bot.timeutil import DAY_MS, TIMEFRAME_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 6. bölüm) ------------------------------------
MAX_OPEN_POSITIONS = 5           # tek hesapta aynı anda en fazla açık pozisyon
MIN_PROFITABLE_COINS = 6         # ölçüt 3: keşifte tek başına kârlı coin sayısı
MIN_TRADES = 100                 # ölçüt 4: portföyün keşif + doğrulama işlem sayısı
FEASIBILITY_EQUITY = 50.0       # bilgi: küçük hesap ve gerçek minimum işlem tutarlarıyla
MIN_NOTIONAL_SKIP = "boyut çok küçük (min. tutar)"   # motorun "minimum tutarın altında" atlama nedeni


def tur2_candidates() -> list[A.Candidate]:
    """T2 = S2, T3 = S3 kuralları; tek fark çok coinli hesabın pozisyon sınırı."""
    base = {c.code: c for c in A.protocol_candidates()}
    out = []
    for code, source in (("T2", "S2"), ("T3", "S3")):
        c = base[source]
        out.append(dataclasses.replace(
            c, code=code, min_trades=MIN_TRADES,
            overrides=A._merge(c.overrides, {"risk": {"max_open_positions": MAX_OPEN_POSITIONS}}),
        ))
    return out


def load_tur2_market(cfg: Config) -> tuple[A.Market, dict]:
    universe = load_universe(cfg)
    symbols = list(universe["secilen"])
    if len(symbols) != UNIVERSE_SIZE:
        raise ValueError(f"Evren dosyasında {UNIVERSE_SIZE} coin olmalı (bulunan: {len(symbols)}).")
    raw = {}
    for symbol in symbols:
        path = candle_path(cfg.data_dir, cfg.exchange.id, symbol, TIMEFRAME)
        if not path.exists():
            raise FileNotFoundError(path.name)
        raw[symbol] = {TIMEFRAME: load_candles(path)}
    return A.prepare_market(raw, parse_utc(cfg.data.exam_start)), universe


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------
@dataclass
class Tur2Report:
    candidate: A.Candidate
    portfolio: dict[str, A.PeriodRun]              # dönem -> portföy (tek hesap)
    coins: dict[str, dict[str, A.PeriodRun]]       # sembol -> dönem -> tek coinlik hesap
    robustness: list[tuple[str, dict]]
    monte_carlo: dict
    feasibility: dict[str, dict]                   # dönem -> 50 USDT ile sonuç
    correlation: dict[str, float]                  # dönem -> coin bazlı getirilerin ortalama korelasyonu
    checks: dict[str, tuple[bool, str]]

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.checks.values())


@dataclass
class Tur2Study:
    created: str
    universe: dict
    initial_equity: float
    data_range: tuple[str, str]
    reports: list[Tur2Report]
    winner: str | None
    bench_curves: dict[str, pd.Series]


def mean_correlation(runs: list[A.PeriodRun]) -> float:
    returns = pd.concat([daily_equity(run.result.equity).pct_change() for run in runs], axis=1)
    corr = returns.corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]       # hiç işlem yapmayan coinin korelasyonu tanımsız
    return float(finite.mean()) if finite.size else float("nan")


def evaluate_tur2(c: A.Candidate, portfolio: dict[str, A.PeriodRun], coins: dict[str, dict[str, A.PeriodRun]],
                  robustness: list[tuple[str, dict]]) -> dict[str, tuple[bool, str]]:
    """Protokoldeki beş ölçüt (portföy düzeyinde): (geçti mi, kısa açıklama)."""
    k, d = portfolio["kesif"].metrics, portfolio["dogrulama"].metrics
    problems = []
    if not k["profit_factor"] >= A.EXPLORE_MIN_PF:
        problems.append(f"PF {A._ratio(k['profit_factor'])} < {A.EXPLORE_MIN_PF}")
    if not k["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not k["max_drawdown"] <= A.MAX_DRAWDOWN:
        problems.append(f"düşüş %{k['max_drawdown'] * 100:.1f}")
    checks = {"kesif": (not problems, "; ".join(problems) or "tümü sağlandı")}

    problems = []
    if not d["net_profit"] > 0:
        problems.append("net getiri ≤ 0")
    if not d["profit_factor"] >= A.VALIDATE_MIN_PF:
        problems.append(f"PF {A._ratio(d['profit_factor'])} < {A.VALIDATE_MIN_PF}")
    if not d["max_drawdown"] <= A.MAX_DRAWDOWN:
        problems.append(f"düşüş %{d['max_drawdown'] * 100:.1f}")
    checks["dogrulama"] = (not problems, "; ".join(problems) or "tümü sağlandı")

    winners = sum(1 for runs in coins.values() if runs["kesif"].metrics["net_profit"] > 0)
    checks["genislik"] = (winners >= MIN_PROFITABLE_COINS,
                          f"{winners}/{len(coins)} coin kârlı (en az {MIN_PROFITABLE_COINS})")

    n = k["n_trades"] + d["n_trades"]
    checks["islem"] = (n >= c.min_trades, f"{n} işlem (en az {c.min_trades})")

    weak = [label for label, m in robustness if not m["profit_factor"] >= A.ROBUST_MIN_PF]
    detail = f"PF < {A.ROBUST_MIN_PF}: {', '.join(weak)}" if weak else "tüm varyantlarda PF ≥ 1"
    checks["saglamlik"] = (bool(robustness) and not weak, detail)
    return checks


def run_candidate_tur2(c: A.Candidate, cfg: Config, market: A.Market, rules: dict[str, MarketRules],
                       progress: A.Progress) -> Tur2Report:
    cfg_c = A.apply_overrides(cfg, c.overrides)
    signals = A.build_signals(c, cfg_c, market)
    trade_start, data_end = A.trade_window(signals, cfg_c.entry_ms)
    split = parse_utc(A.VALIDATION_START)
    if not trade_start < split < data_end:
        raise ValueError(f"{c.code}: veri keşif ve doğrulama dönemlerini kapsamıyor.")
    bounds = {"kesif": (trade_start, split), "dogrulama": (split, min(data_end, parse_utc(cfg.data.exam_start)))}

    portfolio = {}
    for name, label in A.PERIODS:
        t0 = time.perf_counter()
        portfolio[name] = A.run_period(cfg_c, signals, market, c.entry_tf, *bounds[name], rules)
        m = portfolio[name].metrics
        progress(f"  {c.code} portföy {label:<10} {m['n_trades']:>4} işlem  {A._pct(m['total_return']):>8}  "
                 f"({time.perf_counter() - t0:.1f} sn)")

    coins = {
        symbol: {name: A.run_period(cfg_c, {symbol: signals[symbol]}, {symbol: market[symbol]}, c.entry_tf,
                                    *bounds[name], rules)
                 for name, _ in A.PERIODS}
        for symbol in signals
    }
    progress(f"  {c.code} coin bazında: {len(coins)} coin")

    initial = cfg.backtest.initial_equity
    robustness = []
    for label, variant in A.robustness_variants(c):
        cfg_v = A.apply_overrides(cfg, variant.overrides)
        signals_v = A.build_signals(variant, cfg_v, market)
        start = max(A.trade_window(signals_v, cfg_v.entry_ms)[0], trade_start)
        result = run_backtest(cfg_v, signals_v, start, split, initial, rules)
        robustness.append((label, compute_metrics(result.equity, result.trades, initial, result.total_fees,
                                                  result.exposure)))
    progress(f"  {c.code} sağlamlık: {len(robustness)} varyant")

    trades = portfolio["kesif"].result.trades + portfolio["dogrulama"].result.trades
    mc = A.monte_carlo([t.r_multiple for t in trades], cfg_c.risk.risk_per_trade)

    feasibility = {}
    for name, _ in A.PERIODS:
        result = run_backtest(cfg_c, signals, *bounds[name], FEASIBILITY_EQUITY, rules)
        feasibility[name] = {
            "metrics": compute_metrics(result.equity, result.trades, FEASIBILITY_EQUITY, result.total_fees,
                                       result.exposure),
            "signals": result.signals_seen,
            "orders": result.orders_submitted,
            "skipped_min_notional": int(result.skipped.get(MIN_NOTIONAL_SKIP, 0)),
            "skipped": dict(result.skipped),
        }
    correlation = {name: mean_correlation([coins[s][name] for s in coins]) for name, _ in A.PERIODS}
    return Tur2Report(c, portfolio, coins, robustness, mc, feasibility, correlation,
                      evaluate_tur2(c, portfolio, coins, robustness))


def choose_winner(reports: list[Tur2Report]) -> str | None:
    """İkisi de geçerse doğrulama dönemindeki getiri/düşüş oranı yüksek olan (protokol)."""
    passed = [r for r in reports if r.passed]
    if not passed:
        return None
    return max(passed, key=lambda r: r.portfolio["dogrulama"].metrics["return_to_drawdown"]).candidate.code


def run_tur2(cfg: Config, market: A.Market, universe: dict, rules: dict[str, MarketRules],
             candidates: list[A.Candidate] | None = None, progress: A.Progress | None = None) -> Tur2Study:
    progress = progress or (lambda _msg: None)
    reports = []
    for c in candidates or tur2_candidates():
        progress(f"{c.code} · {c.name}")
        reports.append(run_candidate_tur2(c, cfg, market, rules, progress))
    hourly = {s: f[TIMEFRAME] for s, f in market.items()}
    bench_curves = {}
    for name, _ in A.PERIODS:
        start = min(r.portfolio[name].start for r in reports)
        end = max(r.portfolio[name].end for r in reports)
        bench_curves[name] = buy_and_hold(hourly, start, end, cfg.backtest.initial_equity, cfg.costs.taker_fee,
                                          cfg.costs.slippage, TIMEFRAME_MS[TIMEFRAME])
    first = max(int(f[TIMEFRAME]["ts"].iloc[0]) for f in market.values())
    last = min(int(f[TIMEFRAME]["ts"].iloc[-1]) for f in market.values()) + TIMEFRAME_MS[TIMEFRAME]
    return Tur2Study(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), universe, cfg.backtest.initial_equity,
                     (fmt_utc(first, False), fmt_utc(last, False)), reports, choose_winner(reports), bench_curves)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
_pct, _ratio, _mark, _table = A._pct, A._ratio, A._mark, A._table


def _base(symbol: str) -> str:
    return symbol.split("/")[0]


def _signed(x: float) -> str:
    return "-" if x != x else f"{x:+.2f}"


def format_tur2(study: Tur2Study) -> str:
    u = study.universe
    out = [
        f"### 2. tur sonuçları ({study.created})",
        "",
        f"Evren ({len(u['secilen'])} coin): {', '.join(_base(s) for s in u['secilen'])}. "
        + ("Elenenler: " + "; ".join(f"{_base(s)} ({why})" for s, why in u["elenen"].items()) + ". "
           if u["elenen"] else "")
        + f"Veri (1 saatlik): {study.data_range[0]} → {study.data_range[1]}; sınav dönemi kullanılmadı. "
        f"Başlangıç özsermayesi her dönemde {study.initial_equity:,.0f} USDT; komisyon ve kayma dahil.",
        "",
    ]
    for name, label in A.PERIODS:
        rows = []
        for r in study.reports:
            run = r.portfolio[name]
            m, b = run.metrics, run.benchmark
            rows.append([
                f"{r.candidate.code} {r.candidate.name}",
                f"{fmt_utc(run.start, False)} → {fmt_utc(run.end, False)}",
                _pct(m["total_return"]), _pct(m["max_drawdown"], False), _ratio(m["profit_factor"]),
                _ratio(m["pre_cost_profit_factor"]),
                _pct((m["fees_closed"] + m["slippage_closed"]) / m["initial_equity"], False),
                str(m["n_trades"]), _pct(m["win_rate"], False), f"{m['avg_r']:+.2f}",
                f"{_pct(b['total_return'])} / {_pct(b['max_drawdown'], False)}",
                fmt_utc(run.result.halted_at, False) if run.result.halted_at else "-",
            ])
        out += [f"**Portföy: {label.lower()} dönemi (10 coin, tek hesap, en fazla {MAX_OPEN_POSITIONS} pozisyon)**", ""]
        out += _table(["Aday", "Dönem", "Net getiri", "Maks. düşüş", "Kâr faktörü", "Maliyetsiz KF", "Maliyet",
                       "İşlem", "Kazanma", "Ort. R", "Al-tut getiri / düşüş", "Şalter"], rows)
        out.append("")

    rows = []
    for r in study.reports:
        cells = [f"{_mark(ok)} ({text})" for ok, text in r.checks.values()]
        rows.append([r.candidate.code, *cells, "**GEÇTİ**" if r.passed else "KALDI"])
    out += ["**Ölçütler**", ""]
    out += _table(["Aday", "1. Keşif", "2. Doğrulama", "3. Genişlik", "4. İşlem sayısı", "5. Sağlamlık", "Sonuç"],
                  rows)
    out.append("")

    for r in study.reports:
        rows = []
        for symbol, runs in r.coins.items():
            k, d = runs["kesif"].metrics, runs["dogrulama"].metrics
            rows.append([_base(symbol),
                         _pct(k["total_return"]), _ratio(k["profit_factor"]), str(k["n_trades"]),
                         _pct(d["total_return"]), _ratio(d["profit_factor"]), str(d["n_trades"])])
        out += [f"**{r.candidate.code}: coin bazında (her coin ayrı hesapta)**", ""]
        out += _table(["Coin", "Keşif net", "Keşif KF", "Keşif işlem", "Doğrulama net", "Doğrulama KF",
                       "Doğrulama işlem"], rows)
        out.append("")

    rows = []
    for r in study.reports:
        base = r.portfolio["kesif"].metrics
        rows.append([r.candidate.code, "ana parametreler", _ratio(base["profit_factor"]),
                     _pct(base["total_return"]), str(base["n_trades"])])
        for label, m in r.robustness:
            rows.append(["", label, _ratio(m["profit_factor"]), _pct(m["total_return"]), str(m["n_trades"])])
    out += ["**Sağlamlık (portföy, keşif dönemi, parametreler tek tek ±%25)**", ""]
    out += _table(["Aday", "Değişiklik", "Kâr faktörü", "Net getiri", "İşlem"], rows)
    out.append("")

    rows = []
    for r in study.reports:
        mc = r.monte_carlo
        rows.append([r.candidate.code, str(mc["n"]), f"{mc['avg_r']:+.2f}", _pct(mc["actual_dd"], False),
                     _pct(mc["median_dd"], False), _pct(mc["p95_dd"], False)])
    out += [f"**Monte Carlo (portföy işlemleri, sıra {A.MC_RUNS} kez karıştırıldı, işlem başına %1 risk)**", ""]
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

    rows = [[r.candidate.code, *[_signed(r.correlation[name]) for name, _ in A.PERIODS]] for r in study.reports]
    out += ["**Coinler arası ortalama korelasyon (coin bazında günlük strateji getirileri)**", ""]
    out += _table(["Aday", *[label for _, label in A.PERIODS]], rows)
    out.append("")

    if study.winner:
        out.append(f"**Karar: {study.winner} ölçütlerin tamamını geçti ve seçildi.** Sıradaki adım: bota bağlamak, "
                   "ardından sınav (bir kez) ve paper trading.")
    else:
        out.append("**Karar: hiçbir aday ölçütlerin tamamını geçemedi.** Protokol gereği paper trading başlamaz.")
    return "\n".join(out)


def tur2_summary(study: Tur2Study) -> dict:
    def period(run: A.PeriodRun) -> dict:
        return {"start": fmt_utc(run.start, False), "end": fmt_utc(run.end, False), "metrics": run.metrics,
                "symbol_net": run.symbol_net, "buy_and_hold": run.benchmark, "halt": run.result.halt_reason or None}

    return A._clean({
        "created": study.created,
        "universe": study.universe,
        "initial_equity": study.initial_equity,
        "data_range": list(study.data_range),
        "winner": study.winner,
        "protocol": {"max_open_positions": MAX_OPEN_POSITIONS, "min_profitable_coins": MIN_PROFITABLE_COINS,
                     "min_trades": MIN_TRADES, "explore_min_pf": A.EXPLORE_MIN_PF,
                     "validate_min_pf": A.VALIDATE_MIN_PF, "max_drawdown": A.MAX_DRAWDOWN,
                     "robust_min_pf": A.ROBUST_MIN_PF, "feasibility_equity": FEASIBILITY_EQUITY},
        "candidates": [{
            "code": r.candidate.code,
            "name": r.candidate.name,
            "params": asdict(r.candidate.params) if r.candidate.params else None,
            "overrides": r.candidate.overrides,
            "passed": r.passed,
            "checks": {k: {"passed": ok, "detail": text} for k, (ok, text) in r.checks.items()},
            "portfolio": {name: period(run) for name, run in r.portfolio.items()},
            "coins": {s: {name: period(run) for name, run in runs.items()} for s, runs in r.coins.items()},
            "robustness": [{"variant": label, "metrics": m} for label, m in r.robustness],
            "monte_carlo": r.monte_carlo,
            "feasibility": r.feasibility,
            "correlation": r.correlation,
        } for r in study.reports],
    })


def save_tur2(out_dir: Path, study: Tur2Study, text: str, plot: bool = True) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(tur2_summary(study), indent=2, ensure_ascii=False),
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
    plot_tur2(chart, study)
    return chart


def plot_tur2(path: Path, study: Tur2Study) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"T2": "#55A868", "T3": "#C44E52"}
    fig, axes = plt.subplots(len(A.PERIODS), 1, figsize=(12, 9))
    for ax, (name, label) in zip(axes, A.PERIODS, strict=True):
        for r in study.reports:
            curve = daily_equity(r.portfolio[name].result.equity) / study.initial_equity * 100
            dates = pd.to_datetime((curve.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True)
            ax.plot(dates, curve.to_numpy(), linewidth=1.6, color=colors.get(r.candidate.code),
                    label=f"{r.candidate.code} {r.candidate.name}")
        bench = daily_equity(study.bench_curves[name]) / study.initial_equity * 100
        ax.plot(pd.to_datetime((bench.index.to_numpy() + 1) * DAY_MS, unit="ms", utc=True), bench.to_numpy(),
                color="#DD8452", linewidth=1.1, linestyle="--", label="Al-tut (10 coin eşit)")
        ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_title(f"{label} dönemi · 10 coinlik portföy (başlangıç = 100) · komisyon ve kayma dahil",
                     loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="2. araştırma turu: trend takibi, geniş coin evreni.")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    try:
        market, universe = load_tur2_market(cfg)
    except FileNotFoundError as exc:
        print(f"[HATA] Bulunamadı: {exc}\nÖnce 7_tur2_veri_indir.bat çalıştırın.")
        return 2
    except ValueError as exc:
        print(f"[HATA] {exc}")
        return 2
    print(f"Evren: {', '.join(_base(s) for s in universe['secilen'])}")
    print(f"Veri yüklendi; {cfg.data.exam_start} (sınav) sonrası mumlar atıldı.")
    rules = load_market_rules(cfg.data_dir, list(market), cfg.market_rules_default)
    try:
        study = run_tur2(cfg, market, universe, rules, progress=print)
    except ValueError as exc:
        print(f"[HATA] {exc}")
        return 2
    text = format_tur2(study)
    print("\n" + text)
    out_dir = cfg.reports_dir / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_arastirma_tur2"
    chart = save_tur2(out_dir, study, text, plot=not args.no_plot)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
