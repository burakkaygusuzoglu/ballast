"""Backtest raporu: Türkçe konsol özeti, dosyalar (TXT/JSON/CSV) ve grafik."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pencere açmadan dosyaya çizim
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from bot.backtester import BacktestResult  # noqa: E402
from bot.metrics import GateCheck  # noqa: E402
from bot.timeutil import fmt_utc  # noqa: E402

PERIOD_LABELS = {"dev": "GELİŞTİRME DÖNEMİ", "exam": "SINAV DÖNEMİ"}
# Not: Python'da "İ".lower() iki karakter üretir; Türkçe başlıkları ayrı yazıyoruz.
PERIOD_TITLES = {"dev": "Geliştirme dönemi", "exam": "Sınav dönemi"}
_BLUE, _ORANGE = "#4C72B0", "#DD8452"
LINE = "=" * 76


def _pct(x: float | None, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _num(x: float) -> str:
    return "sonsuz" if math.isinf(x) else f"{x:.2f}"


def _verdict(period: str, passed: bool) -> str:
    if period == "dev":
        return ("GELİŞTİRME KAPISI GEÇİLDİ - parametreleri kilitleyip sınava hazırlanın" if passed
                else "GELİŞTİRME KAPISI GEÇİLEMEDİ - strateji bu haliyle sınava hazır değil")
    return ("SINAV KAPISI GEÇİLDİ - paper trading aşamasına geçilebilir" if passed
            else "SINAV KAPISI GEÇİLEMEDİ - canlıya geçilmez")


def format_report(period: str, symbols: list[str], result: BacktestResult, m: dict, bench: dict,
                  gate: list[GateCheck], funnels: dict[str, list[tuple[str, int]]]) -> str:
    out = [
        LINE,
        f" BACKTEST RAPORU - {PERIOD_LABELS.get(period, period)}",
        LINE,
        f" Dönem      : {fmt_utc(result.start_ts, False)} - {fmt_utc(result.end_ts, False)} ({m['days']:.0f} gün)",
        f" Semboller  : {', '.join(symbols)}   (komisyon ve kayma dahil)",
        f" Özsermaye  : {m['initial_equity']:,.2f} -> {m['final_equity']:,.2f} USDT",
        "",
        " SONUÇ                            Strateji      Al-tut",
        f"   Net getiri                    {_pct(m['total_return']):>9}   {_pct(bench['total_return']):>9}",
        f"   Yıllık getiri (CAGR)          {_pct(m['cagr']):>9}   {_pct(bench['cagr']):>9}",
        f"   Maks. düşüş                   {_pct(-m['max_drawdown']):>9}   {_pct(-bench['max_drawdown']):>9}",
        f"   Getiri / düşüş oranı          {_num(m['return_to_drawdown']):>9}   {_num(bench['return_to_drawdown']):>9}",
        f"   Sharpe (günlük, yıllık)       {m['sharpe']:>9.2f}   {bench['sharpe']:>9.2f}",
        "",
        " İŞLEMLER",
        f"   İşlem sayısı                  {m['n_trades']} (haftada {m['trades_per_week']:.1f})",
        f"   İsabet oranı                  {_pct(m['win_rate'], signed=False)}",
        f"   Kâr faktörü                   {_num(m['profit_factor'])}",
        f"   Ortalama R                    {m['avg_r']:+.2f}R",
        f"   Ort. kazanç / kayıp           {m['avg_win']:+,.2f} / {m['avg_loss']:+,.2f} USDT",
        f"   En iyi / en kötü işlem        {m['best_trade']:+,.2f} / {m['worst_trade']:+,.2f} USDT",
        f"   Pozisyonda geçen zaman        {_pct(result.exposure, signed=False)}",
        f"   Dönem sonunda açık            {len(result.open_positions)} pozisyon",
        "",
        " MALİYETLER (kapanan işlemler)",
        f"   Maliyet öncesi kâr/zarar      {m['pre_cost_pnl']:+,.2f} USDT",
        f"   Komisyon                      {-m['fees_closed']:+,.2f} USDT",
        f"   Kayma (tahmini)               {-m['slippage_closed']:+,.2f} USDT",
        f"   İşlem başına ort. maliyet     {m['avg_cost_r']:.2f}R",
    ]
    share = m["cost_share"]
    if share is not None:
        out.append(f"   -> Maliyetlerin maliyet öncesi kâra oranı: %{share * 100:.0f}")
    else:
        out.append("   -> Maliyet öncesinde de kâr yok: sorun komisyon değil, sinyal kalitesi")

    if funnels:
        out += ["", " SİNYAL HUNİSİ (her filtreden sonra kalan mum sayısı)"]
        header = "   " + f"{'Filtre':<28}" + "".join(f"{s:>12}" for s in funnels)
        out.append(header)
        rows = list(zip(*funnels.values(), strict=True))
        for row in rows:
            label = row[0][0]
            out.append("   " + f"{label:<28}" + "".join(f"{count:>12,}" for _, count in row))

    out += ["", " RİSK YÖNETİCİSİ"]
    out.append(f"   Sinyal {result.signals_seen} | emir {result.orders_submitted} | "
               f"dolmadan iptal {result.entries_cancelled}")
    for reason, count in result.skipped.most_common():
        out.append(f"   Reddedildi: {reason:<36} {count}")
    if result.halted_at:
        out.append(f"   [!] ŞALTER: {result.halt_reason}")

    out += ["", f" KARAR KAPISI ({PERIOD_TITLES.get(period, period).lower()})"]
    for check in gate:
        out.append(f"   [{'GEÇTİ' if check.passed else 'KALDI'}] {check.name:<44} {check.detail}")
    out.append(f"   SONUÇ: {_verdict(period, all(c.passed for c in gate))}")
    out.append(LINE)
    return "\n".join(out)


def _to_dt(index: pd.Index) -> pd.DatetimeIndex:
    return pd.to_datetime(index.to_numpy(dtype="int64"), unit="ms", utc=True)


def _thin(series: pd.Series, max_points: int = 6000) -> pd.Series:
    step = max(1, len(series) // max_points)
    thinned = series.iloc[::step]
    if len(series) and thinned.index[-1] != series.index[-1]:
        thinned = pd.concat([thinned, series.iloc[-1:]])
    return thinned


def plot_equity(path: Path, equity: pd.Series, benchmark: pd.Series, title: str, subtitle: str) -> None:
    strat = _thin(equity / equity.iloc[0] * 100)
    bench = _thin(benchmark / benchmark.iloc[0] * 100) if len(benchmark) else benchmark
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(_to_dt(strat.index), strat.to_numpy(), color=_BLUE, linewidth=1.6, label="Strateji")
    if len(bench):
        ax1.plot(_to_dt(bench.index), bench.to_numpy(), color=_ORANGE, linewidth=1.2, linestyle="--",
                 label="Al-tut (semboller eşit ağırlık)")
    ax1.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
    values = np.concatenate([strat.to_numpy(), bench.to_numpy()]) if len(bench) else strat.to_numpy()
    if values.min() > 0 and values.max() / values.min() > 4:
        # Eğriler çok farklı büyüklükteyse logaritmik ölçek ikisini de okunur tutar.
        ax1.set_yscale("log")
        ax1.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
        ax1.yaxis.set_minor_formatter(NullFormatter())
        ax1.set_ylabel("Özsermaye (başlangıç = 100, log ölçek)")
    else:
        ax1.set_ylabel("Özsermaye (başlangıç = 100)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.set_title(f"{title}\n{subtitle}", fontsize=11, loc="left")

    dd = (strat / strat.cummax() - 1) * 100
    ax2.fill_between(_to_dt(dd.index), dd.to_numpy(), 0, color=_BLUE, alpha=0.35, label="Strateji")
    if len(bench):
        bdd = (bench / bench.cummax() - 1) * 100
        ax2.plot(_to_dt(bdd.index), bdd.to_numpy(), color=_ORANGE, linewidth=1.0, linestyle="--", label="Al-tut")
    ax2.set_ylabel("Düşüş (%)")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and (math.isinf(v) or math.isnan(v)) else v) for k, v in d.items()}


def save_report(out_dir: Path, period: str, symbols: list[str], result: BacktestResult, metrics: dict,
                bench_metrics: dict, benchmark: pd.Series, gate: list[GateCheck], text: str,
                config_hash: str, plot: bool = True) -> Path | None:
    """Rapor dosyalarını yazar; çizildiyse grafiğin yolunu döndürür."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rapor.txt").write_text(text + "\n", encoding="utf-8")
    trades = pd.DataFrame([asdict(t) for t in result.trades])
    if not trades.empty:
        for column in ("entry_ts", "exit_ts"):
            trades[column.replace("_ts", "_utc")] = pd.to_datetime(trades[column], unit="ms", utc=True)
    trades.to_csv(out_dir / "islemler.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"close_ts": result.equity.index, "equity": result.equity.to_numpy()}).to_csv(
        out_dir / "ozsermaye.csv", index=False)

    summary = {
        "period": period,
        "symbols": symbols,
        "config_sha256": config_hash,
        "start": fmt_utc(result.start_ts),
        "end": fmt_utc(result.end_ts),
        "strategy": _clean(metrics),
        "buy_and_hold": _clean(bench_metrics),
        "gate": [asdict(c) for c in gate],
        "gate_passed": all(c.passed for c in gate),
        "halted": result.halt_reason or None,
    }
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if plot and len(result.equity) > 1:
        title = (f"Strateji {_pct(metrics['total_return'])} | Al-tut {_pct(bench_metrics['total_return'])} | "
                 f"Maks. düşüş %{metrics['max_drawdown'] * 100:.1f} vs %{bench_metrics['max_drawdown'] * 100:.1f}")
        subtitle = (f"{PERIOD_TITLES.get(period, period)} · {fmt_utc(result.start_ts, False)} - "
                    f"{fmt_utc(result.end_ts, False)} · {', '.join(symbols)} · komisyon ve kayma dahil")
        chart = out_dir / "grafik.png"
        plot_equity(chart, result.equity, benchmark, title, subtitle)
        return chart
    return None
