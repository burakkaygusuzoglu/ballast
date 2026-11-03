"""Geçmiş veride backtest (komisyon ve kayma dahil).

Kullanım:
    python backtest.py                  # GELİŞTİRME dönemi (data.exam_start öncesi) - varsayılan
    python backtest.py --period exam    # SINAV dönemi. Parametreler kilitlendikten sonra, BİR KEZ.

Her çalıştırma reports/<zaman>_<dönem>/ altına rapor.txt, ozet.json, islemler.csv,
ozsermaye.csv ve grafik.png yazar. Sınav çalıştırmaları reports/sinav_kayitlari.log'a
ayar dosyasının parmak iziyle birlikte kaydedilir (dürüstlük kaydı).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime

import pandas as pd

from bot.backtester import run_backtest
from bot.cli import config_hash, confirm, load_config_or_exit, open_file, setup_console
from bot.config import Config
from bot.data import load_market_rules, load_pair
from bot.logger import close_logging, setup_logging
from bot.metrics import buy_and_hold, compute_metrics, evaluate_gate
from bot.report import format_report, save_report
from bot.signals import build_market_signals, conditions, describe
from bot.strategy import first_ready_ts, signal_funnel
from bot.timeutil import fmt_utc, parse_utc

EXAM_LOG = "sinav_kayitlari.log"


def split_periods(cfg: Config, signals: dict[str, pd.DataFrame]) -> dict[str, tuple[int, int]]:
    """Geliştirme / sınav dönemi sınırları (UTC ms, [başlangıç, bitiş)). Sınav tarihi sabittir."""
    data_start = max(int(df["ts"].iloc[0]) for df in signals.values())
    data_end = min(int(df["ts"].iloc[-1]) for df in signals.values()) + cfg.entry_ms
    ready = [first_ready_ts(df) for df in signals.values()]
    if any(r is None for r in ready):
        raise ValueError("Göstergelerin ısınması için veri yetersiz; daha uzun geçmiş indirin.")
    trade_start = max([data_start, *[int(r) for r in ready if r is not None]])
    exam_start = parse_utc(cfg.data.exam_start)
    periods: dict[str, tuple[int, int]] = {}
    if exam_start > trade_start:
        periods["dev"] = (trade_start, min(exam_start, data_end))
    if data_end > exam_start:
        periods["exam"] = (max(exam_start, trade_start), data_end)
    return periods


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Geçmiş veride backtest (komisyon ve kayma dahil).")
    parser.add_argument("--period", choices=["dev", "exam"], default="dev",
                        help="dev: geliştirme (varsayılan), exam: sınav dönemi")
    parser.add_argument("--yes", action="store_true", help="Sınav onay sorusunu atla")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    setup_logging(cfg.log_dir, cfg.logging.level, "backtest.log", console=False)
    try:
        return _run(cfg, args)
    finally:
        close_logging()


def _run(cfg: Config, args: argparse.Namespace) -> int:
    candles: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for symbol in cfg.symbols:
        try:
            candles[symbol] = load_pair(cfg, symbol)
        except FileNotFoundError as exc:
            print(f"[HATA] Veri bulunamadı: {exc}\nÖnce 2_veri_indir.bat çalıştırın.")
            return 2

    t0 = time.perf_counter()
    print("Göstergeler ve sinyaller hesaplanıyor...")
    print(f"Strateji: {describe(cfg)} | {len(cfg.symbols)} sembol")
    signals = build_market_signals(cfg, {s: e for s, (e, _) in candles.items()},
                                   {s: t for s, (_, t) in candles.items()})
    try:
        periods = split_periods(cfg, signals)
    except ValueError as exc:
        print(f"[HATA] {exc}")
        return 2
    for name, label in (("dev", "Geliştirme"), ("exam", "Sınav")):
        if name in periods:
            a, b = periods[name]
            print(f"  {label:<11}: {fmt_utc(a, False)} -> {fmt_utc(b, False)}")
    if args.period not in periods:
        if args.period == "exam":
            print(f"[HATA] Sınav dönemi için veri yok (data.exam_start: {cfg.data.exam_start}). Önce veriyi güncelleyin.")
        else:
            print("[HATA] Geliştirme dönemi oluşmadı: data.exam_start, veri başlangıcından yeterince sonra olmalı.")
        return 2
    start, end = periods[args.period]

    if args.period == "exam" and not args.yes:
        print("\n" + "!" * 76)
        print(" SINAV VERİSİNİ AÇMAK ÜZERESİNİZ")
        print(" Bu dönem stratejiyi değerlendirmek içindir, ayarlamak için değil. Sonucu")
        print(" gördükten sonra ayar değiştirip tekrar sınava sokmak sınavı geçersiz kılar.")
        print(f" Her sınav çalıştırması {cfg.backtest.reports_dir}/{EXAM_LOG} dosyasına kaydedilir.")
        print("!" * 76)
        if not confirm("Devam etmek için 'evet' yazın:", "evet"):
            print("İptal edildi.")
            return 1

    rules = load_market_rules(cfg.data_dir, cfg.symbols, cfg.market_rules_default)
    print(f"Simülasyon çalışıyor ({args.period})...")
    result = run_backtest(cfg, signals, start, end, cfg.backtest.initial_equity, rules)
    benchmark = buy_and_hold({s: c[0] for s, c in candles.items()}, start, end, cfg.backtest.initial_equity,
                             cfg.costs.taker_fee, cfg.costs.slippage, cfg.entry_ms)
    metrics = compute_metrics(result.equity, result.trades, result.initial_equity, result.total_fees, result.exposure)
    bench_metrics = compute_metrics(benchmark, [], cfg.backtest.initial_equity, 0.0)
    gate = evaluate_gate(metrics, bench_metrics, cfg.gate, args.period)
    funnels = {s: signal_funnel(sig, start, end, conditions(cfg)) for s, sig in signals.items()}

    text = format_report(args.period, cfg.symbols, result, metrics, bench_metrics, gate, funnels)
    print("\n" + text)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    digest = config_hash()
    out_dir = cfg.reports_dir / f"{stamp}_{args.period}"
    chart = save_report(out_dir, args.period, cfg.symbols, result, metrics, bench_metrics, benchmark,
                        gate, text, digest, plot=not args.no_plot)
    if args.period == "exam":
        record = (f"{stamp} UTC | config_sha256={digest[:16]} | net={metrics['net_profit']:+.2f} | "
                  f"PF={metrics['profit_factor']:.2f} | maxDD={metrics['max_drawdown']:.3f} | "
                  f"kapı={'GEÇTİ' if all(c.passed for c in gate) else 'KALDI'}\n")
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.reports_dir / EXAM_LOG, "a", encoding="utf-8") as fh:
            fh.write(record)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.1f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
