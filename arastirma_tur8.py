"""8. araştırma turu: çok vadeli trend topluluğu (long ve long/short), BTC ve ETH, günlük.

Protokol: docs/ARASTIRMA.md, 21. bölüm (aynı metin: docs/tur8_protokol.md). Adaylar, ölçütler ve seçim kuralı
sonuçlar görülmeden önce yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey değiştirmek testi
geçersiz kılar.

- T1-25 / T1-15: yalnızca long, spot, hedef volatilite %25 / %15
- T2-25 / T2-15: long/short, vadeli (brüt en fazla %200), hedef volatilite %25 / %15
- Referans: al-tut (yarı BTC, yarı ETH), VY (volatiliteye göre al-tut), R1 sabit %0,5 (bilgi)

    python arastirma_tur8.py   # veri: tur8_veri.py (V7_tur8_veri_indir.bat)

Çıktı: reports/<zaman>_arastirma_tur8 altında sonuc.md, ozet.json, islemler.csv ve grafik.png.
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

from bot.cli import disable_quick_edit, load_config_or_exit, open_file, setup_console
from bot.config import PROJECT_ROOT, Config
from bot.data import candle_path, load_candles, load_market_rules
from bot.hedef_agirlik import (
    FUTURES_COSTS,
    SPOT_COSTS,
    SymbolRules,
    TWResult,
    buy_and_hold,
    run_target_weights,
    tw_metrics,
)
from bot.timeutil import DAY_MS, fmt_utc, parse_utc
from bot.trend_toplulugu import coin_exposure, portfolio_weights

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 21. bölüm) ------------------------------------
PROTOCOL_FILE = PROJECT_ROOT / "docs" / "tur8_protokol.md"
PROTOCOL_SHA256 = "aeade125c5f5d847d5febad9c9dc0c3b1367e7b56c4dfdad8bb8f0563b0de7d1"
COINS = ("BTC/USDT", "ETH/USDT")
PERIODS = (("P1", "2018-09-01", "2022-01-01"), ("P2", "2022-01-01", "2025-01-01"), ("P3", "2025-01-01", None))
FULL_START = "2018-09-01"
INITIAL = 10_000.0
REAL_EQUITY = 50.0
MAX_DD = 0.25
BAND = 0.20
SPOT_GROSS, FUTURES_GROSS = 1.0, 2.0
CANDIDATES = {"T1-25": ("T1", 0.25), "T1-15": ("T1", 0.15), "T2-25": ("T2", 0.25), "T2-15": ("T2", 0.15)}
MC_BLOCK_DAYS, MC_PATHS, MC_SEED, MC_DAYS = 20, 10_000, 20261017, 365
FUNDING_DIR = "tur8"


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
@dataclass
class Data8:
    prices: dict[str, pd.DataFrame]                         # ortak zaman çizgisinde günlük mumlar
    funding: dict[str, tuple[np.ndarray, np.ndarray]]
    funding_start: dict[str, int]
    spot_rules: dict[str, SymbolRules]
    futures_rules: dict[str, SymbolRules]

    @property
    def ts(self) -> np.ndarray:
        return self.prices[COINS[0]]["ts"].to_numpy(dtype="int64")

    @property
    def end_ts(self) -> int:
        return int(self.ts[-1]) + DAY_MS


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Bütün coinlerin ortak günleri (iç birleşim)."""
    common = None
    for df in frames.values():
        s = set(df["ts"].astype("int64"))
        common = s if common is None else common & s
    keep = np.array(sorted(common), dtype="int64")
    out = {}
    for name, df in frames.items():
        d = df[df["ts"].isin(keep)].sort_values("ts").reset_index(drop=True)
        out[name] = d
    return out


def load_funding(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, float_precision="round_trip")
    times = (df["ts"].to_numpy(dtype="int64") // 60_000) * 60_000          # dakikaya aşağı yuvarla
    return times, df["rate"].to_numpy(dtype="float64")


def load_data8(cfg: Config) -> Data8:
    frames = {}
    for coin in COINS:
        path = candle_path(cfg.data_dir, cfg.exchange.id, coin, "1d")
        if not path.exists():
            raise FileNotFoundError(path.name)
        frames[coin] = load_candles(path)
    funding, funding_start = {}, {}
    for coin in COINS:
        path = cfg.data_dir / FUNDING_DIR / f"funding_{coin.replace('/', '')}.csv"
        if not path.exists():
            raise FileNotFoundError(path.name)
        times, rates = load_funding(path)
        funding[coin] = (times, rates)
        funding_start[coin] = int(times[0])
    spot = load_market_rules(cfg.data_dir, list(COINS), cfg.market_rules_default)
    spot_rules = {c: SymbolRules(qty_step=r.qty_step, min_notional=r.min_notional) for c, r in spot.items()}
    raw = json.loads((cfg.data_dir / FUNDING_DIR / "vadeli_kurallar.json").read_text(encoding="utf-8"))
    futures_rules = {c: SymbolRules(qty_step=float(raw[c]["miktar_adimi"] or 0),
                                    min_notional=float(raw[c]["en_kucuk_tutar"] or 0)) for c in COINS}
    return Data8(align(frames), funding, funding_start, spot_rules, futures_rules)


# ---------------------------------------------------------------------------
# Hedef ağırlıklar
# ---------------------------------------------------------------------------
def targets_for(kind: str, vol: float, data: Data8) -> dict[str, np.ndarray]:
    """T1: long-only spot; T2: long/short vadeli (short yalnızca sözleşmenin ilk funding kaydından sonra);
    VY: her zaman long, volatiliteye göre (spot)."""
    exposures = {}
    for coin in COINS:
        close = data.prices[coin]["close"].to_numpy(dtype="float64")
        if kind == "T1":
            exposures[coin] = coin_exposure(close, vol)
        elif kind == "VY":
            exposures[coin] = coin_exposure(close, vol, always_long=True)
        elif kind == "T2":
            long_only = coin_exposure(close, vol)
            long_short = coin_exposure(close, vol, allow_short=True)
            exposures[coin] = np.where(data.ts >= data.funding_start[coin], long_short, long_only)
        else:
            raise ValueError(kind)
    return portfolio_weights(exposures, FUTURES_GROSS if kind == "T2" else SPOT_GROSS)


def period_bounds(data: Data8) -> dict[str, tuple[int, int]]:
    bounds = {name: (parse_utc(a), parse_utc(b) if b else data.end_ts) for name, a, b in PERIODS}
    bounds["TAM"] = (parse_utc(FULL_START), data.end_ts)
    if data.end_ts <= parse_utc("2025-02-01"):
        raise ValueError("Veri P3 dönemini kapsamıyor; önce tur8_veri.py çalıştırılmalı.")
    if int(data.ts[0]) + 361 * DAY_MS > parse_utc(FULL_START):
        raise ValueError("Isınma için 2018-09-01'den önce en az 361 gün veri gerekli.")
    return bounds


def run(kind: str, vol: float, data: Data8, start: int, end: int, initial: float = INITIAL,
        real_rules: bool = False, targets: dict[str, np.ndarray] | None = None) -> TWResult:
    targets = targets if targets is not None else targets_for(kind, vol, data)
    futures = kind == "T2"
    rules = (data.futures_rules if futures else data.spot_rules) if real_rules else None
    return run_target_weights(data.prices, targets, start, end, initial, FUTURES_COSTS if futures else SPOT_COSTS,
                              futures, BAND, data.funding if futures else None,
                              data.funding_start if futures else None, rules)


def metrics_of(res: TWResult, initial: float = INITIAL) -> dict:
    return tw_metrics(res.equity, initial, res.trades, res.fees, res.funding, res.gross_exposure) | {
        "kucuk_islem_atlanan": res.skipped_small}


# ---------------------------------------------------------------------------
# Aşama: bütün koşular, ölçütler ve seçim
# ---------------------------------------------------------------------------
@dataclass
class Study8:
    created: str
    bounds: dict[str, tuple[int, int]]
    runs: dict[tuple[str, str], TWResult]                   # (kurulum, dönem)
    metrics: dict[tuple[str, str], dict]
    real: dict[str, dict]                                   # kurulum -> 50 USDT tam dönem metrikleri
    bh: dict[str, dict]                                     # dönem -> al-tut metrikleri
    bh_curves: dict[str, pd.Series]
    r1: dict[str, dict] = field(default_factory=dict)       # dönem -> R1 metrikleri (bilgi)
    checks: dict[str, dict[str, tuple[bool, str]]] = field(default_factory=dict)
    chosen_small: str | None = None
    chosen_large: str | None = None
    monte_carlo: dict[str, dict] = field(default_factory=dict)

    def passed(self, code: str) -> bool:
        return all(ok for ok, _ in self.checks[code].values())


def protocol_check() -> str:
    digest = hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise AssertionError(f"Protokol dosyası değişmiş: {digest} != {PROTOCOL_SHA256}")
    return digest


def evaluate(s: Study8, code: str) -> dict[str, tuple[bool, str]]:
    kind, vol = CANDIDATES[code]
    m = {p: s.metrics[(code, p)] for p in ("P1", "P2", "P3", "TAM")}
    pct = lambda x: f"{x * 100:+.1f}%"
    dd = lambda x: f"{x * 100:.1f}%"
    checks = {
        "1_kar": (all(m[p]["net_profit"] > 0 for p in ("P1", "P2", "P3")),
                  ", ".join(f"{p} {pct(m[p]['total_return'])}" for p in ("P1", "P2", "P3"))),
        "2_dusus": (all(m[p]["max_drawdown"] <= MAX_DD for p in ("P1", "P2", "P3")),
                    ", ".join(f"{p} {dd(m[p]['max_drawdown'])}" for p in ("P1", "P2", "P3")) + " (sınır %25)"),
    }
    bh = s.bh["TAM"]
    checks["3_al_tut"] = (m["TAM"]["sharpe"] > bh["sharpe"] and m["TAM"]["max_drawdown"] < 0.5 * bh["max_drawdown"],
                          (f"Sharpe {m['TAM']['sharpe']:.2f} / al-tut {bh['sharpe']:.2f}; düşüş {dd(m['TAM']['max_drawdown'])}"
                          f" / al-tut yarısı {dd(0.5 * bh['max_drawdown'])}"))
    vy = s.metrics[(f"VY-{round(vol * 100)}", "TAM")]
    checks["4_trend_degeri"] = (m["TAM"]["sharpe"] > vy["sharpe"],
                                f"Sharpe {m['TAM']['sharpe']:.2f} / VY {vy['sharpe']:.2f}")
    if kind == "T2":
        t1 = f"T1-{round(vol * 100)}"
        wins = sum(m[p]["total_return"] > s.metrics[(t1, p)]["total_return"] for p in ("P1", "P2", "P3"))
        checks["5_short_degeri"] = (m["TAM"]["sharpe"] > s.metrics[(t1, "TAM")]["sharpe"] and wins >= 2,
                                    (f"Sharpe {m['TAM']['sharpe']:.2f} / {t1} {s.metrics[(t1, 'TAM')]['sharpe']:.2f}; "
                                    f"{wins}/3 dönemde daha yüksek getiri"))
    else:
        r = s.real[code]
        checks["6_56usdt"] = (r["net_profit"] > 0 and r["max_drawdown"] <= MAX_DD,
                              f"50 → {r['final']:.2f} USDT ({pct(r['total_return'])}), düşüş {dd(r['max_drawdown'])}")
    return checks


def run_study(data: Data8, r1: dict[str, dict] | None = None,
              progress: Callable[[str], None] | None = None) -> Study8:
    progress = progress or (lambda _msg: None)
    bounds = period_bounds(data)
    s = Study8(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), bounds, {}, {}, {}, {}, {}, r1 or {})
    setups = dict(CANDIDATES) | {"VY-25": ("VY", 0.25), "VY-15": ("VY", 0.15)}
    for code, (kind, vol) in setups.items():
        targets = targets_for(kind, vol, data)
        for name, (start, end) in bounds.items():
            res = run(kind, vol, data, start, end, targets=targets)
            s.runs[(code, name)] = res
            s.metrics[(code, name)] = metrics_of(res)
        if kind != "VY":
            res = run(kind, vol, data, *bounds["TAM"], initial=REAL_EQUITY, real_rules=True, targets=targets)
            s.real[code] = metrics_of(res, REAL_EQUITY)
        m = s.metrics[(code, "TAM")]
        progress(f"  {code:<6} tam dönem: getiri {m['total_return'] * 100:+.1f}%, CAGR {m['cagr'] * 100:+.1f}%, "
                 f"düşüş {m['max_drawdown'] * 100:.1f}%, Sharpe {m['sharpe']:.2f}")
    for name, (start, end) in bounds.items():
        curve = buy_and_hold(data.prices, start, end, INITIAL)
        s.bh_curves[name] = curve
        s.bh[name] = tw_metrics(curve, INITIAL)
    for code in CANDIDATES:
        s.checks[code] = evaluate(s, code)
    small = [c for c in CANDIDATES if CANDIDATES[c][0] == "T1" and s.passed(c)]
    large = [c for c in CANDIDATES if CANDIDATES[c][0] == "T2" and s.passed(c)]
    s.chosen_small = max(small, key=lambda c: CANDIDATES[c][1]) if small else None
    s.chosen_large = max(large, key=lambda c: CANDIDATES[c][1]) if large else None
    for code in {c for c in (s.chosen_small, s.chosen_large) if c}:
        s.monte_carlo[code] = block_bootstrap(s.runs[(code, "TAM")].equity, INITIAL)
    return s


def block_bootstrap(equity: pd.Series, initial: float, days: int = MC_DAYS, paths: int = MC_PATHS,
                    block: int = MC_BLOCK_DAYS, seed: int = MC_SEED) -> dict:
    curve = np.concatenate([[initial], equity.to_numpy()])
    rets = np.diff(curve) / curve[:-1]
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(days / block)
    starts = rng.integers(0, len(rets) - block + 1, size=(paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(paths, -1)[:, :days]
    curves = np.cumprod(1 + rets[idx], axis=1)
    finals = curves[:, -1]
    peaks = np.maximum.accumulate(np.concatenate([np.ones((paths, 1)), curves], axis=1), axis=1)[:, 1:]
    dd = np.max(1 - curves / peaks, axis=1)
    return {"medyan": float(np.median(finals)), "p10": float(np.percentile(finals, 10)),
            "p90": float(np.percentile(finals, 90)), "dusus25": float(np.mean(dd >= MAX_DD)),
            "zarar": float(np.mean(finals < 1.0)), "medyan_dusus": float(np.median(dd))}


# ---------------------------------------------------------------------------
# R1 (bilgi): mevcut 4 saatlik motorla, verisi olan dönemlerde
# ---------------------------------------------------------------------------
def r1_reference(cfg: Config, data_end: int) -> dict[str, dict]:
    import arastirma as A
    import arastirma_tur3 as T3
    import arastirma_tur5 as T5
    raw = T5.load_raw(cfg)
    end = min(T5.last_full_day(raw), data_end)
    market = A.prepare_market(raw, end)
    t3 = {c.code: c for c in T3.tur3_candidates()}
    cfg_c = A.apply_overrides(cfg, t3["R1"].overrides)
    sig = T3.build_signals(t3["R1"], cfg_c, market)
    out = {}
    for name, a, b in (("P2*", "2022-07-20", "2025-01-01"), ("P3*", "2025-01-01", None)):
        start, stop = parse_utc(a), (parse_utc(b) if b else end)
        run_ = A.run_period(cfg_c, sig, market, "4h", start, stop, {})
        m = run_.metrics
        out[name] = {"donem": f"{fmt_utc(start, False)} → {fmt_utc(stop - DAY_MS, False)}",
                     "total_return": m["total_return"], "max_drawdown": m["max_drawdown"], "sharpe": m["sharpe"]}
    return out


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
LABELS = {"1_kar": "1. Kâr", "2_dusus": "2. Düşüş", "3_al_tut": "3. Al-tut'a göre", "4_trend_degeri": "4. Trendin değeri",
          "5_short_degeri": "5. Short'un değeri", "6_56usdt": "6. 50 USDT"}


def _p(x: float) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def _d(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_study(s: Study8) -> str:
    out = [f"### 8. tur: çok vadeli trend topluluğu, BTC ve ETH ({s.created})", "",
           ("Her dönem 10.000 USDT ile ayrı başlar; spot %0,1 + %0,05, vadeli %0,05 + %0,05 ve gerçek funding. Karar "
            "günlük kapanışta, işlem sonraki açılışta; işlem bandı %20."),
           f"Protokol parmak izi: {PROTOCOL_SHA256[:16]}… (docs/tur8_protokol.md)", ""]
    names = ["P1", "P2", "P3", "TAM"]
    spans = {n: f"{fmt_utc(a, False)} → {fmt_utc(b - DAY_MS, False)}" for n, (a, b) in s.bounds.items()}
    out += ["**Dönemler:** " + "; ".join(f"{n} {spans[n]}" for n in names), ""]
    rows = []
    for code in list(CANDIDATES) + ["VY-25", "VY-15"]:
        cells = [code]
        for n in names:
            m = s.metrics[(code, n)]
            cells.append(f"{_p(m['total_return'])} / {_d(m['max_drawdown'])} / {m['sharpe']:.2f}")
        rows.append(cells)
    rows.append(["Al-tut (yarı BTC, yarı ETH)"] + [f"{_p(s.bh[n]['total_return'])} / {_d(s.bh[n]['max_drawdown'])} / "
                                                  f"{s.bh[n]['sharpe']:.2f}" for n in names])
    out += ["**Net getiri / en kötü düşüş / Sharpe**", ""]
    out += _table(["Kurulum"] + names, rows) + [""]
    rows = []
    for code in CANDIDATES:
        m = s.metrics[(code, "TAM")]
        r = s.real[code]
        rows.append([code, _p(m["cagr"]), f"{m['avg_gross'] * 100:.0f}%", str(m["n_trades"]), f"{m['fees']:,.0f}",
                     f"{m['funding']:,.0f}", f"{r['final']:.2f} ({_p(r['total_return'])}, düşüş {_d(r['max_drawdown'])})",
                     str(r["kucuk_islem_atlanan"])])
    out += ["**Tam dönem ayrıntıları (10.000 USDT) ve 50 USDT ile gerçek kurallar**", ""]
    out += _table(["Kurulum", "CAGR", "Ort. brüt pozisyon", "İşlem", "Komisyon (USDT)", "Funding (USDT)",
                   "50 USDT → son bakiye", "Küçük olduğu için atlanan işlem"], rows) + [""]
    if s.r1:
        out += ["**Bilgi: R1 sabit %0,5 (12 coin, 4 saat), verisi olan dönemler**", ""]
        out += _table(["Dönem", "Tarihler", "Net getiri", "En kötü düşüş", "Sharpe"],
                      [[k, v["donem"], _p(v["total_return"]), _d(v["max_drawdown"]), f"{v['sharpe']:.2f}"]
                       for k, v in s.r1.items()]) + [""]
    out += ["**Ölçütler**", ""]
    rows = []
    for code in CANDIDATES:
        cells = [f"{LABELS[k]}: {'GEÇTİ' if ok else 'KALDI'} ({detail})" for k, (ok, detail) in s.checks[code].items()]
        rows.append([code, " · ".join(cells), "GEÇTİ" if s.passed(code) else "KALDI"])
    out += _table(["Aday", "Ölçütler", "Sonuç"], rows) + [""]
    out += [f"**Karar (50 USDT, spot):** {s.chosen_small or 'hiçbir T1 adayı geçmedi'}.",
            f"**Karar (daha büyük sermaye, vadeli):** {s.chosen_large or 'hiçbir T2 adayı geçmedi'}.", ""]
    for code, mc in s.monte_carlo.items():
        out += [f"**Bilgi: Monte Carlo, 1 yıl ({code}; tam dönem günlük getirileri, 20 günlük bloklar, 10.000 yol)**", ""]
        out += _table(["Medyan", "%10 dilim", "%90 dilim", "Medyan düşüş", "Düşüş ≥ %25", "Yılı zararla bitirme",
                       "50 USDT ile medyan kâr"],
                      [[f"{mc['medyan']:.2f}x", f"{mc['p10']:.2f}x", f"{mc['p90']:.2f}x", _d(mc["medyan_dusus"]),
                        _d(mc["dusus25"]), _d(mc["zarar"]), f"{REAL_EQUITY * (mc['medyan'] - 1):+.1f}"]]) + [""]
    return "\n".join(out)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def _clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def study_summary(s: Study8) -> dict:
    return _clean({"olusturuldu": s.created, "protokol_sha256": PROTOCOL_SHA256,
                   "donemler": {n: [fmt_utc(a, False), fmt_utc(b - DAY_MS, False)] for n, (a, b) in s.bounds.items()},
                   "secim_56usdt": s.chosen_small, "secim_buyuk_sermaye": s.chosen_large,
                   "olcutler": {c: {k: [ok, d] for k, (ok, d) in v.items()} for c, v in s.checks.items()},
                   "metrikler": {f"{c}|{n}": m for (c, n), m in s.metrics.items()},
                   "gercek_56usdt": s.real, "al_tut": s.bh, "r1": s.r1, "monte_carlo": s.monte_carlo})


def save_report(out_dir: Path, s: Study8, text: str, plot: bool = True) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(study_summary(s), indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "islemler.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kurulum", "donem", "gun", "coin", "miktar", "fiyat", "komisyon"])
        for (code, name), res in s.runs.items():
            if name != "TAM" or code.startswith("VY"):
                continue
            for t in res.trades:
                w.writerow([code, name, fmt_utc(t["ts"], False), t["coin"], f"{t['qty']:.8g}", f"{t['price']:.8g}",
                            f"{t['fee']:.6f}"])
    if not plot:
        return None
    chart = out_dir / "grafik.png"
    plot_study(chart, s)
    return chart


def plot_study(path: Path, s: Study8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"T1-25": ("#55A868", "-"), "T1-15": ("#55A868", ":"), "T2-25": ("#C44E52", "-"),
              "T2-15": ("#C44E52", ":"), "VY-25": ("#8172B3", "-")}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for code, (color, style) in styles.items():
        eq = s.runs[(code, "TAM")].equity
        x = pd.to_datetime(eq.index.to_numpy(), unit="ms", utc=True)
        ax1.plot(x, eq.to_numpy() / INITIAL * 100, color=color, linestyle=style, linewidth=1.4, label=code)
        curve = eq.to_numpy()
        ax2.plot(x, -(1 - curve / np.maximum.accumulate(np.maximum(curve, 1e-9))) * 100, color=color, linestyle=style,
                 linewidth=1.0)
    bh = s.bh_curves["TAM"]
    x = pd.to_datetime(bh.index.to_numpy(), unit="ms", utc=True)
    ax1.plot(x, bh.to_numpy() / INITIAL * 100, color="gray", linestyle="--", linewidth=1.0, label="Al-tut BTC+ETH")
    ax1.set_yscale("log")
    ax1.set_title("Tam dönem özsermaye (başlangıç 100, logaritmik)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.axhline(-MAX_DD * 100, color="black", linewidth=0.7, linestyle=":")
    ax2.set_title("Tepeden düşüş (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()
    parser = argparse.ArgumentParser(description="8. araştırma turu: çok vadeli trend topluluğu.")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)
    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    protocol_check()
    try:
        data = load_data8(cfg)
    except FileNotFoundError as exc:
        print(f"[HATA] Veri yok: {exc}. Önce V7_tur8_veri_indir.bat çalıştırılmalı.")
        return 2
    try:
        r1 = r1_reference(cfg, data.end_ts)
    except FileNotFoundError:
        r1 = {}
    s = run_study(data, r1, progress=print)
    text = format_study(s)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.reports_dir / f"{stamp}_arastirma_tur8"
    chart = save_report(out_dir, s, text, plot=not args.no_plot)
    print("\n" + text)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
