"""9. araştırma turu: büyüme-optimal kaldıraç (Kelly), "Büyüme Motoru".

Protokol: docs/ARASTIRMA.md, 22. bölüm (aynı metin: docs/tur9_protokol.md). Izgara, ölçütler ve seçim kuralı
sonuçlar görülmeden önce yazıldı; bu dosya onları aynen uygular. Sonuca bakarak burada bir şey değiştirmek testi
geçersiz kılar.

- Temel: 8. turun T1-25 (L, yalnızca long) ve T2-25 (LS, long/short) hedef ağırlıkları, vadelide (çapraz marjin).
- Kaldıraç çarpanı k ∈ {1; 1,5; 2; 3; 4; 6; 8}: ağırlıkların hepsi k ile çarpılır.
- k*: seçim döneminin (P1 + P2) günlük getirileriyle ve yarıya indirilmiş avantajla, 10 yıllık medyan logaritmik
  büyümeyi en yüksek yapan k. Profiller: Büyüme (yarım Kelly) ve Korumalı (%25 sınırı). Doğrulama: P3.

    python arastirma_tur9.py   # veri: tur8_veri.py (V7_tur8_veri_indir.bat)

Çıktı: reports/<zaman>_arastirma_tur9 altında sonuc.md, ozet.json ve grafik.png.
"""
from __future__ import annotations

import argparse
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

import arastirma_tur8 as T8
from bot import buyume as B
from bot.cli import disable_quick_edit, load_config_or_exit, open_file, setup_console
from bot.config import PROJECT_ROOT
from bot.hedef_agirlik import FUTURES_COSTS, TWResult, run_target_weights, tw_metrics
from bot.timeutil import DAY_MS, fmt_utc, parse_utc

# ---- Protokol sabitleri (docs/ARASTIRMA.md, 22. bölüm) ------------------------------------
PROTOCOL_FILE = PROJECT_ROOT / "docs" / "tur9_protokol.md"
PROTOCOL_SHA256 = "46bcf77905d7d4f18b0f119eef48d8c7e73e880a7684d189b93e65efc62628cc"
GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DIRECTIONS = {"L": ("T1", "T1-25"), "LS": ("T2", "T2-25")}          # yön -> (8. tur türü, 8. tur temel adayı)
ELIGIBILITY = {"L": ("1_kar", "3_al_tut", "4_trend_degeri"),
               "LS": ("1_kar", "3_al_tut", "4_trend_degeri", "5_short_degeri")}
BASE_VOL = 0.25
MMR = 0.02                                                            # bakım marjini (kötümser)
SPANS = (("P1", "2018-09-01", "2022-01-01"), ("P2", "2022-01-01", "2025-01-01"),
         ("S", "2018-09-01", "2025-01-01"), ("P3", "2025-01-01", None), ("TAM", "2018-09-01", None))
INITIAL = 10_000.0
REAL_EQUITY = 50.0
TARGET = 1000 * REAL_EQUITY                                           # 50.000 USDT
MAX_DD = 0.25
MC_PATHS, MC_BLOCK, MC_SEED = 10_000, 20, 20261018
SEL_DAYS, PROJ_DAYS = B.horizon_days(10), B.horizon_days(20)          # 3.653 ve 7.305 gün
HAIRCUT_SHARE = 0.5
CONTRIBUTIONS = (0.0, 50.0, 100.0)
CONTRIB_EVERY = 30
MULTIPLES = (10.0, 100.0, 1000.0)
TARGET_YEARS = (5, 10, 15, 20)
DIRECTION_MARGIN = 0.01                                               # yıllık 1 puan
PROFILES = {"K": "Korumalı (%25 sınırı)", "B": "Büyüme (yarım Kelly)"}


def kname(k: float) -> str:
    return f"{k:g}".replace(".", ",")


def code_of(direction: str, k: float) -> str:
    return f"{direction}-k{kname(k)}"


def parse_code(code: str) -> tuple[str, float]:
    d, k = code.split("-k")
    return d, float(k.replace(",", "."))


# ---------------------------------------------------------------------------
# Koşular
# ---------------------------------------------------------------------------
def spans9(data: T8.Data8) -> dict[str, tuple[int, int]]:
    T8.period_bounds(data)                                           # veri kapsamı kontrolü (8. turla aynı)
    return {name: (parse_utc(a), parse_utc(b) if b else data.end_ts) for name, a, b in SPANS}


def base_targets(data: T8.Data8) -> dict[str, dict[str, np.ndarray]]:
    return {d: T8.targets_for(kind, BASE_VOL, data) for d, (kind, _) in DIRECTIONS.items()}


def scaled(targets: dict[str, np.ndarray], k: float) -> dict[str, np.ndarray]:
    return {c: k * np.asarray(w, dtype="float64") for c, w in targets.items()}


def run9(data: T8.Data8, targets: dict[str, np.ndarray], start: int, end: int, initial: float = INITIAL,
         mode: str = "stop", real_rules: bool = False) -> TWResult:
    return run_target_weights(data.prices, targets, start, end, initial, FUTURES_COSTS, True, T8.BAND,
                              data.funding, data.funding_start, data.futures_rules if real_rules else None,
                              liquidation_mmr=MMR, on_liquidation=mode)


def metrics_of(res: TWResult, initial: float = INITIAL) -> dict:
    m = tw_metrics(res.equity, initial, res.trades, res.fees, res.funding, res.gross_exposure)
    attempted = len(res.trades) + res.skipped_small
    return m | {"likidasyon": [fmt_utc(t, False) for t in res.liquidations],
                "kucuk_islem_atlanan": res.skipped_small,
                "atlanan_oran": res.skipped_small / attempted if attempted else 0.0}


# ---------------------------------------------------------------------------
# Protokol kuralları (saf fonksiyonlar; testlerle doğrulanır)
# ---------------------------------------------------------------------------
def eligibility(s8_checks: dict[str, dict[str, tuple[bool, str]]]) -> dict[str, tuple[bool, str]]:
    out = {}
    for d, (_, cand) in DIRECTIONS.items():
        checks = s8_checks[cand]
        failed = [g for g in ELIGIBILITY[d] if not checks[g][0]]
        detail = "gerekli ölçütlerin hepsi geçti" if not failed else "kalan: " + ", ".join(T8.LABELS[g] for g in failed)
        out[d] = (not failed, f"{cand}: {detail}")
    return out


def pick_k_star(g: dict[float, float]) -> float:
    """g'yi en yüksek yapan ızgara değeri; eşitlikte düşük olan."""
    best_k, best = GRID[0], g[GRID[0]]
    for k in GRID[1:]:
        if g[k] > best:
            best_k, best = k, g[k]
    return best_k


def growth_profile(k_star: float) -> float:
    """Büyüme (yarım Kelly): k*/2'yi aşmayan en büyük ızgara değeri; en az 1."""
    fits = [k for k in GRID if k <= k_star / 2]
    return max(fits) if fits else GRID[0]


def protected_profile(k_growth: float, ok: dict[float, bool]) -> float | None:
    """Korumalı: Büyüme profilini aşmayan, k = 1'den başlayarak koşulları kesintisiz sağlayan en büyük değer."""
    best = None
    for k in GRID:
        if k > k_growth or not ok[k]:
            break
        best = k
    return best


def choose_direction(cands: dict[str, float]) -> str | None:
    """cands: yön -> g (S, yarıya indirilmiş). İkisi de geçtiyse g'si yüksek olan; fark 1 puandan azsa L."""
    if not cands:
        return None
    if len(cands) == 1:
        return next(iter(cands))
    g_l, g_ls = cands["L"], cands["LS"]
    if g_l >= g_ls or abs(g_l - g_ls) < DIRECTION_MARGIN:
        return "L"
    return "LS"


# ---------------------------------------------------------------------------
# Aşama
# ---------------------------------------------------------------------------
@dataclass
class Study9:
    created: str
    bounds: dict[str, tuple[int, int]]
    eligible: dict[str, tuple[bool, str]]
    runs: dict[tuple[str, str], TWResult] = field(default_factory=dict)          # (kod, dönem), tarihsel koşu
    metrics: dict[tuple[str, str], dict] = field(default_factory=dict)
    material: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)    # (kod, S|TAM) -> günlük getiri
    h: dict[str, float] = field(default_factory=dict)                             # yön -> h (S)
    h_tam: dict[str, float] = field(default_factory=dict)                         # yön -> h (TAM, projeksiyon)
    selection: dict[str, dict[float, dict]] = field(default_factory=dict)         # yön -> k -> bootstrap özeti
    k_star: dict[str, float] = field(default_factory=dict)
    k_star_hist: dict[str, float] = field(default_factory=dict)                  # bilgi
    profiles: dict[tuple[str, str], float | None] = field(default_factory=dict)  # (yön, K|B) -> k
    validation: dict[tuple[str, str], dict[str, tuple[bool, str]]] = field(default_factory=dict)
    recommended: dict[str, str | None] = field(default_factory=dict)             # K|B -> kod
    projections: dict[str, dict] = field(default_factory=dict)
    real: dict[str, dict] = field(default_factory=dict)

    def adopted(self, d: str, p: str) -> bool:
        k = self.profiles.get((d, p))
        return (k is not None and k > 1 and self.eligible[d][0]
                and all(ok for ok, _ in self.validation[(d, p)].values()))


def protocol_check() -> str:
    digest = hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise AssertionError(f"Protokol dosyası değişmiş: {digest} != {PROTOCOL_SHA256}")
    return digest


def _pct(x: float) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def _dd(x: float) -> str:
    return f"{x * 100:.1f}%"


def validate(s: Study9, d: str, k: float, profile: str) -> dict[str, tuple[bool, str]]:
    c, c1 = code_of(d, k), code_of(d, 1.0)
    m3, m3_base = s.metrics[(c, "P3")], s.metrics[(c1, "P3")]
    liq = {sp: s.metrics[(c, sp)]["likidasyon"] for sp in ("P1", "P2", "S", "P3")}
    any_liq = any(liq.values())
    checks = {
        "1_p3_kar": (m3["net_profit"] > 0, f"P3 {_pct(m3['total_return'])}"),
        "2_likidasyon": (not any_liq, "yok" if not any_liq else
                         "; ".join(f"{sp} {', '.join(v)}" for sp, v in liq.items() if v)),
        "3_kaldirac_degeri": (m3["cagr"] > m3_base["cagr"],
                              f"P3 CAGR {_pct(m3['cagr'])} / k=1 {_pct(m3_base['cagr'])}"),
    }
    if profile == "K":
        checks["4_p3_dusus"] = (m3["max_drawdown"] <= MAX_DD, f"P3 düşüş {_dd(m3['max_drawdown'])} (sınır %25)")
    return checks


def selection_stats(material: np.ndarray, k: float, h: float, starts: np.ndarray) -> dict:
    row = {}
    for scen, series in (("yari", B.haircut(material, k, h)), ("tarihsel", material)):
        st = B.simulate(series, starts, SEL_DAYS, MC_BLOCK, dd_days=SEL_DAYS)
        row[f"g_{scen}"] = B.median_log_growth(st.final, SEL_DAYS)
        row[f"lik10_{scen}"] = B.within(st.ruined_day, SEL_DAYS)
        row[f"dd50_{scen}"] = float(np.mean(st.max_dd >= 0.5))
    return row


def projection(material: np.ndarray, k: float, h: float, starts: np.ndarray) -> dict:
    d10 = B.horizon_days(10)
    out = {}
    for scen, series in (("tarihsel", material), ("yari", B.haircut(material, k, h))):
        st = B.simulate(series, starts, PROJ_DAYS, MC_BLOCK, levels=MULTIPLES, dd_days=d10)
        years = PROJ_DAYS / B.YEAR_DAYS
        row = {"lik10": B.within(st.ruined_day, d10), "dd50_10": float(np.mean(st.max_dd >= 0.5)),
               "medyan_cagr20": float(np.median(st.final)) ** (1 / years) - 1}
        for m in MULTIPLES:
            row[f"x{m:g}_10"] = B.within(st.first_hit[m], d10)
            row[f"x{m:g}_20"] = B.within(st.first_hit[m], PROJ_DAYS)
        targets = {}
        for c in CONTRIBUTIONS:
            ct = B.simulate(series, starts, PROJ_DAYS, MC_BLOCK, initial=REAL_EQUITY, contribution=c,
                            every=CONTRIB_EVERY, levels=(TARGET,), snapshot_days=(d10, PROJ_DAYS))
            hit = ct.first_hit[TARGET]
            targets[f"{c:g}"] = {"olasilik": {str(y): B.within(hit, B.horizon_days(y)) for y in TARGET_YEARS},
                                 "medyan_yil": B.median_years_to_hit(hit),
                                 "medyan_bakiye_10y": float(np.median(ct.snapshots[d10])),
                                 "medyan_bakiye_20y": float(np.median(ct.snapshots[PROJ_DAYS])),
                                 "toplam_katki_20y": c * (PROJ_DAYS // CONTRIB_EVERY)}
        row["hedef"] = targets
        out[scen] = row
    return out


def run_study9(data: T8.Data8, s8_checks: dict[str, dict[str, tuple[bool, str]]],
               progress: Callable[[str], None] | None = None) -> Study9:
    progress = progress or (lambda _msg: None)
    bounds = spans9(data)
    s = Study9(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), bounds, eligibility(s8_checks))
    bases = base_targets(data)
    for d in DIRECTIONS:
        for k in GRID:
            c = code_of(d, k)
            targets = scaled(bases[d], k)
            for span, (start, end) in bounds.items():
                res = run9(data, targets, start, end)
                s.runs[(c, span)] = res
                s.metrics[(c, span)] = metrics_of(res)
                if span in ("S", "TAM"):
                    mat = res if not res.liquidations else run9(data, targets, start, end, mode="restart")
                    s.material[(c, span)] = mat.returns.to_numpy(dtype="float64")
            m = s.metrics[(c, "TAM")]
            liq = f", likidasyon {m['likidasyon'][0]}" if m["likidasyon"] else ""
            progress(f"  {c:<8} tam dönem: getiri {m['total_return'] * 100:+.1f}%, düşüş {m['max_drawdown'] * 100:.1f}%"
                     f", Sharpe {m['sharpe']:.2f}{liq}")
    # Seçim (S): bütün adaylar aynı blok sırasıyla.
    n_s = len(s.material[(code_of("L", 1.0), "S")])
    starts = B.block_starts(n_s, SEL_DAYS, MC_PATHS, MC_BLOCK, MC_SEED)
    for d in DIRECTIONS:
        s.h[d] = B.haircut_step(s.material[(code_of(d, 1.0), "S")], HAIRCUT_SHARE)
        s.selection[d] = {k: selection_stats(s.material[(code_of(d, k), "S")], k, s.h[d], starts) for k in GRID}
        s.k_star[d] = pick_k_star({k: r["g_yari"] for k, r in s.selection[d].items()})
        s.k_star_hist[d] = pick_k_star({k: r["g_tarihsel"] for k, r in s.selection[d].items()})
        k_b = growth_profile(s.k_star[d])
        ok = {k: (s.metrics[(code_of(d, k), "P1")]["max_drawdown"] <= MAX_DD
                  and s.metrics[(code_of(d, k), "P2")]["max_drawdown"] <= MAX_DD
                  and not s.metrics[(code_of(d, k), "S")]["likidasyon"]) for k in GRID}
        s.profiles[(d, "B")] = k_b
        s.profiles[(d, "K")] = protected_profile(k_b, ok)
        for p in PROFILES:
            k = s.profiles[(d, p)]
            if k is not None and k > 1:
                s.validation[(d, p)] = validate(s, d, k, p)
        progress(f"  {d}: k* = {kname(s.k_star[d])} (tarihsel avantajla {kname(s.k_star_hist[d])}); "
                 f"Büyüme k = {kname(k_b)}, Korumalı k = "
                 f"{kname(s.profiles[(d, 'K')]) if s.profiles[(d, 'K')] is not None else 'yok'}")
    for p in PROFILES:
        cands = {d: s.selection[d][s.profiles[(d, p)]]["g_yari"] for d in DIRECTIONS if s.adopted(d, p)}
        best = choose_direction(cands)
        s.recommended[p] = code_of(best, s.profiles[(best, p)]) if best else None
    # Bilgi: projeksiyon ve 50 USDT (her yönde k = 1 ve geçen profiller).
    codes = [code_of(d, 1.0) for d in DIRECTIONS]
    for d in DIRECTIONS:
        for p in PROFILES:
            c = code_of(d, s.profiles[(d, p)]) if s.adopted(d, p) else None
            if c and c not in codes:
                codes.append(c)
    n_t = len(s.material[(code_of("L", 1.0), "TAM")])
    starts_t = B.block_starts(n_t, PROJ_DAYS, MC_PATHS, MC_BLOCK, MC_SEED)
    s.h_tam = {d: B.haircut_step(s.material[(code_of(d, 1.0), "TAM")], HAIRCUT_SHARE) for d in DIRECTIONS}
    for c in codes:
        d, k = parse_code(c)
        s.projections[c] = projection(s.material[(c, "TAM")], k, s.h_tam[d], starts_t)
        res = run9(data, scaled(bases[d], k), *bounds["TAM"], initial=REAL_EQUITY, real_rules=True)
        s.real[c] = metrics_of(res, REAL_EQUITY)
    return s


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
VLABELS = {"1_p3_kar": "1. P3 kâr", "2_likidasyon": "2. Likidasyon", "3_kaldirac_degeri": "3. Kaldıracın değeri",
           "4_p3_dusus": "4. P3 düşüş"}


def _g(x: float) -> str:
    """Logaritmik büyüme -> yıllık bileşik yüzde."""
    return "batış" if x == -math.inf else f"{(math.exp(x) - 1) * 100:+.1f}%"


def _pr(x: float) -> str:
    return f"{x * 100:.1f}%"


def _years(x: float) -> str:
    return "20 yıldan uzun" if math.isinf(x) else f"{x:.1f} yıl"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def format_study9(s: Study9) -> str:
    out = [f"### 9. tur: büyüme-optimal kaldıraç, BTC ve ETH ({s.created})", "",
           ("8. turun T1-25 (L) ve T2-25 (LS) ağırlıkları × k; vadeli, çapraz marjin; %0,05 komisyon + %0,05 kayma, "
            "gerçek funding; gün içi en kötü fiyatla likidasyon (bakım marjini %2). Her koşu 10.000 USDT ile ayrı "
            "başlar."),
           f"Protokol parmak izi: {PROTOCOL_SHA256[:16]}… (docs/tur9_protokol.md)", ""]
    paths = f"{MC_PATHS:,}".replace(",", ".")
    out += ["**Dönemler:** " + "; ".join(f"{n} {fmt_utc(a, False)} → {fmt_utc(b - DAY_MS, False)}"
                                         for n, (a, b) in s.bounds.items()), ""]
    out += ["**Uygunluk (8. turdan):** " + "; ".join(f"{d} {'UYGUN' if ok else 'UYGUN DEĞİL'} ({detail})"
                                                    for d, (ok, detail) in s.eligible.items()), ""]
    rows = []
    for d in DIRECTIONS:
        for k in GRID:
            c = code_of(d, k)
            cells = [c]
            for span in ("P1", "P2", "S", "P3"):
                m = s.metrics[(c, span)]
                liq = " LİKİDE" if m["likidasyon"] else ""
                cells.append(f"{_pct(m['total_return'])} / {_dd(m['max_drawdown'])} / {m['sharpe']:.2f}{liq}")
            m = s.metrics[(c, "TAM")]
            cells += [_pct(m["cagr"]), f"{m['avg_gross'] * 100:.0f}%", f"{m['funding']:,.0f}",
                      ", ".join(m["likidasyon"]) or "-"]
            rows.append(cells)
    out += ["**Net getiri / en kötü düşüş / Sharpe** (LİKİDE: o dönemde likidasyon; yol orada biter)", ""]
    out += _table(["Kurulum", "P1", "P2", "S", "P3", "TAM CAGR", "TAM ort. brüt", "TAM funding (USDT)",
                   "TAM likidasyon"], rows) + [""]
    out += [(f"**Seçim dönemi (S) bootstrap: {paths} yol × 10 yıl, 20 günlük bloklar** "
             "(g: medyan yıllık bileşik büyüme; 'yarı' = yarıya indirilmiş avantaj)"), ""]
    rows = []
    for d in DIRECTIONS:
        for k in GRID:
            r = s.selection[d][k]
            mark = " ← k*" if k == s.k_star[d] else ""
            rows.append([code_of(d, k) + mark, _g(r["g_yari"]), _g(r["g_tarihsel"]),
                         f"{_pr(r['lik10_yari'])} / {_pr(r['lik10_tarihsel'])}",
                         f"{_pr(r['dd50_yari'])} / {_pr(r['dd50_tarihsel'])}"])
    out += _table(["Kurulum", "g (yarı)", "g (tarihsel)", "10 yılda likidasyon (yarı / tarihsel)",
                   "10 yılda ≥%50 düşüş (yarı / tarihsel)"], rows) + [""]
    out += ["**k\\* (yarıya indirilmiş avantaj):** " + "; ".join(f"{d} = {kname(k)}" for d, k in s.k_star.items())
            + ". Bilgi, tarihsel avantajla: " + "; ".join(f"{d} = {kname(k)}" for d, k in s.k_star_hist.items())
            + ".", ""]
    rows = []
    for d in DIRECTIONS:
        for p, label in PROFILES.items():
            k = s.profiles[(d, p)]
            if k is None:
                rows.append([d, label, "yok", "-", "-"])
                continue
            if k == 1:
                rows.append([d, label, "1", "kaldıraç önerilmez (8. turun kararı geçerli)", "-"])
                continue
            checks = s.validation[(d, p)]
            detail = " · ".join(f"{VLABELS[g]}: {'GEÇTİ' if ok else 'KALDI'} ({txt})" for g, (ok, txt) in checks.items())
            if not s.eligible[d][0]:
                result = "yön uygun değil (bilgi)"
            else:
                result = "GEÇTİ" if s.adopted(d, p) else "KALDI"
            rows.append([d, label, kname(k), detail, result])
    out += ["**Profiller ve doğrulama (P3)**", ""]
    out += _table(["Yön", "Profil", "k", "Doğrulama", "Sonuç"], rows) + [""]
    for p, label in PROFILES.items():
        out.append(f"**Karar, {label}:** {s.recommended[p] or 'kaldıraç önerilmez'}.")
    out += [("Son karar Burak'ındır. Büyüme profili %25'ten büyük düşüş demek olabilir; ancak açıkça kabul "
             "edilirse kullanılır."), ""]
    out += [(f"**Bilgi: projeksiyon (TAM'ın günlük getirileri, {paths} yol × 20 yıl). Tahmin değildir; geçmişin "
             "aynı karışımla tekrarlanacağını varsayar.**"), ""]
    rows = []
    for c, pr in s.projections.items():
        for scen, label in (("tarihsel", "tarihsel"), ("yari", "yarı avantaj")):
            r = pr[scen]
            rows.append([c, label, _pct(r["medyan_cagr20"]),
                         " · ".join(f"{m:g}×: {_pr(r[f'x{m:g}_10'])} / {_pr(r[f'x{m:g}_20'])}" for m in MULTIPLES),
                         _pr(r["lik10"]), _pr(r["dd50_10"])])
    out += _table(["Kurulum", "Senaryo", "Medyan yıllık büyüme (20 yıl)", "Ulaşma olasılığı (10 yıl / 20 yıl)",
                   "10 yılda likidasyon", "10 yılda ≥%50 düşüş"], rows) + [""]
    rows = []
    for c, pr in s.projections.items():
        for scen, label in (("tarihsel", "tarihsel"), ("yari", "yarı avantaj")):
            for amount, t in pr[scen]["hedef"].items():
                rows.append([c, label, f"{float(amount):.0f} USDT", " / ".join(_pr(t["olasilik"][str(y)])
                                                                         for y in TARGET_YEARS),
                             _years(t["medyan_yil"]), f"{t['medyan_bakiye_10y']:,.0f} / {t['medyan_bakiye_20y']:,.0f}",
                             f"{t['toplam_katki_20y']:,.0f}"])
    out += ["**Bilgi: 50 USDT'den 50.000 USDT'ye (1.000 kat), 30 günde bir katkıyla**", ""]
    out += _table(["Kurulum", "Senaryo", "Katkı", "Ulaşma olasılığı (5 / 10 / 15 / 20 yıl)", "Medyan süre",
                   "Medyan bakiye (10 / 20 yıl, USDT)", "20 yılda toplam katkı (USDT)"], rows) + [""]
    rows = []
    for c, r in s.real.items():
        rows.append([c, f"{r['final']:.2f}", _pct(r["total_return"]), _dd(r["max_drawdown"]),
                     f"{r['kucuk_islem_atlanan']} ({_pr(r['atlanan_oran'])})", ", ".join(r["likidasyon"]) or "-"])
    out += ["**Bilgi: 50 USDT ile gerçek vadeli kurallar (TAM)**", ""]
    out += _table(["Kurulum", "Son bakiye (USDT)", "Getiri", "En kötü düşüş", "Küçük olduğu için atlanan işlem",
                   "Likidasyon"], rows) + [""]
    return "\n".join(out)


def _clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def study_summary(s: Study9) -> dict:
    return _clean({
        "olusturuldu": s.created, "protokol_sha256": PROTOCOL_SHA256,
        "donemler": {n: [fmt_utc(a, False), fmt_utc(b - DAY_MS, False)] for n, (a, b) in s.bounds.items()},
        "uygunluk": {d: [ok, detail] for d, (ok, detail) in s.eligible.items()},
        "h": s.h, "h_tam": s.h_tam, "k_yildiz": s.k_star, "k_yildiz_tarihsel": s.k_star_hist,
        "secim": {d: {kname(k): r for k, r in rows.items()} for d, rows in s.selection.items()},
        "profiller": {f"{d}|{p}": k for (d, p), k in s.profiles.items()},
        "dogrulama": {f"{d}|{p}": {g: [ok, txt] for g, (ok, txt) in v.items()} for (d, p), v in s.validation.items()},
        "oneri": s.recommended,
        "metrikler": {f"{c}|{span}": m for (c, span), m in s.metrics.items()},
        "projeksiyon": s.projections, "gercek_56usdt": s.real})


def save_report(out_dir: Path, s: Study9, text: str, plot: bool = True) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "ozet.json").write_text(json.dumps(study_summary(s), indent=2, ensure_ascii=False), encoding="utf-8")
    if not plot:
        return None
    chart = out_dir / "grafik.png"
    plot_study9(chart, s)
    return chart


def plot_study9(path: Path, s: Study9) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"L": "#55A868", "LS": "#C44E52"}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 13), gridspec_kw={"height_ratios": [1.2, 0.8, 1.4]})
    x = np.arange(len(GRID))
    for d, color in colors.items():
        for scen, style in (("yari", "-"), ("tarihsel", ":")):
            y = [s.selection[d][k][f"g_{scen}"] for k in GRID]
            y = [(math.exp(v) - 1) * 100 if v > -math.inf else np.nan for v in y]
            label = f"{d}, {'yarıya indirilmiş avantaj' if scen == 'yari' else 'tarihsel avantaj'}"
            ax1.plot(x, y, color=color, linestyle=style, marker="o", label=label)
        ax1.axvline(GRID.index(s.k_star[d]), color=color, linewidth=0.8, alpha=0.5)
        ax2.plot(x, [s.selection[d][k]["lik10_yari"] * 100 for k in GRID], color=color, marker="o",
                 label=f"{d}: 10 yılda likidasyon")
        ax2.plot(x, [s.selection[d][k]["dd50_yari"] * 100 for k in GRID], color=color, marker="s", linestyle="--",
                 label=f"{d}: 10 yılda ≥%50 düşüş")
    for ax in (ax1, ax2):
        ax.set_xticks(x, [kname(k) for k in GRID])
        ax.set_xlabel("Kaldıraç çarpanı k (8. turun temel riskinin katı)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_title("Seçim dönemi: medyan yıllık bileşik büyüme (10 yıllık bootstrap, %) — dikey çizgi: k*")
    ax2.set_title("Seçim dönemi, yarıya indirilmiş avantaj: olasılıklar (%)")
    for c in s.projections:
        d, k = parse_code(c)
        eq = s.runs[(c, "TAM")].equity
        eq = eq[eq > 0]
        t = pd.to_datetime(eq.index.to_numpy(), unit="ms", utc=True)
        ax3.plot(t, eq.to_numpy() / INITIAL * 100, color=colors[d], linewidth=1.6 if k > 1 else 1.0,
                 linestyle="-" if k > 1 else ":", label=c)
    ax3.set_yscale("log")
    ax3.set_title("Tam dönem özsermaye (başlangıç 100, logaritmik): k = 1 ve geçen profiller")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()
    parser = argparse.ArgumentParser(description="9. araştırma turu: büyüme-optimal kaldıraç (Kelly).")
    parser.add_argument("--no-plot", action="store_true", help="Grafik çizme")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args(argv)
    cfg = load_config_or_exit()
    t0 = time.perf_counter()
    T8.protocol_check()
    protocol_check()
    try:
        data = T8.load_data8(cfg)
    except FileNotFoundError as exc:
        print(f"[HATA] Veri yok: {exc}. Önce V7_tur8_veri_indir.bat çalıştırılmalı.")
        return 2
    print("8. tur ölçütleri hesaplanıyor (uygunluk için)...")
    s8 = T8.run_study(data)
    print("9. tur çalıştırılıyor...")
    s = run_study9(data, s8.checks, progress=print)
    text = format_study9(s)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.reports_dir / f"{stamp}_arastirma_tur9"
    chart = save_report(out_dir, s, text, plot=not args.no_plot)
    print("\n" + text)
    print(f"\nRapor klasörü: {out_dir}")
    print(f"Süre: {time.perf_counter() - t0:.0f} sn")
    if chart is not None and cfg.backtest.open_chart and not args.no_open:
        open_file(chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
