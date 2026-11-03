"""Büyüme-optimal risk analizi (9. tur): ortak blok bootstrap, yarıya indirilmiş avantaj ve hedefe ulaşma.

- Bütün adaylar aynı blok sırasını kullanır (ortak rastgele sayılar). Böylece adaylar arasındaki fark örnekleme
  şansından değil, stratejiden gelir.
- Getirisi -1 olan gün (likidasyon), örneklendiği yolu sıfırlar. Katkı varsa sonraki katkılar sıfırdan devam eder.
- "Ulaşma": o süre içinde herhangi bir gün o seviyeye çıkmak (ilk ulaşma günü).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

YEAR_DAYS = 365.25


def horizon_days(years: float) -> int:
    """Yıl -> gün (yukarı yuvarlanır): 10 yıl = 3.653 gün, 20 yıl = 7.305 gün."""
    return math.ceil(years * YEAR_DAYS)


def block_starts(n: int, days: int, paths: int, block: int, seed: int) -> np.ndarray:
    """Her yolun blok başlangıçları (yol x blok sayısı). Bloklar malzemenin içinde kalır."""
    if n < block:
        raise ValueError("Bootstrap malzemesi bir bloktan kısa.")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n - block + 1, size=(paths, math.ceil(days / block)))


def haircut_step(base_returns: np.ndarray, share: float = 0.5) -> float:
    """h = share x (k = 1 serisinin ortalama günlük getirisi). Ortalama negatifse h = 0."""
    return share * max(float(np.mean(np.asarray(base_returns, dtype="float64"))), 0.0)


def haircut(returns: np.ndarray, k: float, h: float) -> np.ndarray:
    """Avantajı azaltılmış getiriler: r - k x h, en fazla -1'e kadar. Likidasyon günleri -1 kalır."""
    r = np.asarray(returns, dtype="float64")
    out = np.maximum(r - k * h, -1.0)
    out[r <= -1.0] = -1.0
    return out


@dataclass
class PathStats:
    final: np.ndarray                                    # yol sonu servet
    ruined_day: np.ndarray                               # ilk likidasyon günü (0'dan sayılır; yoksa -1)
    max_dd: np.ndarray                                   # ilk dd_days gün içindeki en büyük tepeden düşüş
    first_hit: dict[float, np.ndarray] = field(default_factory=dict)   # seviye -> ilk ulaşma günü (yoksa -1)
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)     # gün sayısı -> o günün sonundaki servet


def simulate(returns: np.ndarray, starts: np.ndarray, days: int, block: int, initial: float = 1.0,
             contribution: float = 0.0, every: int = 30, levels: tuple[float, ...] = (), dd_days: int = 0,
             snapshot_days: tuple[int, ...] = ()) -> PathStats:
    """Yolların servetini gün gün yürütür. Katkı, her `every` günün sonunda eklenir (ilk katkı `every`. günün sonunda).
    Düşüş yalnızca katkısız yollar için anlamlıdır (katkı tepeyi bozar)."""
    r = np.asarray(returns, dtype="float64")
    paths = starts.shape[0]
    if starts.shape[1] * block < days:
        raise ValueError("Blok sayısı istenen gün sayısına yetmiyor.")
    wealth = np.full(paths, float(initial))
    peak = wealth.copy()
    max_dd = np.zeros(paths)
    ruined = np.full(paths, -1, dtype=np.int64)
    hits = {float(lv): np.full(paths, -1, dtype=np.int64) for lv in levels}
    snaps: dict[int, np.ndarray] = {}
    snap_set = set(snapshot_days)
    for d in range(days):
        x = r[starts[:, d // block] + d % block]
        wealth *= 1.0 + x
        dead = x <= -1.0
        if dead.any():
            wealth[dead] = 0.0
            ruined[dead & (ruined < 0)] = d
        if contribution and (d + 1) % every == 0:
            wealth += contribution
        if d < dd_days:
            np.maximum(peak, wealth, out=peak)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = np.where(peak > 0, 1.0 - wealth / peak, 0.0)
            np.maximum(max_dd, dd, out=max_dd)
        for lv, arr in hits.items():
            arr[(arr < 0) & (wealth >= lv)] = d
        if d + 1 in snap_set:
            snaps[d + 1] = wealth.copy()
    return PathStats(wealth, ruined, max_dd, hits, snaps)


def median_log_growth(final: np.ndarray, days: int, initial: float = 1.0) -> float:
    """Yolların yıllıklandırılmış logaritmik büyümesinin medyanı. Batan yol (servet 0) -inf sayılır."""
    with np.errstate(divide="ignore"):
        g = np.log(np.asarray(final, dtype="float64") / initial) / (days / YEAR_DAYS)
    return float(np.median(g))


def within(first_day: np.ndarray, days: int) -> float:
    """İlk olayı ilk `days` gün içinde olan yolların oranı."""
    return float(np.mean((first_day >= 0) & (first_day < days)))


def median_years_to_hit(first_day: np.ndarray) -> float:
    """İlk ulaşma süresinin (yıl) medyanı; ulaşamayan yol sonsuz sayılır. Yolların yarısından azı ulaşırsa inf
    (alt medyan: tam yarısı ulaşırsa sonlu)."""
    t = np.where(first_day >= 0, (first_day + 1) / YEAR_DAYS, np.inf)
    return float(np.percentile(t, 50, method="lower"))
