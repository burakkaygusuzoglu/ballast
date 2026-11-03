"""İzleme paneli: bütün botların sağlığı, performansı ve ileriye dönük test ölçütleri tek raporda.

Yalnızca OKUR. Hiçbir botun dosyasına yazmaz, hiçbir botu durdurmaz ya da başlatmaz:
- Canlı proje (varsayılan: bu klasörün yanındaki kripto-bot): gerçek, testnet ve paper botların durum
  veritabanları ve günlükleri. Veritabanları salt okunur modda açılır.
- v2 (bu klasör): gölge botlar R1, R3 ve durdurulan R3A.

Rapor reports/izleme/ altına yazılır: <zaman>_izleme.html (tarayıcıda açılır) ve son_durum.json.

Kullanım:
    python izleme.py                  # rapor (internet gerekmez)
    python izleme.py --esdeger        # ayrıca: paper/gölge botlar aynı dönemin backtest'iyle birebir aynı mı?
    python izleme.py --canli-klasor C:\\yol\\canli-bot
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot.backtester import run_backtest
from bot.cli import disable_quick_edit, open_file, setup_console
from bot.config import PROJECT_ROOT, Config, ConfigError, MarketRules, load_config
from bot.data import candles_from_rows, drop_unclosed, load_market_rules
from bot.metrics import compute_metrics
from bot.portfolio import Trade
from bot.signals import build_market_signals, uses_trend
from bot.storage import read_trades
from bot.timeutil import DAY_MS, fmt_local, fmt_utc, now_ms, parse_utc

# ---- İleriye dönük test (docs/ARASTIRMA.md, 11. bölüm) ---------------------------------------
FORWARD_START = parse_utc("2026-10-09")    # gölge botların başlangıcı: 2026-10-09 00:00 UTC sonrası mumlar
EARLIEST_WEEKS = 12                        # değerlendirme en erken (≈ 2027-01-01)
DEADLINE_WEEKS = 26                        # ölçütler bu sürede sağlanmazsa aday bırakılır
MIN_TRADES = 20
MIN_PF = 1.2
WEEK_MS = 7 * DAY_MS
GRACE_MS = 15 * 60_000                     # mum kapandıktan sonra botun işlemesi için beklenen süre
WARN_DAYS = 7                              # günlükte son kaç günün uyarıları gösterilir
REPORT_DIR = Path("reports") / "izleme"


@dataclass(frozen=True)
class BotSpec:
    key: str
    name: str
    role: str
    project: str                    # "canli" (C:\yol\canli-bot) | "v2" (bu klasör)
    config: str
    state: str
    log: str
    stopped_at: int | None = None   # bilinçli olarak durdurulduğu an (yalnızca kayıt gösterilir)
    criteria: str | None = None     # "R1" | "R3": 11. bölüm ölçüt takibi
    parity: bool = False            # paper/gölge: aynı dönemin backtest'iyle karşılaştırılabilir
    reference: bool = False         # ölçütlerdeki karşılaştırma tabanı (C1 paper)


BOTS: tuple[BotSpec, ...] = (
    BotSpec("gercek", "C1 gerçek hesap", "Gerçek para", "canli", "config/config.yaml", "data/canli_state.db",
            "logs/canli.log"),
    BotSpec("testnet", "C1 testnet", "Testnet: sahte para, gerçek emir akışı", "canli", "config/config.yaml",
            "data/testnet_state.db", "logs/testnet.log"),
    BotSpec("paper", "C1 paper", "Paper: karşılaştırma tabanı", "canli", "config/config.yaml", "data/paper_state.db",
            "logs/paper.log", parity=True, reference=True),
    BotSpec("golge_r1", "R1 gölge", "Gölge test: C1 + BTC filtresi", "v2", "config/golge_r1.yaml",
            "data/golge_r1_state.db", "logs/golge_r1.log", criteria="R1", parity=True),
    BotSpec("golge_r3", "R3 gölge", "Gölge test: momentum rotasyonu", "v2", "config/golge_r3.yaml",
            "data/golge_r3_state.db", "logs/golge_r3.log", criteria="R3", parity=True),
    BotSpec("golge_r3a", "R3A", "Durduruldu (5. tur sınavında kaldı)", "v2", "config/golge_r3a.yaml",
            "data/golge_r3a_state.db", "logs/golge_r3a.log", stopped_at=parse_utc("2026-10-09T03:45"), parity=True),
)


# ---------------------------------------------------------------------------
# Okuma (salt okunur)
# ---------------------------------------------------------------------------
@dataclass
class StateView:
    snapshot: dict[str, Any]
    trades: list[Trade]
    equity: list[tuple[int, float]]            # (mum kapanışı, özsermaye)

    @property
    def initial(self) -> float:
        return float(self.snapshot["portfolio"]["initial_cash"])

    @property
    def last_ts(self) -> int | None:
        value = self.snapshot.get("last_ts")
        return None if value is None else int(value)

    @property
    def halted(self) -> bool:
        return bool(self.snapshot.get("risk", {}).get("halted"))

    @property
    def halt_reason(self) -> str:
        return str(self.snapshot.get("risk", {}).get("halt_reason") or "")

    @property
    def positions(self) -> list[str]:
        return [p["symbol"] for p in self.snapshot["portfolio"].get("positions", [])]

    @property
    def current_equity(self) -> float:
        return self.equity[-1][1] if self.equity else self.initial


def read_state(path: Path) -> StateView | None:
    """Botun SQLite kaydını SALT OKUNUR açar (bot çalışırken de güvenlidir)."""
    if not path.exists():
        return None
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=10)) as conn:
        row = conn.execute("SELECT value FROM kv WHERE key = 'snapshot'").fetchone()
        if row is None:
            return None
        trades = read_trades(conn)
        equity = [(int(t), float(e)) for t, e in conn.execute("SELECT ts, equity FROM equity ORDER BY ts")]
    return StateView(json.loads(row[0]), trades, equity)


_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})Z \| (\w+)\s*\| (.*)$")


def scan_log(path: Path, now: int, days: int = WARN_DAYS) -> tuple[list[str], int | None]:
    """(son günlerin uyarı/hata satırları, en fazla 5; günlüğün son satırının zamanı)."""
    if not path.exists():
        return [], None
    warnings, last = [], None
    since = now - days * DAY_MS
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LOG_LINE.match(line)
        if not match:
            continue
        ts = parse_utc(match.group(1).replace(" ", "T"))
        last = ts
        if match.group(2) in ("WARNING", "ERROR", "CRITICAL") and ts >= since:
            warnings.append(f"{match.group(1)[5:16]} {match.group(3).strip()}")
    return warnings[-5:], last


# ---------------------------------------------------------------------------
# Sağlık
# ---------------------------------------------------------------------------
@dataclass
class Health:
    status: str              # ZAMANINDA | GECİKMİŞ | DURMUŞ | DURDURULDU | HÂLÂ ÇALIŞIYOR | KAYIT YOK | HATA
    level: str               # ok | warn | bad | off
    last_close: int | None   # son işlenen mumun kapanışı (henüz karar yoksa başlangıç noktası)
    lag_bars: int
    detail: str
    warnings: list[str] = field(default_factory=list)
    next_decision: int | None = None


def expected_last_open(now: int, tf_ms: int, grace_ms: int = GRACE_MS) -> int:
    """Şu ana kadar işlenmiş olması gereken son mumun açılış zamanı."""
    boundary = now // tf_ms * tf_ms
    if now - boundary < grace_ms:
        boundary -= tf_ms
    return boundary - tf_ms


def health(spec: BotSpec, view: StateView | None, tf_ms: int, now: int, warnings: list[str]) -> Health:
    stopped = spec.stopped_at is not None
    if view is None or view.last_ts is None:
        level = "off" if stopped else "bad"
        return Health("KAYIT YOK", level, None, 0, "durum dosyası yok ya da bot hiç başlamadı", warnings)
    last_close = view.last_ts + tf_ms
    lag = int(max(0, (expected_last_open(now, tf_ms) - view.last_ts) // tf_ms))
    if stopped:
        if last_close > spec.stopped_at:
            return Health("HÂLÂ ÇALIŞIYOR", "warn", last_close, lag,
                          "durdurulmuştu ama yeni mum işlemiş: penceresini kapatın", warnings)
        return Health("DURDURULDU", "off", last_close, lag, "bilinçli olarak durduruldu", warnings)
    upcoming = last_close + tf_ms
    first = "" if view.equity else f"henüz ilk kararını vermedi; ilk karar {fmt_local(upcoming)}"
    if lag == 0:
        return Health("ZAMANINDA", "ok", last_close, 0, first or "son kararı zamanında verdi", warnings, upcoming)
    if lag == 1:
        return Health("GECİKMİŞ", "warn", last_close, 1, "1 karar kaçmış olabilir: pencere açık mı?", warnings)
    return Health("DURMUŞ", "bad", last_close, lag, f"{lag} karar kaçmış: bot çalışmıyor", warnings)


# ---------------------------------------------------------------------------
# Performans
# ---------------------------------------------------------------------------
def window_metrics(view: StateView | None, start: int) -> dict | None:
    """[start, son] penceresinde getiri, düşüş, işlemler. Bot henüz mum işlemediyse None."""
    if view is None:
        return None
    points = [(t, e) for t, e in view.equity if t > start]
    if not points:
        return None
    before = [e for t, e in view.equity if t <= start]
    base = before[-1] if before else view.initial
    series = pd.Series([base] + [e for _, e in points],
                       index=pd.Index([start] + [t for t, _ in points], dtype="int64"))
    trades = [t for t in view.trades if t.exit_ts >= start]
    m = compute_metrics(series, trades, base, 0.0)
    m["start"] = start
    return m


def start_of(view: StateView | None, tf_ms: int) -> int | None:
    """Botun ilk işlediği mumun açılış zamanı."""
    if view is None or not view.equity:
        return None
    return view.equity[0][0] - tf_ms


# ---------------------------------------------------------------------------
# İleriye dönük test ölçütleri (11. bölüm)
# ---------------------------------------------------------------------------
@dataclass
class Criteria:
    code: str
    weeks: float
    checks: list[tuple[str, bool | None, str]]
    verdict: str
    level: str


def _rd(m: dict | None) -> float | None:
    return None if m is None else float(m["return_to_drawdown"])


def _fmt_rd(x: float | None) -> str:
    if x is None:
        return "-"
    return "sonsuz" if math.isinf(x) else f"{x:+.2f}"


def forward_criteria(code: str, view: StateView | None, reference: StateView | None, now: int) -> Criteria:
    m = window_metrics(view, FORWARD_START)
    ref = window_metrics(reference, FORWARD_START)
    weeks = max(0.0, (now - FORWARD_START) / WEEK_MS)
    n = m["n_trades"] if m else 0
    net = m["net_profit"] if m else 0.0
    pf = m["profit_factor"] if m and n else None
    rd, ref_rd = _rd(m), _rd(ref)
    checks: list[tuple[str, bool | None, str]] = [
        (f"En az {MIN_TRADES} kapanmış işlem", n >= MIN_TRADES, f"{n}/{MIN_TRADES}"),
        ("Net kâr > 0", net > 0, f"{net:+,.2f} USDT"),
        (f"Kâr faktörü ≥ {MIN_PF:g}".replace(".", ","), pf is not None and pf >= MIN_PF,
         "-" if pf is None else ("sonsuz" if math.isinf(pf) else f"{pf:.2f}")),
        ("Getiri/düşüş C1 paper'dan yüksek", rd is not None and ref_rd is not None and rd > ref_rd,
         f"{_fmt_rd(rd)} / C1 paper {_fmt_rd(ref_rd)}"),
        ("Şaltere takılmamış", view is not None and not view.halted,
         (view.halt_reason or "evet") if view is not None else "kayıt yok"),
    ]
    if code == "R3":
        checks.append(("Gerçek sermaye en az 250 USDT", None, "R3 gerçek paraya geçerse ayrıca gerekir"))
    passed = all(ok for _, ok, _ in checks if ok is not None)
    earliest = fmt_utc(FORWARD_START + EARLIEST_WEEKS * WEEK_MS, False)
    deadline = fmt_utc(FORWARD_START + DEADLINE_WEEKS * WEEK_MS, False)
    if weeks < EARLIEST_WEEKS:
        verdict, level = f"Erken: {weeks:.1f}/{EARLIEST_WEEKS} hafta. Değerlendirme en erken {earliest}.", "info"
    elif passed:
        verdict, level = "Ölçütler sağlandı: gerçek paraya geçiş önerilebilir (Burak'ın onayıyla).", "ok"
    elif weeks > DEADLINE_WEEKS:
        verdict, level = "26 hafta doldu, ölçütler sağlanmadı: aday bırakılır (11. bölüm).", "bad"
    else:
        verdict, level = f"Henüz sağlanmadı ({weeks:.1f} hafta; son tarih {deadline}).", "warn"
    return Criteria(code, weeks, checks, verdict, level)


# ---------------------------------------------------------------------------
# Eşdeğerlik: paper/gölge bot = aynı dönemin backtest'i mi?
# ---------------------------------------------------------------------------
@dataclass
class Parity:
    status: str              # AYNI | FARKLI | BEKLİYOR | YAPILAMADI
    level: str
    detail: str
    n_bot: int = 0
    n_backtest: int = 0


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def compare_trades(bot: list[Trade], backtest: list[Trade]) -> list[str]:
    if len(bot) != len(backtest):
        return [f"işlem sayısı farklı: bot {len(bot)}, backtest {len(backtest)}"]
    diffs = []
    for i, (a, b) in enumerate(zip(bot, backtest, strict=True), start=1):
        same_keys = (a.symbol, a.entry_ts, a.exit_ts, a.exit_reason) == (b.symbol, b.entry_ts, b.exit_ts, b.exit_reason)
        same_numbers = all(_close(getattr(a, f), getattr(b, f)) for f in ("qty", "entry_price", "exit_price", "net_pnl"))
        if not (same_keys and same_numbers):
            diffs.append(f"{i}. işlem: bot {a.symbol} {fmt_utc(a.entry_ts)} → {fmt_utc(a.exit_ts)} {a.net_pnl:+.4f}; "
                         f"backtest {b.symbol} {fmt_utc(b.entry_ts)} → {fmt_utc(b.exit_ts)} {b.net_pnl:+.4f}")
    return diffs


def parity_check(cfg: Config, view: StateView | None, source, rules: dict[str, MarketRules]) -> Parity:
    if view is None or not view.equity or view.last_ts is None:
        return Parity("BEKLİYOR", "info", "henüz mum işlenmedi")
    tf = cfg.entry_ms
    first_open = view.equity[0][0] - tf
    end = view.last_ts + tf
    history = cfg.paper.history_bars

    def closed(symbol: str, timeframe: str, tf_ms: int) -> pd.DataFrame:
        # Bot o anda yalnızca kapanmış mumları görmüştü: bitişten sonra kapanan mum atılır.
        rows = source.fetch_ohlcv_range(symbol, timeframe, first_open - history * tf_ms, end)
        return drop_unclosed(candles_from_rows(rows), tf_ms, end)

    entry = {s: closed(s, cfg.timeframes.entry, tf) for s in cfg.symbols}
    trend = {s: closed(s, cfg.timeframes.trend, cfg.trend_ms) for s in cfg.symbols} if uses_trend(cfg) else None
    result = run_backtest(cfg, build_market_signals(cfg, entry, trend), first_open, end, view.initial, rules)
    diffs = compare_trades(view.trades, result.trades)
    open_bot, open_bt = sorted(view.positions), sorted(p.symbol for p in result.open_positions)
    if open_bot != open_bt:
        diffs.append(f"açık pozisyonlar farklı: bot {open_bot or '-'}, backtest {open_bt or '-'}")
    final_bt = float(result.equity.iloc[-1]) if len(result.equity) else view.initial
    if not math.isclose(view.current_equity, final_bt, rel_tol=1e-6, abs_tol=1e-6):
        diffs.append(f"son özsermaye farklı: bot {view.current_equity:,.4f}, backtest {final_bt:,.4f}")
    window = f"{fmt_utc(first_open + tf)} → {fmt_utc(end)}"
    if diffs:
        return Parity("FARKLI", "bad", f"{window}: " + " | ".join(diffs[:3]), len(view.trades), len(result.trades))
    return Parity("AYNI", "ok", f"{window}: {len(view.trades)} işlem, son özsermaye birebir aynı",
                  len(view.trades), len(result.trades))


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------
@dataclass
class BotReport:
    spec: BotSpec
    cfg: Config | None
    view: StateView | None
    health: Health
    full: dict | None            # başlangıçtan beri
    forward: dict | None         # 2026-10-09'dan beri (ileriye dönük test penceresi)
    error: str = ""
    parity: Parity | None = None


@dataclass
class Panel:
    created: int
    roots: dict[str, Path]
    bots: list[BotReport]
    criteria: list[Criteria]


def project_root(spec: BotSpec, roots: dict[str, Path]) -> Path:
    return roots[spec.project]


def collect(roots: dict[str, Path], now: int, specs: tuple[BotSpec, ...] = BOTS) -> Panel:
    bots = []
    for spec in specs:
        root = project_root(spec, roots)
        error, cfg, view = "", None, None
        try:
            cfg = load_config(root / spec.config)
        except ConfigError as exc:
            error = f"ayar dosyası okunamadı: {exc}"
        try:
            view = read_state(root / spec.state)
        except (sqlite3.Error, json.JSONDecodeError, KeyError, TypeError) as exc:
            error = f"durum dosyası okunamadı: {exc}"
        warnings, _ = scan_log(root / spec.log, now)
        tf = cfg.entry_ms if cfg is not None else DAY_MS
        h = health(spec, view, tf, now, warnings)
        if error:
            h = Health("HATA", "bad", h.last_close, h.lag_bars, error, warnings)
        start = start_of(view, tf)
        full = window_metrics(view, start) if start is not None else None
        bots.append(BotReport(spec, cfg, view, h, full, window_metrics(view, FORWARD_START), error))
    reference = next((b.view for b in bots if b.spec.reference), None)
    criteria = [forward_criteria(b.spec.criteria, b.view, reference, now) for b in bots if b.spec.criteria]
    return Panel(now, roots, bots, criteria)


def run_parity(panel: Panel, source_factory) -> None:
    """--esdeger: paper ve gölge botlar için backtest karşılaştırması (internet gerekir)."""
    import ccxt

    for b in panel.bots:
        if not b.spec.parity or b.cfg is None:
            continue
        try:
            rules = load_market_rules(project_root(b.spec, panel.roots) / "data", b.cfg.symbols,
                                      b.cfg.market_rules_default)
            b.parity = parity_check(b.cfg, b.view, source_factory(b.cfg), rules)
        except ccxt.BaseError as exc:
            b.parity = Parity("YAPILAMADI", "warn", f"borsaya bağlanılamadı: {exc}")
        except (ValueError, KeyError) as exc:
            b.parity = Parity("YAPILAMADI", "warn", f"veri hatası: {exc}")


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
def _local(ms: int) -> str:
    return fmt_local(ms).split(" yerel")[0]


def _pct(x: float | None, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:+.2f}%" if signed else f"{x * 100:.2f}%"


def _pf(m: dict | None) -> str:
    if m is None or not m["n_trades"]:
        return "-"
    return "sonsuz" if math.isinf(m["profit_factor"]) else f"{m['profit_factor']:.2f}"


def summary_json(panel: Panel) -> dict:
    def metrics(m: dict | None) -> dict | None:
        if m is None:
            return None
        keys = ("initial_equity", "final_equity", "net_profit", "total_return", "max_drawdown", "return_to_drawdown",
                "n_trades", "win_rate", "profit_factor", "start")
        return {k: (None if isinstance(m[k], float) and not math.isfinite(m[k]) else m[k]) for k in keys}

    return {
        "olusturma_utc": fmt_utc(panel.created),
        "ileriye_donuk_baslangic": fmt_utc(FORWARD_START),
        "botlar": [{
            "anahtar": b.spec.key, "ad": b.spec.name, "rol": b.spec.role,
            "saglik": {"durum": b.health.status, "seviye": b.health.level, "detay": b.health.detail,
                       "son_karar_utc": fmt_utc(b.health.last_close) if b.health.last_close else None,
                       "gecikme_mum": b.health.lag_bars, "uyarilar": b.health.warnings},
            "ozsermaye": b.view.current_equity if b.view else None,
            "acik_pozisyonlar": b.view.positions if b.view else [],
            "salter": b.view.halted if b.view else None,
            "baslangictan_beri": metrics(b.full),
            "ileriye_donuk": metrics(b.forward),
            "esdeger": None if b.parity is None else {"durum": b.parity.status, "detay": b.parity.detail},
        } for b in panel.bots],
        "olcutler": [{"aday": c.code, "hafta": round(c.weeks, 2), "karar": c.verdict, "seviye": c.level,
                      "kontroller": [{"olcut": label, "gecti": ok, "deger": detail} for label, ok, detail in c.checks]}
                     for c in panel.criteria],
    }


def format_console(panel: Panel) -> str:
    out = [f"İZLEME PANELİ  {fmt_utc(panel.created)} | {fmt_local(panel.created)}", ""]
    out.append(f"{'Bot':<18}{'Durum':<16}{'Son mum (yerel)':<20}{'Özsermaye':>12}{'Getiri':>10}{'İşlem':>7}  Açık")
    for b in panel.bots:
        last = _local(b.health.last_close) if b.health.last_close else "-"
        eq = f"{b.view.current_equity:,.2f}" if b.view else "-"
        ret = _pct(b.full["total_return"]) if b.full else "-"
        n = str(len(b.view.trades)) if b.view else "-"
        positions = ", ".join(p.split("/")[0] for p in b.view.positions) if b.view and b.view.positions else "-"
        out.append(f"{b.spec.name:<18}{b.health.status:<16}{last:<20}{eq:>12}{ret:>10}{n:>7}  {positions}")
    out.append("")
    for c in panel.criteria:
        done = sum(1 for _, ok, _ in c.checks if ok)
        total = sum(1 for _, ok, _ in c.checks if ok is not None)
        out.append(f"{c.code}: {c.verdict} (şu an {done}/{total} ölçüt)")
    for b in panel.bots:
        if b.parity is not None:
            out.append(f"Eşdeğerlik {b.spec.name}: {b.parity.status} ({b.parity.detail})")
    problems = [b for b in panel.bots if b.health.level in ("bad", "warn")]
    if problems:
        out.append("")
        out.append("[!] DİKKAT: " + ", ".join(f"{b.spec.name} {b.health.status} ({b.health.detail})" for b in problems))
    return "\n".join(out)


_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5f6b7a;--line:#e3e7ed;--ok:#1e7f4f;--okbg:#e5f4ec;
--warn:#8a5a00;--warnbg:#fff3d6;--bad:#a32020;--badbg:#fde6e6;--off:#5f6b7a;--offbg:#eceff3;--info:#1f5fa8;
--infobg:#e6f0fb}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--ink:#e6e9ee;--muted:#9aa5b4;--line:#2c323b;
--ok:#5fd39a;--okbg:#173527;--warn:#f0c062;--warnbg:#3a2e12;--bad:#ff8a8a;--badbg:#3d1a1a;--off:#9aa5b4;
--offbg:#262b33;--info:#8ab8f0;--infobg:#172a40}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 10px}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0;
overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}
td.wrap{white-space:normal}th{color:var(--muted);font-weight:600}td.num,th.num{text-align:right}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:700}
.ok{color:var(--ok);background:var(--okbg)}.warn{color:var(--warn);background:var(--warnbg)}
.bad{color:var(--bad);background:var(--badbg)}.off{color:var(--off);background:var(--offbg)}
.info{color:var(--info);background:var(--infobg)}.grid{display:grid;gap:10px;
grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}.big{font-size:22px;font-weight:700}
img{max-width:100%;height:auto;border-radius:6px}ul{margin:6px 0;padding-left:20px}
"""


def _badge(text: str, level: str) -> str:
    return f'<span class="badge {level}">{html.escape(text)}</span>'


def _chart_png(panel: Panel) -> str | None:
    """2026-10-09'dan beri özsermaye (başlangıç = 100). Yeterli veri yoksa None."""
    series = {}
    for b in panel.bots:
        if b.view is None:
            continue
        points = [(t, e) for t, e in b.view.equity if t > FORWARD_START]
        if not points:
            continue
        before = [e for t, e in b.view.equity if t <= FORWARD_START]
        base = before[-1] if before else b.view.initial
        series[b.spec.name] = ([FORWARD_START] + [t for t, _ in points], [100.0] + [e / base * 100 for _, e in points])
    if not series or max(len(x) for x, _ in series.values()) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for name, (xs, ys) in series.items():
        ax.plot(pd.to_datetime(xs, unit="ms", utc=True), ys, linewidth=1.6, label=name,
                linestyle="--" if "R3A" in name else "-")
    ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_title("İleriye dönük test: 2026-10-09'dan beri özsermaye (başlangıç = 100)", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def format_html(panel: Panel) -> str:
    e = html.escape
    bad = [b for b in panel.bots if b.health.level == "bad"]
    warn = [b for b in panel.bots if b.health.level == "warn"]
    live = [b for b in panel.bots if b.spec.stopped_at is None]
    on_time = sum(1 for b in live if b.health.level == "ok")
    real = next((b for b in panel.bots if b.spec.key == "gercek"), None)
    parts = [
        "<!doctype html><html lang='tr'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>İzleme paneli</title><style>{_CSS}</style></head><body><main>",
        "<h1>Kripto bot izleme paneli</h1>",
        (f"<div class='muted'>{e(fmt_local(panel.created))} · {e(fmt_utc(panel.created))}. Yalnızca okundu; "
         "hiçbir bota dokunulmadı. Yatırım tavsiyesi değildir.</div>"),
        "<div class='grid' style='margin-top:14px'>",
        f"<div class='card'><div class='muted'>Zamanında çalışan</div><div class='big'>{on_time}/{len(live)}</div></div>",
    ]
    if real is not None and real.view is not None:
        parts.append(f"<div class='card'><div class='muted'>Gerçek hesap (bot kaydı)</div>"
                     f"<div class='big'>{real.view.current_equity:,.2f} USDT</div>"
                     f"<div class='muted'>başlangıç {real.view.initial:,.2f}</div></div>")
    alarm = (_badge(f"{len(bad)} sorun", "bad") if bad else
             _badge(f"{len(warn)} uyarı", "warn") if warn else _badge("sorun yok", "ok"))
    parts.append(f"<div class='card'><div class='muted'>Durum</div><div class='big'>{alarm}</div></div></div>")

    parts.append("<h2>Sağlık</h2><div class='card'><table><tr><th>Bot</th><th>Rol</th><th>Durum</th>"
                 "<th>Son işlenen mum</th><th>Sıradaki karar</th><th class='num'>Gecikme</th><th>Not</th></tr>")
    for b in panel.bots:
        last = e(_local(b.health.last_close)) if b.health.last_close else "-"
        upcoming = e(_local(b.health.next_decision)) if b.health.next_decision else "-"
        notes = e(b.health.detail)
        if b.health.warnings:
            notes += "<ul>" + "".join(f"<li>{e(w[:160])}</li>" for w in b.health.warnings) + "</ul>"
        parts.append(f"<tr><td>{e(b.spec.name)}</td><td class='wrap'>{e(b.spec.role)}</td>"
                     f"<td>{_badge(b.health.status, b.health.level)}</td><td>{last}</td><td>{upcoming}</td>"
                     f"<td class='num'>{b.health.lag_bars} mum</td><td class='wrap'>{notes}</td></tr>")
    parts.append("</table></div>")

    parts.append("<h2>Performans</h2><div class='card'><table><tr><th>Bot</th><th>Başlangıç</th>"
                 "<th class='num'>Başlangıç USDT</th><th class='num'>Şimdi USDT</th><th class='num'>Getiri</th>"
                 "<th class='num'>Maks. düşüş</th><th class='num'>İşlem</th><th class='num'>İsabet</th>"
                 "<th class='num'>Kâr faktörü</th><th>Açık pozisyon</th><th>Şalter</th></tr>")
    for b in panel.bots:
        v, m = b.view, b.full
        if v is None:
            parts.append(f"<tr><td>{e(b.spec.name)}</td><td colspan='10' class='muted'>kayıt yok</td></tr>")
            continue
        start = e(fmt_utc(m["start"], False)) if m else "-"
        win = _pct(m["win_rate"], False) if m and m["n_trades"] else "-"
        positions = e(", ".join(p.split("/")[0] for p in v.positions) or "-")
        halt = _badge("AKTİF", "bad") if v.halted else "-"
        parts.append(f"<tr><td>{e(b.spec.name)}</td><td>{start}</td><td class='num'>{v.initial:,.2f}</td>"
                     f"<td class='num'>{v.current_equity:,.2f}</td><td class='num'>{_pct(m['total_return']) if m else '-'}"
                     f"</td><td class='num'>{_pct(m['max_drawdown'], False) if m else '-'}</td>"
                     f"<td class='num'>{len(v.trades)}</td><td class='num'>{win}</td><td class='num'>{_pf(m)}</td>"
                     f"<td>{positions}</td><td>{halt}</td></tr>")
    parts.append("</table></div>")

    parts.append(f"<h2>İleriye dönük test ({e(fmt_utc(FORWARD_START, False))}'dan beri, aynı pencere)</h2>"
                 "<div class='card'><table><tr><th>Bot</th><th class='num'>Getiri</th><th class='num'>Maks. düşüş</th>"
                 "<th class='num'>Getiri/düşüş</th><th class='num'>İşlem</th><th class='num'>Kâr faktörü</th></tr>")
    for b in panel.bots:
        m = b.forward
        if m is None:
            parts.append(f"<tr><td>{e(b.spec.name)}</td><td colspan='5' class='muted'>henüz mum işlenmedi</td></tr>")
            continue
        parts.append(f"<tr><td>{e(b.spec.name)}</td><td class='num'>{_pct(m['total_return'])}</td>"
                     f"<td class='num'>{_pct(m['max_drawdown'], False)}</td><td class='num'>"
                     f"{e(_fmt_rd(m['return_to_drawdown']))}</td><td class='num'>{m['n_trades']}</td>"
                     f"<td class='num'>{_pf(m)}</td></tr>")
    parts.append("</table></div>")
    chart = _chart_png(panel)
    if chart:
        parts.append(f"<div class='card'><img alt='Özsermaye grafiği' src='data:image/png;base64,{chart}'></div>")

    parts.append("<h2>Gerçek paraya geçiş ölçütleri (docs/ARASTIRMA.md, 11. bölüm)</h2>")
    for c in panel.criteria:
        parts.append(f"<div class='card'><b>{e(c.code)}</b> {_badge(c.verdict, c.level)}<table>"
                     "<tr><th>Ölçüt</th><th>Şu an</th><th>Değer</th></tr>")
        for label, ok, detail in c.checks:
            mark = _badge("bilgi", "off") if ok is None else (_badge("sağlanıyor", "ok") if ok else
                                                             _badge("henüz değil", "warn"))
            parts.append(f"<tr><td>{e(label)}</td><td>{mark}</td><td class='wrap'>{e(detail)}</td></tr>")
        parts.append("</table></div>")
    parts.append("<div class='muted'>Ölçütler ancak 12 hafta dolunca karar için kullanılır; ondan önceki "
                 "durum yalnızca bilgidir. Geçiş her durumda Burak'ın onayıyla.</div>")

    checked = [b for b in panel.bots if b.parity is not None]
    if checked:
        parts.append("<h2>Eşdeğerlik kontrolü (bot = aynı dönemin backtest'i mi?)</h2><div class='card'><table>"
                     "<tr><th>Bot</th><th>Sonuç</th><th>Ayrıntı</th></tr>")
        for b in checked:
            parts.append(f"<tr><td>{e(b.spec.name)}</td><td>{_badge(b.parity.status, b.parity.level)}</td>"
                         f"<td class='wrap'>{e(b.parity.detail)}</td></tr>")
        parts.append("</table></div>")
    parts.append("</main></body></html>")
    return "".join(parts)


def save_panel(panel: Panel, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(panel.created / 1000, tz=UTC).strftime("%Y%m%d-%H%M%S")
    page = out_dir / f"{stamp}_izleme.html"
    page.write_text(format_html(panel), encoding="utf-8")
    (out_dir / "son_durum.json").write_text(json.dumps(summary_json(panel), indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    return page


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()                 # pencereye tıklamak çalışmayı dondurmasın
    parser = argparse.ArgumentParser(description="Bütün botların sağlığı ve performansı (yalnızca okur).")
    parser.add_argument("--esdeger", action="store_true",
                        help="Paper ve gölge botları aynı dönemin backtest'iyle karşılaştır (internet gerekir)")
    parser.add_argument("--canli-klasor", type=Path, default=PROJECT_ROOT.parent / "kripto-bot",
                        help="Canlı projenin klasörü (varsayılan: bu klasörün yanındaki kripto-bot)")
    parser.add_argument("--no-open", action="store_true", help="Raporu tarayıcıda açma")
    args = parser.parse_args(argv)

    roots = {"canli": args.canli_klasor, "v2": PROJECT_ROOT}
    if not args.canli_klasor.exists():
        print(f"[UYARI] Canlı proje klasörü bulunamadı: {args.canli_klasor}")
    panel = collect(roots, now_ms())
    if args.esdeger:
        from bot.exchange import ExchangeClient
        print("Eşdeğerlik kontrolü: geçmiş mumlar Binance'ten indiriliyor (yalnızca okuma, anahtar gerekmez)...")
        run_parity(panel, lambda cfg: ExchangeClient(cfg))
    print(format_console(panel))
    page = save_panel(panel, PROJECT_ROOT / REPORT_DIR)
    print(f"\nRapor: {page}")
    if not args.no_open:
        open_file(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
