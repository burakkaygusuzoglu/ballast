"""İzleme paneli: yalnızca okur, sağlığı ve ölçütleri doğru hesaplar, paper = backtest farkını yakalar."""
import dataclasses
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from helpers import D1, H4, FakeSource, loose_rules, make_candles, with_overrides

import izleme as Z
from bot.candidates import resample_candles
from bot.paper import PaperTrader
from bot.portfolio import Trade
from bot.storage import StateStore
from bot.timeutil import DAY_MS, parse_utc

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def snapshot(initial=50.0, last_ts=None, positions=(), halted=False):
    return {"version": 1, "setup": {}, "last_ts": last_ts, "last_prices": {},
            "portfolio": {"initial_cash": initial, "cash": initial, "reserved": 0.0, "total_fees": 0.0,
                          "positions": [{"symbol": s} for s in positions]},
            "broker": {}, "risk": {"halted": halted, "halt_reason": "test şalteri" if halted else ""}}


def trade(symbol="BTC/USDT", entry="2026-10-10", exit_="2026-10-12", pnl=1.0):
    return Trade(symbol, parse_utc(entry), parse_utc(exit_), 0.001, 60_000.0, 61_000.0, 0.06, 0.06, 0.03,
                 pnl + 0.12, pnl, 0.5, pnl / 0.5, "çıkış sinyali")


def make_db(path: Path, snap: dict, trades=(), equity=()) -> Path:
    store = StateStore(path)
    store.save(snap, list(trades), list(equity))
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Salt okuma
# ---------------------------------------------------------------------------
def test_read_state_never_writes(tmp_path):
    db = make_db(tmp_path / "x_state.db", snapshot(last_ts=parse_utc("2026-10-09")),
                 [trade()], [(parse_utc("2026-10-09 04:00"), 57.0)])
    before, files = digest(db), sorted(p.name for p in tmp_path.iterdir())
    view = Z.read_state(db)
    assert view.initial == 50.0 and view.current_equity == 57.0 and len(view.trades) == 1
    assert digest(db) == before and sorted(p.name for p in tmp_path.iterdir()) == files
    uri = db.resolve().as_uri() + "?mode=ro"
    with pytest.raises(sqlite3.OperationalError), sqlite3.connect(uri, uri=True) as conn:   # yazma reddedilir
        conn.execute("DELETE FROM trades")
    assert Z.read_state(tmp_path / "yok.db") is None


def test_scan_log_keeps_recent_warnings_only(tmp_path):
    log = tmp_path / "bot.log"
    log.write_text("\n".join([
        "2026-09-20 04:00:10Z | WARNING | eski uyarı",
        "2026-10-08 04:00:10Z | INFO    | Mum 2026-10-08 04:00 UTC işlendi",
        *[f"2026-10-08 0{i}:00:10Z | WARNING | yeni uyarı {i}" for i in range(7)],
        "2026-10-09 04:00:13Z | INFO    | Mum 2026-10-09 04:00 UTC işlendi",
        "bozuk satır",
    ]), encoding="utf-8")
    warnings, last = Z.scan_log(log, parse_utc("2026-10-09 05:00"))
    assert len(warnings) == 5 and all("yeni uyarı" in w for w in warnings) and warnings[-1].endswith("6")
    assert last == parse_utc("2026-10-09 04:00:13")
    assert Z.scan_log(tmp_path / "yok.log", 0) == ([], None)


# ---------------------------------------------------------------------------
# Sağlık
# ---------------------------------------------------------------------------
def test_health_uses_grace_period_and_counts_missed_decisions():
    spec = Z.BOTS[3]                                             # R1 gölge (4 saatlik)
    view = Z.StateView(snapshot(last_ts=parse_utc("2026-10-08 20:00")), [], [])
    at = parse_utc
    assert Z.health(spec, view, H4, at("2026-10-09 04:05"), []).status == "ZAMANINDA"   # 04:00 mumu için süre var
    late = Z.health(spec, view, H4, at("2026-10-09 04:20"), [])
    assert (late.status, late.lag_bars, late.level) == ("GECİKMİŞ", 1, "warn")
    dead = Z.health(spec, view, H4, at("2026-10-09 12:20"), [])
    assert (dead.status, dead.lag_bars, dead.level) == ("DURMUŞ", 3, "bad")
    stopped = dataclasses.replace(spec, stopped_at=at("2026-10-09 03:45"))
    assert Z.health(stopped, view, H4, at("2026-10-10"), []).status == "DURDURULDU"
    still = Z.StateView(snapshot(last_ts=parse_utc("2026-10-09 04:00")), [], [])     # 08:00 mumunu işlemiş
    running = Z.health(stopped, still, H4, at("2026-10-09 09:00"), [])
    assert (running.status, running.level) == ("HÂLÂ ÇALIŞIYOR", "warn") and "kapatın" in running.detail
    assert Z.BOTS[5].stopped_at == parse_utc("2026-10-09T03:45") and Z.BOTS[5].key == "golge_r3a"
    fresh = Z.health(spec, view, H4, at("2026-10-09 04:05"), [])                    # henüz hiç mum işlemedi
    assert fresh.next_decision == at("2026-10-09 04:00") and "ilk kararını" in fresh.detail
    assert Z.health(spec, None, H4, at("2026-10-10"), []).status == "KAYIT YOK"
    daily = Z.StateView(snapshot(last_ts=parse_utc("2026-10-08")), [], [])
    assert Z.health(Z.BOTS[4], daily, D1, at("2026-10-09 10:00"), []).status == "ZAMANINDA"
    assert Z.health(Z.BOTS[4], daily, D1, at("2026-10-10 00:30"), []).status == "GECİKMİŞ"


# ---------------------------------------------------------------------------
# Ölçütler (11. bölüm)
# ---------------------------------------------------------------------------
def views_for(n_trades: int, end_equity: float, ref_end: float, halted=False):
    days = [Z.FORWARD_START + (i + 1) * DAY_MS for i in range(120)]
    trades = [trade(entry="2026-10-10", exit_="2026-10-12", pnl=2.0 if i % 2 else -1.0) for i in range(n_trades)]
    equity = [(t, 50.0 + (end_equity - 50.0) * (i + 1) / len(days)) for i, t in enumerate(days)]
    equity[60] = (equity[60][0], equity[60][1] * 0.97)                       # biraz düşüş
    ref = [(t, 50.0 + (ref_end - 50.0) * (i + 1) / len(days)) for i, t in enumerate(days)]
    ref[30] = (ref[30][0], ref[30][1] * 0.90)
    v = Z.StateView(snapshot(halted=halted), trades, equity)
    r = Z.StateView(snapshot(), [], ref)
    return v, r


def test_forward_criteria_follow_protocol_timeline():
    v, r = views_for(24, 58.1, 52.8)
    early = Z.forward_criteria("R1", v, r, Z.FORWARD_START + 5 * Z.WEEK_MS)
    assert early.level == "info" and early.verdict.startswith("Erken") and "2027-01-01" in early.verdict
    ready = Z.forward_criteria("R1", v, r, Z.FORWARD_START + 13 * Z.WEEK_MS)
    assert ready.level == "ok" and all(ok for _, ok, _ in ready.checks)
    few, r2 = views_for(10, 58.1, 52.8)
    pending = Z.forward_criteria("R1", few, r2, Z.FORWARD_START + 13 * Z.WEEK_MS)
    assert pending.level == "warn" and pending.checks[0][1] is False
    expired = Z.forward_criteria("R1", few, r2, Z.FORWARD_START + 27 * Z.WEEK_MS)
    assert expired.level == "bad" and "bırakılır" in expired.verdict
    worse, r3 = views_for(24, 51.0, 58.1)                                     # C1 paper'dan kötü
    assert Z.forward_criteria("R1", worse, r3, Z.FORWARD_START + 13 * Z.WEEK_MS).checks[3][1] is False
    halted, r4 = views_for(24, 58.1, 52.8, halted=True)
    assert Z.forward_criteria("R1", halted, r4, Z.FORWARD_START + 13 * Z.WEEK_MS).level == "warn"
    r3_checks = Z.forward_criteria("R3", v, r, Z.FORWARD_START + 13 * Z.WEEK_MS).checks
    assert r3_checks[-1][1] is None and "250" in r3_checks[-1][0]            # bilgi satırı karar dışı
    assert Z.forward_criteria("R1", None, None, Z.FORWARD_START).checks[0][1] is False


def test_window_metrics_use_equity_at_window_start():
    start = Z.FORWARD_START
    view = Z.StateView(snapshot(initial=100.0), [], [(start - DAY_MS, 110.0), (start, 120.0),
                                                     (start + DAY_MS, 108.0), (start + 2 * DAY_MS, 132.0)])
    m = Z.window_metrics(view, start)
    assert m["initial_equity"] == 120.0 and m["total_return"] == pytest.approx(0.10)
    assert m["max_drawdown"] == pytest.approx(0.10)
    assert Z.window_metrics(view, start + 5 * DAY_MS) is None


# ---------------------------------------------------------------------------
# Eşdeğerlik: paper = backtest
# ---------------------------------------------------------------------------
SHADOW = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


@pytest.fixture(scope="module")
def shadow_run(cfg, tmp_path_factory):
    h4 = {s: make_candles(6 * 260, seed=520 + i, start="2023-01-01", base=30.0 * (i + 1), tf_ms=H4, vol=0.012)
          for i, s in enumerate(SHADOW)}
    d1 = {s: resample_candles(df, H4, D1) for s, df in h4.items()}
    c = with_overrides(cfg, timeframes={"entry": "4h", "trend": "1d"}, execution={"entry_order": "market"},
                       risk={"risk_per_trade": 0.005, "stop_atr_mult": 2.0, "trail_atr_mult": 0.0,
                             "max_open_positions": 5, "max_trades_per_day": 50},
                       strategy={"type": "breakout", "exit_on_regime_break": False, "regime_sma": 30},
                       paper={"history_bars": 400, "initial_equity": 10_000.0, "name": "golge_test"})
    c = dataclasses.replace(c, symbols=SHADOW)
    source = FakeSource({**{(s, "4h"): df for s, df in h4.items()}, **{(s, "1d"): df for s, df in d1.items()}})
    db = tmp_path_factory.mktemp("golge") / "golge_test_state.db"
    trader = PaperTrader(c, source, StateStore(db), loose_rules(SHADOW))
    ts = h4["BTC/USDT"]["ts"]
    moment = int(ts.iloc[6 * 120])
    while moment <= int(ts.iloc[-1]):
        trader.sync(moment + 5_000)
        moment += 2 * H4
    return c, source, db


def test_parity_confirms_paper_equals_backtest(shadow_run):
    c, source, db = shadow_run
    view = Z.read_state(db)
    assert len(view.trades) >= 3, "karşılaştırma anlamlı olsun diye işlem olmalı"
    result = Z.parity_check(c, view, source, loose_rules(SHADOW))
    assert result.status == "AYNI" and result.n_bot == result.n_backtest == len(view.trades)


def test_parity_detects_a_changed_trade(shadow_run, tmp_path):
    c, source, db = shadow_run
    copy = tmp_path / "kopya.db"
    shutil.copy(db, copy)
    with sqlite3.connect(copy) as conn:
        conn.execute("UPDATE trades SET net_pnl = net_pnl + 5 WHERE id = (SELECT MIN(id) FROM trades)")
    result = Z.parity_check(c, Z.read_state(copy), source, loose_rules(SHADOW))
    assert result.status == "FARKLI" and "1. işlem" in result.detail
    assert Z.parity_check(c, None, source, {}).status == "BEKLİYOR"


# ---------------------------------------------------------------------------
# Uçtan uca: iki klasör, altı bot, rapor
# ---------------------------------------------------------------------------
@pytest.fixture()
def roots(tmp_path):
    canli, v2 = tmp_path / "kripto-bot", tmp_path / "kripto-bot-v2"
    for root, names in ((canli, ["config.yaml"]), (v2, ["golge_r1.yaml", "golge_r3.yaml", "golge_r3a.yaml"])):
        (root / "config").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "logs").mkdir()
        for name in names:
            shutil.copy(CONFIG_DIR / name, root / "config" / name)
    h4_last = parse_utc("2026-10-09 00:00")                    # 04:00'da kapanan mum
    for name, initial in (("canli", 50.0), ("testnet", 50.0), ("paper", 50.0)):
        make_db(canli / "data" / f"{name}_state.db", snapshot(initial, last_ts=h4_last),
                [], [(parse_utc("2026-10-09 04:00"), initial)])
        (canli / "logs" / f"{name}.log").write_text(
            "2026-10-09 04:00:10Z | INFO    | Mum 2026-10-09 04:00 UTC işlendi (1)\n", encoding="utf-8")
    make_db(v2 / "data" / "golge_r1_state.db", snapshot(last_ts=h4_last), [trade()],
            [(parse_utc("2026-10-09 04:00"), 50.0), (parse_utc("2026-10-12 04:00"), 57.80)])
    make_db(v2 / "data" / "golge_r3_state.db", snapshot(1000.0, last_ts=parse_utc("2026-10-08")))
    (v2 / "logs" / "golge_r1.log").write_text("2026-10-09 04:00:13Z | WARNING | Binance verisi gecikiyor\n",
                                              encoding="utf-8")
    return {"canli": canli, "v2": v2}


def test_collect_and_report_end_to_end(roots, tmp_path):
    before = {p: digest(p) for root in roots.values() for p in root.rglob("*") if p.is_file()}
    now = parse_utc("2026-10-09 05:00")
    panel = Z.collect(roots, now)
    by_key = {b.spec.key: b for b in panel.bots}
    assert [b.spec.key for b in panel.bots] == ["gercek", "testnet", "paper", "golge_r1", "golge_r3", "golge_r3a"]
    assert by_key["gercek"].health.status == "ZAMANINDA" and by_key["gercek"].view.initial == 50.0
    assert by_key["golge_r3"].health.status == "ZAMANINDA"                   # günlük: sıradaki karar 10 Ekim
    assert by_key["golge_r3a"].health.status == "KAYIT YOK" and by_key["golge_r3a"].health.level == "off"
    assert by_key["golge_r1"].health.warnings and by_key["golge_r1"].forward["n_trades"] == 1
    assert [c.code for c in panel.criteria] == ["R1", "R3"] and all(c.level == "info" for c in panel.criteria)
    text = Z.format_console(panel)
    assert "İZLEME PANELİ" in text and "R1: Erken" in text
    page = Z.save_panel(panel, tmp_path / "rapor")
    content = page.read_text(encoding="utf-8")
    for section in ("Sağlık", "Performans", "İleriye dönük test", "Gerçek paraya geçiş ölçütleri"):
        assert section in content
    summary = json.loads((tmp_path / "rapor" / "son_durum.json").read_text(encoding="utf-8"))
    assert summary["botlar"][0]["anahtar"] == "gercek" and len(summary["olcutler"]) == 2
    after = {p: digest(p) for root in roots.values() for p in root.rglob("*") if p.is_file()}
    assert after == before                                     # hiçbir bot dosyası değişmedi
    late = Z.collect(roots, parse_utc("2026-10-09 12:30"))
    assert {b.spec.key: b.health.status for b in late.bots}["gercek"] == "DURMUŞ"
    assert "[!] DİKKAT" in Z.format_console(late)
