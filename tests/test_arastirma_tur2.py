"""2. tur: evren seçimi kurala uyar, sınav verisi kullanılmaz, portföy ve ölçütler doğru işler."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import arastirma as A
import arastirma_tur2 as T
import arastirma_verisi as V
from bot.config import MarketRules
from bot.data import candle_path, load_market_rules
from bot.timeutil import DAY_MS, parse_utc
from helpers import H1, FakeClient, loose_rules, make_candles, with_overrides

COINS = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT"]


# ---------------------------------------------------------------------------
# Evren seçimi (arastirma_verisi.py)
# ---------------------------------------------------------------------------
def test_data_problem_checks_only_timestamps():
    start, end = parse_utc("2022-01-01"), parse_utc("2022-03-01")
    full = make_candles((end - start) // H1, seed=1, start="2022-01-01", tf_ms=H1)
    gap = 7 * DAY_MS
    assert V.data_problem(full, start, end, H1, gap) is None
    assert "başlıyor" in V.data_problem(full.iloc[24:], start, end, H1, gap)
    assert "bitiyor" in V.data_problem(full.iloc[: -24 * 10], start, end, H1, gap)
    holed = full.drop(index=range(500, 500 + 24 * 8)).reset_index(drop=True)
    assert "günlük boşluk" in V.data_problem(holed, start, end, H1, gap)
    short_hole = full.drop(index=range(500, 500 + 24 * 6)).reset_index(drop=True)
    assert V.data_problem(short_hole, start, end, H1, gap) is None
    assert V.data_problem(full.iloc[:0], start, end, H1, gap) == "veri yok"


class UniverseClient(FakeClient):
    def __init__(self, data, now_ms, markets):
        super().__init__(data, now_ms)
        self._markets = markets

    def markets(self):
        return self._markets


def test_universe_takes_first_eligible_coins_in_order(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(V, "CANDIDATES", ("AAA", "LATE", "GONE", "OFF", "BBB", "HOLE", "CCC", "DDD"))
    monkeypatch.setattr(V, "UNIVERSE_SIZE", 3)
    c = with_overrides(cfg, data={"exam_start": "2022-03-01", "dir": str(tmp_path)})
    bars = (parse_utc("2022-03-10") - parse_utc("2022-01-01")) // H1
    full = lambda seed: make_candles(bars, seed=seed, start="2022-01-01", tf_ms=H1)     # noqa: E731
    data = {("AAA/USDT", "1h"): full(1), ("BBB/USDT", "1h"): full(2), ("CCC/USDT", "1h"): full(3),
            ("DDD/USDT", "1h"): full(4), ("LATE/USDT", "1h"): full(5).iloc[24 * 30:],
            ("HOLE/USDT", "1h"): full(6).drop(index=range(300, 300 + 24 * 9)).reset_index(drop=True)}
    markets = {s: {"spot": True, "active": True} for s, _ in data}
    markets["OFF/USDT"] = {"spot": True, "active": False}
    client = UniverseClient(data, parse_utc("2022-03-10"), markets)
    lines = []
    universe = V.select_universe(client, c, echo=lines.append)
    assert universe["secilen"] == ["AAA/USDT", "BBB/USDT", "CCC/USDT"]
    assert set(universe["elenen"]) == {"LATE/USDT", "GONE/USDT", "OFF/USDT", "HOLE/USDT"}
    assert "yok" in universe["elenen"]["GONE/USDT"] and "aktif değil" in universe["elenen"]["OFF/USDT"]
    assert universe["denenmeyen"] == ["DDD/USDT"]
    assert candle_path(tmp_path, "binance", "AAA/USDT", "1h").exists()
    V.save_universe(c, universe)
    assert V.load_universe(c)["secilen"] == universe["secilen"]


# ---------------------------------------------------------------------------
# 2. tur çalıştırması (arastirma_tur2.py)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tur2_cfg(cfg):
    return with_overrides(cfg, data={"exam_start": "2025-03-01"})


@pytest.fixture(scope="module")
def tur2_market():
    raw = {s: {"1h": make_candles(24 * 460, seed=50 + i, start="2024-01-01", tf_ms=H1, vol=0.008)}
           for i, s in enumerate(COINS)}                                        # sınav sonrasına da uzanır
    return A.prepare_market(raw, parse_utc("2025-03-01"))


@pytest.fixture(scope="module")
def tur2_study(tur2_cfg, tur2_market):
    universe = {"secilen": COINS, "elenen": {"XXX/USDT": "Binance'te USDT paritesi yok"}, "denenmeyen": []}
    return T.run_tur2(tur2_cfg, tur2_market, universe, loose_rules(COINS))


def test_candidates_are_round_one_rules_with_position_limit():
    t2, t3 = T.tur2_candidates()
    base = {c.code: c for c in A.protocol_candidates()}
    for new, old in ((t2, base["S2"]), (t3, base["S3"])):
        assert new.params == old.params and new.entry_tf == old.entry_tf
        assert new.overrides["risk"] == {**old.overrides["risk"], "max_open_positions": T.MAX_OPEN_POSITIONS}
        assert {k: v for k, v in new.overrides.items() if k != "risk"} == \
               {k: v for k, v in old.overrides.items() if k != "risk"}


def test_tur2_study_is_complete(tur2_study):
    split, exam = parse_utc(A.VALIDATION_START), parse_utc("2025-03-01")
    assert [r.candidate.code for r in tur2_study.reports] == ["T2", "T3"]
    for r in tur2_study.reports:
        assert r.portfolio["kesif"].end == split and r.portfolio["dogrulama"].end == exam
        assert r.portfolio["kesif"].metrics["n_trades"] > 0, "test anlamlı olsun diye işlem olmalı"
        assert set(r.coins) == set(COINS)
        assert set(r.checks) == {"kesif", "dogrulama", "genislik", "islem", "saglamlik"}
        assert len(r.robustness) == 6
        assert set(r.feasibility) == {"kesif", "dogrulama"}
        assert r.checks["genislik"][0] is False                             # 4 coin < 6 kârlı coin şartı
    t2 = tur2_study.reports[0]
    assert len(t2.portfolio["kesif"].result.open_positions) <= T.MAX_OPEN_POSITIONS


def test_portfolio_never_exceeds_position_limit(tur2_cfg, tur2_market):
    t2 = T.tur2_candidates()[0]
    limit = 2                                                                # 4 coin > 2 yer: sınır zorlanır
    cfg_c = A.apply_overrides(tur2_cfg, A._merge(t2.overrides, {"risk": {"max_open_positions": limit}}))
    signals = A.build_signals(t2, cfg_c, tur2_market)
    for df in signals.values():
        df["entry_signal"] = df["c_ready"]                                   # her mumda sinyal: sınır zorlanır
    start, _ = A.trade_window(signals, cfg_c.entry_ms)
    from bot.backtester import iter_signal_bars
    from bot.brokers.simulated import SimulatedBroker
    from bot.engine import TradingEngine
    from bot.portfolio import Portfolio
    from bot.risk import RiskManager
    portfolio = Portfolio(10_000)
    broker = SimulatedBroker(portfolio, 0.001, 0.001, 0.0005)
    engine = TradingEngine(cfg_c, portfolio, broker, RiskManager(cfg_c, 10_000), loose_rules(COINS))
    most = 0
    for ts, bars in iter_signal_bars(signals, start, start + 200 * cfg_c.entry_ms):
        engine.on_bar(ts, bars)
        most = max(most, len(portfolio.positions) + broker.pending_count())
    assert most == limit


def test_report_summary_and_files(tur2_study, tmp_path):
    text = T.format_tur2(tur2_study)
    for fragment in ("Portföy: keşif dönemi", "Ölçütler", "T2: coin bazında", "Sağlamlık", "Monte Carlo",
                     "Uygulanabilirlik", "korelasyon", "Karar", "XXX (Binance'te USDT paritesi yok)"):
        assert fragment in text, fragment
    json.dumps(T.tur2_summary(tur2_study), allow_nan=False)
    T.save_tur2(tmp_path, tur2_study, text, plot=False)
    trades = pd.read_csv(tmp_path / "islemler.csv")
    assert len(trades) == sum(len(run.result.trades) for r in tur2_study.reports for run in r.portfolio.values())


def test_feasibility_uses_real_minimum_order_size(tur2_cfg, tur2_market):
    t3 = T.tur2_candidates()[1]
    strict = {s: MarketRules(qty_step=1e-8, min_notional=1_000.0) for s in COINS}     # hiçbir emir geçemez
    report = T.run_candidate_tur2(t3, tur2_cfg, tur2_market, strict, progress=lambda _m: None)
    f = report.feasibility["kesif"]
    assert f["orders"] == 0 and f["skipped_min_notional"] > 0
    assert f["metrics"]["n_trades"] == 0


def test_winner_is_best_validation_return_to_drawdown():
    def report(code, passed, rtd):
        run = SimpleNamespace(metrics={"return_to_drawdown": rtd})
        return SimpleNamespace(passed=passed, candidate=SimpleNamespace(code=code), portfolio={"dogrulama": run})
    assert T.choose_winner([report("T2", True, 1.5), report("T3", True, 2.0)]) == "T3"
    assert T.choose_winner([report("T2", True, 1.5), report("T3", False, 9.0)]) == "T2"
    assert T.choose_winner([report("T2", False, 1.5)]) is None


def test_load_market_requires_universe_file(tur2_cfg, tmp_path):
    c = with_overrides(tur2_cfg, data={"dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError):
        T.load_tur2_market(c)
    V.save_universe(c, {"secilen": COINS, "elenen": {}, "denenmeyen": []})
    with pytest.raises(ValueError, match="10 coin"):
        T.load_tur2_market(c)
    assert load_market_rules(tmp_path, COINS, MarketRules(1.0, 5.0))["AAA/USDT"].qty_step == 1.0
