"""4. araştırma turu (docs/ARASTIRMA.md, 13. bölüm): protokolün kodda aynen uygulandığını sınar."""
import dataclasses
import json
from pathlib import Path

import pytest
from helpers import H1, H4, loose_rules, make_candles, with_overrides

import arastirma as A
import arastirma_tur3 as T3
import arastirma_tur4 as T4
import tur4_veri
from bot.config import load_config
from bot.signals import momentum_params
from bot.timeutil import DAY_MS, parse_utc

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
WIDE = ["BTC/USDT", "AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "EEE/USDT"]
BASE = WIDE[:3]
EXAM = "2026-04-01"


def test_candidates_follow_protocol():
    cands = {c.code: c for c in T4.tur4_candidates()}
    assert list(cands) == ["W0", "W0A", "W1", "W2"]
    assert [c.code for c in cands.values() if c.reference] == ["W0", "W0A"]
    assert T4.REFERENCE_OF == {"W1": "W0", "W2": "W0A"}
    assert (cands["W0"].universe, cands["W0A"].universe, cands["W1"].universe, cands["W2"].universe) == (
        "c1", "c1", "genis", "genis")
    assert cands["W0"].max_drawdown == cands["W1"].max_drawdown == 0.20
    assert cands["W0A"].max_drawdown == cands["W2"].max_drawdown == 0.40
    r3 = next(c for c in T3.tur3_candidates() if c.code == "R3")
    for c in cands.values():                                  # parametreler 3. turdaki R3 ile aynı
        assert c.momentum == r3.momentum == T4.MomentumParams()
    assert cands["W0"].overrides == cands["W1"].overrides == r3.overrides   # %0,5 risk: R3'ün kendisi
    assert cands["W0A"].overrides == cands["W2"].overrides
    assert cands["W2"].overrides["risk"] == {**r3.overrides["risk"], "risk_per_trade": 0.02,
                                             "max_position_pct": 1.0, "max_drawdown_halt": 0.999,
                                             "daily_loss_limit": 0.99}


def test_aggressive_rules_equal_shadow_bot_r3a():
    """%2 riskli adaylar 12. bölümdeki ayarın aynısı: G-R3A gölge botunun ayar dosyasıyla birebir."""
    base = load_config(CONFIG_DIR / "config.yaml")
    r3a = load_config(CONFIG_DIR / "golge_r3a.yaml")
    research = A.apply_overrides(base, T4.AGGRESSIVE_RULES)
    assert research.risk == r3a.risk
    assert research.timeframes.entry == r3a.timeframes.entry == "1d"
    assert momentum_params(r3a) == T4.MomentumParams() and r3a.strategy.regime_sma == T4.MomentumParams().sma_n
    assert (research.costs, research.execution.entry_order) == (r3a.costs, r3a.execution.entry_order)


def test_w0_universe_is_c1s_twelve_coins():
    assert list(T4.C1_SYMBOLS) == load_config(CONFIG_DIR / "config.yaml").symbols
    assert len(T4.C1_SYMBOLS) == 12


def test_robustness_variants_match_round3():
    w1 = next(c for c in T4.tur4_candidates() if c.code == "W1")
    r3 = next(c for c in T3.tur3_candidates() if c.code == "R3")
    mine = [(label, v.momentum) for label, v in T4.robustness_variants(w1)]
    theirs = [(label, v.momentum) for label, v in T3.robustness_variants(r3)]
    assert mine == theirs and len(mine) == 4
    assert all(v.overrides == w1.overrides for _, v in T4.robustness_variants(w1))   # risk aynı kalır
    w2 = next(c for c in T4.tur4_candidates() if c.code == "W2")
    assert all(v.overrides["risk"]["risk_per_trade"] == 0.02 for _, v in T4.robustness_variants(w2))
    assert T4.robustness_variants(next(c for c in T4.tur4_candidates() if c.code == "W0")) == []


def test_period_check_uses_candidate_drawdown_limit():
    m = {"net_profit": 10.0, "profit_factor": 1.5, "max_drawdown": 0.30}
    assert T4._period_check(m, 1.2, 0.40) == (True, "tümü sağlandı")
    ok, text = T4._period_check(m, 1.2, 0.20)
    assert not ok and "düşüş" in text
    ok, text = T4._period_check({**m, "net_profit": 0.0, "profit_factor": 1.1}, 1.2, 0.40)
    assert not ok and "net getiri" in text and "KF" in text


# ---------------------------------------------------------------------------
# Veri: evren dosyası ve dönemler
# ---------------------------------------------------------------------------
def test_load_universes_puts_btc_eth_first_and_requires_min_size(cfg, tmp_path):
    c = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        T4.load_universes(c)
    coins = [f"C{i:02d}/USDT" for i in range(tur4_veri.MIN_SIZE)]
    (c.data_dir / tur4_veri.UNIVERSE_FILE).write_text(json.dumps({"secilen": coins}), encoding="utf-8")
    universes = T4.load_universes(c)
    assert universes["genis"] == [*tur4_veri.ALWAYS, *coins] and universes["c1"] == list(T4.C1_SYMBOLS)
    (c.data_dir / tur4_veri.UNIVERSE_FILE).write_text(json.dumps({"secilen": coins[:-1]}), encoding="utf-8")
    with pytest.raises(ValueError):
        T4.load_universes(c)


def synthetic_market(start="2022-01-01", end="2026-05-01"):
    n = (parse_utc(end) - parse_utc(start)) // H1
    raw = {s: {"1h": make_candles(n, seed=300 + i, start=start, base=40.0 * (i + 1), tf_ms=H1, vol=0.008)}
           for i, s in enumerate(WIDE)}                                    # sınav sonrasına da uzanır
    return A.prepare_market(raw, parse_utc(EXAM))


def test_protocol_bounds_reject_data_that_starts_too_late(cfg):
    c = with_overrides(cfg, data={"exam_start": EXAM})
    market = synthetic_market(start="2022-03-01")                         # SMA200 2022-07-20'de hazır olamaz
    with pytest.raises(ValueError, match="kapsamıyor"):
        T4.protocol_bounds(c, market, {"c1": BASE, "genis": WIDE}, T4.tur4_candidates())


@pytest.fixture(scope="module")
def tur4_run(cfg, tmp_path_factory):
    c = with_overrides(cfg, data={"exam_start": EXAM})
    market = synthetic_market()
    rules = loose_rules(WIDE)
    study = T4.run_tur4(c, market, {"c1": BASE, "genis": WIDE}, rules, rules)
    return study, market, tmp_path_factory.mktemp("tur4")


def test_tur4_uses_round3_periods_and_no_exam_data(tur4_run):
    study, market, _ = tur4_run
    assert study.bounds == {"kesif": (parse_utc("2022-07-20"), parse_utc("2025-01-01")),
                            "dogrulama": (parse_utc("2025-01-01"), parse_utc(EXAM))}
    exam = parse_utc(EXAM)
    for frames in market.values():
        for tf, df in frames.items():
            assert int(df["ts"].iloc[-1]) + {"1h": H1, "4h": H4, "1d": DAY_MS}[tf] <= exam
    for r in study.reports:
        for name, run in r.portfolio.items():
            assert all(t.entry_ts >= study.bounds[name][0] for t in run.result.trades)
            assert all(t.exit_ts < study.bounds[name][1] for t in run.result.trades)


def test_tur4_runs_each_candidate_on_its_own_universe_and_reports(tur4_run):
    study, _, out = tur4_run
    by_code = {r.candidate.code: r for r in study.reports}
    assert by_code["W0"].symbols == BASE and by_code["W1"].symbols == WIDE
    assert {t.symbol for r in ("W0", "W0A") for run in by_code[r].portfolio.values()
            for t in run.result.trades} <= set(BASE)
    assert sum(len(run.result.trades) for run in by_code["W1"].portfolio.values()) > 0
    for code in ("W1", "W2"):
        r = by_code[code]
        assert set(r.checks) == {"kesif", "dogrulama", "iyilesme", "saglamlik", "islem"}
        assert len(r.robustness) == 4
        ref = by_code[T4.REFERENCE_OF[code]]
        expected = all(r.portfolio[n].metrics["return_to_drawdown"] > ref.portfolio[n].metrics["return_to_drawdown"]
                       for n in ("kesif", "dogrulama"))
        assert r.checks["iyilesme"][0] == expected and ref.candidate.code in r.checks["iyilesme"][1]
    for code in ("W0", "W0A"):
        assert by_code[code].checks["iyilesme"][0] is None and not by_code[code].passed
    passed = [r.candidate.code for r in study.reports if r.passed]
    assert set(study.shadow) <= set(passed) and len(study.shadow) <= T4.MAX_SHADOW
    text = T4.format_tur4(study)
    assert "Ölçütler" in text and "Şaltersiz tanı" in text and "coin başına katkı" in text
    assert "W1: coin başına katkı" in text and "W0: coin başına katkı" not in text
    T4.save_tur4(out, study, text, plot=False)
    summary = json.loads((out / "ozet.json").read_text(encoding="utf-8"))
    assert [c["code"] for c in summary["candidates"]] == ["W0", "W0A", "W1", "W2"]
    assert (out / "sonuc.md").exists() and (out / "islemler.csv").exists()


def test_tur4_coin_contribution_adds_up_to_net_profit(tur4_run):
    study, _, _ = tur4_run
    for r in study.reports:
        for name, run in r.portfolio.items():
            total = sum(v for _, v, _ in T4.coin_contribution(r, name))
            assert total == pytest.approx(run.metrics["net_profit"], abs=1e-6 * study.initial_equity)
            old, new = T4.new_coin_share(r, name, BASE)
            assert old + new == pytest.approx(total)


def test_aggressive_candidates_really_use_two_percent_risk(tur4_run):
    study, _, _ = tur4_run
    by_code = {r.candidate.code: r for r in study.reports}
    trades = [t for run in by_code["W2"].portfolio.values() for t in run.result.trades]
    assert trades and by_code["W2"].monte_carlo["n"] == len(trades)
    assert by_code["W2"].portfolio["kesif"].result.halted_at is None     # %20 şalteri kapalı (0,999)


def test_tur4_coin_table_marks_only_coins_outside_c1_as_new(tur4_run):
    """Raporda yalnızca C1 evreninde olmayan coinler "(yeni)" diye işaretlenir; toplamlar doğru bölünür."""
    study, _, _ = tur4_run
    text = T4.format_tur4(study)
    section = text.split("**W1: coin başına katkı")[1].split("**W2: coin başına katkı")[0]
    rows = {line.split("|")[1].strip(): line for line in section.splitlines() if line.startswith("| ")}
    for symbol in WIDE:
        coin = symbol.split("/")[0]
        label = coin if symbol in BASE else f"{coin} (yeni)"
        assert label in rows, f"{label} satırı yok"
    w1 = next(r for r in study.reports if r.candidate.code == "W1")
    old, new = T4.new_coin_share(w1, "kesif", BASE)
    assert f"| {old:+,.0f} |" in rows["**C1'in 3 coini toplam**"].replace("|  |", "|")
    assert f"| {new:+,.0f} |" in rows["**Yeni coinler toplam**"].replace("|  |", "|")
