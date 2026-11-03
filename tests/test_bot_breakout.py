"""Kanal kırılımı stratejisinin bota bağlanması: strateji seçimi, paper = backtest, uçtan uca komutlar."""
import dataclasses
import json

import pytest
import yaml

import backtest
import download_data
import run
from bot.backtester import run_backtest
from bot.candidates import BreakoutParams, breakout_signals
from bot.config import CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH, load_config
from bot.paper import PaperTrader
from bot.signals import BREAKOUT_CONDITIONS, build_signals, conditions, describe, uses_trend
from bot.storage import StateStore
from bot.strategy import CONDITIONS as PULLBACK_CONDITIONS
from helpers import D1, H4, FakeClient, FakeSource, loose_rules, make_candles, resample, with_overrides


@pytest.fixture(scope="module")
def bcfg(cfg):
    """Test ayarının kırılım stratejisine çevrilmiş hâli (C1'in strateji ve risk kuralları)."""
    return with_overrides(
        cfg,
        timeframes={"entry": "4h", "trend": "1d"},
        strategy={"type": "breakout", "exit_on_regime_break": False},
        execution={"entry_order": "market"},
        risk={"risk_per_trade": 0.005, "trail_atr_mult": 0.0, "max_open_positions": 5},
        paper={"history_bars": 5000, "initial_equity": 10_000.0},
    )


def test_strategy_selection(cfg, bcfg, market_15m, market_1h):
    assert uses_trend(cfg) and not uses_trend(bcfg)
    assert conditions(cfg) == PULLBACK_CONDITIONS and conditions(bcfg) == BREAKOUT_CONDITIONS
    assert "Kanal kırılımı 20/10" in describe(bcfg)
    h4 = resample(market_15m, H4)
    direct = breakout_signals(h4, BreakoutParams(20, 10, 20), bcfg)
    via = build_signals(bcfg, h4)
    assert via["entry_signal"].equals(direct["entry_signal"]) and via["exit_signal"].equals(direct["exit_signal"])
    with pytest.raises(ValueError, match="trend"):
        build_signals(cfg, market_15m)
    assert build_signals(cfg, market_15m, market_1h)["entry_signal"].sum() >= 1


# ---------------------------------------------------------------------------
# Paper trading = backtest (kırılım, 4 saatlik, 3 sembol)
# ---------------------------------------------------------------------------
SYMBOLS = ["AAA/USDT", "BBB/USDT", "CCC/USDT"]
N_BARS = 700
START_BAR = 250


@pytest.fixture(scope="module")
def h4_markets():
    return {s: make_candles(N_BARS, seed=60 + i, base=100.0 / (i + 1), tf_ms=H4, vol=0.012)
            for i, s in enumerate(SYMBOLS)}


@pytest.fixture(scope="module")
def pbcfg(bcfg):
    return dataclasses.replace(bcfg, symbols=SYMBOLS)


def test_paper_matches_backtest_for_breakout(pbcfg, h4_markets, tmp_path):
    # Yalnızca 4h verisi var: paper trader trend (1d) verisi isteseydi FakeSource KeyError verirdi.
    source = FakeSource({(s, "4h"): df for s, df in h4_markets.items()})
    rules = loose_rules(SYMBOLS)
    trader = PaperTrader(pbcfg, source, StateStore(tmp_path / "p.db"), rules)
    first = int(h4_markets["AAA/USDT"]["ts"].iloc[START_BAR])
    last = int(h4_markets["AAA/USDT"]["ts"].iloc[-1])
    moment = first
    while moment < last:                     # her turda 3 mum (kaçırılan mumları sırayla işleme dahil)
        trader.sync(moment + 5_000)
        moment += 3 * H4
    trader.sync(last + 5_000)
    signals = {s: build_signals(pbcfg, df) for s, df in h4_markets.items()}
    bt = run_backtest(pbcfg, signals, first, trader.last_ts + H4, pbcfg.paper.initial_equity, rules)
    assert len(bt.trades) >= 5, "karşılaştırma anlamlı olsun diye işlem olmalı"
    assert trader.portfolio.trades == bt.trades
    assert trader.engine.equity_points[-1][1] == pytest.approx(float(bt.equity.iloc[-1]), abs=1e-6)
    assert all(tf == "4h" for _, tf in trader.windows)


# ---------------------------------------------------------------------------
# Uçtan uca: botun kendi ayar dosyasıyla (C1) indirme -> backtest -> sınav -> paper
# ---------------------------------------------------------------------------
@pytest.fixture
def c1_project(tmp_path, monkeypatch):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"].update(dir=str(tmp_path / "data"), start="2023-01-01", exam_start="2024-01-01")
    raw["logging"]["dir"] = str(tmp_path / "logs")
    raw["backtest"].update(reports_dir=str(tmp_path / "reports"), open_chart=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))
    data = {}
    for i, symbol in enumerate(raw["symbols"]):
        h4 = make_candles(450 * 6, seed=80 + i, start="2023-01-01", base=10.0 * (i + 1), tf_ms=H4, vol=0.012)
        data[(symbol, "4h")] = h4
        data[(symbol, "1d")] = resample(h4, D1, from_ms=H4)
    end = int(data[("BTC/USDT", "4h")]["ts"].iloc[-1]) + H4
    client = FakeClient(data, now_ms=end + 5_000)
    factory = lambda *args, **kwargs: client  # noqa: E731
    monkeypatch.setattr(download_data, "ExchangeClient", factory)
    monkeypatch.setattr(run, "ExchangeClient", factory)
    return tmp_path, client


def test_c1_end_to_end(c1_project, capsys, monkeypatch):
    tmp_path, client = c1_project
    assert load_config().strategy.type == "breakout"
    assert download_data.main([]) == 0
    capsys.readouterr()
    assert backtest.main(["--no-open"]) == 0
    out = capsys.readouterr().out
    assert "Kanal kırılımı 20/10" in out and "Kanal kırılımı" in out and "KARAR KAPISI" in out
    (dev_dir,) = [p for p in (tmp_path / "reports").iterdir() if p.name.endswith("_dev")]
    summary = json.loads((dev_dir / "ozet.json").read_text(encoding="utf-8"))
    assert summary["strategy"]["n_trades"] > 0
    assert backtest.main(["--period", "exam", "--yes", "--no-plot"]) == 0
    assert "SINAV DÖNEMİ" in capsys.readouterr().out

    end = client.now
    client.now = end - 3 * H4
    assert run.main(["--once"]) == 0
    client.now = end
    assert run.main(["--once"]) == 0
    assert run.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "Kanal kırılımı 20/10" in out and "Semboller (12)" in out
    assert "Paper trading özeti" in out and "Özsermaye : 50.00 ->" in out
