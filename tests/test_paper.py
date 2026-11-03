"""Paper trading: backtest ile birebir tutarlılık, gecikmeli veri ve yeniden başlatma.

Sahte borsa, oluşmakta olan mumu ZEHİRLİ değerlerle döndürür. Paper trader onu bir kez
bile kullansa sonuçlar backtest'ten sapar; bu testler o yüzden birebir eşitlik ister.
"""
import dataclasses

import pytest
from helpers import H1, M15, FakeSource, loose_rules, make_candles, resample, with_overrides

from bot.backtester import run_backtest
from bot.paper import PaperStateMismatch, PaperTrader
from bot.storage import StateStore
from bot.strategy import compute_signals

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
RULES = loose_rules(SYMBOLS)
N_BARS = 1500
START_BAR = 650          # ısınmadan sonra (test ayarında trend EMA50 -> 150 saatlik mum)
SYNC_DELAY = 5_000       # bot, mum kapanışından 5 sn sonra uyanır
STEP = 2 * H1            # testleri hızlandırmak için iki saatte bir uyanır (her turda 8 mum işler)


@pytest.fixture(scope="module")
def markets():
    btc = make_candles(N_BARS, seed=21, base=100.0)
    eth = make_candles(N_BARS, seed=22, base=50.0)
    return {"BTC/USDT": (btc, resample(btc, H1)), "ETH/USDT": (eth, resample(eth, H1))}


@pytest.fixture(scope="module")
def pcfg(cfg):
    # Hız için kısa trend EMA'sı; pencere tüm geçmişi kapsar ki göstergeler backtest ile bit bit aynı olsun.
    return with_overrides(
        cfg,
        strategy={"ema_trend_mid": 20, "ema_trend_slow": 50},
        paper={"history_bars": 5000, "initial_equity": 10_000.0},
    )


def source_for(markets, lag_ms=None) -> FakeSource:
    data = {}
    for symbol, (m15, m1h) in markets.items():
        data[(symbol, "15m")] = m15
        data[(symbol, "1h")] = m1h
    return FakeSource(data, lag_ms)


def bar_ts(markets, i: int) -> int:
    return int(markets["BTC/USDT"][0]["ts"].iloc[i])


def run_clock(trader: PaperTrader, first_moment: int, last_moment: int, step: int = STEP, skip=frozenset()) -> None:
    """Saati ilerletir: her adımda mum kapanışından 5 sn sonra senkronize eder; son anda mutlaka bir tur."""
    moment = first_moment
    while moment < last_moment:
        if moment not in skip:
            trader.sync(moment + SYNC_DELAY)
        moment += step
    trader.sync(last_moment + SYNC_DELAY)


@pytest.fixture(scope="module")
def reference(tmp_path_factory, pcfg, markets):
    """Hiç kesintisiz çalışan paper trader: diğer testlerin karşılaştırma ölçütü."""
    trader = PaperTrader(pcfg, source_for(markets), StateStore(tmp_path_factory.mktemp("ref") / "r.db"), RULES)
    run_clock(trader, bar_ts(markets, START_BAR), bar_ts(markets, N_BARS - 1))
    return trader


def backtest_same_window(pcfg, markets, last_ts: int):
    signals = {s: compute_signals(m15, m1h, pcfg) for s, (m15, m1h) in markets.items()}
    return run_backtest(pcfg, signals, bar_ts(markets, START_BAR), last_ts + M15, pcfg.paper.initial_equity, RULES)


def test_paper_matches_backtest_exactly(reference, pcfg, markets):
    bt = backtest_same_window(pcfg, markets, reference.last_ts)
    assert len(bt.trades) >= 3, "karşılaştırma anlamlı olsun diye işlem olmalı"
    assert reference.portfolio.trades == bt.trades
    assert reference.engine.equity_points[-1][1] == pytest.approx(float(bt.equity.iloc[-1]), abs=1e-6)
    assert reference.last_ts == bar_ts(markets, N_BARS - 1) - M15


@pytest.mark.parametrize(
    "lag_ms",
    [
        {"1h": 60_000},                       # saatlik mum 60 sn geç yayınlanıyor
        {("ETH/USDT", "15m"): 8_000},         # yalnızca ETH'nin 15 dk'lık mumları 8 sn geç
    ],
    ids=["saatlik-gecikme", "tek-sembol-gecikme"],
)
def test_delayed_exchange_data_does_not_change_results(tmp_path, pcfg, markets, reference, lag_ms):
    trader = PaperTrader(pcfg, source_for(markets, lag_ms=lag_ms), StateStore(tmp_path / "lag.db"), RULES)
    lag_seen = False
    moment, end = bar_ts(markets, START_BAR), bar_ts(markets, N_BARS - 1)
    while moment < end:
        trader.sync(moment + SYNC_DELAY)
        lag_seen |= trader.lagging
        moment += 90 * 60_000                       # saat başı ve buçuk anlarına sırayla denk gelir
    trader.sync(end + 120_000)                      # veriler yayınlandıktan sonra son tur
    assert lag_seen, "gecikme senaryosu gerçekten oluşmalı"
    assert trader.last_ts == reference.last_ts
    assert trader.portfolio.trades == reference.portfolio.trades


def test_waits_for_hourly_candle_before_processing(tmp_path, pcfg, markets):
    """Saat başında kapanan 15 dk'lık mum, aynı anda kapanan 1h mum gelmeden İŞLENMEZ."""
    trader = PaperTrader(pcfg, source_for(markets, lag_ms={"1h": 60_000}), StateStore(tmp_path / "w.db"), RULES)
    c = bar_ts(markets, 1000)                          # saat başı
    trader.sync(c - H1 + 120_000)                      # başlangıç: c-1h'ye kadar her şey yayınlandı
    assert trader.last_ts == c - H1 - M15
    assert trader.sync(c + SYNC_DELAY) == [c - H1, c - 3 * M15, c - 2 * M15]   # c-15m bekletildi
    assert trader.lagging
    assert trader.sync(c + 65_000) == [c - M15]        # saatlik mum geldi -> işlendi
    assert not trader.lagging


def test_restart_resumes_identically(tmp_path, pcfg, markets, reference):
    start, middle, end = (bar_ts(markets, i) for i in (START_BAR, 1100, N_BARS - 1))
    store = StateStore(tmp_path / "b.db")
    first = PaperTrader(pcfg, source_for(markets), store, RULES)
    run_clock(first, start, middle)
    second = PaperTrader(pcfg, source_for(markets), store, RULES)    # "bilgisayar yeniden başladı"
    assert second.resumed and second.last_ts == first.last_ts
    run_clock(second, middle + STEP, end)

    assert second.portfolio.trades == reference.portfolio.trades
    assert store.load_trades() == reference.portfolio.trades
    assert second.portfolio.cash == pytest.approx(reference.portfolio.cash)
    assert second.equity() == pytest.approx(reference.equity())


def test_missed_bars_are_caught_up(tmp_path, pcfg, markets, reference):
    offline = {bar_ts(markets, i) for i in range(900, 1100)}      # ~50 saat kapalı kaldı
    gappy = PaperTrader(pcfg, source_for(markets), StateStore(tmp_path / "g.db"), RULES)
    run_clock(gappy, bar_ts(markets, START_BAR), bar_ts(markets, N_BARS - 1), skip=offline)
    assert gappy.portfolio.trades == reference.portfolio.trades


def test_unclosed_candle_is_never_used(tmp_path, pcfg, markets):
    t = bar_ts(markets, 1000)
    trader = PaperTrader(pcfg, source_for(markets), StateStore(tmp_path / "u.db"), RULES)
    trader.sync(t + SYNC_DELAY)                  # başlangıç: t - 15m'ye kadar olanlar ısınma
    assert trader.sync(t + M15 - 1) == []        # t mumu henüz kapanmadı
    assert trader.sync(t + M15 + 1000) == [t]    # kapandıktan sonra işlenir


def test_changed_symbols_are_detected(tmp_path, pcfg, markets):
    store = StateStore(tmp_path / "m.db")
    trader = PaperTrader(pcfg, source_for(markets), store, RULES)
    trader.sync(bar_ts(markets, 1000) + SYNC_DELAY)
    other = dataclasses.replace(pcfg, symbols=["BTC/USDT"])
    with pytest.raises(PaperStateMismatch):
        PaperTrader(other, source_for(markets), store, RULES)
