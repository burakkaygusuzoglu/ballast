"""Veri katmanı: birleştirme, kapanmamış mum, boşluklar, dosya ve artımlı güncelleme."""
import pandas as pd
import pytest

from bot.config import MarketRules
from bot.data import (
    candle_path, candles_from_rows, drop_unclosed, find_gaps, load_candles, load_market_rules,
    merge_candles, save_candles, save_market_rules, update_candles,
)
from helpers import H1, M15, FakeClient, make_candles, resample


def test_rows_are_sorted_and_deduplicated():
    rows = [[2 * M15, 1, 2, 0.5, 1.5, 10], [M15, 1, 2, 0.5, 1.5, 10], [2 * M15, 9, 9, 9, 9, 9]]
    df = candles_from_rows(rows)
    assert df["ts"].tolist() == [M15, 2 * M15]
    assert df["close"].iloc[-1] == 9            # aynı mum tekrar gelirse son hali geçerli


def test_merge_keeps_latest_version():
    old = candles_from_rows([[0, 1, 1, 1, 1, 1], [M15, 2, 2, 2, 2, 2]])
    new = candles_from_rows([[M15, 3, 3, 3, 3, 3], [2 * M15, 4, 4, 4, 4, 4]])
    merged = merge_candles(old, new)
    assert merged["ts"].tolist() == [0, M15, 2 * M15]
    assert merged["close"].tolist() == [1, 3, 4]


def test_drop_unclosed():
    df = candles_from_rows([[0, 1, 1, 1, 1, 1], [M15, 1, 1, 1, 1, 1]])
    assert drop_unclosed(df, M15, 2 * M15 - 1)["ts"].tolist() == [0]
    assert drop_unclosed(df, M15, 2 * M15)["ts"].tolist() == [0, M15]


def test_gaps_are_found():
    df = candles_from_rows([[0, 1, 1, 1, 1, 1], [M15, 1, 1, 1, 1, 1], [4 * M15, 1, 1, 1, 1, 1]])
    assert find_gaps(df, M15) == [(M15, 4 * M15, 2)]


def test_save_load_roundtrip(tmp_path):
    df = make_candles(50, seed=3)
    path = tmp_path / "x.csv"
    save_candles(df, path)
    loaded = load_candles(path)
    pd.testing.assert_frame_equal(loaded, df.astype(loaded.dtypes.to_dict()), check_exact=True)
    assert not path.with_suffix(".tmp").exists()


def test_update_candles_initial_then_incremental(tmp_path):
    m15 = make_candles(3000, seed=4)
    data = {("BTC/USDT", "15m"): m15, ("BTC/USDT", "1h"): resample(m15, H1)}
    start = int(m15["ts"].iloc[0])
    client = FakeClient(data, now_ms=int(m15["ts"].iloc[2000]) + 60_000)   # 2000. mum oluşuyor

    first = update_candles(client, "BTC/USDT", "15m", M15, start, tmp_path)
    assert first["ts"].iloc[-1] == m15["ts"].iloc[1999]                     # oluşan mum alınmadı
    pd.testing.assert_frame_equal(first, m15.iloc[:2000].reset_index(drop=True).astype(first.dtypes.to_dict()))

    client.now = int(m15["ts"].iloc[2999]) + M15                             # sonraki gün
    calls_before = client.source.calls
    second = update_candles(client, "BTC/USDT", "15m", M15, start, tmp_path)
    assert client.source.calls == calls_before + 1                           # yalnızca eksik kısım
    pd.testing.assert_frame_equal(second, m15.reset_index(drop=True).astype(second.dtypes.to_dict()))
    assert candle_path(tmp_path, "binance", "BTC/USDT", "15m").exists()


def test_market_rules_roundtrip_and_default(tmp_path):
    default = MarketRules(qty_step=0.001, min_notional=10.0)
    save_market_rules(tmp_path, {"BTC/USDT": MarketRules(qty_step=0.00001, min_notional=5.0)})
    rules = load_market_rules(tmp_path, ["BTC/USDT", "ETH/USDT"], default)
    assert rules["BTC/USDT"] == MarketRules(qty_step=0.00001, min_notional=5.0)
    assert rules["ETH/USDT"] == default


def test_saving_rules_keeps_other_symbols(tmp_path):
    save_market_rules(tmp_path, {"BTC/USDT": MarketRules(qty_step=0.00001, min_notional=5.0)})
    save_market_rules(tmp_path, {"SOL/USDT": MarketRules(qty_step=0.001, min_notional=5.0)})
    rules = load_market_rules(tmp_path, ["BTC/USDT", "SOL/USDT"], MarketRules(qty_step=1.0, min_notional=1.0))
    assert rules["BTC/USDT"].qty_step == 0.00001 and rules["SOL/USDT"].qty_step == 0.001
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("bad", ["ts,open\n1,2\n"])
def test_corrupt_file_is_reported(tmp_path, bad):
    path = tmp_path / "bad.csv"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="eksik"):
        load_candles(path)
