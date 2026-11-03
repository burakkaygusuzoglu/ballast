"""8. tur verisi: BTC/ETH günlük mumları, funding geçmişi (sayfalı) ve vadeli sembol kuralları."""
import json

import pandas as pd
from helpers import FakeClient, make_candles, with_overrides

import tur8_veri as T8
from bot.data import candle_path
from bot.timeutil import DAY_MS, parse_utc


class FakeFutures:
    def __init__(self, rates: dict[str, list[tuple[int, float]]]):
        self.rates = rates
        self.calls = 0

    def fetch_funding_rate_history(self, perp, since=None, limit=1000):
        self.calls += 1
        rows = [r for r in self.rates[perp] if r[0] >= since][:limit]
        return [{"timestamp": ts, "fundingRate": rate} for ts, rate in rows]

    def load_markets(self):
        return {}

    def market(self, perp):
        return {"id": perp.split(":")[0].replace("/", ""), "limits": {"cost": {"min": 100.0 if "BTC" in perp else 20.0},
                                                                     "amount": {"min": 0.001}},
                "precision": {"amount": 0.001}}


def test_downloads_daily_candles_funding_pages_and_rules(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(T8, "FUNDING_LIMIT", 5)
    c = with_overrides(cfg, data={"dir": str(tmp_path)})
    now = parse_utc("2020-01-10")
    days = (now - parse_utc("2017-08-17")) // DAY_MS
    data = {(s, "1d"): make_candles(days, seed=i, start="2017-08-17", tf_ms=DAY_MS) for i, s in enumerate(T8.SPOT_SYMBOLS)}
    start = parse_utc("2019-09-10")
    rates = {perp: [(start + k * 8 * 3_600_000, 0.0001 * (k % 3 - 1)) for k in range(12)] for perp in T8.PERPS.values()}
    rates["BTC/USDT:USDT"].append((now + 1, 0.5))                      # gelecekteki kayıt alınmaz
    lines = []
    fut = FakeFutures(rates)
    summary = T8.download(FakeClient(data, now), fut, c, echo=lines.append)
    for s in T8.SPOT_SYMBOLS:
        df = pd.read_csv(candle_path(tmp_path, "binance", s, "1d"))
        assert len(df) == days and int(df["ts"].iloc[0]) == parse_utc("2017-08-17")       # kapanmış mumların hepsi
    btc = pd.read_csv(tmp_path / "tur8" / "funding_BTCUSDT.csv")
    assert len(btc) == 12 and btc["ts"].is_monotonic_increasing and btc["ts"].max() < now
    assert fut.calls >= 6                                              # 5'erli sayfalar
    rules = json.loads((tmp_path / "tur8" / "vadeli_kurallar.json").read_text(encoding="utf-8"))
    assert rules["BTC/USDT"]["en_kucuk_tutar"] == 100.0 and rules["ETH/USDT"]["vadeli_sembol"] == "ETHUSDT"
    assert summary["funding"]["ETH/USDT"]["kayit"] == 12 and any("funding" in line for line in lines)


def test_funding_history_stops_without_progress():
    class Stuck:
        def fetch_funding_rate_history(self, perp, since=None, limit=1000):
            return [{"timestamp": 5, "fundingRate": 0.1}] * limit
    assert T8.funding_history(Stuck(), "X", since_ms=10, until_ms=100) == [(5, 0.1)]
