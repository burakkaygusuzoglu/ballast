"""4. tur veri hazırlığı: geniş evren 2. turdaki kuralla seçilir, BTC ve ETH her zaman indirilir."""
from helpers import H1, FakeClient, make_candles, with_overrides

import arastirma_verisi as V
import tur4_veri as T4
from bot.data import candle_path
from bot.timeutil import parse_utc


class UniverseClient(FakeClient):
    def __init__(self, data, now_ms, markets):
        super().__init__(data, now_ms)
        self._markets = markets

    def markets(self):
        return self._markets


def test_round4_downloads_btc_eth_and_selects_by_rank(cfg, tmp_path, monkeypatch):
    for name in ("CANDIDATES", "UNIVERSE_SIZE", "UNIVERSE_FILE"):       # modül ayarları test sonunda geri gelir
        monkeypatch.setattr(V, name, getattr(V, name))
    monkeypatch.setattr(T4, "TUR4_CANDIDATES", ("AAA", "GONE", "BBB", "HOLE", "CCC", "DDD"))
    monkeypatch.setattr(T4, "TUR4_SIZE", 3)
    c = with_overrides(cfg, data={"exam_start": "2022-03-01", "dir": str(tmp_path)})
    bars = (parse_utc("2022-03-10") - parse_utc("2022-01-01")) // H1
    full = lambda seed: make_candles(bars, seed=seed, start="2022-01-01", tf_ms=H1)
    data = {(s, "1h"): full(i) for i, s in enumerate(["BTC/USDT", "ETH/USDT", "AAA/USDT", "BBB/USDT", "CCC/USDT",
                                                      "DDD/USDT"])}
    data[("HOLE/USDT", "1h")] = full(9).drop(index=range(300, 300 + 24 * 9)).reset_index(drop=True)
    markets = {s: {"spot": True, "active": True} for s, _ in data}
    lines = []
    universe = T4.download_round4(UniverseClient(data, parse_utc("2022-03-10"), markets), c, echo=lines.append)
    assert universe["her_zaman"] == ["BTC/USDT", "ETH/USDT"]
    assert universe["secilen"] == ["AAA/USDT", "BBB/USDT", "CCC/USDT"]           # sıradaki ilk 3 uygun coin
    assert set(universe["elenen"]) == {"GONE/USDT", "HOLE/USDT"} and universe["denenmeyen"] == ["DDD/USDT"]
    assert universe["kural"] == "docs/ARASTIRMA.md, 13. bölüm"
    for s in ("BTC/USDT", "ETH/USDT", "AAA/USDT"):
        assert candle_path(tmp_path, "binance", s, "1h").exists()
    V.save_universe(c, universe)
    assert (tmp_path / "evren_tur4.json").exists() and not (tmp_path / "evren_tur2.json").exists()


def test_round4_candidates_extend_round2_list_in_order():
    assert T4.TUR4_CANDIDATES[: len(V.CANDIDATES)] == V.CANDIDATES
    assert len(set(T4.TUR4_CANDIDATES)) == len(T4.TUR4_CANDIDATES) == 33
    assert not {"BTC", "ETH", "USDT", "USDC", "BUSD", "DAI", "UST", "WBTC", "BTCB"} & set(T4.TUR4_CANDIDATES)
