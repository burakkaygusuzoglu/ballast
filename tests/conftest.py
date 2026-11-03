"""Ortak test verileri. Testler internete bağlanmaz."""
from __future__ import annotations

import pandas as pd
import pytest

from bot.config import Config, load_config
from helpers import H1, TEST_CONFIG_PATH, make_candles, resample


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Testlerin sabit ayarı (pullback, BTC/ETH, 15m/1h). Botun kendi ayarı değişse de testler değişmez."""
    return load_config(TEST_CONFIG_PATH)


@pytest.fixture(scope="session")
def market_15m() -> pd.DataFrame:
    return make_candles(6000, seed=7)


@pytest.fixture(scope="session")
def market_1h(market_15m: pd.DataFrame) -> pd.DataFrame:
    return resample(market_15m, H1)
