"""Borsa bağlantısı (ccxt): yeniden deneme, hız limiti, sayfalama ve sunucu saati burada.

Veri indirme ve paper trading için API anahtarı GEREKMEZ (herkese açık piyasa verisi).
Anahtar yalnızca canlı mod (testnet ya da gerçek hesap) için gerekir.
"""
from __future__ import annotations

import logging
import time
from typing import Any
from collections.abc import Callable

import ccxt
from ccxt.base.decimal_to_precision import DECIMAL_PLACES

from bot.config import Config, MarketRules
from bot.timeutil import fmt_utc

# RequestTimeout, ExchangeNotAvailable, DDoSProtection ve RateLimitExceeded bunun alt sınıfıdır.
RETRYABLE_ERRORS = (ccxt.NetworkError,)


class ExchangeClient:
    def __init__(
        self,
        cfg: Config,
        api_key: str | None = None,
        api_secret: str | None = None,
        logger: logging.Logger | None = None,
        max_retries: int = 5,
        testnet: bool = False,
    ):
        exchange_class = getattr(ccxt, cfg.exchange.id, None)
        if exchange_class is None:
            raise ValueError(f"ccxt bu borsayı tanımıyor: {cfg.exchange.id}")
        params: dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": cfg.exchange.timeout_ms,
            "options": {"defaultType": cfg.exchange.market_type},
        }
        if api_key and api_secret:
            params.update(apiKey=api_key, secret=api_secret)
        self.ex = exchange_class(params)
        self.testnet = testnet
        if testnet:
            self.ex.set_sandbox_mode(True)     # Binance spot test ortamı: sahte para, ayrı anahtarlar
        self.id = cfg.exchange.id
        self.log = logger or logging.getLogger("bot.exchange")
        self.max_retries = max_retries
        self.time_offset_ms = 0     # sunucu saati - yerel saat

    def _call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except RETRYABLE_ERRORS as exc:
                if attempt == self.max_retries:
                    raise
                self.log.warning(f"Ağ hatası ({attempt}/{self.max_retries}): {exc} - {delay:.0f} sn sonra tekrar")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("erişilemez")  # pragma: no cover

    # ---- saat ----------------------------------------------------------------
    def milliseconds(self) -> int:
        return int(self.ex.milliseconds())

    def refresh_time_offset(self) -> int:
        """Yerel saat ile Binance saati arasındaki farkı ölçer (ağ gecikmesinin yarısı düşülür)."""
        before = self.milliseconds()
        server = int(self._call(self.ex.fetch_time))
        after = self.milliseconds()
        self.time_offset_ms = server - (before + after) // 2
        return self.time_offset_ms

    def server_now(self) -> int:
        """Sunucu saatine göre 'şimdi'. Mumun kapanıp kapanmadığı buna göre belirlenir."""
        return self.milliseconds() + self.time_offset_ms

    # ---- piyasa verisi ------------------------------------------------------
    def last_price(self, symbol: str) -> float:
        return float(self._call(self.ex.fetch_ticker, symbol)["last"])

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None, limit: int = 1000) -> list[list]:
        return self._call(self.ex.fetch_ohlcv, symbol, timeframe, since, limit)

    def fetch_ohlcv_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, label: str = "") -> list[list]:
        """[since_ms, until_ms) aralığındaki mumları 1000'erli sayfalarla indirir."""
        tf_ms = int(self.ex.parse_timeframe(timeframe) * 1000)
        rows: list[list] = []
        cursor = since_ms
        pages = 0
        while cursor < until_ms:
            batch = self.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            last = int(batch[-1][0])
            if last < cursor:
                break
            cursor = last + tf_ms
            pages += 1
            if label and pages % 25 == 0:
                self.log.info(f"  {label}: {len(rows):,} mum indirildi (son: {fmt_utc(last)})")
        return [r for r in rows if since_ms <= int(r[0]) < until_ms]

    def api_restrictions(self) -> dict[str, Any]:
        """Anahtarın izinleri (yalnızca gerçek hesapta): para çekme, işlem, IP kısıtlaması."""
        return self._call(self.ex.sapi_get_account_apirestrictions)

    def markets(self) -> dict[str, dict]:
        """Borsadaki pariteler (ccxt biçiminde; 'active' ve 'spot' alanlarıyla)."""
        return self._call(self.ex.load_markets)

    def market_rules(self, symbols: list[str], default: MarketRules) -> dict[str, MarketRules]:
        """Miktar adımı ve minimum işlem tutarını borsadan okur."""
        markets = self.markets()
        rules: dict[str, MarketRules] = {}
        for symbol in symbols:
            market = markets.get(symbol)
            if market is None:
                raise ValueError(f"{self.id} üzerinde sembol bulunamadı: {symbol}")
            amount = (market.get("precision") or {}).get("amount")
            if amount is not None and self.ex.precisionMode == DECIMAL_PLACES:
                amount = 10 ** (-int(amount))
            min_cost = ((market.get("limits") or {}).get("cost") or {}).get("min")
            rules[symbol] = MarketRules(
                qty_step=float(amount) if amount else default.qty_step,
                min_notional=float(min_cost) if min_cost else default.min_notional,
            )
        return rules
