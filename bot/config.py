"""Ayarların yüklenmesi ve doğrulanması.

- config/config.yaml : strateji, risk ve çalışma ayarları (git'e girer)
- .env               : yalnızca canlı mod için API anahtarları (asla git'e girmez)

Yükleme sırasında her ayar doğrulanır. Yazım hatası (bilinmeyen anahtar), eksik ayar
ya da mantıksız bir değer, bot hiç çalışmadan açık bir Türkçe mesajla yakalanır.
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from bot.timeutil import TIMEFRAME_MS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
CONFIG_ENV_VAR = "KRIPTO_BOT_CONFIG"   # testler ve ileri kullanım için farklı ayar dosyası
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
MIN_DEV_DAYS = 90


class ConfigError(ValueError):
    """config.yaml içinde eksik, fazla ya da hatalı ayar."""


@dataclass(frozen=True)
class ExchangeConfig:
    id: str
    market_type: str
    timeout_ms: int


@dataclass(frozen=True)
class TimeframeConfig:
    entry: str
    trend: str


@dataclass(frozen=True)
class DataConfig:
    dir: str
    start: str
    exam_start: str


@dataclass(frozen=True)
class CostConfig:
    maker_fee: float
    taker_fee: float
    slippage: float


MARKET_TYPES = ("spot", "futures")
LIQUIDATION_MODELS = ("open", "touch")
STRATEGY_TYPES = ("breakout", "pullback", "momentum")
SIZING_MODES = ("risk", "allocation")
REGIME_SYMBOL = "BTC/USDT"   # rejim filtresi ve momentum için piyasa göstergesi


@dataclass(frozen=True)
class StrategyConfig:
    type: str                 # "breakout": kanal kırılımı (trend takibi) | "pullback": 3 katmanlı geri çekilme
    channel_entry: int        # breakout: kapanış önceki N mumun en yükseğini aşarsa al
    channel_exit: int         # breakout: kapanış önceki N mumun en düşüğünün altına inerse sat
    channel_atr: int          # breakout: ilk stop için ATR periyodu
    ema_trend_mid: int
    ema_trend_slow: int
    ema_fast: int
    rsi_period: int
    rsi_pullback_level: float
    rsi_trigger_level: float
    pullback_lookback: int
    bb_period: int
    bb_std: float
    atr_period: int
    volume_ma_period: int
    obv_ema_period: int
    max_cost_r: float
    exit_on_regime_break: bool
    # 3. tur (isteğe bağlı; yazılmazsa C1 davranışı):
    regime_sma: int = 0             # >0: yeni alım yalnızca BTC günlük kapanışı SMA(n) üstündeyken (0 = kapalı)
    momentum_lookback: int = 28     # momentum: getiri penceresi (gün)
    momentum_top_n: int = 3         # momentum: en güçlü ilk N coin alınır
    momentum_exit_rank: int = 5     # momentum: bu sıranın dışına düşen satılır
    # 6. tur (yalnızca vadeli işlem): aynalı short tarafı. Rejim filtresi açıksa short yalnızca BTC günlük
    # kapanışı SMA'nın ALTINDAYKEN açılır (long yalnızca üstündeyken).
    allow_short: bool = False


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float
    stop_atr_mult: float
    trail_atr_mult: float
    max_position_pct: float
    max_open_positions: int
    max_trades_per_day: int
    cooldown_bars: int
    daily_loss_limit: float
    max_drawdown_halt: float
    # Piramitleme (Kaplumbağa): kârdaki pozisyona birim ekleme. 1 = kapalı (C1). Yalnızca simülasyon broker'ı
    # (backtest/paper) destekler; canlı broker'da etkisizdir.
    pyramid_units: int = 1          # coin başına en fazla birim (ilk alım dahil)
    pyramid_step_atr: float = 0.5   # son alımdan bu kadar ATR yükselince bir birim daha
    max_total_units: int = 10       # hesaptaki toplam birim sınırı
    # Boyutlandırma: "risk" = stop çalışırsa kayıp risk_per_trade kadar (C1); "allocation" = her pozisyon
    # özsermayenin max_position_pct kadarı (L1, uzun vadeli trend; risk_per_trade kullanılmaz).
    sizing: str = "risk"
    # 7. tur: düşüşe göre ölçeklenen risk (Grossman-Zhou). 0 = kapalı (C1). >0 ise işlem başına risk =
    # risk_per_trade x max(0, 1 - tepeden düşüş / dd_scaling); ör. 0.25: %12,5 düşüşte risk yarıya iner.
    dd_scaling: float = 0.0


@dataclass(frozen=True)
class ExecutionConfig:
    entry_order: str
    entry_ttl_bars: int


@dataclass(frozen=True)
class MarketRules:
    qty_step: float
    min_notional: float


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float
    reports_dir: str
    open_chart: bool


@dataclass(frozen=True)
class PaperConfig:
    initial_equity: float
    history_bars: int
    poll_buffer_seconds: int
    name: str = "paper"             # durum ve günlük dosyalarının adı (gölge botlar için farklı ad)


@dataclass(frozen=True)
class LiveConfig:
    testnet: bool             # true: Binance test ortamı (sahte para) | false: GERÇEK PARA
    max_capital: float        # botun kullanabileceği en fazla USDT (spot cüzdandaki serbest bakiyeden)


@dataclass(frozen=True)
class GateConfig:
    min_profit_factor: float
    max_drawdown: float
    min_trades_dev: int
    require_positive_exam: bool
    require_beat_benchmark: bool


@dataclass(frozen=True)
class LoggingConfig:
    dir: str
    level: str


@dataclass(frozen=True)
class FuturesConfig:
    """Vadeli işlem (USDⓈ-M sürekli sözleşme, izole marjin) modeli. Yalnızca exchange.market_type: futures."""
    leverage: float = 3.0               # her pozisyonun kaldıracı: marjin = pozisyon büyüklüğü / kaldıraç
    maintenance_margin: float = 0.02    # bakım marjini oranı (likidasyon fiyatı için; kötümser)
    funding_long_8h: float = 0.0001     # long'un 8 saatte ödediği oran (negatif = alır)
    funding_short_8h: float = -0.0001   # short'un 8 saatte ödediği oran (negatif = alır)
    liquidation: str = "open"           # "open": mum likidasyon fiyatının ötesinde açılırsa | "touch": ucu değerse
    max_stop_to_liq: float = 0.8        # stop mesafesi likidasyon mesafesinin bu oranını aşarsa işlem açılmaz


@dataclass(frozen=True)
class Config:
    exchange: ExchangeConfig
    symbols: list[str]
    timeframes: TimeframeConfig
    data: DataConfig
    costs: CostConfig
    strategy: StrategyConfig
    risk: RiskConfig
    execution: ExecutionConfig
    market_rules_default: MarketRules
    backtest: BacktestConfig
    paper: PaperConfig
    live: LiveConfig
    gate: GateConfig
    logging: LoggingConfig
    futures: FuturesConfig = field(default_factory=FuturesConfig)

    # ---- türetilmiş değerler ------------------------------------------------
    @property
    def is_futures(self) -> bool:
        return self.exchange.market_type == "futures"

    @property
    def entry_ms(self) -> int:
        return TIMEFRAME_MS[self.timeframes.entry]

    @property
    def trend_ms(self) -> int:
        return TIMEFRAME_MS[self.timeframes.trend]

    @property
    def entry_fee_rate(self) -> float:
        """Girişte ödenen maliyet oranı: limit emir (maker) ya da piyasa emri (taker + kayma)."""
        if self.execution.entry_order == "limit":
            return self.costs.maker_fee
        return self.costs.taker_fee + self.costs.slippage

    @property
    def exit_fee_rate(self) -> float:
        """Çıkış maliyeti: stoplar piyasa emri gibi dolar (taker + kayma)."""
        return self.costs.taker_fee + self.costs.slippage

    @property
    def round_trip_cost(self) -> float:
        """Kötümser gidiş-dönüş maliyeti (oran): giriş + çıkış komisyonu ve kayma."""
        return self.entry_fee_rate + self.exit_fee_rate

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.data.dir)

    @property
    def log_dir(self) -> Path:
        return self.resolve(self.logging.dir)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.backtest.reports_dir)


# ---------------------------------------------------------------------------
# Yükleme
# ---------------------------------------------------------------------------
def _coerce(value: Any, hint: Any, where: str) -> Any:
    if get_origin(hint) is list:
        (item_hint,) = get_args(hint)
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{where}: en az bir elemanlı liste olmalı.")
        return [_coerce(v, item_hint, f"{where}[{i}]") for i, v in enumerate(value)]
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: true ya da false olmalı (şu an: {value!r}).")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: tam sayı olmalı (şu an: {value!r}).")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: sayı olmalı (şu an: {value!r}).")
        return float(value)
    if hint is str:
        if isinstance(value, (_dt.date, _dt.datetime)):
            return value.isoformat()
        if not isinstance(value, str):
            raise ConfigError(f"{where}: metin olmalı (şu an: {value!r}).")
        return value
    raise ConfigError(f"{where}: desteklenmeyen ayar türü {hint!r}.")


def _build(cls: type, data: Any, section: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"'{section}' bölümü eksik ya da hatalı.")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(
            f"'{section}' bölümünde tanınmayan ayar(lar): {', '.join(map(str, unknown))} "
            "(yazım hatası olabilir)."
        )
    optional = {f.name for f in fields(cls) if f.default is not MISSING or f.default_factory is not MISSING}
    missing = [n for n in names if n not in data and n not in optional]
    if missing:
        raise ConfigError(f"'{section}' bölümünde eksik ayar(lar): {', '.join(missing)}.")
    kwargs: dict[str, Any] = {}
    for name in names:
        if name not in data:          # varsayılanı olan isteğe bağlı ayar
            continue
        hint = hints[name]
        where = name if section == "config.yaml" else f"{section}.{name}"
        if isinstance(hint, type) and is_dataclass(hint):
            kwargs[name] = _build(hint, data[name], where)
        else:
            kwargs[name] = _coerce(data[name], hint, where)
    return cls(**kwargs)


def _parse_date(text: str, where: str, errors: list[str]) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(text)
    except ValueError:
        errors.append(f"{where}: 'YYYY-AA-GG' biçiminde bir tarih olmalı (örnek: \"2022-01-01\").")
        return None


def _validate(cfg: Config) -> None:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(cfg.exchange.market_type in MARKET_TYPES, f"exchange.market_type: {' ya da '.join(MARKET_TYPES)} olmalı.")
    check(cfg.exchange.timeout_ms >= 1000, "exchange.timeout_ms: en az 1000 olmalı.")
    for i, symbol in enumerate(cfg.symbols):
        check(
            symbol.count("/") == 1 and symbol.endswith("/USDT"),
            f"symbols[{i}]: 'BTC/USDT' biçiminde bir USDT paritesi olmalı (şu an: {symbol!r}).",
        )
    check(len(set(cfg.symbols)) == len(cfg.symbols), "symbols: aynı sembol iki kez yazılmış.")

    tf_ok = cfg.timeframes.entry in TIMEFRAME_MS and cfg.timeframes.trend in TIMEFRAME_MS
    check(tf_ok, f"timeframes: geçerli değerler {', '.join(TIMEFRAME_MS)}.")
    if tf_ok and cfg.strategy.type == "momentum":    # momentum trend zaman dilimini kullanmaz
        check(cfg.trend_ms >= cfg.entry_ms, "timeframes: trend zaman dilimi, giriş zaman diliminden küçük olamaz.")
    elif tf_ok:
        check(
            cfg.trend_ms > cfg.entry_ms and cfg.trend_ms % cfg.entry_ms == 0,
            "timeframes: trend zaman dilimi, giriş zaman diliminden büyük ve onun tam katı olmalı.",
        )

    start = _parse_date(cfg.data.start, "data.start", errors)
    exam = _parse_date(cfg.data.exam_start, "data.exam_start", errors)
    if start and exam:
        check(
            (exam - start).days >= MIN_DEV_DAYS,
            f"data.exam_start: data.start'tan en az {MIN_DEV_DAYS} gün sonra olmalı (geliştirme verisi için).",
        )

    for name in ("maker_fee", "taker_fee", "slippage"):
        check(0 <= getattr(cfg.costs, name) < 0.01, f"costs.{name}: 0 ile 0.01 (%1) arasında olmalı.")

    s = cfg.strategy
    check(s.type in STRATEGY_TYPES, f"strategy.type: {' ya da '.join(STRATEGY_TYPES)} olmalı.")
    for name in (
        "channel_entry", "channel_exit", "channel_atr",
        "ema_trend_mid", "ema_trend_slow", "ema_fast", "rsi_period", "bb_period",
        "atr_period", "volume_ma_period", "obv_ema_period", "pullback_lookback",
    ):
        check(getattr(s, name) >= 2, f"strategy.{name}: en az 2 olmalı.")
    check(s.ema_trend_mid < s.ema_trend_slow, "strategy: ema_trend_mid, ema_trend_slow'dan küçük olmalı.")
    check(
        0 < s.rsi_pullback_level < s.rsi_trigger_level < 100,
        "strategy: 0 < rsi_pullback_level < rsi_trigger_level < 100 olmalı.",
    )
    check(s.bb_std > 0, "strategy.bb_std: pozitif olmalı.")
    check(0 < s.max_cost_r <= 1, "strategy.max_cost_r: 0 ile 1 arasında olmalı (0.25 = riskin %25'i).")
    check(s.regime_sma == 0 or s.regime_sma >= 2, "strategy.regime_sma: 0 (kapalı) ya da en az 2 olmalı.")
    check(s.momentum_lookback >= 2, "strategy.momentum_lookback: en az 2 olmalı.")
    check(1 <= s.momentum_top_n <= s.momentum_exit_rank,
          "strategy: 1 <= momentum_top_n <= momentum_exit_rank olmalı.")
    if s.regime_sma or s.type == "momentum":
        check(REGIME_SYMBOL in cfg.symbols, f"symbols: rejim/momentum için {REGIME_SYMBOL} listede olmalı.")
    if s.regime_sma and s.type == "breakout":
        check(cfg.timeframes.trend == "1d", "timeframes.trend: rejim filtresi günlük mum kullanır ('1d' olmalı).")
    if s.type == "momentum":
        check(cfg.timeframes.entry == "1d", "timeframes.entry: momentum günlük mumlarla çalışır ('1d' olmalı).")
        check(s.regime_sma >= 2, "strategy.regime_sma: momentum BTC rejim filtresi gerektirir (ör. 200).")

    r = cfg.risk
    check(0 < r.risk_per_trade <= 0.05, "risk.risk_per_trade: 0 ile 0.05 (%5) arasında olmalı.")
    check(r.stop_atr_mult > 0, "risk.stop_atr_mult: pozitif olmalı.")
    check(r.trail_atr_mult >= 0, "risk.trail_atr_mult: negatif olamaz (0 = iz süren stop kapalı).")
    check(0 < r.max_position_pct <= 1, "risk.max_position_pct: 0 ile 1 arasında olmalı (1 = özsermayenin tamamı; spotta kaldıraç yok).")
    check(r.max_open_positions >= 1, "risk.max_open_positions: en az 1 olmalı.")
    check(r.max_trades_per_day >= 1, "risk.max_trades_per_day: en az 1 olmalı.")
    check(r.cooldown_bars >= 0, "risk.cooldown_bars: negatif olamaz.")
    check(r.sizing in SIZING_MODES, f"risk.sizing: {' ya da '.join(SIZING_MODES)} olmalı.")
    check(0 <= r.dd_scaling < 1, "risk.dd_scaling: 0 (kapalı) ile 1 arasında olmalı (0.25 = %25 düşüşte risk sıfır).")
    check(r.dd_scaling == 0 or r.dd_scaling <= r.max_drawdown_halt,
          "risk.dd_scaling: düşüş şalterinden (risk.max_drawdown_halt) büyük olamaz.")
    check(0 < r.daily_loss_limit < 1, "risk.daily_loss_limit: 0 ile 1 arasında olmalı (0.03 = %3).")
    check(0 < r.max_drawdown_halt < 1, "risk.max_drawdown_halt: 0 ile 1 arasında olmalı (0.20 = %20).")
    check(r.pyramid_units >= 1, "risk.pyramid_units: en az 1 olmalı (1 = piramitleme kapalı).")
    check(r.pyramid_step_atr > 0, "risk.pyramid_step_atr: pozitif olmalı.")
    check(r.max_total_units >= 1, "risk.max_total_units: en az 1 olmalı.")

    check(cfg.execution.entry_order in ("limit", "market"), "execution.entry_order: 'limit' ya da 'market' olmalı.")
    check(cfg.execution.entry_ttl_bars >= 1, "execution.entry_ttl_bars: en az 1 olmalı.")
    check(cfg.market_rules_default.qty_step > 0, "market_rules_default.qty_step: pozitif olmalı.")
    check(cfg.market_rules_default.min_notional >= 0, "market_rules_default.min_notional: negatif olamaz.")
    check(cfg.backtest.initial_equity > 0, "backtest.initial_equity: pozitif olmalı.")
    check(cfg.paper.initial_equity > 0, "paper.initial_equity: pozitif olmalı.")
    if s.type == "pullback":
        min_history, why = 3 * s.ema_trend_slow + 10, f"trend EMA{s.ema_trend_slow} ısınması için"
    elif s.type == "momentum":
        min_history, why = 3 * max(s.momentum_lookback, s.channel_atr) + 10, "momentum ve ATR ısınması için"
    else:
        min_history, why = 3 * max(s.channel_entry, s.channel_exit, s.channel_atr) + 10, "kanal ve ATR ısınması için"
    if s.regime_sma:
        min_history, why = max(min_history, s.regime_sma + 10), f"{why}; rejim SMA{s.regime_sma} için"
    check(
        min_history <= cfg.paper.history_bars <= 5000,
        f"paper.history_bars: {min_history} ile 5000 arasında olmalı ({why}).",
    )
    check(cfg.paper.poll_buffer_seconds >= 1, "paper.poll_buffer_seconds: en az 1 olmalı.")
    check(bool(cfg.paper.name) and cfg.paper.name.replace("_", "").isalnum(),
          "paper.name: yalnızca harf, rakam ve alt çizgi (ör. golge_r1).")
    check(cfg.live.max_capital > 0, "live.max_capital: pozitif olmalı (botun kullanabileceği en fazla USDT).")

    f = cfg.futures
    if cfg.is_futures:
        check(1 <= f.leverage <= 10, "futures.leverage: 1 ile 10 arasında olmalı (bu projede tavan 10x).")
        check(0 <= f.maintenance_margin < 0.1, "futures.maintenance_margin: 0 ile 0.1 arasında olmalı.")
        for name in ("funding_long_8h", "funding_short_8h"):
            check(abs(getattr(f, name)) < 0.01, f"futures.{name}: -0.01 ile 0.01 arasında olmalı (8 saatlik oran).")
        check(f.liquidation in LIQUIDATION_MODELS, f"futures.liquidation: {' ya da '.join(LIQUIDATION_MODELS)} olmalı.")
        check(0 < f.max_stop_to_liq <= 1, "futures.max_stop_to_liq: 0 ile 1 arasında olmalı.")
        check(cfg.execution.entry_order == "market", "execution.entry_order: vadeli işlemde yalnızca 'market'.")
        check(r.pyramid_units == 1, "risk.pyramid_units: vadeli işlemde piramitleme yok (1 olmalı).")
        check(r.sizing == "risk", "risk.sizing: vadeli işlemde yalnızca 'risk'.")
    check(not s.allow_short or cfg.is_futures, "strategy.allow_short: short yalnızca vadeli işlemde (futures).")
    check(not s.allow_short or s.type == "breakout", "strategy.allow_short: yalnızca kanal kırılımı (breakout).")

    g = cfg.gate
    check(g.min_profit_factor >= 1, "gate.min_profit_factor: en az 1 olmalı.")
    check(0 < g.max_drawdown < 1, "gate.max_drawdown: 0 ile 1 arasında olmalı.")
    check(g.min_trades_dev >= 1, "gate.min_trades_dev: en az 1 olmalı.")
    check(cfg.logging.level.upper() in VALID_LOG_LEVELS, f"logging.level: {', '.join(VALID_LOG_LEVELS)} olmalı.")

    if errors:
        raise ConfigError("config.yaml hataları:\n  - " + "\n  - ".join(errors))


def config_path() -> Path:
    """Kullanılacak ayar dosyası: KRIPTO_BOT_CONFIG ortam değişkeni ya da config/config.yaml."""
    override = os.environ.get(CONFIG_ENV_VAR, "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_config(path: str | Path | None = None) -> Config:
    """config.yaml'ı okur, türlerini dönüştürür ve tüm değerleri doğrular."""
    target = Path(path) if path else config_path()
    if not target.exists():
        raise ConfigError(f"Ayar dosyası bulunamadı: {target}")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml okunamadı (girinti ya da yazım hatası olabilir): {exc}") from exc
    cfg = _build(Config, raw, "config.yaml")
    _validate(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Gizli bilgiler (yalnızca canlı mod)
# ---------------------------------------------------------------------------
def load_api_keys(testnet: bool = False) -> tuple[str, str]:
    """.env dosyasından Binance API anahtarlarını okur (testnet ve gerçek hesap anahtarları ayrıdır)."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    prefix = "BINANCE_TESTNET_API" if testnet else "BINANCE_API"
    key = os.getenv(f"{prefix}_KEY", "").strip()
    secret = os.getenv(f"{prefix}_SECRET", "").strip()
    if not key or not secret:
        raise ConfigError(f".env içinde {prefix}_KEY / {prefix}_SECRET boş. README.md -> 'Canlı mod' adımlarına bakın.")
    return key, secret


def mask_secret(secret: str) -> str:
    """Anahtarları loglarda asla tam göstermeyiz."""
    return f"{secret[:4]}...{secret[-4:]}" if len(secret) > 8 else "****"
