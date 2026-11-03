"""Ayarlar: gerçek dosya geçerli olmalı; hatalı ayarlar bot çalışmadan yakalanmalı."""
import pytest
import yaml

from bot.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def write(tmp_path, mutate) -> str:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_shipped_config_is_locked_candidate_c1():
    """Botun ayarı, sınavdan önce kilitlenen C1 ile aynı olmalı (docs/ARASTIRMA.md, 8. bölüm)."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.symbols == ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT", "DOT/USDT",
                           "AVAX/USDT", "DOGE/USDT", "SHIB/USDT", "UNI/USDT", "ALGO/USDT"]
    assert cfg.timeframes.entry == "4h" and cfg.entry_ms == 4 * 3_600_000
    s, r = cfg.strategy, cfg.risk
    assert (s.type, s.channel_entry, s.channel_exit, s.channel_atr, s.max_cost_r) == ("breakout", 20, 10, 20, 0.25)
    assert s.exit_on_regime_break is False and cfg.execution.entry_order == "market"
    assert (r.risk_per_trade, r.stop_atr_mult, r.trail_atr_mult, r.max_position_pct) == (0.005, 2.0, 0.0, 0.5)
    assert (r.max_open_positions, r.max_trades_per_day, r.cooldown_bars) == (5, 4, 4)
    assert (r.daily_loss_limit, r.max_drawdown_halt) == (0.03, 0.20)
    assert cfg.round_trip_cost == pytest.approx(2 * (0.001 + 0.0005))
    assert cfg.data.exam_start == "2026-04-01"


def test_test_config_is_pullback_baseline():
    from helpers import TEST_CONFIG_PATH
    cfg = load_config(TEST_CONFIG_PATH)
    assert cfg.strategy.type == "pullback" and cfg.symbols == ["BTC/USDT", "ETH/USDT"]
    assert cfg.entry_ms == 900_000 and cfg.risk.risk_per_trade == 0.01


def test_typo_in_key_is_reported(tmp_path):
    def mutate(d):
        d["risk"]["risk_per_trad"] = d["risk"].pop("risk_per_trade")
    with pytest.raises(ConfigError, match="risk_per_trad"):
        load_config(write(tmp_path, mutate))


def test_missing_key_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="eksik"):
        load_config(write(tmp_path, lambda d: d["costs"].pop("taker_fee")))


@pytest.mark.parametrize(
    "section, key, value, fragment",
    [
        ("risk", "risk_per_trade", 0.2, "risk_per_trade"),
        ("risk", "max_position_pct", 2.0, "kaldıraç"),
        ("strategy", "rsi_pullback_level", 60, "rsi_pullback_level"),
        ("strategy", "max_cost_r", 0, "max_cost_r"),
        ("costs", "taker_fee", 0.05, "taker_fee"),
        ("execution", "entry_order", "stop", "entry_order"),
        ("risk", "max_open_positions", "iki", "tam sayı"),
        ("data", "exam_start", "2022-02-01", "exam_start"),
        ("data", "start", "01.01.2022", "data.start"),
        ("risk", "trail_atr_mult", -1.0, "trail_atr_mult"),
        ("strategy", "type", "grid", "strategy.type"),
        ("strategy", "channel_entry", 1, "channel_entry"),
        ("paper", "history_bars", 50, "history_bars"),
        ("risk", "stop_atr_mult", 0, "stop_atr_mult"),
    ],
)
def test_bad_values_are_rejected(tmp_path, section, key, value, fragment):
    def mutate(d):
        d[section][key] = value
    with pytest.raises(ConfigError, match=fragment):
        load_config(write(tmp_path, mutate))


def test_unquoted_yaml_date_is_accepted(tmp_path):
    path = tmp_path / "config.yaml"
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace('start: "2022-01-01"', "start: 2022-01-01")
    path.write_text(text, encoding="utf-8")
    assert load_config(path).data.start == "2022-01-01"


def test_non_usdt_symbol_rejected(tmp_path):
    with pytest.raises(ConfigError, match="USDT"):
        load_config(write(tmp_path, lambda d: d.__setitem__("symbols", ["BTC/EUR"])))


def test_broken_yaml_is_reported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("risk:\n  risk_per_trade: 0.01\n   stop_atr_mult: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="okunamadı"):
        load_config(path)


def test_trailing_stop_zero_is_allowed(tmp_path):
    cfg = load_config(write(tmp_path, lambda d: d["risk"].__setitem__("trail_atr_mult", 0)))
    assert cfg.risk.trail_atr_mult == 0
