"""Risk yöneticisi: elle hesaplanmış senaryolar."""
import pytest

from bot.config import MarketRules
from bot.risk import RiskManager, floor_to_step
from bot.timeutil import DAY_MS, parse_utc
from helpers import M15, with_overrides

RULES = MarketRules(qty_step=0.00001, min_notional=5.0)
DAY0 = parse_utc("2024-03-01")


def test_position_size_includes_costs_hand_calculated(cfg):
    # 1000 USDT, %1 risk, giriş 100, stop 98; giriş maker %0.1, çıkış taker %0.1 + kayma %0.05
    risk = RiskManager(with_overrides(cfg, risk={"max_position_pct": 1.0}), 1000)
    qty, planned = risk.size(1000, 1000, 100.0, 98.0, RULES)
    loss_per_unit = 2.0 + 100 * 0.001 + 98 * 0.0015            # = 2.247
    assert qty == pytest.approx(floor_to_step(10 / loss_per_unit, 0.00001))
    assert planned == pytest.approx(qty * loss_per_unit)
    assert planned <= 10.0 + 1e-9                                # risk asla bütçeyi aşmaz


def test_position_capped_by_max_position_pct(cfg):
    risk = RiskManager(cfg, 1000)                               # max_position_pct = 0.5
    qty, _ = risk.size(1000, 1000, 100.0, 99.9, RULES)          # çok dar stop -> devasa boyut ister
    assert qty * 100.0 <= 500.0 + 1e-6


def test_position_capped_by_free_cash(cfg):
    risk = RiskManager(with_overrides(cfg, risk={"max_position_pct": 1.0}), 1000)
    qty, _ = risk.size(1000, 200, 100.0, 99.0, RULES)           # sadece 200 USDT serbest
    assert qty * 100.0 * (1 + 0.001) <= 200.0


def test_min_notional_and_step(cfg):
    risk = RiskManager(cfg, 1000)
    # 4.45 adet istenir, adım 1 -> 4 adet = 400 USDT; minimum 500 USDT -> işlem yok
    assert risk.size(1000, 1000, 100.0, 98.0, MarketRules(qty_step=1.0, min_notional=500.0)) == (0.0, 0.0)
    qty, _ = risk.size(1000, 1000, 100.0, 98.0, MarketRules(qty_step=1.0, min_notional=5.0))
    assert qty == 4.0


def test_invalid_stop_returns_zero(cfg):
    risk = RiskManager(cfg, 1000)
    assert risk.size(1000, 1000, 100.0, 100.0, RULES) == (0.0, 0.0)
    assert risk.size(1000, 1000, 100.0, 101.0, RULES) == (0.0, 0.0)
    assert risk.size(1000, 1000, 100.0, float("nan"), RULES) == (0.0, 0.0)


def test_floor_to_step_never_rounds_up():
    assert floor_to_step(0.123456789, 0.00001) == 0.12345
    assert floor_to_step(1.99999, 0.001) == 1.999
    assert floor_to_step(5.0, 1.0) == 5.0
    assert floor_to_step(0.30000000000000004, 0.1) == 0.3


def test_daily_loss_limit_blocks_until_next_utc_day(cfg):
    risk = RiskManager(cfg, 10_000)                             # günlük limit %3
    assert risk.on_bar_close(DAY0 + M15, 10_000) == []
    assert risk.on_bar_close(DAY0 + 2 * M15, 9_800) == []      # -%2
    assert risk.on_bar_close(DAY0 + 3 * M15, 9_690) == ["daily_block"]   # -%3.1
    assert risk.can_open("BTC/USDT", DAY0 + 3 * M15, 0, 0) == (False, "günlük zarar limiti")
    events = risk.on_bar_close(DAY0 + DAY_MS + M15, 9_700)     # ertesi UTC günü
    assert "new_day" in events
    assert risk.can_open("BTC/USDT", DAY0 + DAY_MS + M15, 0, 0) == (True, "")


def test_bar_closing_at_midnight_belongs_to_previous_day(cfg):
    risk = RiskManager(cfg, 10_000)
    risk.on_bar_close(DAY0 + DAY_MS - M15, 10_000)             # 23:45'te kapanan
    assert "new_day" not in risk.on_bar_close(DAY0 + DAY_MS, 10_000)   # 00:00'da kapanan (önceki gün)
    assert "new_day" in risk.on_bar_close(DAY0 + DAY_MS + M15, 10_000)


def test_max_drawdown_halts_until_manual_reset(cfg):
    risk = RiskManager(cfg, 10_000)                             # şalter %20
    risk.on_bar_close(DAY0 + M15, 12_000)                      # yeni tepe
    assert "halt" in risk.on_bar_close(DAY0 + 50 * DAY_MS, 9_590)   # 12000'den -%20.1
    assert risk.can_open("BTC/USDT", DAY0 + 50 * DAY_MS, 0, 0)[0] is False
    risk.on_bar_close(DAY0 + 51 * DAY_MS, 9_600)
    assert risk.state.halted                                   # yeni gün şalteri açmaz
    risk.reset_halt()
    assert risk.can_open("BTC/USDT", DAY0 + 51 * DAY_MS, 0, 0) == (True, "")


def test_cooldown_and_limits(cfg):
    risk = RiskManager(cfg, 10_000)                             # bekleme 4 mum, 2 pozisyon, günde 4 işlem
    risk.on_bar_close(DAY0 + M15, 10_000)
    risk.record_exit("BTC/USDT", DAY0)
    assert risk.can_open("BTC/USDT", DAY0 + 3 * M15, 0, 0) == (False, "çıkış sonrası bekleme")
    assert risk.can_open("BTC/USDT", DAY0 + 4 * M15, 0, 0) == (True, "")
    assert risk.can_open("ETH/USDT", DAY0 + M15, 1, 1)[1] == "maks. açık pozisyon"
    for _ in range(4):
        risk.record_entry()
    assert risk.can_open("ETH/USDT", DAY0 + 5 * M15, 0, 0) == (False, "günlük işlem sınırı")


def test_trailing_stop_only_moves_up(cfg):
    risk = RiskManager(cfg, 10_000)                             # iz süren 3 x ATR
    assert risk.trailing_stop(95.0, 110.0, 2.0) == 104.0
    assert risk.trailing_stop(104.0, 105.0, 2.0) == 104.0      # aşağı inmez
    assert risk.initial_stop(100.0, 2.0) == 96.0               # 2 x ATR


def test_trailing_stop_can_be_disabled(cfg):
    risk = RiskManager(with_overrides(cfg, risk={"trail_atr_mult": 0.0}), 10_000)
    assert risk.trailing_stop(95.0, 150.0, 2.0) == 95.0        # 0 = kapalı: stop yerinde kalır
