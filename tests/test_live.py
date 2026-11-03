"""Canlı broker ve emir katmanı: sahte borsada emir zinciri, stop yönetimi, mutabakat, simülasyonla
zamanlama eşdeğerliği ve uçtan uca canlı mod (testnet / gerçek hesap güvenlik kilitleri)."""
import dataclasses

import ccxt
import pytest
import yaml
from fake_exchange import FEE, FakeExchange, FakeLiveClient
from helpers import H4, TEST_CONFIG_PATH, FakeSource, loose_rules, make_candles, with_overrides

import run
from bot.backtester import run_backtest
from bot.brokers import orders as orders_module
from bot.brokers.base import Bar, EntryOrder
from bot.brokers.live import LiveBroker
from bot.brokers.orders import OrderStatusUnknown, SpotOrderAPI, parse_order
from bot.config import CONFIG_ENV_VAR, ConfigError
from bot.paper import PaperTrader
from bot.portfolio import Portfolio
from bot.signals import build_signals
from bot.storage import StateStore

SYM = "SOL/USDT"
BAR = H4


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(orders_module.time, "sleep", lambda _s: None)


def make(usdt=1000.0, price=100.0, stop_market=True, extra=None):
    ex = FakeExchange([SYM], {"USDT": usdt, **(extra or {})}, stop_market=stop_market)
    ex.prices[SYM] = price
    api = SpotOrderAPI(FakeLiveClient(ex))
    portfolio = Portfolio(usdt)
    saves = []
    broker = LiveBroker(portfolio, api, BAR, FEE, on_change=lambda: saves.append(1))
    broker.clock_ts = 10 * BAR
    return ex, portfolio, broker, saves


def entry(qty=1.0, stop=90.0, risk=10.0):
    return EntryOrder(SYM, qty, None, stop, risk_amount=risk, placed_ts=9 * BAR, ttl_bars=1)


# ---------------------------------------------------------------------------
# Emir katmanı
# ---------------------------------------------------------------------------
def test_parse_order_splits_fees_by_currency():
    order = {"id": 7, "status": "closed", "filled": 2.0, "cost": 200.0, "average": 100.0,
             "fees": [{"cost": 0.002, "currency": "SOL"}, {"cost": 0.1, "currency": "USDT"},
                      {"cost": 0.0003, "currency": "BNB"}]}
    r = parse_order(order, SYM)
    assert (r.id, r.status, r.filled, r.average) == ("7", "closed", 2.0, 100.0)
    assert (r.fee_base, r.fee_quote, r.fee_other) == (0.002, 0.1, 0.0003)


def test_network_error_after_send_does_not_duplicate_order():
    ex, _portfolio, broker, _ = make()
    ex.network_error_after_create = True              # emir borsaya ulaştı ama cevap kayboldu
    fills = broker.submit_entry(entry(), ref_price=100.0)
    buys = [o for o in ex.created if o["type"] == "market" and o["side"] == "buy"]
    assert len(buys) == 1 and fills[0].side == "buy"


def test_order_is_never_resent_while_its_fate_is_unknown():
    ex = FakeExchange([SYM], {"USDT": 1000.0})
    ex.prices[SYM] = 100.0
    api = SpotOrderAPI(FakeLiveClient(ex))
    ex.network_error_after_create = True                         # emir borsaya ulaştı, cevap kayboldu
    ex.fail_calls["fetch_order"] = [ccxt.NetworkError("kesinti")] * 3     # sorgular da bir süre çalışmadı
    assert api.market_buy(SYM, 1.0).filled == 1.0 and len(ex.created) == 1

    ex.network_error_after_create = True
    ex.fail_calls["fetch_order"] = [ccxt.NetworkError("kesinti")] * len(SpotOrderAPI.LOOKUP_DELAYS)
    with pytest.raises(OrderStatusUnknown):                      # emin olunamadı: ikinci kez gönderilmez
        api.market_buy(SYM, 1.0)
    assert len(ex.created) == 2


def test_lost_order_is_resent_only_after_recv_window(monkeypatch):
    slept = []
    monkeypatch.setattr(orders_module.time, "sleep", slept.append)
    ex = FakeExchange([SYM], {"USDT": 1000.0})
    ex.prices[SYM] = 100.0
    ex.fail_next = [ccxt.NetworkError("bağlantı yok")]           # emir borsaya hiç ulaşmadı
    result = SpotOrderAPI(FakeLiveClient(ex)).market_buy(SYM, 1.0)
    assert result.filled == 1.0 and len(ex.created) == 1
    assert sum(slept) >= SpotOrderAPI.RECV_WINDOW_S              # "yok" cevabına ancak bu süreden sonra güvenildi


# ---------------------------------------------------------------------------
# Canlı broker
# ---------------------------------------------------------------------------
def test_entry_records_received_coins_and_places_exchange_stop():
    ex, portfolio, broker, saves = make()
    (fill,) = broker.submit_entry(entry(qty=1.0), ref_price=100.0)
    received = 1.0 * (1 - FEE)                        # Binance alış komisyonunu coin olarak keser
    assert fill.side == "buy" and fill.qty == pytest.approx(received) and fill.ts == 10 * BAR
    position = portfolio.positions[SYM]
    assert position.qty == pytest.approx(received) and position.entry_fee == pytest.approx(0.1)
    assert portfolio.cash == pytest.approx(900.0)
    (stop,) = ex.open_stops(SYM)
    assert stop["type"] == "STOP_LOSS" and stop["params"]["stopPrice"] == 90.0
    assert stop["amount"] == pytest.approx(0.999)
    assert len(saves) >= 2                            # alıştan ve stoptan hemen sonra diske yazıldı


def test_stop_limit_is_used_when_stop_market_is_unavailable():
    ex, _, broker, _ = make(stop_market=False)
    broker.submit_entry(entry(), ref_price=100.0)
    (stop,) = ex.open_stops(SYM)
    assert stop["type"] == "STOP_LOSS_LIMIT" and stop["price"] == pytest.approx(89.1)


def test_exchange_stop_fill_is_recorded_on_next_bar():
    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ex.trigger(SYM, low=89.0)
    (fill,) = broker.process_bar(SYM, Bar(10 * BAR, 100.0, 101.0, 89.0, 95.0))
    assert fill.side == "sell" and fill.reason == "stop" and fill.price == 90.0 and fill.ts == 10 * BAR
    (trade,) = portfolio.trades
    assert trade.net_pnl == pytest.approx(0.999 * (90 - 100) - 0.1 - 0.999 * 90 * FEE)
    assert SYM not in portfolio.positions and not ex.open_stops()
    assert portfolio.cash == pytest.approx(900.0 + 0.999 * 90 * (1 - FEE))


def test_exit_signal_cancels_stop_and_sells_only_position():
    ex, portfolio, broker, _ = make(extra={"SOL": 5.0})          # hesapta botun olmayan 5 SOL da var
    broker.submit_entry(entry(), ref_price=100.0)
    ex.prices[SYM] = 110.0
    (fill,) = broker.request_exit(SYM, "çıkış sinyali")
    assert fill.reason == "çıkış sinyali" and fill.qty == pytest.approx(0.999) and fill.price == 110.0
    assert not ex.open_stops() and ex.free["SOL"] == pytest.approx(5.0)
    assert portfolio.trades[0].net_pnl > 0


def test_position_is_closed_if_stop_cannot_be_placed():
    ex, portfolio, broker, _ = make()
    ex.reject_stops = True
    fills = broker.submit_entry(entry(), ref_price=100.0)
    assert [f.side for f in fills] == ["buy", "sell"] and fills[1].reason == "stop konamadı"
    assert not portfolio.positions


def test_unfilled_stop_limit_is_closed_at_market():
    ex, portfolio, broker, _ = make(stop_market=False)
    broker.submit_entry(entry(), ref_price=100.0)
    ex.trigger(SYM, low=85.0, fill=False)                       # sert düşüş: stop-limit dolmadı
    ex.prices[SYM] = 86.0
    (fill,) = broker.process_bar(SYM, Bar(10 * BAR, 100.0, 100.0, 85.0, 86.0))
    assert fill.reason == "stop (piyasa)" and fill.price == 86.0
    assert not portfolio.positions and not ex.open_stops()


def test_rejected_buy_is_skipped_safely():
    ex, portfolio, broker, _ = make()
    ex.fail_next = [ccxt.InsufficientFunds("bakiye yetersiz")]
    assert broker.submit_entry(entry(), ref_price=100.0) == []
    assert not portfolio.positions and portfolio.cash == 1000.0


def test_reconcile_after_restart():
    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ex.trigger(SYM, low=80.0)                                    # bot kapalıyken stop çalıştı
    notes = broker.reconcile()
    assert "stop çalışmış" in notes[0] and portfolio.trades[0].exit_reason == "stop"

    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ex.cancel_order(ex.open_stops(SYM)[0]["id"], SYM)            # stop elle iptal edilmiş
    assert "yeniden kondu" in broker.reconcile()[0] and len(ex.open_stops(SYM)) == 1

    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ex.cancel_order(ex.open_stops(SYM)[0]["id"], SYM)
    ex.free["SOL"] = 0.0                                         # coin elle satılmış
    assert "kayıttan düşüldü" in broker.reconcile()[0] and not portfolio.positions


# ---------------------------------------------------------------------------
# Motor + canlı broker = simülasyonla aynı kararlar
# ---------------------------------------------------------------------------
SYMBOLS = ["AAA/USDT", "BBB/USDT"]


@pytest.fixture(scope="module")
def lcfg(cfg):
    c = with_overrides(
        cfg,
        timeframes={"entry": "4h", "trend": "1d"},
        strategy={"type": "breakout", "exit_on_regime_break": False},
        execution={"entry_order": "market"},
        costs={"slippage": 0.0},
        risk={"risk_per_trade": 0.005, "trail_atr_mult": 0.0, "max_open_positions": 5, "max_trades_per_day": 50},
        paper={"history_bars": 5000},
    )
    return dataclasses.replace(c, symbols=SYMBOLS)


def test_live_trader_makes_same_decisions_as_simulation(lcfg, tmp_path):
    n, start_bar = 700, 250
    data = {s: make_candles(n, seed=90 + i, base=50.0 * (i + 1), tf_ms=H4, vol=0.012) for i, s in enumerate(SYMBOLS)}
    ex = FakeExchange(SYMBOLS, {"USDT": 10_000.0}, step=1e-6)
    client = FakeLiveClient(ex, FakeSource({(s, "4h"): df for s, df in data.items()}))
    api = SpotOrderAPI(client)

    def factory(portfolio, persist):
        return LiveBroker(portfolio, api, lcfg.entry_ms, lcfg.costs.taker_fee, on_change=persist)

    trader = PaperTrader(lcfg, client, StateStore(tmp_path / "l.db"), loose_rules(SYMBOLS),
                         broker_factory=factory, catch_up=False, initial_equity=10_000.0)
    for i in range(start_bar, n):
        for s, df in data.items():                    # mum i kapanmadan önce: borsadaki stoplar tetiklenir
            ex.trigger(s, float(df["low"].iloc[i - 1]))
            ex.prices[s] = float(df["close"].iloc[i - 1])   # şu anki fiyat = son kapanış = yeni mumun açılışı
        trader.sync(int(data["AAA/USDT"]["ts"].iloc[i]) + 5_000)

    signals = {s: build_signals(lcfg, df) for s, df in data.items()}
    first = int(data["AAA/USDT"]["ts"].iloc[start_bar])     # ilk senkron yalnızca başlangıç noktasını belirler
    sim = run_backtest(lcfg, signals, first, trader.last_ts + H4, 10_000.0, loose_rules(SYMBOLS))
    live_trades = [(t.symbol, t.entry_ts, t.exit_ts, t.exit_reason) for t in trader.portfolio.trades]
    sim_trades = [(t.symbol, t.entry_ts, t.exit_ts, t.exit_reason) for t in sim.trades]
    assert len(sim_trades) >= 5, "karşılaştırma anlamlı olsun diye işlem olmalı"
    assert live_trades == sim_trades
    assert set(trader.portfolio.positions) == {p.symbol for p in sim.open_positions}


# ---------------------------------------------------------------------------
# Uçtan uca: run.py --mode live
# ---------------------------------------------------------------------------
@pytest.fixture
def live_project(tmp_path, monkeypatch):
    raw = yaml.safe_load(TEST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["symbols"] = SYMBOLS
    raw["timeframes"] = {"entry": "4h", "trend": "1d"}
    raw["strategy"].update(type="breakout", exit_on_regime_break=False)
    raw["execution"]["entry_order"] = "market"
    raw["risk"].update(risk_per_trade=0.005, trail_atr_mult=0.0)
    raw["data"].update(dir=str(tmp_path / "data"), start="2023-01-01", exam_start="2024-01-01")
    raw["logging"]["dir"] = str(tmp_path / "logs")
    raw["backtest"].update(reports_dir=str(tmp_path / "reports"), open_chart=False)
    config_file = tmp_path / "config.yaml"

    def write(**live):
        raw["live"].update(live)
        config_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    write()
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))
    data = {(s, "4h"): make_candles(600, seed=70 + i, start="2023-01-01", base=20.0 * (i + 1), tf_ms=H4, vol=0.012)
            for i, s in enumerate(SYMBOLS)}
    ex = FakeExchange(SYMBOLS, {"USDT": 500.0})
    end = int(data[("AAA/USDT", "4h")]["ts"].iloc[-1]) + H4
    client = FakeLiveClient(ex, FakeSource(data), now_ms=end + 5_000)
    for s in SYMBOLS:
        ex.prices[s] = float(data[(s, "4h")]["close"].iloc[-1])
    monkeypatch.setattr(run, "ExchangeClient", lambda *a, **k: client)
    monkeypatch.setattr(run, "load_api_keys", lambda testnet=False: ("anahtar", "gizli"))
    return tmp_path, client, write


def test_live_mode_needs_keys(live_project, capsys, monkeypatch):
    def missing(testnet=False):
        raise ConfigError(".env içinde BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET boş.")
    monkeypatch.setattr(run, "load_api_keys", missing)
    assert run.main(["--mode", "live", "--once"]) == 2
    assert "BINANCE_TESTNET_API_KEY" in capsys.readouterr().out


def test_testnet_once_and_status(live_project, capsys):
    tmp_path, client, _ = live_project
    end = client.now
    client.now = end - 2 * H4
    assert run.main(["--mode", "live", "--once"]) == 0
    client.now = end
    assert run.main(["--mode", "live", "--once"]) == 0
    out = capsys.readouterr().out
    assert "TESTNET (sahte para)" in out and (tmp_path / "data" / "testnet_state.db").exists()
    assert run.main(["--mode", "live", "--status"]) == 0
    assert "Testnet özeti" in capsys.readouterr().out
    assert not (tmp_path / "data" / "canli_state.db").exists()


def test_real_mode_refuses_withdrawal_permission(live_project, capsys):
    _tmp_path, client, write = live_project
    write(testnet=False)
    client.restrictions["enableWithdrawals"] = True
    assert run.main(["--mode", "live", "--once"]) == 2
    assert "PARA ÇEKME" in capsys.readouterr().out


def test_real_mode_requires_typed_confirmation(live_project, capsys, monkeypatch):
    tmp_path, _client, write = live_project
    write(testnet=False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "hayir")
    assert run.main(["--mode", "live", "--once"]) == 1
    out = capsys.readouterr().out
    assert "GERÇEK PARA" in out and "İptal" in out
    monkeypatch.setattr("builtins.input", lambda _prompt="": "gercek")
    assert run.main(["--mode", "live", "--once"]) == 0
    assert "CANLI İŞLEM (GERÇEK PARA)" in capsys.readouterr().out
    assert (tmp_path / "data" / "canli_state.db").exists()


def test_live_mode_needs_enough_usdt(live_project, capsys):
    _, client, _ = live_project
    client.ex.free["USDT"] = 3.0
    assert run.main(["--mode", "live", "--once"]) == 2
    assert "Spot cüzdana" in capsys.readouterr().out


def test_order_plumbing_script_on_fake_testnet(capsys, monkeypatch):
    import emir_testi
    ex = FakeExchange(["ETH/USDT"], {"USDT": 100.0, "ETH": 0.5}, step=1e-4)   # hesapta zaten 0.5 ETH var
    ex.prices["ETH/USDT"] = 2_600.0
    monkeypatch.setattr(emir_testi.time, "sleep", lambda _s: None)
    assert emir_testi.main([], client=FakeLiveClient(ex)) == 0
    out = capsys.readouterr().out
    assert "Emir tesisatı çalışıyor" in out and out.count("[OK]") >= 5
    assert ex.free["ETH"] == pytest.approx(0.5, abs=2e-4)              # testten önceki ETH'ye dokunulmadı
    assert "ETH artığı: 0.000095 ETH (~0.25 USDT" in out                # yuvarlama artığı ayrıca gösterilir
    assert "gerçek maliyeti (komisyon + alış-satış farkı): ~0.0237 USDT" in out   # yalnızca iki komisyon
    (stop,) = [o for o in ex.created if o["type"].startswith("STOP")]
    assert stop["amount"] * 2_600.0 * 0.90 >= 5.0                     # stop, minimum tutarın üstündeydi
    assert not ex.open_stops()


def test_live_trader_never_trades_missed_bars(lcfg, tmp_path):
    data = {s: make_candles(400, seed=95 + i, base=40.0, tf_ms=H4, vol=0.012) for i, s in enumerate(SYMBOLS)}
    ex = FakeExchange(SYMBOLS, {"USDT": 10_000.0}, step=1e-6)
    client = FakeLiveClient(ex, FakeSource({(s, "4h"): df for s, df in data.items()}))
    api = SpotOrderAPI(client)
    trader = PaperTrader(lcfg, client, StateStore(tmp_path / "m.db"), loose_rules(SYMBOLS),
                         broker_factory=lambda p, persist: LiveBroker(p, api, lcfg.entry_ms, FEE, on_change=persist),
                         catch_up=False, initial_equity=10_000.0)
    ts = data["AAA/USDT"]["ts"]
    trader.sync(int(ts.iloc[300]) + 5_000)                    # başlangıç
    processed = trader.sync(int(ts.iloc[306]) + 5_000)        # bot 6 mum boyunca kapalıydı
    assert processed == [int(ts.iloc[305])] and trader.skipped_bars == 5


def test_engine_records_immediate_live_fills_in_risk_manager(cfg):
    from bot.brokers.base import Fill
    from bot.brokers.simulated import SimulatedBroker
    from bot.engine import TradingEngine
    from bot.risk import RiskManager
    portfolio = Portfolio(1000.0)
    risk = RiskManager(cfg, 1000.0)
    engine = TradingEngine(cfg, portfolio, SimulatedBroker(portfolio, 0.001, 0.001, 0.0), risk, {})
    engine._apply([Fill(SYM, "buy", 1.0, 100.0, 0.1, 5 * BAR, "giriş"), Fill(SYM, "sell", 1.0, 99.0, 0.1, 7 * BAR, "stop")])
    assert risk.state.trades_today == 1 and risk.state.last_exit_ts[SYM] == 7 * BAR     # bekleme süresi işler
    assert engine._apply(None) == []


def test_local_time_label():
    from datetime import datetime, timedelta, timezone

    from bot.timeutil import fmt_local, utc_offset_label
    assert utc_offset_label(datetime(2026, 10, 8, tzinfo=timezone(timedelta(hours=2)))) == "UTC+2"
    assert utc_offset_label(datetime(2026, 1, 8, tzinfo=timezone(timedelta(hours=5, minutes=30)))) == "UTC+5:30"
    assert utc_offset_label(datetime(2026, 1, 8, tzinfo=timezone(timedelta(hours=-3)))) == "UTC-3"
    text = fmt_local(1_791_424_800_000)
    assert "yerel (UTC" in text and text[:4] == "2026"


# ---------------------------------------------------------------------------
# Gerçek hesaba özgü durumlar: minimum tutar, geçici ve kalıcı satış hataları
# ---------------------------------------------------------------------------
def make_btc(usdt=50.0):
    from bot.config import MarketRules
    ex = FakeExchange(["BTC/USDT"], {"USDT": usdt}, step=1e-5)
    ex.prices["BTC/USDT"] = 83_000.0
    api = SpotOrderAPI(FakeLiveClient(ex))
    portfolio = Portfolio(usdt)
    rules = {"BTC/USDT": MarketRules(qty_step=1e-5, min_notional=5.0)}
    broker = LiveBroker(portfolio, api, BAR, FEE, rules=rules)
    broker.clock_ts = 10 * BAR
    return ex, portfolio, broker


def btc_entry(qty, stop):
    return EntryOrder("BTC/USDT", qty, None, stop, risk_amount=0.3, placed_ts=9 * BAR, ttl_bars=1)


def test_tiny_position_is_sized_to_stay_sellable_after_fee_and_rounding():
    ex, portfolio, broker = make_btc()
    (_fill,) = broker.submit_entry(btc_entry(0.00007, 79_680.0), ref_price=83_000.0)   # ~5,8 USDT
    # 0.00007 BTC alınsaydı komisyon sonrası 0.00006 BTC kalırdı: stopta 4,8 USDT < 5 (satılamazdı).
    assert ex.created[0]["amount"] == pytest.approx(0.00008)
    (stop,) = ex.open_stops()
    assert stop["amount"] * 79_680.0 >= 5.0
    assert portfolio.positions["BTC/USDT"].risk_amount > 0.3                     # risk de oranlı büyüdü


def test_entry_skipped_when_minimum_would_need_too_large_increase():
    ex, portfolio, broker = make_btc()
    assert broker.submit_entry(btc_entry(0.00004, 79_680.0), ref_price=83_000.0) == []   # %50'den fazla artış
    assert not ex.created and not portfolio.positions


def test_temporary_sell_failure_keeps_position_protected_and_retries():
    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ex.fail_next = [ccxt.NetworkError("zaman aşımı")] * 3           # satış 3 denemede de gönderilemedi
    assert broker.request_exit(SYM, "çıkış sinyali") == []
    assert SYM in portfolio.positions and len(ex.open_stops(SYM)) == 1     # stop yeniden kondu
    assert broker.to_dict()["pending_exits"] == {SYM: "çıkış sinyali"}
    ex.prices[SYM] = 105.0
    (fill,) = broker.process_bar(SYM, Bar(10 * BAR, 104.0, 106.0, 103.0, 105.0))   # sonraki mumda tekrar
    assert fill.reason == "çıkış sinyali" and fill.price == 105.0
    assert not portfolio.positions and not ex.open_stops() and not broker.pending_exits


def test_permanently_rejected_sell_drops_position_and_keeps_coins():
    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ex.fail_next = [ccxt.InvalidOrder("Filter failure: NOTIONAL")]
    assert broker.request_exit(SYM, "çıkış sinyali") == []
    assert not portfolio.positions and ex.free["SOL"] == pytest.approx(0.999)   # coinler hesapta


def test_exit_is_retried_while_stop_cannot_be_cancelled():
    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ex.fail_calls["cancel_order"] = [ccxt.NetworkError("zaman aşımı")]
    assert broker.request_exit(SYM, "çıkış sinyali") == []
    assert SYM in portfolio.positions and len(ex.open_stops(SYM)) == 1      # stop borsada: coinler korunuyor
    assert broker.pending_exits == {SYM: "çıkış sinyali"}
    assert not [o for o in ex.created if o["type"] == "market" and o["side"] == "sell"]
    ex.prices[SYM] = 104.0
    (fill,) = broker.process_bar(SYM, Bar(10 * BAR, 103.0, 105.0, 102.0, 104.0))
    assert fill.reason == "çıkış sinyali" and fill.price == 104.0
    assert not portfolio.positions and not ex.open_stops() and not broker.pending_exits


def test_access_error_on_sell_keeps_position_protected():
    ex, portfolio, broker, _ = make()
    broker.submit_entry(entry(), ref_price=100.0)
    ip_error = "Invalid API-key, IP, or permissions for action."     # ör. ev IP'si değişti
    ex.fail_calls["fetch_balance"] = [ccxt.AuthenticationError(ip_error)]
    assert broker.request_exit(SYM, "çıkış sinyali") == []
    assert SYM in portfolio.positions and len(ex.open_stops(SYM)) == 1      # stop yeniden kondu
    ex.fail_next = [ccxt.PermissionDenied(ip_error)]                         # bir sonraki mumda satış da reddedildi
    assert broker.process_bar(SYM, Bar(10 * BAR, 100.0, 101.0, 99.0, 100.0)) == []
    assert SYM in portfolio.positions and len(ex.open_stops(SYM)) == 1
    (fill,) = broker.process_bar(SYM, Bar(11 * BAR, 100.0, 101.0, 99.0, 100.0))   # erişim düzelince satıldı
    assert fill.reason == "çıkış sinyali" and not portfolio.positions and not ex.open_stops()


def test_explicit_target_flags_override_config(live_project, capsys, monkeypatch):
    tmp_path, client, write = live_project
    targets = []                                                   # hangi ortamın anahtarı/bağlantısı istendi
    monkeypatch.setattr(run, "ExchangeClient", lambda *a, **k: targets.append(("bağlantı", k["testnet"])) or client)
    monkeypatch.setattr(run, "load_api_keys", lambda testnet=False: targets.append(("anahtar", testnet)) or ("a", "g"))
    write(testnet=False)                                           # ayar dosyası gerçek hesap dese bile
    end = client.now
    client.now = end - 2 * H4
    assert run.main(["--mode", "live", "--testnet", "--once"]) == 0
    client.now = end
    assert run.main(["--mode", "live", "--testnet", "--once"]) == 0
    assert "TESTNET (sahte para)" in capsys.readouterr().out
    assert targets == [("anahtar", True), ("bağlantı", True)] * 2
    assert not (tmp_path / "data" / "canli_state.db").exists()

    targets.clear()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "hayir")
    write(testnet=True)                                            # ayar testnet dese bile --gercek onay ister
    assert run.main(["--mode", "live", "--gercek", "--once"]) == 1
    assert "GERÇEK PARA" in capsys.readouterr().out
    assert targets == [("anahtar", False), ("bağlantı", False)]

    assert run.main(["--mode", "live", "--status"]) == 0           # bayraksız durum: ikisini de gösterir
    out = capsys.readouterr().out
    assert "Testnet özeti" in out and "Henüz kayıtlı canlı işlem sonucu yok" in out

