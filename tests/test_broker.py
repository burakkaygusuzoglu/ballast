"""Simülasyon broker'ı: kötümser dolum kuralları ve muhasebe."""
import pytest

from bot.brokers.base import Bar, EntryOrder
from bot.brokers.simulated import InsufficientCash, SimulatedBroker
from bot.portfolio import Portfolio

MAKER, TAKER, SLIP = 0.001, 0.001, 0.0005
SYM = "BTC/USDT"


def make_broker(cash: float = 10_000.0) -> tuple[Portfolio, SimulatedBroker]:
    portfolio = Portfolio(cash)
    return portfolio, SimulatedBroker(portfolio, MAKER, TAKER, SLIP)


def limit_order(qty=1.0, limit=100.0, stop=95.0, ttl=1) -> EntryOrder:
    return EntryOrder(SYM, qty, limit, stop, risk_amount=5.0, placed_ts=0, ttl_bars=ttl)


def test_limit_requires_trade_through_not_touch():
    portfolio, broker = make_broker()
    broker.submit_entry(limit_order(), ref_price=100.0)
    assert portfolio.reserved == pytest.approx(100.0 * (1 + MAKER))
    fills = broker.process_bar(SYM, Bar(1, 101.0, 102.0, 100.0, 101.5))   # dibe değdi ama altına inmedi
    assert fills == [] and SYM not in portfolio.positions
    assert portfolio.reserved == 0 and portfolio.cash == pytest.approx(10_000.0)   # TTL=1: iptal, nakit iade


def test_limit_fills_at_limit_price_with_maker_fee():
    portfolio, broker = make_broker()
    broker.submit_entry(limit_order(), ref_price=100.0)
    (fill,) = broker.process_bar(SYM, Bar(1, 101.0, 102.0, 99.9, 101.5))
    assert fill.side == "buy" and fill.price == 100.0 and fill.fee == pytest.approx(0.1)
    assert portfolio.cash == pytest.approx(10_000.0 - 100.0 - 0.1)
    assert portfolio.positions[SYM].stop_price == 95.0
    assert portfolio.positions[SYM].entry_slippage == 0.0       # limit emirde kayma yok


def test_market_entry_pays_slippage_and_taker():
    portfolio, broker = make_broker()
    order = EntryOrder(SYM, 1.0, None, 95.0, risk_amount=5.0, placed_ts=0, ttl_bars=1)
    broker.submit_entry(order, ref_price=100.0)
    (fill,) = broker.process_bar(SYM, Bar(1, 100.0, 101.0, 99.0, 100.5))
    assert fill.price == pytest.approx(100.0 * (1 + SLIP))
    assert fill.fee == pytest.approx(100.0 * (1 + SLIP) * TAKER)
    assert portfolio.positions[SYM].entry_slippage == pytest.approx(100.0 * SLIP)


def test_order_waits_until_ttl():
    _portfolio, broker = make_broker()
    broker.submit_entry(limit_order(ttl=2), ref_price=100.0)
    broker.process_bar(SYM, Bar(1, 101, 102, 100.5, 101))
    assert broker.has_pending_entry(SYM)
    broker.process_bar(SYM, Bar(2, 101, 102, 100.5, 101))
    assert not broker.has_pending_entry(SYM) and broker.cancelled_entries == 1


def test_stop_gap_fills_at_open_with_slippage_and_taker_fee():
    portfolio, broker = make_broker()
    broker.submit_entry(limit_order(), ref_price=100.0)
    broker.process_bar(SYM, Bar(1, 101, 102, 99.5, 101))           # giriş @100
    (fill,) = broker.process_bar(SYM, Bar(2, 90.0, 91.0, 89.0, 90.5))   # stop 95 ama 90'dan açıldı
    expected_price = 90.0 * (1 - SLIP)
    assert fill.side == "sell" and fill.price == pytest.approx(expected_price)
    trade = portfolio.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.net_pnl == pytest.approx((expected_price - 100.0) - 0.1 - expected_price * TAKER)
    assert trade.slippage_cost == pytest.approx(90.0 * SLIP)
    assert trade.r_multiple == pytest.approx(trade.net_pnl / 5.0)


def test_same_bar_entry_then_stop():
    portfolio, broker = make_broker()
    broker.submit_entry(limit_order(), ref_price=100.0)
    fills = broker.process_bar(SYM, Bar(1, 101.0, 101.0, 94.0, 96.0))   # 100'ü de 95'i de geçti
    assert [f.side for f in fills] == ["buy", "sell"]
    assert fills[1].price == pytest.approx(95.0 * (1 - SLIP))           # limit dolumu mum içinde: boşluk yok
    assert not portfolio.positions


def test_requested_exit_executes_at_next_open():
    _portfolio, broker = make_broker()
    broker.submit_entry(limit_order(), ref_price=100.0)
    broker.process_bar(SYM, Bar(1, 101, 102, 99.5, 101))
    broker.request_exit(SYM, "rejim bozuldu")
    (fill,) = broker.process_bar(SYM, Bar(2, 103.0, 104.0, 102.0, 103.5))
    assert fill.price == pytest.approx(103.0 * (1 - SLIP)) and fill.reason == "rejim bozuldu"


def test_stop_only_moves_up():
    portfolio, broker = make_broker()
    broker.submit_entry(limit_order(), ref_price=100.0)
    broker.process_bar(SYM, Bar(1, 101, 102, 99.5, 101))
    assert broker.update_stop(SYM, 97.0) is True
    assert broker.update_stop(SYM, 96.0) is False
    assert portfolio.positions[SYM].stop_price == 97.0


def test_equity_is_conserved_without_price_change():
    portfolio, broker = make_broker()
    broker.submit_entry(limit_order(), ref_price=100.0)
    assert portfolio.equity({}) == pytest.approx(10_000.0)              # ayrılan nakit özsermayede
    broker.process_bar(SYM, Bar(1, 101, 102, 99.5, 100.0))
    assert portfolio.equity({SYM: 100.0}) == pytest.approx(10_000.0 - 0.1)   # sadece komisyon kadar azaldı


def test_cannot_overspend():
    _portfolio, broker = make_broker(cash=50.0)
    with pytest.raises(InsufficientCash):
        broker.submit_entry(limit_order(qty=1.0, limit=100.0), ref_price=100.0)


def test_state_roundtrip():
    portfolio, broker = make_broker()
    broker.submit_entry(limit_order(ttl=3), ref_price=100.0)
    restored = SimulatedBroker(portfolio, MAKER, TAKER, SLIP)
    restored.load_state(broker.to_dict())
    assert restored.pending_entries == broker.pending_entries
