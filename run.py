"""Paper trading (canlı veri + sanal bakiye) ve canlı işlem.

Kullanım:
    python run.py                  # paper trading başlat (durdurmak için Ctrl+C)
    python run.py --status         # kayıtlı paper sonuçlarının özeti (internet gerekmez)
    python run.py --reset-halt     # maks. düşüş şalterinden sonra elle yeniden açma
    python run.py --reset-state    # tüm paper geçmişini silip sıfırdan başlama
    python run.py --mode live --testnet   # canlı işlem, Binance TESTNET (sahte para)
    python run.py --mode live --gercek    # canlı işlem, GERÇEK hesap (her başlatmada onay ister)
    python run.py --mode live --status    # testnet ve gerçek hesap özetleri

Bot her giriş mumu (C1'de 4 saatlik) kapandıktan birkaç saniye sonra uyanır, yeni mumu işler ve
durumu data/ altındaki SQLite dosyasına yazar. Kapatıp açtığınızda kaldığı yerden devam eder.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import ccxt

from bot.brokers.live import LiveBroker
from bot.brokers.orders import SpotOrderAPI
from bot.cli import confirm, disable_quick_edit, load_config_or_exit, setup_console
from bot.config import ConfigError, load_api_keys
from bot.data import load_market_rules, save_market_rules
from bot.exchange import ExchangeClient
from bot.logger import close_logging, setup_logging
from bot.metrics import compute_metrics
from bot.paper import PaperStateMismatch, PaperTrader
from bot.signals import describe
from bot.storage import StateStore
from bot.timeutil import fmt_local, fmt_utc

STATE_FILE = "paper_state.db"      # paper.name = "paper" iken (gölge botlarda <ad>_state.db)
LIVE_STATE_FILES = {True: "testnet_state.db", False: "canli_state.db"}
MIN_LIVE_CAPITAL = 10.0          # USDT; daha azıyla 5 USDT minimum emir sınırı yüzünden işlem yapılamaz
LAG_RETRY_SECONDS = 10
LAG_MAX_RETRIES = 6
OFFSET_REFRESH_MS = 3_600_000
NETWORK_RETRY_SECONDS = 30


def paper_state_file(cfg) -> str:
    return STATE_FILE if cfg.paper.name == "paper" else f"{cfg.paper.name}_state.db"


def paper_title(cfg) -> str:
    return "Paper trading" if cfg.paper.name == "paper" else f"Gölge bot ({cfg.paper.name})"


def print_status(store: StateStore, title: str = "Paper trading") -> None:
    import pandas as pd

    snapshot = store.load_snapshot()
    trades = store.load_trades()
    points = store.load_equity()
    if snapshot is None or not points:
        print(f"Henüz kayıtlı {title.lower()} sonucu yok (en az bir mum işlenmiş olmalı).")
        return
    initial = float(snapshot["portfolio"]["initial_cash"])
    equity = pd.Series([e for _, e in points], index=pd.Index([t for t, _ in points], dtype="int64"))
    m = compute_metrics(equity, trades, initial, snapshot["portfolio"]["total_fees"])
    print(f"{title} özeti ({fmt_utc(points[0][0])} - {fmt_utc(points[-1][0])})")
    print(f"  Özsermaye : {initial:,.2f} -> {m['final_equity']:,.2f} USDT ({m['total_return'] * 100:+.2f}%)")
    pf = "-" if m["n_trades"] == 0 else ("sonsuz" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}")
    print(f"  İşlem     : {m['n_trades']} | isabet %{m['win_rate'] * 100:.0f} | kâr faktörü {pf}")
    print(f"  Komisyon  : {m['fees_paid']:,.2f} USDT | maks. düşüş %{m['max_drawdown'] * 100:.1f}")
    print(f"  Açık poz. : {', '.join(p['symbol'] for p in snapshot['portfolio']['positions']) or '-'}")
    if snapshot["risk"]["halted"]:
        print(f"  [!] ŞALTER AKTİF: {snapshot['risk']['halt_reason']}  (açmak için: --reset-halt)")


def main(argv: list[str] | None = None) -> int:
    setup_console()
    disable_quick_edit()                 # pencereye tıklamak botu dondurmasın
    parser = argparse.ArgumentParser(description="Paper trading (canlı veri + sanal bakiye).")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--status", action="store_true", help="Kayıtlı sonuçların özetini göster")
    parser.add_argument("--reset-halt", action="store_true", help="Maks. düşüş şalterini elle kapat")
    parser.add_argument("--reset-state", action="store_true", help="Paper geçmişini sil ve sıfırdan başla")
    parser.add_argument("--once", action="store_true", help="Bir kez senkronize et ve çık (test için)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--testnet", action="store_true", help="Canlı mod: Binance test ortamı (sahte para)")
    target.add_argument("--gercek", action="store_true", help="Canlı mod: GERÇEK hesap")
    args = parser.parse_args(argv)

    cfg = load_config_or_exit()
    live = args.mode == "live"
    if args.testnet or args.gercek:                  # komut satırı, ayar dosyasındaki live.testnet'ten önce gelir
        cfg = dataclasses.replace(cfg, live=dataclasses.replace(cfg.live, testnet=args.testnet))
    if live and args.status and not (args.testnet or args.gercek):
        for testnet, name in ((True, "Testnet"), (False, "Canlı işlem")):
            path = cfg.data_dir / LIVE_STATE_FILES[testnet]
            if path.exists():
                print_status(StateStore(path), name)
                print()
        return 0
    title = ("Testnet" if cfg.live.testnet else "Canlı işlem") if live else paper_title(cfg)
    store = StateStore(cfg.data_dir / (LIVE_STATE_FILES[cfg.live.testnet] if live else paper_state_file(cfg)))
    if args.status:
        print_status(store, title)
        return 0
    if args.reset_state:
        snapshot = store.load_snapshot()
        if live and snapshot and snapshot["portfolio"]["positions"]:
            print("[HATA] Açık pozisyon varken canlı kayıtlar silinemez; önce pozisyonları kapatın.")
            return 2
        if not confirm(f"Tüm {title.lower()} geçmişi silinecek. Onaylamak için 'sil' yazın:", "sil"):
            print("İptal edildi.")
            return 1
        store.reset()
        print(f"[OK] {title} durumu sıfırlandı.")

    log_name = ("testnet.log" if cfg.live.testnet else "canli.log") if live else f"{cfg.paper.name}.log"
    log = setup_logging(cfg.log_dir, cfg.logging.level, log_name)
    try:
        return _run_live(cfg, args, store, log, title) if live else _run(cfg, args, store, log)
    finally:
        close_logging()


def _run(cfg, args, store: StateStore, log) -> int:
    client = ExchangeClient(cfg, logger=log)
    try:
        offset = client.refresh_time_offset()
    except ccxt.BaseError as exc:
        print(f"[HATA] Binance'e bağlanılamadı: {exc}\nİnternet bağlantınızı kontrol edip tekrar deneyin.")
        return 1
    if abs(offset) > 2000:
        log.warning(f"Bilgisayar saati Binance saatinden {offset / 1000:+.1f} sn farklı; bot Binance saatini kullanır.")
    try:
        rules = client.market_rules(cfg.symbols, cfg.market_rules_default)
        save_market_rules(cfg.data_dir, rules)
    except ccxt.BaseError as exc:
        log.warning(f"Sembol kuralları borsadan alınamadı ({exc}); kayıtlı değerler kullanılacak.")
        rules = load_market_rules(cfg.data_dir, cfg.symbols, cfg.market_rules_default)

    try:
        trader = PaperTrader(cfg, client, store, rules, log, label=paper_title(cfg))
    except PaperStateMismatch as exc:
        print(f"[HATA] {exc}")
        return 2
    if args.reset_halt:
        trader.risk.reset_halt()
        store.save(trader.snapshot(), [], [])
        print("[OK] Şalter elle kapatıldı.")

    print("=" * 76)
    print(f" {paper_title(cfg).upper()}  |  {describe(cfg)}")
    print(f" Semboller ({len(cfg.symbols)}): {', '.join(s.split('/')[0] for s in cfg.symbols)}")
    print(f" Durum: {'kaldığı yerden devam' if trader.resumed else 'yeni başlangıç'}  |  "
          f"Sanal özsermaye: {trader.equity():,.2f} USDT")
    print(" Gerçek emir GÖNDERİLMEZ. Durdurmak için Ctrl+C.")
    _print_clock(trader, client)
    print("=" * 76)
    return _loop(trader, client, store, cfg, args, log, paper_title(cfg))


def _print_clock(trader: PaperTrader, client) -> None:
    now = client.server_now()
    print(f" Saat: {fmt_local(now)} | {fmt_utc(now)} (kayıtlardaki zamanlar UTC)")
    print(f" Sıradaki karar: {fmt_local(trader.next_wakeup_ms(now), with_date=False)}")


def _run_live(cfg, args, store: StateStore, log, title: str) -> int:
    testnet = cfg.live.testnet
    try:
        key, secret = load_api_keys(testnet)
    except ConfigError as exc:
        print(f"[HATA] {exc}")
        return 2
    client = ExchangeClient(cfg, key, secret, logger=log, testnet=testnet)
    try:
        client.refresh_time_offset()
        if not testnet:
            permissions = client.api_restrictions()
            if permissions.get("enableWithdrawals"):
                print("[HATA] API anahtarında PARA ÇEKME izni açık. Güvenlik için bot çalışmayı reddediyor.\n"
                      "Binance -> API Yönetimi'nden bu izni kapatıp tekrar deneyin.")
                return 2
            if not permissions.get("enableSpotAndMarginTrading"):
                print("[HATA] API anahtarında spot işlem izni yok.")
                return 2
            if not permissions.get("ipRestrict"):
                print("[UYARI] API anahtarında IP kısıtlaması kapalı. Açmanız şiddetle önerilir.")
        markets = client.markets()
        symbols = [s for s in cfg.symbols if (markets.get(s) or {}).get("active", False)]
        missing = [s for s in cfg.symbols if s not in symbols]
        if missing and not testnet:
            print(f"[HATA] Binance'te işlem görmeyen sembol(ler): {', '.join(missing)}")
            return 2
        if missing:
            print(f"[UYARI] Testnet'te olmayan coinler atlanıyor: {', '.join(s.split('/')[0] for s in missing)}")
        if not symbols:
            print("[HATA] İşlem yapılabilecek sembol yok.")
            return 2
        cfg = dataclasses.replace(cfg, symbols=symbols)
        rules = client.market_rules(symbols, cfg.market_rules_default)
        api = SpotOrderAPI(client)
        usdt = api.free_balance("USDT")
    except ccxt.BaseError as exc:
        print(f"[HATA] Binance'e bağlanılamadı ya da anahtar geçersiz: {exc}")
        return 1

    resumed = store.load_snapshot() is not None
    capital = min(usdt, cfg.live.max_capital)
    if not resumed and capital < MIN_LIVE_CAPITAL:
        print(f"[HATA] Spot cüzdanda yeterli USDT yok ({usdt:,.2f}). En az {MIN_LIVE_CAPITAL:.0f} USDT gerekir; "
              "Earn'deki USDT'yi önce Spot cüzdana aktarın.")
        return 2
    if not testnet:
        print("!" * 76)
        print(" GERÇEK PARA ile işlem yapılacak. Bot, Binance'e gerçek alım-satım emirleri gönderir.")
        print(f" Kullanılacak en fazla tutar: {capital:,.2f} USDT. Zarar edebilirsiniz; kâr garantisi yoktur.")
        print("!" * 76)
        if not confirm("Devam etmek için 'gercek' yazın:", "gercek"):
            print("İptal edildi.")
            return 1

    def factory(portfolio, persist):
        return LiveBroker(portfolio, api, cfg.entry_ms, cfg.costs.taker_fee, log, persist, rules)

    try:
        trader = PaperTrader(cfg, client, store, rules, log, broker_factory=factory, catch_up=False,
                             initial_equity=capital, label=title)
        notes = trader.broker.reconcile()
    except PaperStateMismatch as exc:
        print(f"[HATA] {exc}")
        return 2
    except ccxt.BaseError as exc:
        print(f"[HATA] Borsa ile mutabakat yapılamadı: {exc}")
        return 1
    for note in notes:
        print(f"[MUTABAKAT] {note}")
    if args.reset_halt:
        trader.risk.reset_halt()
        trader.persist()
        print("[OK] Şalter elle kapatıldı.")

    print("=" * 76)
    print(f" {'TESTNET (sahte para)' if testnet else 'CANLI İŞLEM (GERÇEK PARA)'}  |  {describe(cfg)}")
    print(f" Semboller ({len(cfg.symbols)}): {', '.join(s.split('/')[0] for s in cfg.symbols)}")
    print(f" Durum: {'kaldığı yerden devam' if trader.resumed else 'yeni başlangıç'}  |  "
          f"Özsermaye: {trader.equity():,.2f} USDT")
    print(" Koruyucu stoplar borsada bekler. Durdurmak için Ctrl+C.")
    _print_clock(trader, client)
    print("=" * 76)
    return _loop(trader, client, store, cfg, args, log, title)


def _loop(trader: PaperTrader, client, store: StateStore, cfg, args, log, title: str) -> int:
    last_offset_refresh = client.milliseconds()
    lag_retries = 0
    try:
        while True:
            try:
                if client.milliseconds() - last_offset_refresh > OFFSET_REFRESH_MS:
                    client.refresh_time_offset()
                    last_offset_refresh = client.milliseconds()
                processed = trader.sync(client.server_now())
                if processed:
                    positions = ", ".join(
                        f"{p.symbol} (stop {p.stop_price:,.2f})" for p in trader.portfolio.positions.values()
                    ) or "-"
                    halted = " | ŞALTER AKTİF" if trader.risk.state.halted else ""
                    closed = processed[-1] + cfg.entry_ms
                    log.info(f"Mum {fmt_utc(closed)} [{fmt_local(closed, with_date=False)}] işlendi ({len(processed)}) | "
                             f"özsermaye {trader.equity():,.2f} | açık: {positions}{halted}")
            except ccxt.BaseError as exc:
                if args.once:
                    print(f"[HATA] Borsa/ağ hatası: {exc}")
                    return 1
                log.warning(f"Borsa/ağ hatası: {exc}. {NETWORK_RETRY_SECONDS} sn sonra tekrar denenecek.")
                time.sleep(NETWORK_RETRY_SECONDS)
                continue
            if args.once:
                return 0
            if trader.lagging and lag_retries < LAG_MAX_RETRIES:
                lag_retries += 1
                time.sleep(LAG_RETRY_SECONDS)
                continue
            if trader.lagging:
                log.warning("Binance verisi gecikiyor; bir sonraki mumda tekrar denenecek.")
            lag_retries = 0
            wake = trader.next_wakeup_ms(client.server_now())
            while client.server_now() < wake:
                time.sleep(1)
    except KeyboardInterrupt:
        # Son tutarlı durum her senkronizasyonun sonunda zaten yazıldı; yarım işlenmiş
        # bir mum varsa yeniden başlatıldığında aynı sonuçla tekrar işlenir.
        print("\nDurduruldu. Durum kayıtlı; tekrar başlattığınızda kaldığı yerden devam eder.")
        print_status(store, title)
        return 0


if __name__ == "__main__":
    sys.exit(main())
