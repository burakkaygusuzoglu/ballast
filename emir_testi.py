"""Emir tesisatı testi: botun Binance'e emir gönderme zincirini birkaç saniyede baştan sona dener.

Kullanım:
    python emir_testi.py            # Binance TESTNET (sahte para) - önce bunu çalıştırın
    python emir_testi.py --gercek   # GERÇEK hesap: yaklaşık 12 USDT'lik alım ve hemen satış (onay ister)

Adımlar: bakiye -> küçük piyasa alışı -> koruyucu stop emri -> stopu iptal -> satış -> bakiye.
Gerçek hesapta maliyeti komisyon ve alış-satış farkıdır (birkaç sent). Komisyon coin olarak kesildiği
ve miktar borsa adımına (0,0001 ETH) yuvarlandığı için hesapta en fazla bir adımlık ETH artığı kalır;
bu artık kaybolmaz, hesaptadır. Test sonunda ayrıca gösterilir.
"""
from __future__ import annotations

import argparse
import sys
import time

import ccxt

from bot.brokers.orders import SpotOrderAPI, base_asset
from bot.cli import confirm, load_config_or_exit, setup_console
from bot.config import ConfigError, load_api_keys
from bot.exchange import ExchangeClient

TEST_SYMBOL = "ETH/USDT"         # adımı küçük (0.0001 ETH): komisyon sonrası artık önemsiz kalır
TARGET_NOTIONAL = 12.0           # USDT; komisyon ve yuvarlamadan sonra stop fiyatında bile minimumun üstünde
STOP_DISTANCE = 0.90             # test stopu fiyatın %10 altında
FEE_RATE = 0.001


def step(text: str) -> None:
    print(f"\n>> {text}")


def main(argv: list[str] | None = None, client: ExchangeClient | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Binance emir tesisatı testi.")
    parser.add_argument("--gercek", action="store_true", help="GERÇEK hesapta dene (varsayılan: testnet)")
    args = parser.parse_args(argv)
    testnet = not args.gercek
    cfg = load_config_or_exit()
    label = "TESTNET (sahte para)" if testnet else "GERÇEK HESAP"
    print("=" * 76)
    print(f" EMİR TESTİ - {label} - {TEST_SYMBOL}, yaklaşık {TARGET_NOTIONAL:.0f} USDT")
    print("=" * 76)

    if client is None:
        try:
            key, secret = load_api_keys(testnet)
        except ConfigError as exc:
            print(f"[HATA] {exc}")
            return 2
        client = ExchangeClient(cfg, key, secret, testnet=testnet)
    api = SpotOrderAPI(client)
    base = base_asset(TEST_SYMBOL)
    try:
        step("Bağlantı ve izinler")
        client.refresh_time_offset()
        if not testnet:
            permissions = client.api_restrictions()
            if permissions.get("enableWithdrawals"):
                print("[HATA] Anahtarda PARA ÇEKME izni açık. Kapatmadan devam edilmez.")
                return 2
            print(f"[OK]    Para çekme kapalı | spot işlem: {permissions.get('enableSpotAndMarginTrading')} | "
                  f"IP kısıtlaması: {permissions.get('ipRestrict')}")
            if not confirm(f"Gerçek hesapta ~{TARGET_NOTIONAL:.0f} USDT'lik ETH alınıp hemen satılacak. "
                           "'gercek' yazın:", "gercek"):
                print("İptal edildi.")
                return 1
        usdt_before = api.free_balance("USDT")
        print(f"[OK]    Bağlantı tamam. Serbest bakiye: {usdt_before:,.2f} USDT")

        step("Piyasa alışı")
        price = float(client.ex.fetch_ticker(TEST_SYMBOL)["last"])
        rule = client.market_rules([TEST_SYMBOL], cfg.market_rules_default)[TEST_SYMBOL]
        need = rule.min_notional * 1.05 / (price * STOP_DISTANCE)     # stop fiyatında satılabilecek en az miktar
        qty = api.amount(TEST_SYMBOL, max(TARGET_NOTIONAL / price, need / (1 - FEE_RATE)))
        for _ in range(5):
            if api.amount(TEST_SYMBOL, qty * (1 - FEE_RATE)) >= need:
                break
            qty = api.amount(TEST_SYMBOL, qty + rule.qty_step)
        if qty * price * 1.01 > usdt_before:
            print(f"[HATA] Test için yaklaşık {qty * price * 1.01:.2f} USDT serbest bakiye gerekir.")
            return 2
        buy = api.market_buy(TEST_SYMBOL, qty)
        print(f"[OK]    {buy.filled:.6f} {base} @ {buy.average:,.2f} (tutar {buy.cost:,.2f} USDT, "
              f"komisyon {buy.fee_base:.8f} {base} + {buy.fee_quote:.4f} USDT) | durum: {buy.status}")

        step("Koruyucu stop emri (fiyatın %10 altında)")
        # Yalnızca bu testte alınan miktar kullanılır; hesaptaki diğer coinlere dokunulmaz.
        own = api.amount(TEST_SYMBOL, min(buy.filled - buy.fee_base, api.free_balance(base)))
        stop_id = api.place_stop(TEST_SYMBOL, own, price * STOP_DISTANCE)
        stop = api.fetch(TEST_SYMBOL, stop_id)
        print(f"[OK]    Stop emri borsada: kimlik {stop_id}, durum: {stop.status}")

        step("Stop emrini iptal")
        api.cancel(TEST_SYMBOL, stop_id)
        time.sleep(0.5)
        print(f"[OK]    Stop durumu: {api.fetch(TEST_SYMBOL, stop_id).status}")

        step("Piyasa satışı (yalnızca testte alınan miktar)")
        sell = api.market_sell(TEST_SYMBOL, own)
        print(f"[OK]    {sell.filled:.6f} {base} @ {sell.average:,.2f} (tutar {sell.cost:,.2f} USDT) | "
              f"durum: {sell.status}")

        usdt_after = api.free_balance("USDT")
        residue = max(0.0, buy.filled - buy.fee_base - sell.filled)      # borsa adımına yuvarlamadan kalan
        residue_value = residue * sell.average
        print(f"\nSerbest bakiye: {usdt_before:,.2f} -> {usdt_after:,.2f} USDT")
        print(f"Hesapta kalan {base} artığı: {residue:.6f} {base} (~{residue_value:.2f} USDT; sizindir, kaybolmadı)")
        print(f"Testin gerçek maliyeti (komisyon + alış-satış farkı): ~{usdt_before - usdt_after - residue_value:.4f} USDT")
        print("[OK] Emir tesisatı çalışıyor: alış, stop, iptal ve satış başarılı.")
        return 0
    except ccxt.AuthenticationError as exc:
        print(f"[HATA] Anahtar geçersiz ya da yetkisiz: {exc}")
        return 2
    except ccxt.BaseError as exc:
        print(f"[HATA] Borsa hatası: {exc}")
        print("Hesapta test için alınan ETH kalmış olabilir; Binance'te kontrol edin.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
