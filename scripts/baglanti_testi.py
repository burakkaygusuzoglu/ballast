"""Binance bağlantı testi (API anahtarı gerekmez). Kurulumun 6. adımı bu dosyayı çalıştırır.

Elle çalıştırmak için:  .venv\\Scripts\\python.exe scripts\\baglanti_testi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.cli import load_config_or_exit, setup_console  # noqa: E402
from bot.exchange import ExchangeClient  # noqa: E402


def main() -> int:
    setup_console()
    cfg = load_config_or_exit()
    client = ExchangeClient(cfg, max_retries=3)
    try:
        offset = client.refresh_time_offset()
        print(f"[OK]    Binance sunucu saati alındı (bilgisayar saati farkı {offset / 1000:+.2f} sn)")
        if abs(offset) > 1000:
            print("[UYARI] Bilgisayar saati 1 saniyeden fazla farklı. Bot Binance saatini kullanır; yine de "
                  "Windows Ayarlar > Saat ve dil > Tarih ve saat > 'Şimdi eşitle' (Sync now) önerilir.")
        for symbol in cfg.symbols:
            print(f"[OK]    {symbol} anlık fiyat: {client.last_price(symbol):,.2f} USDT")
        rules = client.market_rules(cfg.symbols, cfg.market_rules_default)
        summary = ", ".join(f"{s} adım {r.qty_step:g} / min. {r.min_notional:g} USDT" for s, r in rules.items())
        print(f"[OK]    Sembol kuralları alındı: {summary}")
    except Exception as exc:  # tanı aracı: her hatayı anlaşılır biçimde göster
        print(f"[HATA]  {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
