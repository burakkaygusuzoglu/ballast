"""Gerçek bot ve testnet için güvenli başlatma penceresi (bilgisayar kapanıp açıldıktan sonra).

Canlı bot, kapalıyken kaçırdığı kararları geriye dönük vermez; açıldığı anda en son kapanan 4 saatlik mumu
işler. Karar saatinden (UTC 00, 04, 08, 12, 16, 20) en az 1, en fazla 30 dakika sonra açılırsa bu mum tazedir.
Daha geç açılırsa saatlerce eski bir sinyalle, değişmiş fiyattan alım yapılabilir.

Bu araç yalnızca saate bakar; hiçbir botu kendisi başlatmaz ve hiçbir dosyaya yazmaz.

Kullanım:
    python baslat_penceresi.py           # çıkış kodu 0 = pencere açık, 1 = beklenmeli
    python baslat_penceresi.py --bilgi   # yalnızca bilgi (çıkış kodu her zaman 0)
"""
from __future__ import annotations

import argparse
import sys

from bot.cli import setup_console
from bot.timeutil import fmt_local, now_ms

DECISION_MS = 4 * 3_600_000              # C1: 4 saatlik mum, kararlar UTC 00/04/08/12/16/20
OPEN_AFTER_MS = 60_000                   # mum verisinin yayınlanması için 1 dakika
CLOSE_AFTER_MS = 30 * 60_000             # karar saatinden en fazla 30 dakika sonra


def window(now: int) -> tuple[bool, int, int]:
    """(pencere şu an açık mı, pencerenin başlangıcı, bitişi) - UTC milisaniye."""
    close = now // DECISION_MS * DECISION_MS
    if close + OPEN_AFTER_MS <= now <= close + CLOSE_AFTER_MS:
        return True, close + OPEN_AFTER_MS, close + CLOSE_AFTER_MS
    if now < close + OPEN_AFTER_MS:                       # karar saatinin ilk dakikası: az sonra açılır
        return False, close + OPEN_AFTER_MS, close + CLOSE_AFTER_MS
    upcoming = close + DECISION_MS
    return False, upcoming + OPEN_AFTER_MS, upcoming + CLOSE_AFTER_MS


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Gerçek bot ve testnet için güvenli başlatma penceresi.")
    parser.add_argument("--bilgi", action="store_true", help="Yalnızca bilgi ver (çıkış kodu her zaman 0)")
    args = parser.parse_args(argv)
    is_open, start, end = window(now_ms())
    if is_open:
        print(f"[OK] Gerçek bot ve testnet için başlatma penceresi açık ({fmt_local(end, with_date=False)} "
              "kadar).")
        return 0
    print(f"[BEKLE] Gerçek bot ve testnet şu an başlatılmamalı. Sıradaki pencere: {fmt_local(start)} ile "
          f"{fmt_local(end, with_date=False)} arası.")
    print("        Neden: kapalıyken kaçırılan kararlar geriye dönük verilmez; bot açılınca en son kapanan mumu "
          "işler. Eski bir sinyalle, değişmiş fiyattan alım yapılmasın diye karar saatinden hemen sonra açıyoruz.")
    return 0 if args.bilgi else 1


if __name__ == "__main__":
    sys.exit(main())
