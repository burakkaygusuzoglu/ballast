"""Zaman yardımcıları.

Sistemin içinde tüm zamanlar UTC milisaniyedir (Unix epoch). Bot hangi ülkede çalışırsa çalışsın
hesaplar (mum sınırları, günlük zarar limiti) UTC'ye göredir. Ekranda kolaylık için bilgisayarın
kendi saat dilimi de gösterilir (ör. Üsküp: kışın UTC+1, yazın UTC+2; İstanbul: UTC+3).
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
DAY_MS = 86_400_000


def timeframe_ms(timeframe: str) -> int:
    """'15m' -> 900000. Bilinmeyen zaman diliminde açık hata verir."""
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        valid = ", ".join(TIMEFRAME_MS)
        raise ValueError(f"Desteklenmeyen zaman dilimi: {timeframe!r} (geçerli: {valid})") from None


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_utc(text: str) -> int:
    """'2022-01-01' ya da ISO 8601 metnini UTC milisaniyeye çevirir."""
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def fmt_utc(ms: int, with_time: bool = True) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M UTC") if with_time else dt.strftime("%Y-%m-%d")


def utc_day(ms: int) -> int:
    """UTC gün numarası. Günlük zarar limiti bu değer değiştiğinde sıfırlanır."""
    return ms // DAY_MS


def utc_offset_label(dt: datetime) -> str:
    offset = dt.utcoffset() or timedelta(0)
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    hours, rest = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours}" + (f":{rest:02d}" if rest else "")


def fmt_local(ms: int, with_date: bool = True) -> str:
    """Bilgisayarın kendi saat dilimiyle gösterim: '2026-10-08 06:00 yerel (UTC+2)'."""
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC).astimezone()
    text = dt.strftime("%Y-%m-%d %H:%M" if with_date else "%H:%M")
    return f"{text} yerel ({utc_offset_label(dt)})"
