"""Loglama: konsol + dönüşümlü (rotating) UTF-8 dosya. Zaman damgaları UTC'dir."""
from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)sZ | %(levelname)-7s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path, level: str = "INFO", filename: str = "bot.log", console: bool = True) -> logging.Logger:
    """'bot' kök logger'ını kurar. Tekrar çağrılırsa eski handler'ları kapatıp temizler."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    formatter.converter = time.gmtime  # UTC

    root = logging.getLogger("bot")
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = RotatingFileHandler(log_dir / filename, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
    root.propagate = False
    return root


def close_logging() -> None:
    """Dosyaları serbest bırakır (Windows'ta açık log dosyası silinemez)."""
    root = logging.getLogger("bot")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
