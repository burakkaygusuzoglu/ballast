"""Komut satırı araçları için ortak yardımcılar."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from bot.config import Config, ConfigError, config_path, load_config


def setup_console() -> None:
    """Windows konsolunda yazdırılamayan bir karakter programı asla çökertmesin."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080


def disable_quick_edit() -> None:
    """Windows konsolunda "Hızlı Düzenleme" kapatılır. Açıkken pencereye tıklamak metin seçimi başlatır ve
    program bir sonraki yazdırmada donar (başlıkta "Seç/Select" görünür); sürekli çalışan bot için tehlikeli."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-10)                  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return                                           # konsol değil (ör. yönlendirilmiş giriş)
        kernel32.SetConsoleMode(handle, (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE)
    except Exception:                                        # en kötü ihtimalle ayar değişmez
        pass


def load_config_or_exit() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        print(f"[HATA] {exc}")
        raise SystemExit(2) from None


def config_hash() -> str:
    return hashlib.sha256(config_path().read_bytes()).hexdigest()


def confirm(question: str, expected: str) -> bool:
    try:
        answer = input(f"{question} ").strip().lower()
    except EOFError:
        return False
    return answer == expected.lower()


def open_file(path: Path) -> None:
    """Dosyayı Windows'ta varsayılan programla açar; başka sistemlerde sessizce geçer."""
    if sys.platform == "win32":
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError:
            pass
