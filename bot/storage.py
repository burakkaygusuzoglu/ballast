"""Paper trading durumunun SQLite'ta saklanması.

Bot kapanıp açıldığında (elektrik, yeniden başlatma, Ctrl+C) kaldığı yerden devam eder.
Durum, yeni işlemler ve özsermaye noktaları TEK bir işlemde (transaction) yazılır;
yarım kalmış bir kayıt oluşmaz.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import MISSING, astuple, fields
from pathlib import Path
from typing import Any

from bot.portfolio import Trade

_TRADE_FIELDS = [f.name for f in fields(Trade)]
# Sonradan eklenen sütunlar (eski veritabanlarında yok): yazarken eklenir, okurken varsayılan değer kullanılır.
_TRADE_DEFAULTS = {f.name: f.default for f in fields(Trade) if f.default is not MISSING}


def trade_columns(conn: sqlite3.Connection) -> list[str]:
    """trades tablosunda gerçekten var olan Trade sütunları (Trade'deki sırayla)."""
    present = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    return [name for name in _TRADE_FIELDS if name in present]


def read_trades(conn: sqlite3.Connection) -> list[Trade]:
    """İşlemleri okur; eski şemada olmayan sütunlar Trade'in varsayılan değerini alır (salt okunur bağlantıda da)."""
    columns = trade_columns(conn)
    missing = [name for name in _TRADE_FIELDS if name not in columns]
    if any(name not in _TRADE_DEFAULTS for name in missing):
        raise ValueError(f"trades tablosunda zorunlu sütun eksik: {missing}")
    rows = conn.execute(f"SELECT {', '.join(columns)} FROM trades ORDER BY id").fetchall()
    return [Trade(**dict(zip(columns, row, strict=True))) for row in rows]
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, {", ".join(_TRADE_FIELDS)});
CREATE TABLE IF NOT EXISTS equity (ts INTEGER PRIMARY KEY, equity REAL NOT NULL);
"""


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.executescript(_SCHEMA)
            present = set(trade_columns(conn))
            for name, default in _TRADE_DEFAULTS.items():      # eski veritabanı: yeni sütunları ekle
                if name not in present:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {name} DEFAULT {default!r}")

    def save(self, snapshot: dict[str, Any], new_trades: list[Trade], new_equity: list[tuple[int, float]]) -> None:
        placeholders = ", ".join("?" for _ in _TRADE_FIELDS)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO kv(key, value) VALUES ('snapshot', ?)", (json.dumps(snapshot),))
            conn.executemany(
                f"INSERT INTO trades({', '.join(_TRADE_FIELDS)}) VALUES ({placeholders})",
                [astuple(t) for t in new_trades],
            )
            conn.executemany("INSERT OR REPLACE INTO equity(ts, equity) VALUES (?, ?)", new_equity)

    def load_snapshot(self) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute("SELECT value FROM kv WHERE key = 'snapshot'").fetchone()
        return json.loads(row[0]) if row else None

    def load_trades(self) -> list[Trade]:
        with closing(sqlite3.connect(self.path)) as conn:
            return read_trades(conn)

    def load_equity(self) -> list[tuple[int, float]]:
        with closing(sqlite3.connect(self.path)) as conn:
            return [(int(t), float(e)) for t, e in conn.execute("SELECT ts, equity FROM equity ORDER BY ts")]

    def reset(self) -> None:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("DELETE FROM kv")
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM equity")
