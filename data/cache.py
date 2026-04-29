"""
SQLite-backed cache for NBA API responses.
Prevents repeated slow API calls for the same team within a session.
Default TTL: 6 hours.
"""
import pickle
import sqlite3
import time
from contextlib import contextmanager

from config import CACHE_DB

TTL_DEFAULT = 6 * 3600  # 6 hours


@contextmanager
def _conn():
    db = sqlite3.connect(CACHE_DB)
    try:
        yield db
    finally:
        db.close()


def init():
    with _conn() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                key      TEXT PRIMARY KEY,
                payload  BLOB NOT NULL,
                expires  REAL NOT NULL
            )
        """)
        db.commit()


def get(key: str):
    """Return cached value or None if missing / expired."""
    try:
        with _conn() as db:
            row = db.execute(
                "SELECT payload, expires FROM api_cache WHERE key = ?", (key,)
            ).fetchone()
        if row and row[1] > time.time():
            return pickle.loads(row[0])
    except Exception:
        pass
    return None


def set(key: str, value, ttl: int = TTL_DEFAULT):
    """Store value in cache with expiry."""
    try:
        with _conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO api_cache (key, payload, expires) VALUES (?, ?, ?)",
                (key, pickle.dumps(value), time.time() + ttl),
            )
            db.commit()
    except Exception:
        pass


def clear_expired():
    try:
        with _conn() as db:
            db.execute("DELETE FROM api_cache WHERE expires <= ?", (time.time(),))
            db.commit()
    except Exception:
        pass


def clear_all():
    try:
        with _conn() as db:
            db.execute("DELETE FROM api_cache")
            db.commit()
    except Exception:
        pass


def stats() -> dict:
    """Return cache stats for display in UI."""
    try:
        with _conn() as db:
            total   = db.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
            expired = db.execute(
                "SELECT COUNT(*) FROM api_cache WHERE expires <= ?", (time.time(),)
            ).fetchone()[0]
        return {"total": total, "live": total - expired, "expired": expired}
    except Exception:
        return {"total": 0, "live": 0, "expired": 0}


# Auto-init on import
try:
    init()
except Exception:
    pass
