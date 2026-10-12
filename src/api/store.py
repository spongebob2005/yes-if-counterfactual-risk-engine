"""Small key-value storage for checkout orders and rate limits.

On Vercel the filesystem is read-only and instances come and go, so state lives in Upstash
Redis (REST API, already provisioned for the waitlist). Locally it falls back to SQLite.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol


class Store(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def put(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None: ...
    def hit(self, key: str, window: int) -> int: ...


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL, expires REAL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT v, expires FROM kv WHERE k = ?", (key,)).fetchone()
        if row is None or (row[1] is not None and row[1] < time.time()):
            return None
        return json.loads(row[0])

    def put(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        expires = time.time() + ttl if ttl else None
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO kv (k, v, expires) VALUES (?, ?, ?)", (key, json.dumps(value), expires))

    def hit(self, key: str, window: int) -> int:
        with self._lock:
            current = self.get(key) or {"n": 0}
            current["n"] += 1
            if current["n"] == 1:
                self.put(key, current, window)
            else:
                with self._connect() as db:
                    db.execute("UPDATE kv SET v = ? WHERE k = ?", (json.dumps(current), key))
            return current["n"]


class UpstashStore:
    def __init__(self, url: str, token: str, session: Any = None):
        if session is None:
            import requests

            session = requests.Session()
        self.url, self.token, self.session = url.rstrip("/"), token, session

    def _call(self, *command: Any) -> Any:
        response = self.session.post(self.url, json=[str(c) for c in command],
                                     headers={"Authorization": f"Bearer {self.token}"}, timeout=10)
        response.raise_for_status()
        return response.json().get("result")

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._call("GET", key)
        return json.loads(value) if value else None

    def put(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        if ttl:
            self._call("SET", key, json.dumps(value), "EX", ttl)
        else:
            self._call("SET", key, json.dumps(value))

    def hit(self, key: str, window: int) -> int:
        count = int(self._call("INCR", key))
        if count == 1:
            self._call("EXPIRE", key, window)
        return count


def store_from_env() -> Store:
    url = os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        return UpstashStore(url, token)
    return SQLiteStore(os.getenv("CHECKOUT_DATABASE_PATH", ".data/yes-if.sqlite3"))
