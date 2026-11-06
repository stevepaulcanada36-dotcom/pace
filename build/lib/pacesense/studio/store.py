"""An in-memory session store with idle expiry, a size cap and one lock per session.

Sessions hold a live model (weights, steering history), so they stay in process memory; run one worker process
and scale with threads (the model work is numpy and short). The per-session lock serialises requests for one
session without blocking other coaches.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class SessionNotFound(KeyError):
    pass


@dataclass
class Entry(Generic[T]):
    value: T
    created: float
    touched: float
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionStore(Generic[T]):
    def __init__(self, ttl_s: float = 3600, max_size: int = 200, clock=time.monotonic) -> None:
        self.ttl_s, self.max_size, self._clock = ttl_s, max_size, clock
        self._items: OrderedDict[str, Entry[T]] = OrderedDict()
        self._lock = threading.Lock()
        self.created_total = 0
        self.expired_total = 0

    def _sweep(self, now: float) -> None:
        while self._items:
            sid, e = next(iter(self._items.items()))
            if now - e.touched <= self.ttl_s and len(self._items) <= self.max_size:
                break
            del self._items[sid]
            self.expired_total += 1

    def add(self, value: T) -> str:
        sid = secrets.token_urlsafe(16)
        now = self._clock()
        with self._lock:
            self._items[sid] = Entry(value, now, now)
            self.created_total += 1
            self._sweep(now)
        return sid

    def entry(self, sid: str) -> Entry[T]:
        now = self._clock()
        with self._lock:
            self._sweep(now)
            e = self._items.get(sid)
            if e is None:
                raise SessionNotFound(sid)
            e.touched = now
            self._items.move_to_end(sid)
            return e

    def remove(self, sid: str) -> None:
        with self._lock:
            if self._items.pop(sid, None) is None:
                raise SessionNotFound(sid)

    def __len__(self) -> int:
        with self._lock:
            self._sweep(self._clock())
            return len(self._items)

    def stats(self) -> dict:
        return {"active": len(self), "created_total": self.created_total, "expired_total": self.expired_total,
                "ttl_s": self.ttl_s, "max_size": self.max_size}


__all__ = ["Entry", "SessionNotFound", "SessionStore"]
