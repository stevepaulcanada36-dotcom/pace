"""SQLite storage for study sessions and trials (one row per coach x runner)."""
from __future__ import annotations

import json
import sqlite3
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    condition       TEXT NOT NULL CHECK (condition IN ('interactive', 'noninteractive')),
    coach_label     TEXT,
    unfamiliar_only INTEGER NOT NULL DEFAULT 0,
    data_source     TEXT,
    questionnaire   TEXT,
    created_at      REAL NOT NULL,
    completed_at    REAL
);
CREATE TABLE IF NOT EXISTS trials (
    session_id         TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq                INTEGER NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('planned', 'pre', 'done')),
    familiar           INTEGER NOT NULL,
    runner_id          TEXT NOT NULL,
    runner             TEXT NOT NULL,          -- JSON: gender, age, name, shown races
    truth_s            REAL,                   -- held-out best race (unfamiliar runners only)
    coach_pre          REAL,
    pace_pre           REAL,
    familiarity        INTEGER,
    runner_motivations TEXT,
    model_initial      REAL,
    model_adapted      REAL,
    weights            TEXT,
    interactions       INTEGER NOT NULL DEFAULT 0,
    coach_post         REAL,
    pace_post          REAL,
    motivation         TEXT,
    measures           TEXT,
    started_at         REAL,
    ended_at           REAL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS trials_status ON trials(status);
-- coach dashboard ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coach_runners (
    id         TEXT PRIMARY KEY,
    data       TEXT NOT NULL,            -- JSON runner snapshot (name, gender, age, races)
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id              TEXT PRIMARY KEY,
    runner_id       TEXT NOT NULL,
    runner_name     TEXT NOT NULL,
    target_race     TEXT NOT NULL,
    race_date       TEXT,
    weights         TEXT NOT NULL,       -- JSON list, one per past race
    paces           TEXT NOT NULL,       -- JSON list of 9 paces (s/km)
    finish_s        REAL NOT NULL,       -- the coach's plan
    model_finish_s  REAL NOT NULL,       -- model with the coach's race weights
    default_finish_s REAL NOT NULL,      -- model with equal weights
    low_s           REAL,
    high_s          REAL,
    notes           TEXT,
    created_at      REAL NOT NULL,
    result_s        REAL,                -- actual finish, recorded after the race
    result_at       REAL
);
CREATE INDEX IF NOT EXISTS plans_runner ON plans(runner_id);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                  -- JSON
);
"""

_JSON_COLS = {"runner", "runner_motivations", "weights", "measures", "questionnaire", "data", "paces"}


class Repository:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        # Close the connection when this repository is garbage-collected, so a throwaway
        # repository (tests, one-off scripts) never leaks an open SQLite handle.
        self._finalizer = weakref.finalize(self, self._conn.close)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock:
            self._conn.executescript(_SCHEMA)  # executescript manages its own transaction
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._finalizer()

    @staticmethod
    def _encode(values: dict[str, Any]) -> dict[str, Any]:
        return {k: (json.dumps(v) if k in _JSON_COLS and v is not None else v) for k, v in values.items()}

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        for k in _JSON_COLS & out.keys():
            if out[k] is not None:
                out[k] = json.loads(out[k])
        return out

    # sessions -----------------------------------------------------------------
    def insert_session(self, **values: Any) -> None:
        v = self._encode(values)
        cols = ", ".join(v)
        with self.tx() as c:
            c.execute(f"INSERT INTO sessions ({cols}) VALUES ({', '.join('?' * len(v))})", tuple(v.values()))

    def update_session(self, session_id: str, **values: Any) -> None:
        v = self._encode(values)
        with self.tx() as c:
            c.execute(f"UPDATE sessions SET {', '.join(f'{k} = ?' for k in v)} WHERE id = ?", (*v.values(), session_id))

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            return self._decode(self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())

    # trials -------------------------------------------------------------------
    def insert_trials(self, rows: list[dict[str, Any]]) -> None:
        with self.tx() as c:
            for r in rows:
                v = self._encode(r)
                c.execute(f"INSERT INTO trials ({', '.join(v)}) VALUES ({', '.join('?' * len(v))})", tuple(v.values()))

    def update_trial(self, session_id: str, seq: int, **values: Any) -> None:
        v = self._encode(values)
        with self.tx() as c:
            c.execute(f"UPDATE trials SET {', '.join(f'{k} = ?' for k in v)} WHERE session_id = ? AND seq = ?",
                      (*v.values(), session_id, seq))

    def increment_interactions(self, session_id: str, seq: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE trials SET interactions = interactions + 1 WHERE session_id = ? AND seq = ?", (session_id, seq))

    def previous_familiar_runners(self, exclude_session: str | None = None) -> list[dict]:
        """Runners other participants selected as their own, most recently selected first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.runner FROM trials t JOIN sessions s ON s.id = t.session_id "
                "WHERE t.familiar = 1 AND t.session_id != ? ORDER BY s.created_at DESC, t.seq",
                (exclude_session or "",)).fetchall()
        seen, out = set(), []
        for r in rows:
            snap = json.loads(r["runner"])
            if snap["runner_id"] not in seen:
                seen.add(snap["runner_id"])
                out.append(snap)
        return out

    def get_trial(self, session_id: str, seq: int) -> dict | None:
        with self._lock:
            return self._decode(self._conn.execute(
                "SELECT * FROM trials WHERE session_id = ? AND seq = ?", (session_id, seq)).fetchone())

    def list_trials(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM trials WHERE session_id = ? ORDER BY seq", (session_id,)).fetchall()
        return [self._decode(r) for r in rows]

    # coach dashboard ----------------------------------------------------------
    def insert_coach_runner(self, runner_id: str, data: dict) -> None:
        import time as _t
        with self.tx() as c:
            c.execute("INSERT INTO coach_runners (id, data, created_at) VALUES (?, ?, ?)",
                      (runner_id, json.dumps(data), _t.time()))

    def coach_runners(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM coach_runners ORDER BY created_at DESC").fetchall()
        return [json.loads(r["data"]) for r in rows]

    def insert_plan(self, **values: Any) -> None:
        v = self._encode(values)
        with self.tx() as c:
            c.execute(f"INSERT INTO plans ({', '.join(v)}) VALUES ({', '.join('?' * len(v))})", tuple(v.values()))

    def update_plan(self, plan_id: str, **values: Any) -> None:
        v = self._encode(values)
        with self.tx() as c:
            c.execute(f"UPDATE plans SET {', '.join(f'{k} = ?' for k in v)} WHERE id = ?", (*v.values(), plan_id))

    def get_plan(self, plan_id: str) -> dict | None:
        with self._lock:
            return self._decode(self._conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone())

    def list_plans(self, runner_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM plans", ()
        if runner_id:
            sql, args = sql + " WHERE runner_id = ?", (runner_id,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at DESC", args).fetchall()
        return [self._decode(r) for r in rows]

    # settings (dashboard preferences) ----------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, json.dumps(value)))

    # export -------------------------------------------------------------------
    def sessions_frame(self) -> pd.DataFrame:
        with self._lock:
            rows = [self._decode(r) for r in self._conn.execute("SELECT * FROM sessions ORDER BY created_at")]
        out = []
        for r in rows:
            q = r.pop("questionnaire") or {}
            out.append({**r, **{f"q_{k}": v for k, v in q.items()}})
        return pd.DataFrame(out)

    def trials_frame(self, completed_only: bool = True) -> pd.DataFrame:
        sql = "SELECT t.*, s.condition, s.coach_label, s.unfamiliar_only FROM trials t JOIN sessions s ON s.id = t.session_id"
        if completed_only:
            sql += " WHERE t.status = 'done'"
        with self._lock:
            rows = [self._decode(r) for r in self._conn.execute(sql + " ORDER BY s.created_at, t.seq")]
        out = []
        for r in rows:
            measures = r.pop("measures") or {}
            runner = r.pop("runner") or {}
            r["n_races_shown"] = len(runner.get("races", []))
            r["race_years"] = "|".join(str(x["year"]) for x in runner.get("races", []))
            r["weights"] = "|".join(f"{w:.1f}" for w in (r["weights"] or []))
            r["runner_motivations"] = "|".join(r["runner_motivations"] or [])
            out.append({**r, **measures})
        return pd.DataFrame(out)
