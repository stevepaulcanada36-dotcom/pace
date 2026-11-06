"""Loading race records from CSV and generating a synthetic demo database.

CSV schema (one row per race, header names case-insensitive)::

    runner_id, name, gender, age, event, year, finish, s5, s10, s15, s20, s25, s30, s35, s40

``name`` and the split columns are optional. Times may be ``h:mm:ss`` or seconds.
Splits are elapsed times at each 5 km point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .domain import MARATHON_KM, N_SEGMENTS, SEGMENT_KM, Race, Runner, ValidationError, parse_time

log = logging.getLogger(__name__)

SPLIT_COLS = ("s5", "s10", "s15", "s20", "s25", "s30", "s35", "s40")
_ALIASES = {
    "runner_id": ("runner_id", "runner", "id"),
    "name": ("name", "runner_name"),
    "gender": ("gender", "sex"),
    "age": ("age",),
    "event": ("event", "race", "marathon"),
    "year": ("year",),
    "finish": ("finish", "finish_time", "time"),
}


@dataclass(frozen=True)
class LoadReport:
    runners: int
    races: int
    rows_skipped: int


def _resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = {c.strip().lower(): c for c in df.columns}
    out: dict[str, str | None] = {}
    for key, aliases in _ALIASES.items():
        out[key] = next((cols[a] for a in aliases if a in cols), None)
    for s in SPLIT_COLS:
        out[s] = cols.get(s)
    for required in ("runner_id", "finish"):
        if out[required] is None:
            raise ValidationError(f"CSV is missing a '{required}' column")
    return out


def runners_from_frame(df: pd.DataFrame) -> tuple[list[Runner], LoadReport]:
    cols = _resolve_columns(df)
    has_splits = all(cols[s] is not None for s in SPLIT_COLS)
    grouped: dict[str, dict] = {}
    skipped = 0
    for rec in df.to_dict("records"):
        get = lambda k: rec.get(cols[k]) if cols.get(k) else None  # noqa: E731
        rid = get("runner_id")
        finish = parse_time(get("finish"))
        if rid is None or (isinstance(rid, float) and np.isnan(rid)) or str(rid).strip() == "":
            skipped += 1
            continue
        rid = str(rid).strip()
        splits = None
        if has_splits:
            vals = [parse_time(get(s)) for s in SPLIT_COLS]
            if all(v is not None for v in vals):
                splits = vals
        age_raw = get("age")
        age = int(age_raw) if age_raw is not None and pd.notna(age_raw) else None
        year_raw = get("year")
        year = int(year_raw) if year_raw is not None and pd.notna(year_raw) else 0
        try:
            race = Race.from_cumulative_splits(str(get("event") or "Marathon"), year, finish, splits, age)
        except (ValidationError, TypeError):
            try:  # keep the result even if splits are inconsistent
                race = Race(str(get("event") or "Marathon"), year, finish, age, None)
            except (ValidationError, TypeError):
                skipped += 1
                continue
        g = str(get("gender") or "M").strip().upper()[:1]
        entry = grouped.setdefault(rid, {"gender": "F" if g == "F" else "M", "name": str(get("name") or ""), "races": []})
        entry["races"].append(race)
    runners = [Runner(rid, e["gender"], e["races"], name="" if e["name"] in ("nan", "None") else e["name"]) for rid, e in grouped.items()]
    report = LoadReport(len(runners), sum(len(r.races) for r in runners), skipped)
    log.info("loaded %d runners, %d races (%d rows skipped)", report.runners, report.races, report.rows_skipped)
    return runners, report


def load_csv(path: str | Path) -> tuple[list[Runner], LoadReport]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return runners_from_frame(df)


def runners_to_frame(runners: list[Runner]) -> pd.DataFrame:
    rows = []
    for r in runners:
        for race in r.races:
            row = {"runner_id": r.runner_id, "name": r.name, "gender": r.gender, "age": race.age,
                   "event": race.event, "year": race.year, "finish": round(race.finish_s, 1)}
            if race.segments_s is not None:
                cum = np.cumsum(race.segments_s)[:8]
                row.update({s: round(float(v), 1) for s, v in zip(SPLIT_COLS, cum)})
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_runners(n: int = 8000, seed: int = 20240318) -> list[Runner]:
    """Simulated runners resembling the Dutch marathon data used in the paper
    (Amsterdam, Eindhoven, Rotterdam, 2008-2019). For demos and tests only."""
    rng = np.random.default_rng(seed)
    events = ("Amsterdam", "Eindhoven", "Rotterdam")
    seg = np.asarray(SEGMENT_KM)
    j = np.arange(N_SEGMENTS)
    runners: list[Runner] = []
    for i in range(n):
        gender = "F" if rng.random() < 0.3 else "M"
        age0 = 20 + int(rng.integers(0, 42))
        base = (14800 if gender == "F" else 13600) * np.exp(rng.normal(0, 0.14))
        fade_tendency = 0.015 + rng.random() * 0.05
        n_races = 2 + int(rng.random() ** 1.4 * 5)
        year = 2008 + int(rng.integers(0, max(1, 12 - n_races)))
        learn = 1.0
        races: list[Race] = []
        for k in range(n_races):
            if year > 2019:
                break
            age = age0 + (year - 2008)
            age_factor = 1 + 0.006 * (age - 40) if age > 40 else 1.0
            bad = rng.random() < 0.13
            t = base * learn * age_factor * np.exp(rng.normal(0, 0.03))
            fade = fade_tendency + rng.normal(0, 0.01)
            if bad:
                t *= 1.05 + rng.random() * 0.1
                fade += 0.05 + rng.random() * 0.06
            prof = 1 + fade * (j / 8 - 0.45) + (np.where(j >= 5, fade * 0.6 * (j - 4) / 4, 0) if bad else 0)
            prof = prof + rng.normal(0, 0.006, N_SEGMENTS)
            prof /= np.dot(prof, seg) / MARATHON_KM
            segments = prof * seg * (t / MARATHON_KM)
            finish = float(segments.sum())
            if 7200 <= finish <= 25200:
                races.append(Race(events[int(rng.integers(0, 3))], year, finish, age, tuple(float(x) for x in segments)))
            learn *= 1 - (0.015 + rng.random() * 0.02) * (1 if k < 3 else 0.3)
            year += 1 + int(rng.random() * 2.2)
        if len(races) >= 2:
            runners.append(Runner(f"R{i + 1:05d}", gender, races))
    return runners
