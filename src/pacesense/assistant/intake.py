"""Checking and correcting runner details before they are stored.

Used by the assistant when a coach describes a runner in their own words ("Anna, female, 34, ran Amsterdam
2023 in 3h35"), and usable on its own. Nothing here talks to an AI or to the database: it takes a draft,
returns the cleaned-up runner, every correction it made, every problem it found and every missing field.
The coach always confirms before anything is saved.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from ..domain import MAX_FINISH_S, MIN_FINISH_S, format_time

GENDERS = {"f": "F", "female": "F", "woman": "F", "w": "F", "v": "F", "m": "M", "male": "M", "man": "M"}
_UNIT_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in(?:utes?|s)?)?)?\s*"
                      r"(?:(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?|s)?)?)?\s*$", re.I)
SPLIT_KM = (5, 10, 15, 20, 25, 30, 35, 40)
MIN_PACE_S, MAX_PACE_S = 150, 900          # 2:30 to 15:00 per km, as in the planner


def parse_duration(value: Any) -> float | None:
    """Seconds from '3:35:48', '3:35' (h:mm), '3h35m48s', '3h35', '215 min', '215' (minutes) or a number.

    Plain numbers below 1000 are read as minutes, larger ones as seconds, which is how coaches tend to
    write them. Returns None when the text is not a time.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v * 60 if v < 1000 else v
    text = str(value).strip().lower().replace(",", ".")
    if not text:
        return None
    if ":" in text:
        try:
            parts = [float(p) for p in text.split(":")]
        except ValueError:
            return None
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = parts[0], parts[1], 0.0
        else:
            return None
        if m >= 60 or s >= 60 or min(parts) < 0:
            return None
        return h * 3600 + m * 60 + s
    try:
        v = float(text)
        return v * 60 if v < 1000 else v
    except ValueError:
        pass
    text = re.sub(r"(h(?:ours?|rs?)?)\s*(\d+(?:\.\d+)?)$", r"\1 \2m", text)  # '3h35', '3 hours 58' -> minutes
    text = re.sub(r"(m(?:in)?)\s*(\d+(?:\.\d+)?)$", r"\1\2s", text)     # '3h35m48' -> '3h35m48s'
    m = _UNIT_RE.match(text)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s


def _split_time(value: Any) -> float | None:
    """A split is usually under 3 hours and often written m:ss or mm:ss ('24:10'), so two-part times
    below 3:00 hours that look like minutes are read as minutes:seconds."""
    text = str(value).strip()
    if text.count(":") == 1:
        a, b = text.split(":")
        try:
            a_i, b_i = float(a), float(b)
        except ValueError:
            return None
        if a_i >= 4 and b_i < 60:          # '24:10' = 24 min 10 s, not 24 hours
            return a_i * 60 + b_i
    return parse_duration(value)


def _clock(s: float) -> str:
    r = int(round(s))
    return f"{r // 3600}:{r % 3600 // 60:02d}:{r % 60:02d}"


def check_runner(draft: dict[str, Any], today: dt.date | None = None) -> dict[str, Any]:
    """Validate and normalise a runner draft.

    draft = {"name": str, "gender": "F"/"female"/..., "age": int?, "races": [{"event", "year", "finish",
    "splits": [8 times at 5..40 km]?}, ...]}

    Returns {"ok", "runner" (cleaned, ready for CoachService.add_runner via runner_payload_races),
    "corrections", "problems", "missing"}. ok is True only with no problems and nothing missing.
    """
    today = today or dt.date.today()
    corrections: list[str] = []
    problems: list[str] = []
    missing: list[str] = []

    name = " ".join(str(draft.get("name") or "").split())
    if not name:
        missing.append("the runner's name")
    elif name != str(draft.get("name")):
        corrections.append(f"Tidied the name to '{name}'.")

    g_raw = str(draft.get("gender") or "").strip()
    gender = GENDERS.get(g_raw.lower())
    if not g_raw:
        missing.append("gender (the model compares runners of the same gender)")
    elif gender is None:
        problems.append(f"Gender '{g_raw}' is not recognised; use female or male (the race data only has these two).")
    elif g_raw.lower() not in ("f", "m", "female", "male"):
        corrections.append(f"Read gender '{g_raw}' as {gender}.")

    age = draft.get("age")
    if age in ("", None):
        age = None
    else:
        try:
            age_f = float(age)
            age = int(round(age_f))
            if age_f != age:
                corrections.append(f"Rounded age {age_f:g} to {age}.")
            if not 10 <= age <= 100:
                problems.append(f"Age {age} is outside 10 to 100.")
        except (TypeError, ValueError):
            problems.append(f"Age '{age}' is not a number.")
            age = None

    races_in = draft.get("races") or []
    if not isinstance(races_in, list):
        problems.append("the races must be a list of marathons")
        races_in = []
    if not races_in:
        missing.append("at least one past marathon (event, year and finish time)")
    races: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for n, r in enumerate(races_in, 1):
        if not isinstance(r, dict):
            problems.append(f"Race {n}: '{r}' is not a race with an event, year and finish time.")
            continue
        event = " ".join(str(r.get("event") or "").split()) or "Marathon"
        label = f"Race {n} ({event})"
        year: int | None
        try:
            year = int(r.get("year")) if r.get("year") not in (None, "") else None
        except (TypeError, ValueError):
            problems.append(f"{label}: year '{r.get('year')}' is not a year.")
            year = None
        if year is None:
            missing.append(f"the year of {label.lower()}")
        elif not 1980 <= year <= today.year:
            problems.append(f"{label}: year {year} is not between 1980 and {today.year}.")
        label = f"{event} {year}" if year else label

        raw_finish = r.get("finish")
        finish = parse_duration(raw_finish)
        if finish is None:
            if raw_finish in (None, ""):
                missing.append(f"the finish time of {label}")
            else:
                problems.append(f"{label}: finish '{raw_finish}' is not a time like 3:35:48.")
        else:
            text = _clock(finish)
            if str(raw_finish).strip() != text:
                corrections.append(f"{label}: read finish '{raw_finish}' as {text}.")
            if not MIN_FINISH_S <= finish <= MAX_FINISH_S:
                problems.append(f"{label}: a finish of {text} is outside {format_time(MIN_FINISH_S)} to "
                                f"{format_time(MAX_FINISH_S)}; is it a marathon time?")
        if year is not None:
            if (event.lower(), year) in seen:
                problems.append(f"{label} appears twice.")
            seen.add((event.lower(), year))

        splits = _check_splits(r.get("splits"), finish, label, corrections, problems)
        races.append({"event": event, "year": year or 0, "finish": _clock(finish) if finish else None,
                      "splits": splits})

    runner = {"name": name, "gender": gender, "age": age,
              "races": sorted(races, key=lambda x: (x["year"], x["finish"] or ""))}
    return {"ok": not problems and not missing, "runner": runner, "corrections": corrections,
            "problems": problems, "missing": missing}


def _check_splits(raw: Any, finish: float | None, label: str, corrections: list[str],
                  problems: list[str]) -> list[str] | None:
    if raw in (None, "", []):
        return None
    if isinstance(raw, str):
        items = re.split(r"[,;\s]+", raw.strip())
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        problems.append(f"{label}: splits must be a list of 8 times.")
        return None
    items = [x for x in items if str(x).strip()]
    if len(items) != 8:
        problems.append(f"{label}: {len(items)} split times given; enter 8 (at 5, 10 … 40 km) or none.")
        return None
    secs = [_split_time(x) for x in items]
    if any(s is None for s in secs):
        problems.append(f"{label}: split times must be times like 24:10 or 1:12:30.")
        return None
    increasing = all(b > a for a, b in zip(secs, secs[1:]))
    if not increasing and all(s < 3600 for s in secs):
        # Looks like the time for each 5 km section rather than the elapsed time at each point.
        cum, total = [], 0.0
        for s in secs:
            total += s
            cum.append(total)
        secs = cum
        corrections.append(f"{label}: the splits looked like times per 5 km section, so they were added up "
                           f"into elapsed times at 5 … 40 km.")
    elif not increasing:
        problems.append(f"{label}: elapsed split times must go up at each point.")
        return None
    if finish is not None and secs[-1] >= finish:
        problems.append(f"{label}: the 40 km split ({_clock(secs[-1])}) is not before the finish ({_clock(finish)}).")
        return None
    prev, prev_km = 0.0, 0
    for km, s in zip(SPLIT_KM, secs):
        pace = (s - prev) / (km - prev_km)
        if not MIN_PACE_S <= pace <= MAX_PACE_S:
            problems.append(f"{label}: {prev_km}–{km} km works out at {int(pace // 60)}:{int(pace % 60):02d}/km, "
                            f"which is not a realistic marathon pace. Check the splits.")
            return None
        prev, prev_km = s, km
    if finish is not None:
        last = (finish - secs[-1]) / 2.195
        if not MIN_PACE_S <= last <= MAX_PACE_S:
            problems.append(f"{label}: the last 2.2 km works out at {int(last // 60)}:{int(last % 60):02d}/km; "
                            f"check the 40 km split and the finish time.")
            return None
    return [_clock(s) for s in secs]
