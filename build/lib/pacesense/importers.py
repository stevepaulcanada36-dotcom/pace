"""Import real marathon results (Kaggle and similar) into PaceSense's race CSV.

    pacesense import data/raw/marathon_results_2015.csv data/raw/marathon_results_2016.csv \
                     data/raw/marathon_results_2017.csv --out data/boston.csv

The model needs the same runner in more than one race (an earlier race paired with a later personal best), but
public results files list one row per finisher per year with no runner id. So runners are linked across files by

    normalised name (word order, accents and punctuation ignored)  +  gender  +  estimated birth year (year - age, ±1)

Rows that cannot be linked safely (two finishers with the same name, gender and birth year in one race) are kept
as separate one-race runners rather than guessed. Everything dropped is counted in the report, and the report is
written next to the output CSV so the import itself is reproducible and checkable.

Columns are found by name (case and punctuation ignored), with presets for common files:

    Boston Marathon 2015-2017 (Kaggle):  Name, M/F, Age, 5K, 10K, ..., 40K, Official Time
    Chicago / Berlin / generic files:    name, gender|sex, age, 5k ... 40k, finish|time|net time|chip time

Use --map to pass your own column names if a file uses different ones.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .domain import MAX_FINISH_S, MIN_FINISH_S, parse_time

SPLIT_KM = (5, 10, 15, 20, 25, 30, 35, 40)
EVENTS = ("boston", "chicago", "berlin", "london", "new york", "nyc", "tokyo", "amsterdam", "rotterdam", "eindhoven",
          "toronto", "paris", "valencia", "sydney")

SYNONYMS: dict[str, tuple[str, ...]] = {
    "name": ("name", "runner", "runner name", "full name", "athlete", "participant"),
    "first": ("first name", "firstname", "first"),
    "last": ("last name", "lastname", "surname", "last"),
    "gender": ("m/f", "gender", "sex", "mf"),
    "age": ("age", "age at race"),
    "finish": ("official time", "finish", "finish time", "net time", "chip time", "time", "official", "final time",
               "result"),
    "year": ("year", "race year", "season"),
    "event": ("event", "race", "marathon"),
}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@dataclass
class ColumnMap:
    name: str | None = None
    first: str | None = None
    last: str | None = None
    gender: str | None = None
    age: str | None = None
    finish: str | None = None
    year: str | None = None
    event: str | None = None
    splits: dict[int, str] = field(default_factory=dict)

    @classmethod
    def detect(cls, columns: Iterable[str], override: dict[str, str] | None = None) -> "ColumnMap":
        cols = {_key(c): c for c in columns}
        m = cls()
        for fld, names in SYNONYMS.items():
            for n in names:
                if _key(n) in cols:
                    setattr(m, fld, cols[_key(n)])
                    break
        for km in SPLIT_KM:
            for cand in (f"{km}k", f"{km}km", f"split {km}k", f"{km} km", f"k{km}", f"s{km}"):
                if _key(cand) in cols:
                    m.splits[km] = cols[_key(cand)]
                    break
        for k, v in (override or {}).items():
            if k.startswith("s") and k[1:].isdigit():
                m.splits[int(k[1:])] = v
            elif hasattr(m, k):
                setattr(m, k, v)
        return m


@dataclass
class ImportReport:
    files: list[str] = field(default_factory=list)
    rows_read: int = 0
    rows_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    runners: int = 0
    runners_with_2plus: int = 0
    ambiguous_names: int = 0
    with_splits: int = 0
    events: dict[str, int] = field(default_factory=dict)
    columns: dict[str, dict] = field(default_factory=dict)

    def text(self) -> str:
        lines = [f"Read {self.rows_read:,} rows from {len(self.files)} file(s); kept {self.rows_kept:,} races "
                 f"({self.with_splits:,} with 5 km splits).",
                 f"Runners: {self.runners:,}, of whom {self.runners_with_2plus:,} ran 2+ races (these build the case "
                 f"base).", "Races per event/year: " + ", ".join(f"{k} {v:,}" for k, v in sorted(self.events.items()))]
        if self.dropped:
            lines.append("Dropped: " + ", ".join(f"{k} {v:,}" for k, v in sorted(self.dropped.items())))
        if self.ambiguous_names:
            lines.append(f"{self.ambiguous_names:,} rows share name, gender and birth year with another finisher in "
                         f"the same race; kept unlinked.")
        return "\n".join(lines)


def normalise_name(name: str) -> str:
    """'Desisa, Lelisa' and 'Lelisa Desisa' -> 'desisa lelisa' (accents and punctuation removed, words sorted)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    words = re.findall(r"[a-z]+", s)
    return " ".join(sorted(words))


def _gender(v) -> str | None:
    s = str(v).strip().upper()[:1]
    return {"F": "F", "W": "F", "M": "M"}.get(s)


def _guess(path: str, pattern: str, options: Sequence[str] = ()) -> str | None:
    name = Path(path).stem.lower().replace("_", " ").replace("-", " ")
    if options:
        return next((o for o in options if o in name), None)
    m = re.search(pattern, name)
    return m.group(0) if m else None


def import_files(paths: Sequence[str], event: str | None = None, year: int | None = None,
                 column_map: dict[str, str] | None = None) -> tuple[pd.DataFrame, ImportReport]:
    rep = ImportReport(files=[str(p) for p in paths])
    drop: Counter = Counter()
    rows: list[dict] = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
        rep.rows_read += len(df)
        m = ColumnMap.detect(df.columns, column_map)
        rep.columns[str(path)] = {k: v for k, v in asdict(m).items() if v}
        if m.finish is None:
            raise ValueError(f"{path}: no finish-time column found (columns: {list(df.columns)[:15]}…). "
                             f"Pass --map finish=YOUR_COLUMN")
        keys = {_key(c) for c in df.columns}
        layout = "boston" if {"officialtime", "projtime", "bib"} <= keys else None   # BAA results format (Kaggle)
        f_event = event or (_guess(str(path), "", EVENTS) or layout or "marathon").title()
        if f_event.lower() in ("nyc", "new york"):
            f_event = "New York"
        y_guess = _guess(str(path), r"(19|20)\d\d")
        for rec in df.to_dict("records"):
            fin = parse_time(rec.get(m.finish))
            if fin is None or not MIN_FINISH_S <= fin <= MAX_FINISH_S:
                drop["finish missing or outside 2:00-7:00"] += 1
                continue
            g = _gender(rec.get(m.gender)) if m.gender else None
            if g is None:
                drop["gender missing"] += 1
                continue
            yr = year or (int(str(rec.get(m.year))[:4]) if m.year and str(rec.get(m.year))[:4].isdigit()
                          else int(y_guess) if y_guess else None)
            if yr is None:
                drop["year unknown (put it in the file name or use --year)"] += 1
                continue
            try:
                age = int(float(rec.get(m.age))) if m.age and str(rec.get(m.age)).strip() else None
            except ValueError:
                age = None
            if age is not None and not 14 <= age <= 95:
                age = None
            if m.name:
                raw_name = rec.get(m.name)
            else:
                raw_name = f"{rec.get(m.first, '')} {rec.get(m.last, '')}" if (m.first or m.last) else ""
            ev = (str(rec.get(m.event)).strip() or f_event) if m.event else f_event
            splits = []
            for km in SPLIT_KM:
                col = m.splits.get(km)
                splits.append(parse_time(rec.get(col)) if col else None)
            ok_splits = (all(s is not None and s > 0 for s in splits)
                         and all(a < b for a, b in zip(splits, splits[1:])) and splits[-1] < fin)
            if any(s is not None for s in splits) and not ok_splits:
                drop["splits incomplete or out of order (race kept without splits)"] += 1
            rows.append({"name": str(raw_name).strip(), "nkey": normalise_name(raw_name), "gender": g, "age": age,
                         "event": ev, "year": yr, "finish": round(fin, 1),
                         **{f"s{km}": (round(s, 1) if ok_splits else None) for km, s in zip(SPLIT_KM, splits)}})
    # ---- link runners across races ----
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["nkey"] and r["age"] is not None:
            groups[(r["nkey"], r["gender"])].append(i)
    rid = [None] * len(rows)
    next_id = 0

    def new_id() -> str:
        nonlocal next_id
        next_id += 1
        return f"K{next_id:06d}"

    for key in sorted(groups):
        idx = groups[key]
        same_race = Counter((rows[i]["event"], rows[i]["year"], rows[i]["year"] - rows[i]["age"]) for i in idx)
        safe, unsafe = [], []
        for i in idx:
            r = rows[i]
            (unsafe if same_race[(r["event"], r["year"], r["year"] - r["age"])] > 1 else safe).append(i)
        rep.ambiguous_names += len(unsafe)
        for i in unsafe:
            rid[i] = new_id()
        # chain by estimated birth year (±1 year: age is on race day, birthdays fall either side)
        safe.sort(key=lambda i: (rows[i]["year"] - rows[i]["age"], rows[i]["year"]))
        current, last_b, seen = None, None, set()
        for i in safe:
            b = rows[i]["year"] - rows[i]["age"]
            race = (rows[i]["event"], rows[i]["year"])
            if current is None or b - last_b > 1 or race in seen:
                current, seen = new_id(), set()
            rid[i], last_b = current, b
            seen.add(race)
    for i in range(len(rows)):
        if rid[i] is None:
            if not rows[i]["nkey"]:
                drop["no name (kept as a one-race runner)"] += 1
            rid[i] = new_id()
    out = pd.DataFrame([{"runner_id": rid[i], "name": r["name"], "gender": r["gender"], "age": r["age"],
                         "event": r["event"], "year": r["year"], "finish": r["finish"],
                         **{f"s{km}": r[f"s{km}"] for km in SPLIT_KM}} for i, r in enumerate(rows)])
    if len(out):
        out = out.sort_values(["runner_id", "year", "event"]).reset_index(drop=True)
        per = out.groupby("runner_id").size()
        rep.runners, rep.runners_with_2plus = int(len(per)), int((per >= 2).sum())
        rep.with_splits = int(out["s40"].notna().sum())
        rep.events = {f"{e} {y}": int(n) for (e, y), n in out.groupby(["event", "year"]).size().items()}
    rep.rows_kept = len(out)
    rep.dropped = dict(drop)
    return out, rep


def write_import(paths: Sequence[str], out_path: str, **kw) -> ImportReport:
    df, rep = import_files(paths, **kw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    Path(str(out_path) + ".import.json").write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    return rep


__all__ = ["ColumnMap", "ImportReport", "import_files", "write_import", "normalise_name"]
