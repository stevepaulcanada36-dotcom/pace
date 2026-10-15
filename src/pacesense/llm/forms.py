"""The inputs that voice and the AI can fill, and how each is checked.

A form says which fields exist (so the AI knows what to fill), what the refined version is for, and how to
check the result. The checks are plain Python, so a model's answer never goes straight into the database.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..assistant.intake import check_runner, parse_duration
from ..assistant.offline import parse_runner_text
from ..domain import MAX_FINISH_S, MIN_FINISH_S


@dataclass
class FormCheck:
    fields: dict[str, Any]
    corrections: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.missing


@dataclass(frozen=True)
class Form:
    key: str
    title: str
    purpose: str                       # what the refined version is for (goes into the refine prompt)
    schema: str                        # the JSON the extract model must return
    check: Callable[[dict], FormCheck]
    offline: Callable[[str], dict]     # field filling without AI (simple rules)


def _clock(s: float) -> str:
    r = int(round(s))
    return f"{r // 3600}:{r % 3600 // 60:02d}:{r % 60:02d}"


def _check_question(f: dict) -> FormCheck:
    q = " ".join(str(f.get("question") or "").split())
    return FormCheck({"question": q}, missing=[] if q else ["the question"])


def _check_runner(f: dict) -> FormCheck:
    r = check_runner(f)
    return FormCheck(r["runner"], r["corrections"], r["problems"], r["missing"])


def _check_plan(f: dict) -> FormCheck:
    out = FormCheck({"target_race": " ".join(str(f.get("target_race") or "").split()),
                     "race_date": None, "notes": str(f.get("notes") or "").strip()})
    if not out.fields["target_race"]:
        out.missing.append("the target race")
    d = f.get("race_date")
    if d:
        try:
            out.fields["race_date"] = dt.date.fromisoformat(str(d)[:10]).isoformat()
        except ValueError:
            out.problems.append(f"race date '{d}' is not a date like 2026-10-18")
    return out


def _check_result(f: dict) -> FormCheck:
    raw = f.get("finish")
    s = parse_duration(raw) if raw not in (None, "") else None
    out = FormCheck({"finish": _clock(s) if s else None})
    if s is None:
        (out.missing if raw in (None, "") else out.problems).append(
            "the finish time" if raw in (None, "") else f"'{raw}' is not a finish time like 3:31:20")
    elif not MIN_FINISH_S <= s <= MAX_FINISH_S:
        out.problems.append(f"a finish of {_clock(s)} is outside 2:00:00 to 7:00:00")
    elif str(raw).strip() != _clock(s):
        out.corrections.append(f"read '{raw}' as {_clock(s)}")
    return out


_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
                   r"|\b\d+\s*h(?:ours?|rs?)?(?:\s*\d+\s*(?:m(?:in(?:utes?|s)?)?)?(?:\s*\d+\s*(?:s(?:ec(?:onds?|s)?)?)?)?)?\b",
                   re.I)


def _offline_plan(text: str) -> dict:
    d = _DATE.search(text)
    race = re.search(r"\b(?:race|for|running|run)\s+(?:the\s+)?([A-Z][\w' ]*?(?:Marathon|marathon)(?:\s+\d{4})?)", text)
    return {"target_race": race.group(1).strip() if race else "", "race_date": d.group(0) if d else None,
            "notes": text.strip()}


def _offline_result(text: str) -> dict:
    m = _TIME.search(text)
    return {"finish": m.group(0) if m else None}


FORMS: dict[str, Form] = {f.key: f for f in (
    Form("question", "A question for Ask PaceSense",
         "a question to a marathon planning assistant about a runner, a plan or saved results",
         '{"question": "the question, in one or two sentences"}',
         _check_question, lambda text: {"question": text.strip()}),
    Form("runner", "A new runner",
         "a coach's description of a runner to add: name, gender, age, and past marathons with year and finish "
         "time (and 5 km splits if given)",
         '{"name": str, "gender": "female" or "male" or null, "age": int or null, "races": [{"event": str, '
         '"year": int or null, "finish": "h:mm:ss" or as said, "splits": [8 elapsed times at 5..40 km] or null}]}',
         _check_runner, parse_runner_text),
    Form("plan_details", "Plan details",
         "the target race, race date and notes a coach saves with a marathon pacing plan",
         '{"target_race": str, "race_date": "YYYY-MM-DD" or null, "notes": str}', _check_plan, _offline_plan),
    Form("result", "A race result", "the actual finish time of a marathon",
         '{"finish": "h:mm:ss" or null}', _check_result, _offline_result),
)}


# ---- facts in a text: used to check that a refined version kept the facts ---------------------------
_NUM = re.compile(r"(?<![\d:])\d+(?:\.\d+)?(?![\d:])")


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|" \
       r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MDY = re.compile(_MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", re.I)
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MON + r"(?:,?\s+(\d{4}))?\b", re.I)


def _month(name: str) -> int:
    return next(i for i, m in enumerate(_MONTHS, 1) if m.startswith(name.lower().rstrip(".")[:3]))


def facts(text: str) -> set[str]:
    """Dates, times (as seconds) and other numbers in a text, written the same way whatever the style:
    '3h35', '3:35' and '3:35:00' are all the fact 't12900'; 'October 18th 2026' and '2026-10-18' are both
    'md10-18' plus 'n2026'."""
    out: set[str] = set()
    rest = text or ""

    def date(m: int, d: int, y: str | None) -> str:
        out.add(f"md{m}-{d}")
        if y:
            out.add(f"n{int(y)}")
        return " "
    rest = _ISO.sub(lambda m: date(int(m.group(2)), int(m.group(3)), m.group(1)), rest)
    rest = _MDY.sub(lambda m: date(_month(m.group(1)), int(m.group(2)), m.group(3)), rest)
    rest = _DMY.sub(lambda m: date(_month(m.group(2)), int(m.group(1)), m.group(3)), rest)
    for m in _TIME.finditer(rest):
        s = parse_duration(m.group(0))
        if s is not None:
            out.add(f"t{int(round(s))}")
    rest = _TIME.sub(" ", rest)
    for m in _NUM.finditer(rest):
        v = float(m.group(0))
        out.add(f"n{v:g}")
    return out


def fact_text(f: str) -> str:
    if f.startswith("md"):
        m, d = f[2:].split("-")
        return f"{_MONTHS[int(m) - 1].capitalize()} {int(d)}"
    return _clock(float(f[1:])) if f.startswith("t") else f[1:]
