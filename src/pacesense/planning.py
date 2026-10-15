"""A coach's planning session for one runner, independent of any user interface.

`PlanSession` holds the state of "Plan a race": the race weights, the pacing plan, where the plan came from,
the model's current prediction and the plan check. Every action returns a `Change` that records what was
changed and what it did to the model's finish time and to the plan's finish time. The Streamlit pages only
call these methods and display the results, so all of the behaviour is testable without a browser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .coach import MAX_PACE_S, MIN_PACE_S, PACE_UNITS, CoachService, finish_from_paces
from .domain import SEGMENT_KM, SEGMENT_LABELS, ValidationError, format_time

DEFAULT_WEIGHT = 0.5
LABELS = tuple(lbl.replace("-", "–") for lbl in SEGMENT_LABELS)


def pace_text(s_per_km: float, unit: str = "km") -> str:
    """'4:59' for a pace in seconds per km, shown per km or per mile (rounded first: never '4:60')."""
    r = int(round(s_per_km * PACE_UNITS[unit]))
    return f"{r // 60}:{r % 60:02d}"


def parse_pace(text: str, unit: str = "km") -> float:
    """'4:59' (minutes:seconds per unit) -> seconds per km."""
    try:
        m, s = str(text).strip().split(":")
        secs = int(m) * 60 + int(s)
        if not 0 <= int(s) < 60:
            raise ValueError
    except ValueError:
        raise ValidationError(f"pace must look like 4:59, got {text!r}") from None
    return secs / PACE_UNITS[unit]


def duration_text(seconds: float) -> str:
    """'1:22' for a duration under an hour, '1:02:03' above."""
    r = int(round(abs(seconds)))
    return f"{r // 3600}:{r % 3600 // 60:02d}:{r % 60:02d}" if r >= 3600 else f"{r // 60}:{r % 60:02d}"


def change_text(delta_s: float) -> str:
    r = round(delta_s)
    if r == 0:
        return "no change"
    return f"{duration_text(r)} {'faster' if r < 0 else 'slower'}"


@dataclass
class Change:
    key: str
    subject: str
    old: str | None
    new: str
    model_before: float
    plan_before: float
    model_after: float
    plan_after: float
    note: str = ""

    @property
    def model_delta(self) -> float:
        return self.model_after - self.model_before

    @property
    def plan_delta(self) -> float:
        return self.plan_after - self.plan_before

    def is_noop(self) -> bool:
        return self.old == self.new and round(self.model_delta) == 0 and round(self.plan_delta) == 0

    def text(self) -> str:
        what = f"{self.subject}: {self.new}." if self.old is None else f"{self.subject} changed from {self.old} to {self.new}."
        effects = []
        if round(self.model_delta):
            effects.append(f"model {change_text(self.model_delta)} ({format_time(self.model_after)})")
        if round(self.plan_delta):
            effects.append(f"your plan {change_text(self.plan_delta)} ({format_time(self.plan_after)})")
        tail = (", ".join(effects)[0].upper() + ", ".join(effects)[1:] + ".") if effects else "No change to the finish times."
        return f"{what} {tail}{(' ' + self.note) if self.note else ''}"


@dataclass
class PlanSession:
    coach: CoachService
    runner_id: str
    unit: str = "km"
    overview: dict = field(init=False)
    weights: list[float] = field(init=False)
    paces: list[float] = field(init=False)
    source: str = field(init=False, default="model")     # "model", a strategy key, or "custom"
    prediction: dict = field(init=False)
    check: dict = field(init=False)
    changes: list[Change] = field(init=False, default_factory=list)   # newest first
    version: int = field(init=False, default=0)          # bumps on every change (useful as a widget key)
    saved_plan_id: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.unit not in PACE_UNITS:
            raise ValidationError(f"unit must be one of {sorted(PACE_UNITS)}")
        self.overview = self.coach.overview(self.runner_id)
        self.weights = [DEFAULT_WEIGHT] * len(self.overview["races"])
        self._refresh_prediction(reset_plan=True)

    @classmethod
    def from_saved_plan(cls, coach: CoachService, plan: dict, unit: str = "km") -> "PlanSession":
        s = cls(coach, plan["runner_id"], unit)
        if len(plan["weights"]) == len(s.weights):
            s.weights = list(plan["weights"])
        s._refresh_prediction(reset_plan=False)
        s.paces, s.source, s.saved_plan_id = list(plan["paces"]), "custom", plan["id"]
        s._refresh_check()
        return s

    # ---- read-only views ---------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.overview["name"]

    @property
    def model_finish(self) -> float:
        return self.prediction["finish_s"]

    @property
    def plan_finish(self) -> float:
        return self.check["finish_s"]

    @property
    def strategies(self) -> list[dict]:
        return self.prediction["scenarios"]

    def strategy_name(self, key: str) -> str:
        if key == "custom":
            return "Your edited plan"
        return next((s["name"] for s in self.strategies if s["key"] == key), key)

    def interval_rows(self) -> list[dict[str, Any]]:
        """Pace, time and elapsed time per interval (times add up to the plan's finish)."""
        rows, elapsed = [], 0.0
        for label, km, p in zip(LABELS, SEGMENT_KM, self.paces):
            t = p * km
            elapsed += t
            rows.append({"interval": f"{label} km", "distance_km": km, "pace": pace_text(p, self.unit),
                         "time": duration_text(t), "elapsed": format_time(elapsed)})
        return rows

    # ---- internal ------------------------------------------------------------------------------
    def _refresh_prediction(self, reset_plan: bool) -> None:
        self.prediction = self.coach.predict(self.runner_id, self.weights, self.unit)
        if reset_plan or self.source != "custom":
            key = "model" if reset_plan else self.source
            chosen = next((s for s in self.strategies if s["key"] == key), self.strategies[0])
            self.paces = list(chosen["paces"])
            if reset_plan:
                self.source = "model"
        self._refresh_check()

    def _refresh_check(self) -> None:
        self.check = self.coach.evaluate_plan(self.runner_id, self.weights, self.paces, self.unit)

    def _record(self, key: str, subject: str, old: str | None, new: str, before: tuple[float, float],
                note: str = "") -> Change | None:
        after = (self.model_finish, self.plan_finish)
        last = self.changes[0] if self.changes else None
        if last and last.key == key:            # repeated edits of one control merge into one entry
            last.new, last.model_after, last.plan_after, last.note = new, *after, note
            change = last
        else:
            change = Change(key, subject, old, new, *before, *after, note=note)
            self.changes.insert(0, change)
        if change.is_noop():
            self.changes.remove(change)
            return None
        self.version += 1
        return change

    # ---- actions -------------------------------------------------------------------------------
    def set_unit(self, unit: str) -> None:
        if unit not in PACE_UNITS:
            raise ValidationError(f"unit must be one of {sorted(PACE_UNITS)}")
        if unit != self.unit:
            self.unit = unit
            self.prediction = self.coach.predict(self.runner_id, self.weights, unit)
            self._refresh_check()
            self.version += 1

    def set_weight(self, i: int, weight: float) -> Change | None:
        if not 0 <= i < len(self.weights):
            raise ValidationError("no such race")
        weight = round(float(weight), 1)
        if not 0.0 <= weight <= 1.0:
            raise ValidationError("weights must be between 0 and 1")
        old = self.weights[i]
        if weight == old:
            return None
        before = (self.model_finish, self.plan_finish)
        self.weights[i] = weight
        try:
            self._refresh_prediction(reset_plan=False)
        except ValidationError:
            self.weights[i] = old
            raise
        race = self.overview["races"][i]
        note = ("Your edited pacing plan stayed as it was; pick a strategy to rebuild it from the new prediction."
                if self.source == "custom" else "")
        return self._record(f"w{i}", f"{race['event']} {race['year']} weight", f"{old:.1f}", f"{weight:.1f}", before, note)

    def reset_weights(self) -> Change | None:
        if all(w == DEFAULT_WEIGHT for w in self.weights):
            return None
        before = (self.model_finish, self.plan_finish)
        self.weights = [DEFAULT_WEIGHT] * len(self.weights)
        self._refresh_prediction(reset_plan=False)
        return self._record("reset", "All race weights", None, "back to equal (0.5)", before)

    def choose_strategy(self, key: str) -> Change | None:
        chosen = next((s for s in self.strategies if s["key"] == key), None)
        if chosen is None:
            raise ValidationError(f"unknown strategy {key!r}")
        if key == self.source:
            return None
        before, old = (self.model_finish, self.plan_finish), self.strategy_name(self.source)
        self.paces, self.source = list(chosen["paces"]), key
        self._refresh_check()
        return self._record("strategy", "Pacing strategy", old, chosen["name"], before)

    def set_pace(self, j: int, s_per_km: float) -> Change | None:
        if not 0 <= j < len(self.paces):
            raise ValidationError("no such interval")
        factor = PACE_UNITS[self.unit]
        new = round(min(MAX_PACE_S, max(MIN_PACE_S, float(s_per_km))) * factor) / factor  # whole seconds in the unit read
        old = self.paces[j]
        if abs(new - old) < 0.01:
            return None
        before = (self.model_finish, self.plan_finish)
        self.paces[j], self.source = new, "custom"
        self._refresh_check()
        u = self.unit
        return self._record(f"pace{j}", f"Pace for {LABELS[j]} km", f"{pace_text(old, u)}/{u}", f"{pace_text(new, u)}/{u}", before)

    def set_pace_texts(self, texts: Sequence[str]) -> list[Change]:
        """Apply edited 'm:ss' paces (one per interval, in the session's unit). Invalid input changes nothing."""
        if len(texts) != len(self.paces):
            raise ValidationError(f"expected {len(self.paces)} paces")
        parsed = [parse_pace(t, self.unit) for t in texts]          # validate everything first
        changes = []
        for j, p in enumerate(parsed):
            if pace_text(p, self.unit) != pace_text(self.paces[j], self.unit):
                c = self.set_pace(j, p)
                if c:
                    changes.append(c)
        return changes

    def save(self, target_race: str, race_date: str | None = None, notes: str = "") -> dict:
        plan = self.coach.save_plan(self.runner_id, target_race, race_date, self.weights, self.paces, notes)
        self.saved_plan_id = plan["id"]
        return plan

    def plan_text(self, target_race: str = "", race_date: str | None = None, notes: str = "") -> str:
        """The plan as plain text a coach can send to the runner."""
        lines = [f"{self.name}: {target_race or 'next marathon'}{f' ({race_date})' if race_date else ''}",
                 f"Target finish: {format_time(self.plan_finish)}", "",
                 f"{'Interval':<14}{'Pace':>10}{'Time':>9}{'Elapsed':>10}"]
        for r in self.interval_rows():
            lines.append(f"{r['interval']:<14}{r['pace'] + '/' + self.unit:>10}{r['time']:>9}{r['elapsed']:>10}")
        if notes:
            lines += ["", f"Notes: {notes}"]
        return "\n".join(lines) + "\n"

    def total_check(self) -> bool:
        """The interval times always add up to the plan's finish time."""
        return abs(finish_from_paces(self.paces) - self.plan_finish) < 0.5
