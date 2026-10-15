"""Coach dashboard service: plan a runner's next marathon with the model.

This is the day-to-day tool built on the same model as the study. A coach picks a runner,
weights the past races (the paper's interaction), shapes the pacing plan interval by
interval, compares strategies, saves the plan, and records the result after the race.
Recorded results form the track record and the store a governed learning loop can use.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .cbr import DEFAULT_WEIGHT, RANGE_LABEL, MarathonModel, Prediction
from .domain import (MARATHON_KM, SEGMENT_KM, SEGMENT_LABELS, Race, Runner, ValidationError,
                     format_time, is_valid_finish)
from .storage import Repository
from .study import NotFound, race_from_dict, race_to_dict

_SEG = np.asarray(SEGMENT_KM)
MIN_PACE_S, MAX_PACE_S = 150.0, 900.0      # 2:30 to 15:00 per km
FIRST_HALF = slice(0, 4)                    # 0-20 km
SECOND_HALF = slice(4, 9)                   # 20-42.2 km

# Pace units shown to the coach. Everything is computed in seconds per km; only text is converted.
PACE_UNITS = {"km": 1.0, "mi": 1.609344}

# Pacing strategies, as per-interval offsets in seconds per km applied to the model's plan.
# The description template takes {a} and {b} (the two offsets, converted to the chosen unit) and {u}.
SCENARIOS: tuple[tuple[str, str, str, tuple[float, ...]], ...] = (
    ("model", "Model plan", "Pacing of similar runners' best races", (0,) * 9),
    ("even", "Even pace", "Same pace all the way", ()),            # computed separately
    ("fast_start", "Faster first half", "{a} s/{u} faster to 20 km, {b} s/{u} slower after", (-5,) * 4 + (5,) * 5),
    ("conservative", "Conservative start", "{a} s/{u} slower to 10 km, {b} s/{u} faster after", (10,) * 2 + (-3,) * 7),
    ("negative", "Negative split", "{a} s/{u} slower to 20 km, {b} s/{u} faster after", (5,) * 4 + (-5,) * 5),
)

# Dashboard preferences: allowed values and defaults. "experience" stays None until the coach picks one,
# which is how the dashboard knows to show the first-visit welcome.
PREFERENCE_CHOICES: dict[str, tuple] = {
    "experience": (None, "new", "experienced"),
    "view": ("guided", "full"),
    "explanations": (True, False),
    "unit": tuple(PACE_UNITS),
    "overlay_last_race": (True, False),
}
DEFAULT_PREFERENCES: dict[str, Any] = {"experience": None, "view": "guided", "explanations": True,
                                       "unit": "km", "overlay_last_race": True}


def finish_from_paces(paces: Sequence[float]) -> float:
    return float(np.dot(np.asarray(paces, float), _SEG))


def race_paces(race: Race) -> list[float] | None:
    if race.segments_s is None:
        return None
    return [float(s / km) for s, km in zip(race.segments_s, SEGMENT_KM)]


def _unit(unit: str) -> str:
    if unit not in PACE_UNITS:
        raise ValidationError(f"unit must be one of {sorted(PACE_UNITS)}")
    return unit


def _pace_txt(s_per_km: float, unit: str = "km") -> str:
    """'4:59' for a pace given in seconds per km, shown per km or per mile."""
    r = int(round(s_per_km * PACE_UNITS[unit]))
    return f"{r // 60}:{r % 60:02d}"


def _secs(s_per_km: float, unit: str) -> str:
    """A pace difference in whole seconds of the chosen unit."""
    return f"{abs(s_per_km) * PACE_UNITS[unit]:.0f}"


@dataclass
class CoachService:
    model: MarathonModel
    runners: Sequence[Runner]
    repo: Repository

    def __post_init__(self) -> None:
        self._db = {r.runner_id: r for r in self.runners}

    # runners -----------------------------------------------------------------------
    def _own(self) -> dict[str, Runner]:
        out = {}
        for d in self.repo.coach_runners():
            races = [race_from_dict(x) for x in d["races"]]
            out[d["runner_id"]] = Runner(d["runner_id"], d["gender"], races, name=d.get("name", ""), age=d.get("age"))
        return out

    def runner(self, runner_id: str) -> Runner:
        r = self._own().get(runner_id) or self._db.get(runner_id)
        if r is None:
            raise NotFound(f"runner {runner_id!r} not found")
        return r

    @staticmethod
    def _summary(r: Runner, own: bool) -> dict:
        best = min(r.races, key=lambda x: x.finish_s)
        return {"runner_id": r.runner_id, "name": r.display_name, "gender": r.gender, "age": r.age, "own": own,
                "n_races": len(r.races), "best": format_time(best.finish_s), "best_race": f"{best.event} {best.year}"}

    def search(self, query: str = "", limit: int = 12) -> list[dict]:
        q = query.strip().lower()
        own = [self._summary(r, True) for r in self._own().values()
               if not q or q in r.display_name.lower() or q in r.runner_id.lower()]
        if len(q) < 2:
            return own[:limit]
        hits = []
        for r in self.runners:
            if q in r.runner_id.lower() or (r.name and q in r.name.lower()):
                hits.append(self._summary(r, False))
                if len(hits) + len(own) >= limit:
                    break
        return (own + hits)[:limit]

    def add_runner(self, name: str, gender: str, age: int | None, races: Sequence[Race]) -> dict:
        name = (name or "").strip()
        if not name:
            raise ValidationError("enter the runner's name")
        if gender not in ("F", "M"):
            raise ValidationError("gender must be F or M")
        if not races:
            raise ValidationError("add at least one previous marathon")
        rid = "C-" + uuid.uuid4().hex[:10]
        runner = Runner(rid, gender, list(races), name=name[:120], age=age)
        self.repo.insert_coach_runner(rid, {"runner_id": rid, "name": runner.name, "gender": gender, "age": age,
                                            "races": [race_to_dict(x) for x in runner.races]})
        return self._summary(runner, True)

    # history: how the model's target compared with each later race -------------------
    def history(self, r: Runner) -> list[dict]:
        rows = []
        for k in range(1, len(r.races)):
            before = r.races[max(0, k - 5):k]
            try:
                pred = self.model.predict(r, before)
            except ValidationError:
                continue
            actual = r.races[k]
            rows.append({"event": actual.event, "year": actual.year, "finish_s": actual.finish_s,
                         "finish": format_time(actual.finish_s), "model_s": pred.finish_s,
                         "model": format_time(pred.finish_s), "difference_s": actual.finish_s - pred.finish_s,
                         "in_range": bool(pred.low_s <= actual.finish_s <= pred.high_s)})
        return rows[::-1]

    def overview(self, runner_id: str) -> dict:
        r = self.runner(runner_id)
        races = r.recent(5)
        return {**self._summary(r, runner_id.startswith("C-")),
                "races": [{"index": i, "event": x.event, "year": x.year, "finish_s": x.finish_s,
                           "finish": format_time(x.finish_s), "paces": race_paces(x)} for i, x in enumerate(races)],
                "history": self.history(r),
                "plans": [self._plan_view(p) for p in self.repo.list_plans(runner_id)]}

    # prediction, plan and scenarios ------------------------------------------------
    def _predict(self, r: Runner, weights: Sequence[float] | None) -> tuple[Prediction, Prediction]:
        races = r.recent(5)
        per = self.model.per_race(r, races)
        default = MarathonModel.combine(per, None)
        chosen = MarathonModel.combine(per, weights) if weights is not None else default
        return chosen, default

    def predict(self, runner_id: str, weights: Sequence[float] | None = None, unit: str = "km") -> dict:
        unit = _unit(unit)
        r = self.runner(runner_id)
        if weights is not None:
            weights = [round(float(w), 1) for w in weights]
        pred, default = self._predict(r, weights)
        paces = list(pred.profile * pred.finish_s / MARATHON_KM)
        return {
            "finish_s": pred.finish_s, "finish": format_time(pred.finish_s),
            "low_s": pred.low_s, "high_s": pred.high_s, "range_label": RANGE_LABEL,
            "default_finish_s": default.finish_s, "weights": list(pred.weights), "n_cases": pred.n_cases,
            "paces": paces,
            "per_race": [None if p is None else {"finish_s": p.finish_s, "n_cases": p.n_neighbours} for p in pred.per_race],
            "unit": unit,
            "scenarios": self.scenarios(paces, pred, unit),
            "insights": self.insights(r, paces, pred, unit),
        }

    def scenarios(self, model_paces: Sequence[float], pred: Prediction, unit: str = "km") -> list[dict]:
        base = np.asarray(model_paces, float)
        out = []
        for key, name, desc, offsets in SCENARIOS:
            paces = np.full(9, pred.finish_s / MARATHON_KM) if key == "even" else base + np.asarray(offsets, float)
            f = finish_from_paces(paces)
            if offsets and "{a}" in desc:
                desc = desc.format(a=_secs(offsets[0], unit), b=_secs(offsets[-1], unit), u=unit)
            out.append({"key": key, "name": name, "description": desc, "paces": paces.tolist(),
                        "finish_s": f, "finish": format_time(f), "in_range": bool(pred.low_s <= f <= pred.high_s)})
        return out

    def evaluate_plan(self, runner_id: str, weights: Sequence[float] | None, paces: Sequence[float],
                      unit: str = "km") -> dict:
        """Finish time of a coach-edited pacing plan, and how realistic it is."""
        unit = _unit(unit)
        r = self.runner(runner_id)
        if len(paces) != 9:
            raise ValidationError("a pacing plan needs 9 paces, one per interval")
        p = np.asarray([float(x) for x in paces])
        if np.any(p < MIN_PACE_S) | np.any(p > MAX_PACE_S):
            raise ValidationError("paces must be between 2:30 and 15:00 per km")
        pred, _ = self._predict(r, weights)
        finish = finish_from_paces(p)
        # Warn only where the plan goes clearly beyond BOTH the model's own plan and anything the runner has
        # run on that section. The model aims for a personal best, so its plan is usually faster than the
        # runner's past splits already; warning on that alone would flag almost every interval.
        model_paces = pred.profile * pred.finish_s / MARATHON_KM
        flags = []
        past = [race_paces(x) for x in r.recent(5)]
        past = [x for x in past if x]
        fastest = np.min(np.asarray(past), axis=0) if past else None
        for j in np.where(p < model_paces * 0.97)[0]:
            if fastest is not None and p[j] >= fastest[j]:
                continue
            ran = f" and faster than this runner has run it ({_pace_txt(fastest[j], unit)}/{unit})" if fastest is not None else ""
            flags.append({"interval": SEGMENT_LABELS[j],
                          "message": f"{SEGMENT_LABELS[j].replace('-', '–')} km at {_pace_txt(p[j], unit)}/{unit} is more than 3% faster than "
                                     f"the model's plan for that section ({_pace_txt(model_paces[j], unit)}/{unit}){ran}."})
        if finish < pred.low_s:
            verdict, note = "ambitious", "Faster than 80% of similar runners went on to run. Possible, but a stretch."
        elif finish > pred.high_s:
            verdict, note = "cautious", "Slower than 80% of similar runners went on to run. A safe target."
        else:
            verdict, note = "realistic", "Within the range similar runners went on to run."
        return {"finish_s": finish, "finish": format_time(finish), "model_finish_s": pred.finish_s,
                "difference_s": finish - pred.finish_s, "low_s": pred.low_s, "high_s": pred.high_s,
                "verdict": verdict, "note": note, "flags": flags,
                "interval_times_s": (p * _SEG).tolist(), "cumulative_s": np.cumsum(p * _SEG).tolist()}

    def insights(self, r: Runner, plan_paces: Sequence[float], pred: Prediction, unit: str = "km") -> list[str]:
        out = []
        plan = np.asarray(plan_paces, float)
        rel_plan = plan / (pred.finish_s / MARATHON_KM)
        with_splits = [x for x in r.recent(5) if x.segments_s]
        if with_splits:
            closest = min(with_splits, key=lambda x: float(np.sqrt(np.mean((x.profile() - rel_plan) ** 2))))
            out.append(f"The plan's pacing is closest to {closest.event} {closest.year} ({format_time(closest.finish_s)}).")
        first = float(np.dot(plan[FIRST_HALF], _SEG[FIRST_HALF]) / _SEG[FIRST_HALF].sum())
        second = float(np.dot(plan[SECOND_HALF], _SEG[SECOND_HALF]) / _SEG[SECOND_HALF].sum())
        d = second - first
        if abs(d) < 2:
            out.append("The plan runs both halves at about the same pace.")
        elif d > 0:
            out.append(f"The plan runs the second half {_secs(d, unit)} s/{unit} slower than the first.")
        else:
            out.append(f"The plan is a negative split: the second half is {_secs(d, unit)} s/{unit} faster.")
        if with_splits:
            fades = sum(1 for x in with_splits if np.mean(race_paces(x)[6:9]) > np.mean(race_paces(x)[0:2]) * 1.05)
            if fades:
                out.append(f"Slowed by more than 5% after 30 km in {fades} of {len(with_splits)} past races.")
            else:
                out.append(f"Held pace to the finish in all {len(with_splits)} past races with splits.")
        width = pred.high_s - pred.low_s
        if np.isfinite(width):
            out.append(f"Based on {pred.n_cases} similar runners; their results span "
                       f"{format_time(pred.low_s)} to {format_time(pred.high_s)} ({int(round(width / 60))} min).")
        return out

    # preferences --------------------------------------------------------------------
    def preferences(self) -> dict:
        stored = self.repo.get_setting("preferences", {}) or {}
        return {k: stored.get(k, v) if stored.get(k, v) in PREFERENCE_CHOICES[k] else v
                for k, v in DEFAULT_PREFERENCES.items()}

    def update_preferences(self, changes: dict[str, Any]) -> dict:
        if not isinstance(changes, dict) or not changes:
            raise ValidationError("send at least one preference to change")
        unknown = sorted(set(changes) - set(PREFERENCE_CHOICES))
        if unknown:
            raise ValidationError(f"unknown preference: {', '.join(unknown)}")
        for k, v in changes.items():
            # bools must be real bools (JSON true/false), not 0/1
            if v not in PREFERENCE_CHOICES[k] or (isinstance(v, (int, float)) and not isinstance(v, bool)
                                                   and PREFERENCE_CHOICES[k] == (True, False)):
                raise ValidationError(f"{k} must be one of {list(PREFERENCE_CHOICES[k])}")
        merged = {**self.preferences(), **changes}
        self.repo.set_setting("preferences", merged)
        return merged

    # plans and outcomes ------------------------------------------------------------
    def save_plan(self, runner_id: str, target_race: str, race_date: str | None, weights: Sequence[float] | None,
                  paces: Sequence[float], notes: str = "") -> dict:
        r = self.runner(runner_id)
        if not (target_race or "").strip():
            raise ValidationError("enter the target race")
        ev = self.evaluate_plan(runner_id, weights, paces)
        pred, default = self._predict(r, weights)
        pid = uuid.uuid4().hex[:12]
        self.repo.insert_plan(id=pid, runner_id=runner_id, runner_name=r.display_name, target_race=target_race.strip()[:120],
                              race_date=(race_date or None), weights=list(pred.weights), paces=[float(x) for x in paces],
                              finish_s=ev["finish_s"], model_finish_s=pred.finish_s, default_finish_s=default.finish_s,
                              low_s=pred.low_s, high_s=pred.high_s, notes=(notes or "")[:5000], created_at=time.time())
        return self._plan_view(self.repo.get_plan(pid))

    def update_notes(self, plan_id: str, notes: str) -> dict:
        self._plan(plan_id)
        self.repo.update_plan(plan_id, notes=(notes or "")[:5000])
        return self._plan_view(self.repo.get_plan(plan_id))

    def record_result(self, plan_id: str, finish_s: float) -> dict:
        self._plan(plan_id)
        if not is_valid_finish(finish_s):
            raise ValidationError("the result must be between 2:00:00 and 7:00:00")
        self.repo.update_plan(plan_id, result_s=float(finish_s), result_at=time.time())
        return self._plan_view(self.repo.get_plan(plan_id))

    def _plan(self, plan_id: str) -> dict:
        p = self.repo.get_plan(plan_id)
        if p is None:
            raise NotFound(f"plan {plan_id!r} not found")
        return p

    @staticmethod
    def _plan_view(p: dict) -> dict:
        v = {k: p[k] for k in ("id", "runner_id", "runner_name", "target_race", "race_date", "weights", "paces",
                               "finish_s", "model_finish_s", "default_finish_s", "low_s", "high_s", "notes",
                               "created_at", "result_s")}
        v["finish"] = format_time(p["finish_s"])
        v["result"] = format_time(p["result_s"]) if p["result_s"] else None
        if p["result_s"]:
            res = p["result_s"]
            v["errors_pct"] = {k: 100 * abs(res - p[f]) / res for k, f in
                               (("plan", "finish_s"), ("model_with_weights", "model_finish_s"), ("model_default", "default_finish_s"))}
            v["in_range"] = bool(p["low_s"] <= res <= p["high_s"]) if p["low_s"] else None
        return v

    def list_plans(self) -> list[dict]:
        return [self._plan_view(p) for p in self.repo.list_plans()]

    def track_record(self) -> dict:
        done = [self._plan_view(p) for p in self.repo.list_plans() if p["result_s"]]
        if not done:
            return {"n": 0}
        err = lambda k: float(np.mean([d["errors_pct"][k] for d in done]))  # noqa: E731
        better = sum(1 for d in done if d["errors_pct"]["plan"] < d["errors_pct"]["model_default"])
        inside = [d["in_range"] for d in done if d["in_range"] is not None]
        return {"n": len(done), "plan_error_pct": err("plan"), "model_with_weights_error_pct": err("model_with_weights"),
                "model_default_error_pct": err("model_default"), "plan_better_than_model": better,
                "in_range_pct": 100 * float(np.mean(inside)) if inside else None}


def runner_payload_races(items: Sequence[dict[str, Any]], age: int | None, parse) -> list[Race]:
    """Races from an API payload: [{event, year, finish, splits?}] with times as text or seconds."""
    races = []
    for it in items:
        splits = it.get("splits")
        races.append(Race.from_cumulative_splits(str(it.get("event") or "Marathon"), int(it.get("year") or 0),
                                                 parse(it.get("finish"), "finish"),
                                                 [parse(x, "split") for x in splits] if splits else None, age))
    return races
