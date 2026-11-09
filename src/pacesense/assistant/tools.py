"""The PaceSense functions the assistant may call.

The AI never calculates a finish time, a range or a pace itself. When it needs one, it asks for one of the
functions below, PaceSense runs it with the same code the dashboard uses, and the AI words the answer from
the result. Every call and its result is shown to the coach under the answer ("What PaceSense calculated"),
so each number can be traced back.

Two deliberate limits:
- No function reads the research study's data. Coaches taking part in the study must see the paper's
  procedure, not an AI's commentary, and participants' answers are not sent to an outside service.
- No function saves anything. propose_runner only prepares a runner; the coach adds it with a button.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import racewatch
from ..coach import CoachService
from ..domain import ValidationError, format_time, parse_time
from ..study import NotFound
from ..explain import HELP
from ..planning import LABELS, PlanSession, parse_pace, pace_text
from .intake import check_runner, parse_duration

RACE_SCHEMA = {
    "type": "object",
    "properties": {
        "event": {"type": "string", "description": "Marathon name, e.g. Amsterdam"},
        "year": {"type": "integer"},
        "finish": {"type": "string", "description": "Finish time as the coach wrote it, e.g. 3:35:48 or 3h35"},
        "splits": {"type": "array", "items": {"type": "string"},
                   "description": "Optional: 8 elapsed times at 5, 10 ... 40 km"},
    },
    "required": ["event", "finish"],
}
RUNNER_PROPS = {
    "name": {"type": "string"},
    "gender": {"type": "string", "description": "female or male, as the coach said it"},
    "age": {"type": "integer"},
    "races": {"type": "array", "items": RACE_SCHEMA},
}


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}


SPECS: list[dict] = [
    _fn("find_runners", "Search the coach's own runners and the race database by name or runner number.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("runner_details", "A runner's recent marathons (finish, pace per 5 km section) and how the model's target "
        "compared with each later race.", {"runner_id": {"type": "string"}}, ["runner_id"]),
    _fn("predict", "The model's challenging-yet-realistic finish time, likely range (about 8 in 10 similar runners "
        "finished inside it), number of similar runners, prediction per past race, pacing strategies and insights. "
        "weights: one number 0 to 1 per recent race, oldest first; leave out for equal weights (0.5 each).",
        {"runner_id": {"type": "string"}, "weights": {"type": "array", "items": {"type": "number"}}}, ["runner_id"]),
    _fn("check_pacing_plan", "Finish time of a pacing plan and whether it is realistic, ambitious or cautious, with "
        "warnings. paces: 9 paces as m:ss in the coach's unit, for 0-5, 5-10 ... 35-40, 40-42.2 km.",
        {"runner_id": {"type": "string"}, "paces": {"type": "array", "items": {"type": "string"}},
         "weights": {"type": "array", "items": {"type": "number"}}}, ["runner_id", "paces"]),
    _fn("current_plan", "The plan the coach has open on the Plan a race page: runner, race weights, paces, finish, "
        "check and the list of changes the coach made."),
    _fn("watch_moment", "What the race-day watch shows at a moment in the race for the open plan (or a saved plan): "
        "ahead or behind, target pace, projected finish and the cue.",
        {"distance_km": {"type": "number"}, "elapsed": {"type": "string", "description": "h:mm:ss"},
         "plan_id": {"type": "string", "description": "a saved plan; leave out for the open plan"}},
        ["distance_km", "elapsed"]),
    _fn("saved_plans", "The coach's saved plans, recorded results, and the track record (average error of the "
        "plans and of the model)."),
    _fn("check_runner_details", "Check and correct runner details the coach typed or pasted, before adding the "
        "runner: reads times written in different ways, finds mistakes and lists what is missing. Saves nothing.",
        RUNNER_PROPS, []),
    _fn("propose_runner", "Prepare a new runner for the coach to confirm with a button on screen. Call only when "
        "check_runner_details found no problems and nothing missing. Saves nothing by itself.",
        RUNNER_PROPS, ["name", "gender", "races"]),
    _fn("explain", "PaceSense's own explanation of a term, including whether it comes from the paper.",
        {"topic": {"type": "string", "enum": sorted(HELP)}}, ["topic"]),
]


def _clean(value: Any) -> Any:
    """Make results JSON-safe and short: no NaN/inf, floats rounded."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 2)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


@dataclass
class AssistantTools:
    coach: CoachService
    plan: Callable[[], PlanSession | None] = lambda: None
    unit: str = "km"
    pending_runner: dict | None = field(default=None)

    @property
    def specs(self) -> list[dict]:
        return SPECS

    def call(self, name: str, arguments: dict | str | None) -> dict:
        """Run one function. Errors come back as {"error": ...} so the AI can tell the coach what went wrong."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {"error": "the arguments were not valid JSON"}
        fn = getattr(self, f"_t_{name}", None)
        if fn is None or name not in {s["function"]["name"] for s in SPECS}:
            return {"error": f"there is no function called {name!r}"}
        try:
            return _clean(fn(**(arguments or {})))
        except (ValidationError, NotFound) as e:
            return {"error": str(e)}
        except TypeError as e:            # wrong or missing arguments from the model
            return {"error": f"wrong arguments: {e}"}

    # ---- helpers -------------------------------------------------------------------------------
    def _pace(self, s: float) -> str:
        return f"{pace_text(s, self.unit)}/{self.unit}"

    def _weights(self, runner_id: str, weights: list | None) -> list[float] | None:
        if weights is None:
            return None
        n = len(self.coach.overview(runner_id)["races"])
        if len(weights) != n:
            raise ValidationError(f"this runner has {n} recent races, so give {n} weights (oldest first)")
        if any(not 0 <= float(w) <= 1 for w in weights):
            raise ValidationError("weights must be between 0 and 1")
        return [float(w) for w in weights]

    # ---- the functions ---------------------------------------------------------------------------
    def _t_find_runners(self, query: str) -> dict:
        hits = self.coach.search(query, limit=10)
        return {"runners": [{k: h[k] for k in ("runner_id", "name", "gender", "age", "n_races", "best", "best_race", "own")}
                            for h in hits], "note": None if hits else "no runners match"}

    def _t_runner_details(self, runner_id: str) -> dict:
        o = self.coach.overview(runner_id)
        return {
            "runner_id": o["runner_id"], "name": o["name"], "gender": o["gender"], "age": o["age"],
            "recent_races_oldest_first": [
                {"index": r["index"], "race": f"{r['event']} {r['year']}", "finish": r["finish"],
                 "pace_per_section": None if not r["paces"] else dict(zip(LABELS, (self._pace(p) for p in r["paces"])))}
                for r in o["races"]],
            "model_target_vs_later_races": [
                {"race": f"{h['event']} {h['year']}", "actual": h["finish"], "model_target": h["model"],
                 "difference": f"{'+' if h['difference_s'] >= 0 else '-'}{format_time(abs(h['difference_s']))}",
                 "inside_likely_range": h["in_range"]} for h in o["history"]],
            "saved_plans": len(o["plans"]),
        }

    def _t_predict(self, runner_id: str, weights: list | None = None) -> dict:
        w = self._weights(runner_id, weights)
        p = self.coach.predict(runner_id, w, self.unit)
        races = self.coach.overview(runner_id)["races"]
        return {
            "finish": p["finish"], "likely_range": f"{format_time(p['low_s'])} to {format_time(p['high_s'])}",
            "range_meaning": ("The interval uses the 20th to 80th percentiles of similar runners' implied "
                              "next-marathon finishes. On held-out demo races, the actual result fell inside "
                              "it for about 80% of runners; this is validation evidence, not a guarantee."), "similar_runners": p["n_cases"],
            "with_equal_weights": format_time(p["default_finish_s"]),
            "weights_used": dict(zip((f"{r['event']} {r['year']}" for r in races), p["weights"])),
            "prediction_from_each_race": {f"{r['event']} {r['year']}": None if x is None else format_time(x["finish_s"])
                                          for r, x in zip(races, p["per_race"])},
            "model_pacing": dict(zip(LABELS, (self._pace(s) for s in p["paces"]))),
            "strategies": [{"name": s["name"], "finish": s["finish"], "inside_likely_range": s["in_range"]}
                           for s in p["scenarios"]],
            "insights": p["insights"],
        }

    def _t_check_pacing_plan(self, runner_id: str, paces: list, weights: list | None = None) -> dict:
        if len(paces) != 9:
            raise ValidationError("give 9 paces, one per interval: 0-5, 5-10 ... 35-40, 40-42.2 km")
        secs = [parse_pace(p, self.unit) for p in paces]
        ev = self.coach.evaluate_plan(runner_id, self._weights(runner_id, weights), secs, self.unit)
        return {"finish": ev["finish"], "model_finish": format_time(ev["model_finish_s"]),
                "difference_from_model": f"{'+' if ev['difference_s'] >= 0 else '-'}{format_time(abs(ev['difference_s']))}",
                "likely_range": f"{format_time(ev['low_s'])} to {format_time(ev['high_s'])}",
                "verdict": ev["verdict"], "verdict_meaning": ev["note"], "warnings": [f["message"] for f in ev["flags"]]}

    def _t_current_plan(self) -> dict:
        p = self.plan()
        if p is None:
            return {"open_plan": None, "note": "No plan is open. The coach can open one on Plan a race."}
        return {
            "runner_id": p.runner_id, "runner": p.name, "unit": p.unit,
            "race_weights": {f"{r['event']} {r['year']}": w for r, w in zip(p.overview["races"], p.weights)},
            "model_finish": format_time(p.model_finish), "plan_finish": format_time(p.plan_finish),
            "likely_range": f"{format_time(p.prediction['low_s'])} to {format_time(p.prediction['high_s'])}",
            "pacing_source": p.strategy_name(p.source) if p.source != "model" else "the model's pacing",
            "paces": dict(zip(LABELS, (self._pace(s) for s in p.paces))),
            "check": p.check["verdict"], "check_meaning": p.check["note"],
            "warnings": [f["message"] for f in p.check["flags"]],
            "changes_newest_first": [c.text() for c in p.changes][:10],
        }

    def _t_watch_moment(self, distance_km: float, elapsed: str, plan_id: str | None = None) -> dict:
        if plan_id:
            saved = next((x for x in self.coach.list_plans() if x["id"] == plan_id), None)
            if saved is None:
                raise NotFound(f"plan {plan_id!r} not found")
            paces, label = saved["paces"], f"{saved['runner_name']}, {saved['target_race']}"
        else:
            p = self.plan()
            if p is None:
                return {"error": "no plan is open; give a saved plan_id or open a plan first"}
            paces, label = p.paces, p.name
        t = parse_time(elapsed) if ":" in str(elapsed) else parse_duration(elapsed)
        if t is None:
            raise ValidationError("elapsed must be a time like 1:40:00")
        s = racewatch.status(paces, float(distance_km), t, self.unit)
        return {"plan": label, "distance_km": s.distance_km, "elapsed": format_time(s.elapsed_s),
                "plan_elapsed_here": format_time(s.target_elapsed_s),
                "ahead_or_behind": "on plan" if s.level == "on" else
                (f"{format_time(abs(s.delta_s))} {'ahead' if s.delta_s < 0 else 'behind'}"),
                "target_pace_now": self._pace(s.target_pace_s), "projected_finish": format_time(s.projected_finish_s),
                "plan_finish": format_time(s.plan_finish_s), "status": s.level, "watch_cue": s.cue}

    def _t_saved_plans(self) -> dict:
        plans = self.coach.list_plans()
        tr = self.coach.track_record()
        return {"plans": [{"plan_id": p["id"], "runner": p["runner_name"], "runner_id": p["runner_id"],
                           "race": p["target_race"], "date": p["race_date"], "plan_finish": p["finish"],
                           "model_finish": format_time(p["model_finish_s"]), "result": p["result"]} for p in plans],
                "track_record": tr if tr.get("n") else {"n": 0, "note": "no results recorded yet"}}

    def _t_check_runner_details(self, **draft: Any) -> dict:
        return check_runner(draft)

    def _t_propose_runner(self, **draft: Any) -> dict:
        result = check_runner(draft)
        if not result["ok"]:
            return {"proposed": False, "reason": "fix these first", **result}
        self.pending_runner = result["runner"]
        return {"proposed": True, "corrections": result["corrections"],
                "next": "The coach now sees the runner on screen and adds it with the 'Add this runner' button."}

    def _t_explain(self, topic: str) -> dict:
        if topic not in HELP:
            raise ValidationError(f"topic must be one of {sorted(HELP)}")
        h = HELP[topic]
        return {"title": h.title, "explanation": h.text, "source": h.paper}
