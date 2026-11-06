"""JSON routes for coach planning: /api/coach/*."""
from __future__ import annotations

from ..coach import CoachService, runner_payload_races
from ..domain import ValidationError
from .core import Request, Router, time_field

R = r"(?P<rid>[A-Za-z0-9\-]{1,40})"
P = r"(?P<pid>[0-9a-f]{12})"


def _weights(value) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        raise ValidationError("weights must be a list of numbers between 0 and 1")
    return [float(x) for x in value]


def _paces(value) -> list[float]:
    if not isinstance(value, list) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        raise ValidationError("paces must be a list of 9 numbers (seconds per km)")
    return [float(x) for x in value]


def _unit(value) -> str:
    if value is None:
        return "km"
    if not isinstance(value, str):
        raise ValidationError("unit must be 'km' or 'mi'")
    return value


def register(router: Router, coach: CoachService) -> None:
    def add_runner(req: Request):
        b = req.body
        age = b.get("age")
        if age is not None and (not isinstance(age, int) or not 10 <= age <= 100):
            raise ValidationError("age must be a whole number from 10 to 100")
        races = runner_payload_races(b.get("races") or [], age, time_field)
        return 201, coach.add_runner(str(b.get("name", "")), str(b.get("gender", "")), age, races)

    def save_plan(req: Request):
        b = req.body
        return 201, coach.save_plan(str(b.get("runner_id", "")), str(b.get("target_race", "")), b.get("race_date"),
                                    _weights(b.get("weights")), _paces(b.get("paces")), str(b.get("notes", "")))

    r = router.add
    r("GET", "/api/coach/runners", lambda req: {"results": coach.search(req.query.get("q", ""))})
    r("POST", "/api/coach/runners", add_runner)
    r("GET", f"/api/coach/runners/{R}", lambda req, rid: coach.overview(rid))
    r("POST", f"/api/coach/runners/{R}/prediction",
      lambda req, rid: coach.predict(rid, _weights(req.body.get("weights")), _unit(req.body.get("unit"))))
    r("POST", f"/api/coach/runners/{R}/plan-check",
      lambda req, rid: coach.evaluate_plan(rid, _weights(req.body.get("weights")), _paces(req.body.get("paces")),
                                           _unit(req.body.get("unit"))))
    r("GET", "/api/coach/preferences", lambda req: coach.preferences())
    r("PUT", "/api/coach/preferences", lambda req: coach.update_preferences(req.body))
    r("GET", "/api/coach/plans", lambda req: {"plans": coach.list_plans()})
    r("POST", "/api/coach/plans", save_plan)
    r("PUT", f"/api/coach/plans/{P}/notes", lambda req, pid: coach.update_notes(pid, str(req.body.get("notes", ""))))
    r("PUT", f"/api/coach/plans/{P}/result",
      lambda req, pid: coach.record_result(pid, time_field(req.body.get("finish"), "finish")))
    r("GET", "/api/coach/track-record", lambda req: coach.track_record())
