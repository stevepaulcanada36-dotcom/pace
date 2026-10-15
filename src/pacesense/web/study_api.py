"""JSON routes for the research study (the paper's procedure): /api/sessions/*, /api/admin/*."""
from __future__ import annotations

import hmac
import uuid

from ..analysis import analyze_repository
from ..domain import Race, Runner, ValidationError
from ..study import StudyService
from .core import HTTPError, Request, Response, Router, time_field

S = r"(?P<sid>[0-9a-f]{32})"
N = r"(?P<seq>[1-4])"


def _hand_runner(d: dict) -> Runner:
    try:
        races = [Race.from_cumulative_splits(str(r.get("event") or "Marathon"), int(r.get("year") or 0),
                                             time_field(r["finish"], "finish"),
                                             [time_field(x, "split") for x in r["splits"]] if r.get("splits") else None,
                                             d.get("age")) for r in d["races"]]
        return Runner(f"H-{uuid.uuid4().hex[:12]}", str(d.get("gender", "M"))[:1].upper(), races,
                      name=str(d.get("name", ""))[:200], age=d.get("age"))
    except (KeyError, TypeError) as e:
        raise ValidationError(f"invalid runner entered by hand: {e}") from None


def register(router: Router, service: StudyService, admin_token: str | None, default_assignment: str) -> None:
    def admin(req: Request) -> None:
        if not admin_token:
            raise HTTPError(403, "admin endpoints are disabled; set PACESENSE_ADMIN_TOKEN")
        given = req.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given, admin_token):
            raise HTTPError(401, "missing or wrong admin token")

    def plan(req: Request, sid: str):
        b = req.body
        familiar = list(b.get("familiar_ids", [])) + [_hand_runner(d) for d in b.get("familiar_runners", [])]
        return 201, {"runners": service.plan_runners(sid, familiar, unfamiliar_only=bool(b.get("unfamiliar_only")))}

    def estimate(req: Request, sid: str, seq: str):
        b = req.body
        return service.submit_pre(sid, int(seq), time_field(b.get("coach_pre"), "coach_pre"), float(b.get("pace", 0)),
                                  b.get("familiarity"), b.get("runner_motivations", []))

    def weights(req: Request, sid: str, seq: str):
        w = req.body.get("weights")
        if not isinstance(w, list) or not all(isinstance(x, (int, float)) for x in w):
            raise ValidationError("weights must be a list of numbers between 0 and 1")
        return service.set_weights(sid, int(seq), w)

    def advice(req: Request, sid: str, seq: str):
        b = req.body
        return service.submit_post(sid, int(seq), time_field(b.get("coach_post"), "coach_post"),
                                   float(b.get("pace", 0)), str(b.get("motivation", "")))

    def report(req: Request):
        admin(req)
        try:
            return Response(200, analyze_repository(service.repo).to_markdown(), "text/markdown; charset=utf-8")
        except ValueError as e:
            raise HTTPError(409, str(e)) from None

    def csv(kind: str):
        def handler(req: Request):
            admin(req)
            df = service.repo.trials_frame(completed_only=False) if kind == "trials" else service.repo.sessions_frame()
            return Response(200, df.to_csv(index=False), "text/csv; charset=utf-8")
        return handler

    r = router.add
    r("GET", "/api/runners", lambda req: {"results": service.search_runners(req.query.get("q", ""))})
    r("POST", "/api/sessions", lambda req: (201, service.create_session(req.body.get("assignment", default_assignment),
                                                                       coach_label=req.body.get("coach_label"))))
    r("GET", f"/api/sessions/{S}", lambda req, sid: service.session_summary(sid))
    r("POST", f"/api/sessions/{S}/runners", plan)
    r("GET", f"/api/sessions/{S}/runners/{N}", lambda req, sid, seq: service.trial_view(sid, int(seq)))
    r("POST", f"/api/sessions/{S}/runners/{N}/estimate", estimate)
    r("PUT", f"/api/sessions/{S}/runners/{N}/weights", weights)
    r("POST", f"/api/sessions/{S}/runners/{N}/advice", advice)
    r("GET", f"/api/sessions/{S}/questionnaire", lambda req, sid: service.questionnaire_for(sid))
    r("POST", f"/api/sessions/{S}/questionnaire", lambda req, sid: service.submit_questionnaire(sid, req.body))
    r("GET", "/api/admin/report", report)
    r("GET", r"/api/admin/trials\.csv", csv("trials"))
    r("GET", r"/api/admin/sessions\.csv", csv("sessions"))
