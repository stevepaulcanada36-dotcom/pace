"""JSON routes for the command center (/api/hud/*) and the page (/hud).

    POST /api/hud/sessions                 {runner_id | "demo", holdout?: bool, ai_reader?: bool} -> {id, state}
    GET  /api/hud/sessions/<sid>                                                           -> state
    POST /api/hud/sessions/<sid>/command   {text, source?}  -> {reply, actions, turn, card, state, used, steps}
    POST /api/hud/sessions/<sid>/weights   {index, weight}  -> {card, state}
    POST /api/hud/sessions/<sid>/simulate  {temp_c, wind_kmh, profile}                      -> simulation
    POST /api/hud/sessions/<sid>/reveal                                                    -> state
    POST /api/hud/sessions/<sid>/react     {turn, value: agree|unsure|disagree, note?}    -> reaction summary
    GET  /api/hud/track-record                                                             -> coach track record
"""
from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from functools import lru_cache
from importlib import resources

from ..coach import CoachService
from ..domain import ValidationError
from ..hud import HudSession, route
from .core import HTTPError, Request, Response, Router
from .steer_api import demo_runner_id

S = r"(?P<sid>[A-Za-z0-9_\-]{8,40})"
MAX_SESSIONS = 100


@lru_cache(maxsize=1)
def hud_page() -> str:
    return resources.files("pacesense.resources").joinpath("hud.html").read_text(encoding="utf-8")


def register(router: Router, coach: CoachService, ai_factory=None) -> None:
    sessions: "OrderedDict[str, HudSession]" = OrderedDict()
    lock = threading.Lock()

    def get(sid: str) -> HudSession:
        with lock:
            s = sessions.get(sid)
            if s is None:
                raise HTTPError(404, "no such session (it may have expired); open a runner again")
            sessions.move_to_end(sid)
            return s

    def create(req: Request):
        rid = str(req.body.get("runner_id") or "demo")
        holdout = bool(req.body.get("holdout", rid == "demo"))
        if rid == "demo":
            rid = demo_runner_id(coach)
        ai = ai_factory() if ai_factory else None
        with lock:
            s = HudSession(coach, rid, holdout=holdout, ai=ai, ai_reader=bool(req.body.get("ai_reader")))
            sid = secrets.token_urlsafe(12)
            sessions[sid] = s
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
            return 201, {"id": sid, "state": s.state()}

    def command(req: Request, sid: str):
        text = str(req.body.get("text") or "").strip()
        if not text:
            raise ValidationError("text is required")
        if len(text) > 1000:
            raise ValidationError("text is too long (1000 characters at most)")
        source = req.body.get("source") if req.body.get("source") in ("voice", "typed") else "typed"
        s = get(sid)
        with lock:
            r = route(text, s, source)
            sim = None
            for a in r.actions:
                if a["type"] == "reveal" and s.truth_s is not None:
                    s.revealed = True
                if a["type"] == "simulate":
                    sim = s.simulate(a["temp_c"], a["wind_kmh"], a["profile"])
            card = s.feedback_card(r.turn) if r.turn else None
            return {"reply": r.reply, "actions": r.actions, "turn": r.turn, "card": card, "used": r.used,
                    "steps": r.steps, "simulation": sim, "state": s.state()}

    def weights(req: Request, sid: str):
        i, w = req.body.get("index"), req.body.get("weight")
        if not isinstance(i, int) or not isinstance(w, (int, float)) or isinstance(w, bool):
            raise ValidationError("index (whole number) and weight (0-1) are required")
        s = get(sid)
        with lock:
            turn = s.steer.manual(i, float(w)).to_dict()
            return {"turn": turn, "card": s.feedback_card(turn), "state": s.state()}

    def simulate(req: Request, sid: str):
        b = req.body
        prof = b.get("profile") or "on_plan"
        try:
            temp, wind = float(b.get("temp_c", 12)), float(b.get("wind_kmh", 0))
        except (TypeError, ValueError):
            raise ValidationError("temp_c and wind_kmh must be numbers") from None
        return get(sid).simulate(temp, wind, prof)

    def reveal(req: Request, sid: str):
        s = get(sid)
        if s.truth_s is None:
            raise ValidationError("this session has no hidden race (open a demo runner)")
        s.revealed = True
        return s.state()

    def react(req: Request, sid: str):
        b = req.body
        if not isinstance(b.get("turn"), int):
            raise ValidationError("turn (number) is required")
        return get(sid).react(b["turn"], str(b.get("value")), str(b.get("note") or ""))

    r = router.add
    r("GET", "/hud", lambda req: Response(200, hud_page(), "text/html; charset=utf-8"))
    r("POST", "/api/hud/sessions", create)
    r("GET", f"/api/hud/sessions/{S}", lambda req, sid: get(sid).state())
    r("POST", f"/api/hud/sessions/{S}/command", command)
    r("POST", f"/api/hud/sessions/{S}/weights", weights)
    r("POST", f"/api/hud/sessions/{S}/simulate", simulate)
    r("POST", f"/api/hud/sessions/{S}/reveal", reveal)
    r("POST", f"/api/hud/sessions/{S}/react", react)
    r("GET", "/api/hud/track-record", lambda req: coach.track_record())


__all__ = ["register", "hud_page"]
