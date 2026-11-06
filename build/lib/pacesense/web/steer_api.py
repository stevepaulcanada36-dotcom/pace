"""JSON routes for PaceSense Live (/api/steer/*) and the live page itself (/live).

    POST /api/steer/sessions                      {runner_id, ai?: bool}       -> {id, state}
    GET  /api/steer/sessions/<sid>                                             -> state
    POST /api/steer/sessions/<sid>/utterances     {text, source?: voice|typed} -> {turn, state}
    POST /api/steer/sessions/<sid>/audio          {audio_b64, format}          -> {text, turn, state}   (local Whisper)
    POST /api/steer/sessions/<sid>/manual         {index, weight}              -> {turn, state}
    POST /api/steer/sessions/<sid>/undo                                        -> {turn, state}
    GET  /api/steer/sessions/<sid>/transcript                                  -> {transcript}  (replayable)
    GET  /api/steer/demo-runner                                                -> {runner_id}
    GET  /live                                                                 -> the page

Sessions live in memory in this server process (one coach, one laptop). They are not written to the study
database: PaceSense Live is a proposal and does not change the study as the paper ran it.
"""
from __future__ import annotations

import base64
import binascii
import secrets
import threading
from collections import OrderedDict
from functools import lru_cache
from importlib import resources

from ..coach import CoachService
from ..domain import ValidationError
from ..llm.openrouter import LLMError
from ..planning import PlanSession
from ..steer import SteeringSession, race_evidence
from .core import HTTPError, Request, Response, Router

S = r"(?P<sid>[A-Za-z0-9_\-]{8,40})"
MAX_SESSIONS = 200


@lru_cache(maxsize=1)
def live_page() -> str:
    return resources.files("pacesense.resources").joinpath("live.html").read_text(encoding="utf-8")


def demo_runner_id(coach: CoachService) -> str:
    """A fixed, reproducible demo: the first female runner (by id) with five races, one clearly slow."""
    best = None
    for r in sorted(coach.runners, key=lambda x: x.runner_id):
        races = r.recent(5)
        if len(races) < 5:
            continue
        typical_time = sorted(x.finish_s for x in races)[2]
        if not 3 * 3600 <= typical_time <= 4 * 3600 + 15 * 60:     # a relatable 3-4 hour marathoner
            continue
        ev = race_evidence([x.finish_s for x in races])
        slow = [e for e in ev if e.z is not None and e.z >= 2]
        typical = [e for e in ev if e.direction == 0]
        if len(slow) == 1 and len(typical) >= 3 and len({x.year for x in races}) == 5:
            if r.gender == "F":
                return r.runner_id
            best = best or r.runner_id
    if best:
        return best
    raise ValidationError("no runner with five races found for the demo")


def register(router: Router, coach: CoachService, ai_factory=None) -> None:
    sessions: "OrderedDict[str, SteeringSession]" = OrderedDict()
    lock = threading.Lock()

    def get(sid: str) -> SteeringSession:
        with lock:
            s = sessions.get(sid)
            if s is None:
                raise HTTPError(404, "no such live session (it may have expired; start a new one)")
            sessions.move_to_end(sid)
            return s

    def create(req: Request):
        rid = str(req.body.get("runner_id") or "")
        if not rid:
            raise ValidationError("runner_id is required")
        ai = ai_factory() if (req.body.get("ai") and ai_factory) else None
        s = SteeringSession(PlanSession(coach, rid), ai=ai)
        sid = secrets.token_urlsafe(12)
        with lock:
            sessions[sid] = s
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        st = s.state()
        st["ai"] = bool(ai is not None and getattr(ai, "online", False))
        return 201, {"id": sid, "state": st}

    def utter(req: Request, sid: str):
        text = str(req.body.get("text") or "").strip()
        if not text:
            raise ValidationError("text is required")
        if len(text) > 1000:
            raise ValidationError("text is too long (1000 characters at most)")
        source = req.body.get("source") if req.body.get("source") in ("voice", "typed") else "typed"
        s = get(sid)
        with lock:
            turn = s.hear(text, source)
            return {"turn": turn.to_dict(), "state": s.state()}

    def audio(req: Request, sid: str):
        s = get(sid)
        try:
            data = base64.b64decode(str(req.body.get("audio_b64") or ""), validate=True)
        except (binascii.Error, ValueError):
            raise ValidationError("audio_b64 must be base64") from None
        fmt = str(req.body.get("format") or "webm").lower()
        if fmt not in ("webm", "ogg", "wav", "mp4", "m4a", "mp3"):
            raise ValidationError("unsupported audio format")
        from ..llm.local_whisper import transcribe
        try:
            t = transcribe(data, fmt)
        except LLMError as e:
            raise HTTPError(422, f"local Whisper: {e}. Install it with: pip install -e \".[local-stt]\"") from None
        except ImportError:
            raise HTTPError(422, 'local Whisper is not installed: pip install -e ".[local-stt]"') from None
        if not t.text.strip():
            return {"text": "", "turn": None, "state": s.state()}
        with lock:
            turn = s.hear(t.text, "voice")
            return {"text": t.text, "turn": turn.to_dict(), "state": s.state()}

    def manual(req: Request, sid: str):
        i, w = req.body.get("index"), req.body.get("weight")
        if not isinstance(i, int) or not isinstance(w, (int, float)) or isinstance(w, bool):
            raise ValidationError("index (whole number) and weight (0-1) are required")
        s = get(sid)
        with lock:
            turn = s.manual(i, float(w))
            return {"turn": turn.to_dict(), "state": s.state()}

    def undo(req: Request, sid: str):
        s = get(sid)
        with lock:
            turn = s.undo()
            return {"turn": turn.to_dict(), "state": s.state()}

    r = router.add
    r("GET", "/live", lambda req: Response(200, live_page(), "text/html; charset=utf-8"))
    r("GET", "/api/steer/demo-runner", lambda req: {"runner_id": demo_runner_id(coach)})
    r("POST", "/api/steer/sessions", create)
    r("GET", f"/api/steer/sessions/{S}", lambda req, sid: get(sid).state())
    r("POST", f"/api/steer/sessions/{S}/utterances", utter)
    r("POST", f"/api/steer/sessions/{S}/audio", audio)
    r("POST", f"/api/steer/sessions/{S}/manual", manual)
    r("POST", f"/api/steer/sessions/{S}/undo", undo)
    r("GET", f"/api/steer/sessions/{S}/transcript", lambda req, sid: {"transcript": get(sid).transcript()})


__all__ = ["register", "demo_runner_id", "live_page"]
