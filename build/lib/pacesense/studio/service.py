"""The application layer: everything the Studio API does, independent of HTTP.

One `StudioService` per process. It owns the model and data (loaded once at start-up), the session store and
the optional AI factory. Each session is a `HudSession` (model weights, steering history, demo hold-out,
feedback loop, simulator); the methods here lock that session, call it, and return plain dicts for JSON.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import __version__
from ..cbr import CaseBase, MarathonModel
from ..coach import CoachService
from ..data import load_csv, synthetic_runners
from ..domain import ValidationError, format_time, parse_time
from ..hud import HudSession, route
from ..repro import configured_seed
from ..storage import Repository
from ..tuning import configured_cbr
from ..web.steer_api import demo_runner_id
from .settings import Settings
from .store import SessionStore

log = logging.getLogger("pacesense.studio")


@dataclass
class DataInfo:
    source: str
    runners: int
    races: int
    cases: int
    loaded_in_s: float
    fingerprint: str


def load_coach(settings: Settings) -> tuple[CoachService, DataInfo]:
    t0 = time.perf_counter()
    if settings.data_path:
        runners, _ = load_csv(settings.data_path)
        source = settings.data_path
        with open(settings.data_path, "rb") as f:
            fp = hashlib.sha256(f.read()).hexdigest()[:12]
    else:
        seed = settings.seed if settings.seed is not None else configured_seed()
        runners = synthetic_runners(seed=seed)
        source, fp = f"synthetic (seed {seed})", f"synthetic-{seed}"
    model = MarathonModel(CaseBase(runners, configured_cbr()))
    coach = CoachService(model, runners, Repository(settings.db_path))
    info = DataInfo(source, len(runners), sum(len(r.races) for r in runners), len(model.case_base),
                    round(time.perf_counter() - t0, 3), fp)
    log.info("data loaded", extra={"data": info.__dict__})
    return coach, info


class StudioService:
    def __init__(self, coach: CoachService, info: DataInfo | None = None, settings: Settings | None = None,
                 ai_factory: Callable[[], Any] | None = None) -> None:
        self.settings = settings or Settings()
        self.coach = coach
        self.info = info or DataInfo("in-memory", len(coach.runners), sum(len(r.races) for r in coach.runners),
                                     len(coach.model.case_base), 0.0, "in-memory")
        self.ai_factory = ai_factory
        self.sessions: SessionStore[HudSession] = SessionStore(self.settings.session_ttl_s, self.settings.max_sessions)
        self.started = time.time()
        self._demo_id: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> StudioService:
        coach, info = load_coach(settings)
        ai = None
        if settings.ai_enabled:
            from ..llm.service import AIService
            ai = lambda: AIService(coach.repo)
        return cls(coach, info, settings, ai)

    # ---- runners --------------------------------------------------------------------------------
    @property
    def demo_runner(self) -> str:
        if self._demo_id is None:
            self._demo_id = demo_runner_id(self.coach)
        return self._demo_id

    def runners(self, query: str = "", limit: int = 20) -> list[dict]:
        q = query.strip().lower()
        demo = self.coach.runner(self.demo_runner)
        out = []
        if not q or q in "demo example":
            out.append({**CoachService._summary(demo, False), "runner_id": "demo",
                        "name": "Demo runner", "note": f"{demo.runner_id}, best race hidden"})
        if len(q) >= 2:
            out += self.coach.search(q, limit)
        else:
            picks = [r for r in sorted(self.coach.runners, key=lambda r: r.runner_id) if len(r.recent(5)) >= 4]
            out += [CoachService._summary(r, False) for r in picks[: limit - len(out)]]
        return out[:limit]

    # ---- sessions -------------------------------------------------------------------------------
    def create(self, runner_id: str = "demo", holdout: bool | None = None) -> tuple[str, dict]:
        is_demo = runner_id == "demo"
        rid = self.demo_runner if is_demo else runner_id
        self.coach.runner(rid)                                   # 404 early for an unknown id
        s = HudSession(self.coach, rid, holdout=is_demo if holdout is None else holdout,
                       ai=self.ai_factory() if self.ai_factory else None)
        sid = self.sessions.add(s)
        log.info("session created", extra={"session": sid[:6], "runner": rid, "holdout": s.holdout})
        return sid, self.view(sid, s)

    def _with(self, sid: str, fn: Callable[[HudSession], Any]) -> Any:
        e = self.sessions.entry(sid)
        with e.lock:
            return fn(e.value)

    def view(self, sid: str, s: HudSession) -> dict:
        st = s.state()
        st["session"] = {"id": sid, "source_runner": s.source_runner, "holdout": s.holdout,
                         "can_undo": bool(s.steer._undo), "ai": st.pop("ai", False)}
        r = self.coach.runner(s.runner_id)
        st["runner"].update({"gender": r.gender, "age": r.age, "n_races": len(st["races"])})
        return st

    def state(self, sid: str) -> dict:
        return self._with(sid, lambda s: self.view(sid, s))

    def command(self, sid: str, text: str, source: str = "typed") -> dict:
        def run(s: HudSession):
            r = route(text, s, source)
            sim = None
            for a in r.actions:
                if a["type"] == "reveal" and s.truth_s is not None:
                    s.revealed = True
                if a["type"] == "simulate":
                    sim = s.simulate(a["temp_c"], a["wind_kmh"], a["profile"])
            card = s.feedback_card(r.turn) if r.turn else None
            return {"reply": r.reply, "actions": r.actions, "card": card, "used": r.used,
                    "simulation": sim, "state": self.view(sid, s)}
        return self._with(sid, run)

    def set_weight(self, sid: str, index: int, weight: float) -> dict:
        def run(s: HudSession):
            if index >= len(s.steer.races):
                raise ValidationError(f"race index {index} is out of range")
            turn = s.steer.manual(index, round(weight, 2)).to_dict()
            return {"card": s.feedback_card(turn), "state": self.view(sid, s)}
        return self._with(sid, run)

    def undo(self, sid: str) -> dict:
        def run(s: HudSession):
            turn = s.steer.undo().to_dict()
            return {"card": s.feedback_card(turn), "state": self.view(sid, s)}
        return self._with(sid, run)

    def estimate(self, sid: str, finish: str) -> dict:
        secs = parse_time(finish)
        if secs is None or not 2 * 3600 <= secs <= 7 * 3600:
            raise ValidationError("an estimate should be between 2:00 and 7:00")
        return self.command(sid, f"I think she can run {format_time(secs)}")

    def simulate(self, sid: str, temp_c: float, wind_kmh: float, profile: str) -> dict:
        return self._with(sid, lambda s: s.simulate(temp_c, wind_kmh, profile))

    def reveal(self, sid: str) -> dict:
        def run(s: HudSession):
            if s.truth_s is None:
                raise ValidationError("this session has no hidden race; open the demo runner")
            s.revealed = True
            return self.view(sid, s)
        return self._with(sid, run)

    def react(self, sid: str, turn: int, value: str, note: str = "") -> dict:
        return self._with(sid, lambda s: s.react(turn, value, note))

    def export(self, sid: str) -> dict:
        """Everything needed to audit or replay the session (pacesense.steer.replay)."""
        def run(s: HudSession):
            st = self.view(sid, s)
            return {"pacesense_version": __version__, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "data": {"source": self.info.source, "fingerprint": self.info.fingerprint},
                    "runner": st["runner"], "holdout": st["holdout"], "transcript": s.steer.transcript(),
                    "reactions": s.reactions, "final": {"weights": [r["weight"] for r in st["races"]],
                                                        "prediction": st["prediction"], "steering": st["steering"]}}
        return self._with(sid, run)

    def close(self, sid: str) -> None:
        self.sessions.remove(sid)

    # ---- operations -----------------------------------------------------------------------------
    def health(self) -> dict:
        return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - self.started, 1)}

    def ready(self) -> dict:
        ok = self.info.cases > 0
        return {"ready": ok, "data": self.info.__dict__, "sessions": self.sessions.stats()}


__all__ = ["DataInfo", "StudioService", "load_coach"]
