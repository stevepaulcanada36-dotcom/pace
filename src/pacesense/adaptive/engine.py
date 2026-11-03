"""The adaptive engine: runs every feedback loop on every event and turns their adaptations into one interface
directive, a persona summary and a short spoken reply.

State lives in two places in the same SQLite file as everything else (Repository settings table):
    adaptive:coach:<id>     the coach profile (persona, track record, panel usage, layout lock)
    adaptive:population     pooled cue outcomes across all coaches (Shared learning loop)
Both are re-read before and written after every event under one lock, so several sessions (even of the same
coach) never overwrite each other's learning.
"""
from __future__ import annotations

import threading
from typing import Any

from ..storage import Repository
from .events import Adaptation, Event
from .interview import Choice
from .loops import (
    FeedbackLoop,
    InterviewLoop,
    LoopContext,
    SessionContext,
    default_loops,
)
from .persona import Persona

_LOCK = threading.RLock()
LAYOUTS = ("standard", "voice", "analysis", "races", "summary", "guided")


def _key(coach_id: str) -> str:
    return f"adaptive:coach:{coach_id}"


class AdaptiveEngine:
    def __init__(self, repo: Repository, coach_id: str = "anonymous", loops: list[FeedbackLoop] | None = None) -> None:
        self.repo, self.coach_id = repo, coach_id
        self.loops = loops if loops is not None else default_loops()
        self.ctx = LoopContext(Persona(), {}, {}, SessionContext())
        self.log: list[Adaptation] = []
        self.pending: dict | None = None            # the open prompt, if any
        self._load()

    # ---- persistence ----------------------------------------------------------------------------
    def _load(self) -> None:
        profile = self.repo.get_setting(_key(self.coach_id), {}) or {}
        runner = self.ctx.persona.runner
        self.ctx.persona = Persona.from_dict(profile.get("persona"))
        self.ctx.persona.runner = runner
        self.ctx.profile = profile
        self.ctx.population = self.repo.get_setting("adaptive:population", {}) or {}

    def _save(self) -> None:
        self.ctx.profile["persona"] = self.ctx.persona.to_dict()
        self.repo.set_setting(_key(self.coach_id), self.ctx.profile)
        self.repo.set_setting("adaptive:population", self.ctx.population)

    # ---- running the loops ---------------------------------------------------------------------
    def start(self, state: dict) -> dict:
        """A new runner: fresh session context and per-runner persona, same lasting persona and history."""
        with _LOCK:
            self.ctx.session = SessionContext(state=state)
            self.ctx.persona.new_runner()
            self.log = []
            self.pending = None
        return self.process(Event("session_start", source="system"), state)

    def process(self, ev: Event, state: dict | None = None) -> dict:
        with _LOCK:
            self._load()
            if state is not None:
                self.ctx.session.state = state
            if ev.kind == "voice_on":
                self.ctx.session.voice = True
            elif ev.kind == "voice_off":
                self.ctx.session.voice = False
            out: list[Adaptation] = []
            for loop in self.loops:
                out += loop.observe(ev, self.ctx)
            self.log = (self.log + out)[-60:]
            prompts = [a for a in out if a.kind == "prompt"]
            if prompts:
                self.pending = prompts[-1].to_dict()
            elif ev.kind in ("utterance", "slider", "undo", "interview_answer") and self.pending and \
                    not self.pending["data"].get("question"):
                self.pending = None
            self._save()
            return self.result(out)

    def dismiss(self) -> dict:
        with _LOCK:
            self.pending = None
            return self.result([])

    # ---- the interview --------------------------------------------------------------------------
    def interview_loop(self) -> InterviewLoop:
        return next(lp for lp in self.loops if isinstance(lp, InterviewLoop))

    def start_interview(self) -> dict:
        with _LOCK:
            self.ctx.session.interview_on = True
            self.ctx.session.asked = []
            out = self.interview_loop().ask(self.ctx, "you asked for the guided interview")
            self.log = (self.log + out)[-60:]
            self.pending = next((a.to_dict() for a in out if a.kind == "prompt"), None)
            return self.result(out)

    def interview_choice(self, question_id: str, key: str) -> Choice:
        from .interview import questions
        for q in questions(self.ctx.session.state):
            if q.id == question_id:
                return q.choice(key)
        raise KeyError(question_id)

    # ---- what the interface gets ----------------------------------------------------------------
    def ui(self) -> dict:
        p, s = self.ctx.persona, self.ctx.session
        explain = s.explain if s.explain != "auto" else ("full" if p.value("detail") >= 0.6 else
                                                         "brief" if p.value("detail") <= 0.35 else "normal")
        usage = self.ctx.profile.get("panel_usage") or {}
        tab = max(("pacing", "similar", "insights"), key=lambda k: (usage.get(k, 0), k == "pacing"))
        if s.layout == "analysis" and not usage:
            tab = "similar"
        return {"layout": s.layout, "explain": explain, "spotlight": s.spotlight, "analysis_tab": tab,
                "voice": s.voice, "speak": p.value("voice") >= 0.6 or s.voice,
                "speak_style": "full" if explain == "full" else "brief", "interview": s.interview_on,
                "layout_lock": self.ctx.profile.get("layout_lock")}

    def loops_status(self) -> list[dict]:
        return [lp.status(self.ctx) for lp in self.loops]

    def result(self, out: list[Adaptation]) -> dict:
        return {"adaptations": [a.to_dict() for a in out], "ui": self.ui(), "persona": self.ctx.persona.summary(),
                "loops": self.loops_status(), "prompt": self.pending,
                "log": [a.to_dict() for a in self.log[-12:] if a.message]}

    def snapshot(self) -> dict:
        with _LOCK:
            return self.result([])

    # ---- persona edits and layout lock ---------------------------------------------------------
    def edit_persona(self, changes: dict[str, Any]) -> dict:
        with _LOCK:
            self._load()
            lock = changes.pop("layout_lock", "unset")
            if lock != "unset":
                if lock is not None and lock not in LAYOUTS:
                    raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
                self.ctx.profile["layout_lock"] = lock
            if "interview_default" in changes:
                self.ctx.profile["interview_default"] = bool(changes.pop("interview_default"))
            if changes:
                self.ctx.persona.edit(changes)
            self._save()
        return self.process(Event("persona_edit", source="system"))

    def forget(self) -> None:
        """Delete everything learned about this coach (privacy)."""
        with _LOCK:
            self.repo.set_setting(_key(self.coach_id), {})
            self.ctx.profile, self.ctx.persona = {}, Persona()

    # ---- the spoken reply ----------------------------------------------------------------------
    def speak(self, card: dict | None, reply: str = "") -> str:
        style = self.ui()["speak_style"]
        if card and card.get("items"):
            it = card["items"][0]
            word = {"supported": "The data agrees.", "your_knowledge": "That rests on your knowledge.",
                    "against_data": "Her own times disagree.", "undo": "Undone."}.get(it["verdict"], "")
            more = f" and {len(card['items']) - 1} more" if len(card["items"]) > 1 else ""
            s = f"{it['race']}{more} now counts {it['new']:.1f}. {word} Target {card['target']}."
            if style == "full":
                if card.get("range"):
                    s += f" Likely between {card['range'][0]} and {card['range'][1]}."
                if card.get("accuracy"):
                    s += f" That is {card['accuracy']['text']}."
            return s
        if self.pending and self.pending.get("kind") == "prompt":
            return self.pending["message"]
        r = (reply or "").strip()
        return r if style == "full" or len(r) < 160 else r.split(". ")[0] + "."


__all__ = ["LAYOUTS", "AdaptiveEngine"]
