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
CONDITIONS = ("known", "unknown", "ask")
VOICE_PRESETS = ("quick", "balanced", "calm", "handsfree")
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
        self.last_insight = ""                      # what the loops said on the last event (spoken if nothing else)
        self.pending_new = False                    # the last event opened a new question
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
    def start(self, state: dict, condition: str = "ask") -> dict:
        """A new runner: fresh session context and per-runner persona, same lasting persona and history.
        `condition` is the paper's familiar / unfamiliar split: known, unknown, or ask the coach first."""
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {', '.join(CONDITIONS)}")
        with _LOCK:
            self.ctx.session = SessionContext(state=state, condition=condition)
            self.ctx.persona.new_runner({"known": 0.9, "unknown": 0.1}.get(condition))
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
            elif ev.kind == "condition":
                self.ctx.session.condition = ev.data["condition"]
                self.ctx.persona.runner["familiarity"].override = {"known": 0.9, "unknown": 0.1}.get(ev.data["condition"])
            elif ev.kind == "interview_answer" and ev.data.get("familiarity") is not None:
                self.ctx.session.condition = "known" if ev.data["familiarity"] >= 0.7 else "unknown"
            if ev.kind == "prompt_answer" or ev.data.get("prompt_answer"):
                self.pending = None
            out: list[Adaptation] = []
            for loop in self.loops:
                if loop.active(self.ctx)[0] or loop.key == "persona":
                    out += loop.observe(ev, self.ctx)
            self.log = (self.log + out)[-60:]
            said = [a for a in out if a.kind in ("insight", "persona") and a.message]
            self.last_insight = said[-1].message if said else ""
            prompts = [a for a in out if a.kind == "prompt"]
            self.pending_new = bool(prompts)
            if prompts:
                self.pending = prompts[-1].to_dict()
            elif ev.kind in ("utterance", "slider", "undo", "interview_answer") and self.pending and \
                    not self.pending["data"].get("question"):
                self.pending = None
            self._save()
            return self.result(out)

    def dismiss(self) -> dict:
        """The coach closed the current question without acting on it (tap, or said "no" / "skip")."""
        loop = (self.pending or {}).get("loop")
        self.pending = None
        return self.process(Event("prompt_answer", source="system", data={"loop": loop, "choice": "dismiss"}))

    def set_condition(self, condition: str) -> dict:
        if condition not in ("known", "unknown"):
            raise ValueError("condition must be known or unknown")
        self.ctx.session.asked = list(dict.fromkeys(self.ctx.session.asked + ["familiarity"]))
        return self.process(Event("condition", source="system", data={"condition": condition}))

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
        vs = s.voice_style or {"preset": "balanced", "label": "Balanced", "rate": 1.03, "silence_ms": 1000,
                               "reply": "normal", "read_choices": False}
        reply = vs["reply"] if s.voice else ("full" if explain == "full" else "brief")
        return {"layout": s.layout, "explain": explain, "spotlight": s.spotlight, "analysis_tab": tab,
                "voice": s.voice, "speak": p.value("voice") >= 0.6 or s.voice,
                "speak_style": "full" if reply == "full" else "brief", "interview": s.interview_on,
                "layout_lock": self.ctx.profile.get("layout_lock"), "condition": s.condition,
                "voice_style": vs, "voice_preset_lock": self.ctx.profile.get("voice_preset")}

    def loops_status(self) -> list[dict]:
        return [lp.status(self.ctx) for lp in self.loops]

    def active_count(self) -> tuple[int, int]:
        st = self.loops_status()
        return sum(1 for x in st if x["active"]), len(st)

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
            if "voice_preset" in changes:
                vp = changes.pop("voice_preset")
                if vp is not None and vp not in VOICE_PRESETS:
                    raise ValueError(f"voice preset must be one of {', '.join(VOICE_PRESETS)}")
                self.ctx.profile["voice_preset"] = vp
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
    def _choices_aloud(self, prompt: dict) -> str:
        choices = prompt.get("data", {}).get("choices") or []
        if any(c.get("yes") for c in choices):
            return " Say yes or no."
        if (self.ui()["voice_style"].get("read_choices") or self.ctx.session.voice) and 1 < len(choices) <= 5:
            labels = [c["label"] for c in choices]
            return " You can say " + ", ".join(labels[:-1]) + f", or {labels[-1]}."
        return ""

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
            if self.pending_new and self.pending:                     # e.g. the next suggestion
                s += " " + self.pending["message"] + self._choices_aloud(self.pending)
            elif self.last_insight:
                s += " " + self.last_insight
            return s
        if self.pending and self.pending.get("kind") == "prompt":
            return self.pending["message"] + self._choices_aloud(self.pending)
        r = (reply or "").strip()
        if not r:
            return self.last_insight
        return r if style == "full" or len(r) < 160 else r.split(". ")[0] + "."


__all__ = ["CONDITIONS", "LAYOUTS", "VOICE_PRESETS", "AdaptiveEngine"]
