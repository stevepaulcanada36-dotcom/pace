"""The application layer: everything the Studio API does, independent of HTTP.

One `StudioService` per process. It owns the model and data (loaded once at start-up), the session store and
the optional AI factory. Each session is a `HudSession` (model weights, steering history, demo hold-out,
feedback loop, simulator); the methods here lock that session, call it, and return plain dicts for JSON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import __version__
from ..adaptive import AdaptiveEngine, Event
from ..adaptive.answers import is_bare_answer, match_choice
from ..adaptive.explain import explain
from ..cbr import CaseBase, MarathonModel
from ..coach import CoachService
from ..data import load_csv, synthetic_runners
from ..domain import ValidationError, format_time, parse_time
from ..hud import HudSession
from ..hud.commands import Routed, route
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


@dataclass
class Live:
    """One open session: the model session and the adaptive engine that watches it."""
    hud: HudSession
    engine: AdaptiveEngine
    coach_id: str


class StudioService:
    def __init__(self, coach: CoachService, info: DataInfo | None = None, settings: Settings | None = None,
                 ai_factory: Callable[[], Any] | None = None) -> None:
        self.settings = settings or Settings()
        self.coach = coach
        self.info = info or DataInfo("in-memory", len(coach.runners), sum(len(r.races) for r in coach.runners),
                                     len(coach.model.case_base), 0.0, "in-memory")
        self.ai_factory = ai_factory
        self.sessions: SessionStore[Live] = SessionStore(self.settings.session_ttl_s, self.settings.max_sessions)
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

    def runners(self, query: str = "", limit: int = 50, gender: str | None = None,
                min_races: int = 0) -> list[dict]:
        q = query.strip().lower()
        demo = self.coach.runner(self.demo_runner)
        out = []
        if not q or q in "demo example":
            out.append({**CoachService._summary(demo, False), "runner_id": "demo",
                        "name": "Demo runner", "note": f"{demo.runner_id}, best race hidden"})
        if len(q) >= 2:
            out += self.coach.search(q, limit)
        else:
            # Browse the loaded database, not only runners with four or more races.
            records = {r.runner_id: r for r in self.coach.runners}
            records.update(self.coach._own())
            picks = sorted(records.values(), key=lambda r: r.runner_id)
            out += [CoachService._summary(r, r.runner_id in self.coach._own()) for r in picks[: limit - len(out)]]
        if gender:
            out = [r for r in out if r.get("gender") == gender]
        if min_races:
            out = [r for r in out if int(r.get("n_races") or 0) >= min_races]
        return out[:limit]

    # ---- sessions -------------------------------------------------------------------------------
    def create(self, runner_id: str = "demo", holdout: bool | None = None, coach_id: str = "anonymous",
               condition: str = "ask") -> tuple[str, dict]:
        is_demo = runner_id == "demo"
        rid = self.demo_runner if is_demo else runner_id
        self.coach.runner(rid)                                   # 404 early for an unknown id
        hud = HudSession(self.coach, rid, holdout=is_demo if holdout is None else holdout,
                         ai=self.ai_factory() if self.ai_factory else None)
        live = Live(hud, AdaptiveEngine(self.coach.repo, coach_id), coach_id)
        sid = self.sessions.add(live)
        st = self.view(sid, live)
        st["adapt"] = live.engine.start(st, condition)
        st["adapt"]["speak"] = live.engine.speak(None, "")
        log.info("session created", extra={"session": sid[:6], "runner": rid, "holdout": hud.holdout, "coach": coach_id[:8]})
        return sid, st

    def _with(self, sid: str, fn: Callable[[Live], Any]) -> Any:
        e = self.sessions.entry(sid)
        with e.lock:
            return fn(e.value)

    def view(self, sid: str, live: Live, with_adapt: bool = False) -> dict:
        s = live.hud
        st = s.state()
        st["session"] = {"id": sid, "source_runner": s.source_runner, "holdout": s.holdout,
                         "can_undo": bool(s.steer._undo), "ai": st.pop("ai", False), "coach_id": live.coach_id}
        r = self.coach.runner(s.runner_id)
        st["runner"].update({"gender": r.gender, "age": r.age, "n_races": len(st["races"])})
        if with_adapt:
            st["adapt"] = live.engine.snapshot()
        return st

    def state(self, sid: str) -> dict:
        return self._with(sid, lambda live: self.view(sid, live, with_adapt=True))

    # every mutation: do it, then let the feedback loops observe it
    def _observe(self, sid: str, live: Live, ev: Event, out: dict) -> dict:
        st = out.get("state") or self.view(sid, live)
        adapt = live.engine.process(ev, st)
        adapt["speak"] = live.engine.speak(out.get("card"), out.get("reply", ""))
        out["adapt"] = adapt
        return out

    def _interpret_voice_intent(self, text: str, live: Live) -> dict | None:
        """Ask Boson to classify a voice command; validate every proposed action locally."""
        if not self.ai_factory:
            return None
        try:
            ai = self.ai_factory()
            if not getattr(ai, "online", False):
                return None
            from ..hud.voice_actions import validate_voice_intent
            from ..llm.openrouter import parse_json_answer

            model, _ = ai.config("assistant")
            records = {r.runner_id: r for r in self.coach.runners}
            own = self.coach._own()
            records.update(own)
            rows = [{**CoachService._summary(r, r.runner_id in own), "runner_id": r.runner_id}
                    for r in sorted(records.values(), key=lambda x: x.runner_id)]
            rows.append({"runner_id": "demo", "name": "Demo runner"})
            # The catalog is reference material only; the validator still resolves exact records.
            catalog = [{"runner_id": r["runner_id"], "name": r.get("name", "")} for r in rows[:150]]
            system = """You interpret spoken commands for PaceSense, a marathon-coaching application.
Return exactly one JSON object: {"intent": "...", "arguments": {}, "confidence": 0.0, "evidence": ["..."], "clarification": "..."}.
Allowed intents only:
- runner_database: arguments {"query": "optional runner name or ID search text", "gender": "F|M|optional empty string", "min_races": integer 0-20}
When the coach asks for female/male runners, set gender to F/M. When the coach asks for a minimum number of previous marathons, set min_races to that number. Do not invent a filter the coach did not request.
- select_runner: arguments {"runner": "exact existing runner name or ID from the catalog"}
- set_familiarity: arguments {"value": "known" or "unfamiliar"}
- focus_panel or close_panel: arguments {"panel": "races|prediction|pacing|similar|simulator|feedback|steering|hologram|chat|log|record"}
- simulate: arguments {"temp_c": number, "wind_kmh": number, "profile": "on_plan|fast_start|late_fade"}
- reveal or undo: arguments {}
- none: arguments {}
Rules:
Use only one intent. Do not invent runner IDs or names; choose select_runner only for one unambiguous catalog entry. If the coach asks to find/search/browse runners, use runner_database and preserve their search words as query. Known/unfamiliar describes whether the coach personally knows an existing database runner; it never means create or add a runner. Do not invent filters or actions. If ambiguous, unsupported, or missing a necessary value, use none and explain the clarification. Confidence is your estimate that the intent and arguments faithfully match the utterance. Evidence must quote or briefly identify the words that support the intent. Do not execute anything; the application validates and executes separately."""
            user = {"utterance": text, "current_runner_id": live.hud.runner_id,
                    "can_undo": bool(live.hud.steer._undo),
                    "can_reveal": live.hud.truth_s is not None,
                    "existing_runner_catalog": catalog}
            result = ai.client.chat(model, [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ], temperature=0.0, json_mode=True)
            payload = parse_json_answer(result.text)
            return validate_voice_intent(payload, rows, can_undo=bool(live.hud.steer._undo),
                                          can_reveal=live.hud.truth_s is not None)
        except Exception as exc:
            log.warning("voice intent interpretation unavailable; using safe rule-based fallback: %s", exc)
            return None

    @staticmethod
    def _voice_route(text: str, source: str, live: Live, intent: dict) -> Routed:
        """Convert one validated intent into the existing, tested command/action path."""
        name, args = intent["intent"], intent["arguments"]
        if name == "select_runner":
            return Routed(reply=f"Opening {args['name']} from the existing runner database.",
                          actions=[{"type": "runner", "id": args["runner_id"], "name": args["name"]}], used=["ui"])
        if name == "runner_database":
            routed = route("open the runner database", live.hud, source)
            for action in routed.actions:
                if action.get("type") == "runner_database":
                    action["query"] = args["query"]
                    action["gender"] = args["gender"]
                    action["min_races"] = args["min_races"]
            return routed
        if name == "set_familiarity":
            phrase = "I know this runner" if args["condition"] == "known" else "I don't know this runner"
            return route(phrase, live.hud, source)
        if name in ("focus_panel", "close_panel"):
            verb = "open" if name == "focus_panel" else "close"
            return route(f"{verb} {args['panel']}", live.hud, source)
        if name == "simulate":
            phrase = f"simulate {args['temp_c']:g} degrees"
            if args["wind_kmh"]:
                phrase += f" with {args['wind_kmh']:g} km/h headwind"
            if args["profile"] == "fast_start":
                phrase += " and starts too fast"
            elif args["profile"] == "late_fade":
                phrase += " and a late fade"
            return route(phrase, live.hud, source)
        if name == "reveal":
            return route("reveal", live.hud, source)
        if name == "undo":
            return route("undo", live.hud, source)
        return route(text, live.hud, source)

    def _run_command(self, sid: str, live: Live, text: str, source: str, kind: str = "utterance",
                     extra: dict | None = None) -> dict:
        s = live.hud
        voice_intent = self._interpret_voice_intent(text, live) if source == "voice" else None
        if voice_intent and voice_intent.get("status") == "clarify":
            r = Routed(reply=voice_intent["message"], used=["prompt"])
        elif voice_intent and voice_intent.get("status") == "action":
            r = self._voice_route(text, source, live, voice_intent)
        else:
            r = route(text, s, source)
        sim, revealed = None, False
        for a in r.actions:
            if a["type"] == "reveal" and s.truth_s is not None:
                s.revealed = revealed = True
            if a["type"] == "condition":
                live.engine.set_condition(a["value"])
            if a["type"] == "simulate":
                sim = s.simulate(a["temp_c"], a["wind_kmh"], a["profile"])
        card = s.feedback_card(r.turn) if r.turn else None
        state = self.view(sid, live)
        out: dict[str, Any] = {"reply": r.reply, "actions": r.actions, "card": card, "used": r.used,
                               "simulation": sim, "state": state}
        if voice_intent:
            out["voice_intent"] = {key: voice_intent[key] for key in
                                   ("status", "intent", "confidence", "evidence") if key in voice_intent}
        turn = r.turn or {}
        data = {"applied": turn.get("applied", []), "turn": turn.get("n"), "estimate_s": turn.get("estimate_s"),
                "notes": turn.get("notes", []),
                "used": r.used, "ui_only": "ui" in r.used and "steer" not in r.used, **(extra or {})}
        self._observe(sid, live, Event(kind, text, source, data), out)
        if r.used == ["assistant"] and live.engine.pending_new:   # a loop asked something better than the helper's
            out["reply"] = ""                                        # generic answer: show and say the question instead
            out["adapt"]["speak"] = live.engine.speak(None, "")
        if revealed:
            out["adapt"] = live.engine.process(self._reveal_event(live), state) | {"speak": out["adapt"]["speak"]}
        return out

    def command(self, sid: str, text: str, source: str = "typed", voice: dict | None = None) -> dict:
        """What the coach said. If PaceSense has a question open and the words answer it ("yes", "the second
        one", "2010", "undo"), the answer is taken; otherwise it is a new sentence for the interpreter."""
        def run(live: Live):
            pend = live.engine.pending
            if pend:
                c = match_choice(text, pend["data"].get("choices") or [])
                if c is not None:
                    return self._take_choice(sid, live, c, text, source, pend, voice)
            elif is_bare_answer(text):                           # "yes" with nothing to say yes to
                reply = ("There's no question open right now. Tell me about her races, say “guide me”, "
                         "or ask what the likely range means.")
                adapt = live.engine.snapshot() | {"speak": reply}
                return {"reply": reply, "card": None, "actions": [], "used": ["prompt"], "simulation": None,
                        "state": self.view(sid, live), "adapt": adapt}
            return self._run_command(sid, live, text, source, extra={"voice": voice} if voice else None)
        return self._with(sid, run)

    def choose(self, sid: str, index: int) -> dict:
        """The coach tapped choice `index` of the open question (same path as saying it)."""
        def run(live: Live):
            pend = live.engine.pending
            choices = (pend or {}).get("data", {}).get("choices") or []
            if not pend or not 0 <= index < len(choices):
                raise ValidationError("there is no open question with that choice")
            return self._take_choice(sid, live, choices[index], choices[index]["label"], "typed", pend, None)
        return self._with(sid, run)

    def _take_choice(self, sid: str, live: Live, c: dict, said: str, source: str, pend: dict,
                     voice: dict | None) -> dict:
        loop = pend.get("loop")
        if c.get("interview"):
            out = self._interview(sid, live, c["interview"][0], c["interview"][1])
        elif c.get("say"):
            out = self._run_command(sid, live, c["say"], source, extra={"prompt_answer": loop, "voice": voice})
        elif c.get("action") == "undo":
            out = self._undo(sid, live)
        elif c.get("action") == "explain":                       # answer, then ask the same question again
            text = explain(c.get("explain", "check"), self.view(sid, live), c.get("race"))
            out = {"reply": text, "card": None, "actions": [], "used": ["prompt"], "simulation": None,
                   "state": self.view(sid, live), "adapt": live.engine.snapshot()}
            again = pend["message"].split(". ")[-1] if pend.get("message") else ""
            out["adapt"]["speak"] = f"{text} {again}{live.engine._choices_aloud(pend)}".strip()
        else:
            act = c.get("action") or "dismiss"
            reply = "OK, I'll leave it as it is." if act in ("dismiss", "ack") else ""
            actions: list[dict] = []
            if act == "keep" and c.get("turn"):
                live.hud.react(int(c["turn"]), "disagree", "kept against the data")
                reply = "Kept. Your knowledge wins here; I've noted that the data disagreed."
            elif act.startswith("tab:"):
                actions = [{"type": "focus", "panel": act[4:]}]
                reply = "Here is what similar runners went on to run."
            out = {"reply": reply, "card": None, "actions": actions, "used": ["prompt"], "simulation": None,
                   "state": self.view(sid, live)}
            out["adapt"] = live.engine.dismiss()
            out["adapt"]["speak"] = live.engine.speak(None, reply)
        out["choice"] = {"label": c.get("label"), "loop": loop, "said": said}
        return out

    def set_weight(self, sid: str, index: int, weight: float) -> dict:
        def run(live: Live):
            s = live.hud
            if index >= len(s.steer.races):
                raise ValidationError(f"race index {index} is out of range")
            turn = s.steer.manual(index, round(weight, 2)).to_dict()
            out = {"card": s.feedback_card(turn), "state": self.view(sid, live)}
            return self._observe(sid, live, Event("slider", source="manual",
                                                  data={"applied": turn["applied"], "turn": turn["n"]}), out)
        return self._with(sid, run)

    def _undo(self, sid: str, live: Live) -> dict:
        s = live.hud
        last = next((t for t in reversed(s.steer.turns) if t.source != "undo" and t.applied), None)
        undid = [a.verdict for a in last.applied] if last else []
        turn = s.steer.undo().to_dict()
        out = {"card": s.feedback_card(turn), "state": self.view(sid, live), "reply": ""}
        return self._observe(sid, live, Event("undo", "undo", "typed", {"undid_verdicts": undid,
                                                                         "applied": turn["applied"]}), out)

    def undo(self, sid: str) -> dict:
        return self._with(sid, lambda live: self._undo(sid, live))

    def estimate(self, sid: str, finish: str) -> dict:
        secs = parse_time(finish)
        if secs is None or not 2 * 3600 <= secs <= 7 * 3600:
            raise ValidationError("an estimate should be between 2:00 and 7:00")
        return self.command(sid, f"I think she can run {format_time(secs)}")

    def simulate(self, sid: str, temp_c: float, wind_kmh: float, profile: str) -> dict:
        return self._with(sid, lambda live: live.hud.simulate(temp_c, wind_kmh, profile))

    def _reveal_event(self, live: Live) -> Event:
        st = live.hud.state()["holdout"] or {}
        return Event("reveal", source="system", data={"error_pct": st.get("error_pct"), "runner": live.hud.source_runner,
                                                      "equal_weights_error_pct": st.get("equal_weights_error_pct")})

    def reveal(self, sid: str) -> dict:
        def run(live: Live):
            s = live.hud
            if s.truth_s is None:
                raise ValidationError("this session has no hidden race; open the demo runner")
            s.revealed = True
            st = self.view(sid, live)
            st["adapt"] = live.engine.process(self._reveal_event(live), st)
            return st
        return self._with(sid, run)

    def react(self, sid: str, turn: int, value: str, note: str = "") -> dict:
        def run(live: Live):
            counts = live.hud.react(turn, value, note)
            t = next((x for x in live.hud.steer.turns if x.n == turn), None)
            verdict = t.applied[0].verdict if t and t.applied else None
            adapt = live.engine.process(Event("reaction", source="typed", data={"value": value, "verdict": verdict,
                                                                                "turn": turn}), self.view(sid, live))
            return {**counts, "adapt": adapt}
        return self._with(sid, run)

    # ---- adaptive layer --------------------------------------------------------------------------
    def event(self, sid: str, kind: str, panel: str | None = None) -> dict:
        def run(live: Live):
            if kind == "dismiss":
                return {"adapt": live.engine.dismiss()}
            adapt = live.engine.process(Event(kind, source="system", data={"panel": panel} if panel else {}),
                                        self.view(sid, live))
            return {"adapt": adapt}
        return self._with(sid, run)

    def interview_start(self, sid: str) -> dict:
        def run(live: Live):
            live.engine.ctx.session.state = self.view(sid, live)
            adapt = live.engine.start_interview()
            adapt["speak"] = (adapt.get("prompt") or {}).get("message", "")
            return {"adapt": adapt}
        return self._with(sid, run)

    def _interview(self, sid: str, live: Live, question: str, choice: str) -> dict:
        try:
            c = live.engine.interview_choice(question, choice)
        except KeyError:
            raise ValidationError(f"unknown question or choice: {question}/{choice}") from None
        extra = {"question": question, "choice": c.key, "familiarity": c.familiarity}
        if c.say:
            out = self._run_command(sid, live, c.say, "typed", kind="interview_answer", extra=extra)
        else:
            out = {"reply": "Noted.", "card": None, "actions": [], "used": [], "simulation": None,
                   "state": self.view(sid, live)}
            self._observe(sid, live, Event("interview_answer", c.label, "typed", extra), out)
        if not (out.get("card") and out["card"].get("items")):
            out["adapt"]["speak"] = live.engine.speak(None, "")
        return out

    def interview_answer(self, sid: str, question: str, choice: str) -> dict:
        return self._with(sid, lambda live: self._interview(sid, live, question, choice))

    def set_condition(self, sid: str, condition: str) -> dict:
        def run(live: Live):
            live.engine.ctx.session.state = self.view(sid, live)
            try:
                adapt = live.engine.set_condition(condition)
            except ValueError as e:
                raise ValidationError(str(e)) from None
            adapt["speak"] = live.engine.speak(None, "")
            return {"adapt": adapt}
        return self._with(sid, run)

    def profile(self, coach_id: str) -> dict:
        eng = AdaptiveEngine(self.coach.repo, coach_id)
        snap = eng.snapshot()
        return {"coach_id": coach_id, "persona": snap["persona"], "outcomes": eng.ctx.profile.get("outcomes", []),
                "panel_usage": eng.ctx.profile.get("panel_usage", {}), "layout_lock": eng.ctx.profile.get("layout_lock"),
                "interview_default": bool(eng.ctx.profile.get("interview_default")),
                "voice_preset": eng.ctx.profile.get("voice_preset"),
                "runner_notes": eng.ctx.profile.get("runner_notes", {})}

    def edit_profile(self, coach_id: str, changes: dict, sid: str | None = None) -> dict:
        if sid:
            def run(live: Live):
                if live.coach_id != coach_id:
                    raise ValidationError("that session belongs to another coach")
                try:
                    return {"adapt": live.engine.edit_persona(dict(changes))}
                except ValueError as e:
                    raise ValidationError(str(e)) from None
            return self._with(sid, run)
        try:
            AdaptiveEngine(self.coach.repo, coach_id).edit_persona(dict(changes))
        except ValueError as e:
            raise ValidationError(str(e)) from None
        return self.profile(coach_id)

    def forget_coach(self, coach_id: str) -> None:
        AdaptiveEngine(self.coach.repo, coach_id).forget()

    def export(self, sid: str) -> dict:
        """Everything needed to audit or replay the session (pacesense.steer.replay), plus what adapted and why."""
        def run(live: Live):
            s = live.hud
            st = self.view(sid, live)
            snap = live.engine.snapshot()
            return {"pacesense_version": __version__, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "data": {"source": self.info.source, "fingerprint": self.info.fingerprint},
                    "runner": st["runner"], "holdout": st["holdout"], "transcript": s.steer.transcript(),
                    "reactions": s.reactions, "final": {"weights": [r["weight"] for r in st["races"]],
                                                        "prediction": st["prediction"], "steering": st["steering"]},
                    "adaptive": {"persona": snap["persona"], "loops": snap["loops"],
                                 "adaptations": [a.to_dict() for a in live.engine.log]}}
        return self._with(sid, run)

    def close(self, sid: str) -> None:
        self.sessions.remove(sid)

    # ---- operations -----------------------------------------------------------------------------
    def health(self) -> dict:
        return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - self.started, 1)}

    def ready(self) -> dict:
        ok = self.info.cases > 0
        return {"ready": ok, "data": self.info.__dict__, "sessions": self.sessions.stats()}


__all__ = ["DataInfo", "Live", "StudioService", "load_coach"]
