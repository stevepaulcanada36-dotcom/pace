"""The application layer: everything the Studio API does, independent of HTTP.

One `StudioService` per process. It owns the model and data (loaded once at start-up), the session store and
the optional AI factory. Each session is a `HudSession` (model weights, steering history, demo hold-out,
feedback loop, simulator); the methods here lock that session, call it, and return plain dicts for JSON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
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
from ..hud.commands import Routed, route, ui_commands
from ..repro import configured_seed
from ..storage import Repository
from ..study import NotFound
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
    voice_offer: dict | None = None      # a control PaceSense offered and is waiting for "yes" or "no" on


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
        from ..llm.config import boson_api_key, openrouter_api_key
        has_key = bool(boson_api_key()[0] or openrouter_api_key()[0])
        # A configured provider key is sufficient to enable the assistant; PACESENSE_AI=1
        # remains a compatibility opt-in. AIService still honors PACESENSE_AI=offline.
        if settings.ai_enabled or has_key:
            from ..llm.service import AIService
            ai = lambda: AIService(coach.repo)
            try:
                probe = ai()
                log.info("AI provider", extra={"data": {
                    "online": probe.online, "provider": probe.provider if probe.online else None,
                    "backup": probe.backup_provider, "key_source": probe.key_source,
                    "assistant_model": probe.config("assistant")[0] if probe.online else None,
                    "transcribe_model": probe.config("transcribe")[0] if probe.online else None}})
            except Exception as exc:  # noqa: BLE001 - a startup log line must never stop the server
                log.warning("could not describe the AI provider: %s", exc)
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
            own = self.coach._own()
            records.update(own)
            picks = sorted(records.values(), key=lambda r: r.runner_id)
            out += [CoachService._summary(r, r.runner_id in own) for r in picks[: limit - len(out)]]
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

    @staticmethod
    def _assistant_context(live: Live) -> str:
        """Ground free-form voice/chat turns in the active runner and any open adaptive question."""
        hud = live.hud
        state = hud.state()
        runner = state.get("runner") or {}
        try:
            gender = str(hud.coach.runner(hud.runner_id).gender or "").upper()[:1]
        except Exception:  # noqa: BLE001 - the pronoun hint is optional context
            gender = ""
        sex = {"M": "; male, say he/him", "F": "; female, say she/her"}.get(gender, "")
        who = f"Current runner: {runner.get('name') or hud.runner_id} (ID {hud.runner_id}{sex})."
        lines = [who, f"Runner familiarity condition: {live.engine.ctx.session.condition}."]
        races = state.get("races") or []
        if races:
            lines.append("Existing marathon records: " + "; ".join(
                f"{r.get('event', 'Marathon')} {r.get('year', '')}".strip() for r in races))
            lines.append("Race details (these are past results, not predictions): " + "; ".join(
                f"{r.get('event', 'Marathon')} {r.get('year', '')}: finished in "
                f"{r.get('finish', 'an unknown time')}, "
                f"{(r.get('evidence') or {}).get('label') or 'no comparison available'}, "
                f"current weight {float(r.get('weight', 0.5)):.2f}" for r in races))
        prediction = state.get("prediction") or {}
        if prediction.get("finish") and prediction.get("low_s") and prediction.get("high_s"):
            lines.append("Current model target for the next marathon: " + str(prediction["finish"]))
            lines.append("Current likely range: " + format_time(prediction["low_s"]) +
                         " to " + format_time(prediction["high_s"]))
        notes = (live.engine.ctx.profile.get("runner_notes") or {}).get(hud.source_runner, [])
        if notes:
            lines.append("Previously captured coach context: " + " | ".join(
                str(n.get("text", "")) for n in notes[-5:] if isinstance(n, dict)))
        pending = live.engine.pending
        if pending:
            choices = pending.get("data", {}).get("choices") or []
            labels = [str(choice.get("label", "")) for choice in choices if choice.get("label")]
            lines.append("PaceSense's currently open follow-up question: " + str(pending.get("message", "")))
            if labels:
                lines.append("Available answers: " + "; ".join(labels))
        lines.append("Treat trial runs, relays, training and injuries as coach-provided context, not marathon records.")
        return "\n".join(lines)

    def _interpret_voice_intent(self, text: str, live: Live, voice_signals: dict | None = None) -> dict | None:
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
            system = """You interpret one spoken command for PaceSense Studio.
Return exactly one JSON object: {"intent":"...","arguments":{},"confidence":0.0,"evidence":["words from the transcript"],"clarification":""}.
Allowed intents only:
- runner_database: {"query":"only a spoken runner name or ID, otherwise empty","gender":"F|M|empty","min_races":integer 0-20}
- select_runner: {"runner":"the exact name or ID the coach actually said; the local validator resolves it against the full database"}
- set_familiarity: {"value":"known|unfamiliar","runner":"optional exact current runner name or ID"}
- focus_panel or close_panel: {"panel":"races|prediction|pacing|similar|simulator|feedback|steering|chat|log|record"}
- simulate: {"temp_c":number,"wind_kmh":number,"profile":"on_plan|fast_start|late_fade"}
- reveal or undo: {}
- none: {}
Rules:
Interpret the transcript, not assumptions. Use only one intent. A question about data is not a UI action: use none unless the coach explicitly asks to open, close, search, filter, select, simulate, reveal or undo. For runner_database, query contains only an actual name or ID spoken by the coach; set gender or min_races only when explicitly requested. Do not treat familiar/unfamiliar/known/unknown as database filters: familiarity applies only to the currently selected runner. Never omit or invent a filter. For select_runner, require an explicit request to open/load/switch/select/choose/pick one name or ID actually spoken by the coach. The supplied catalog is only a preview; the local validator checks the full database. Never invent an ID or name. Both familiar and unfamiliar runners already exist in the database. Familiarity means whether the coach personally knows the currently selected runner; unfamiliar never means create a runner or add race data. If another runner is named for a familiarity change, include it in runner; do not change a different active runner. When selecting a different runner, do not copy the current runner's familiarity: the UI asks the coach again. For simulate, use current_ui_state for every condition the coach did not explicitly change; never invent temperature, wind or pacing conditions. A question, a comment about a race, or information about the runner (an injury, training, how a race went) is not a screen control: use none with an empty clarification, and PaceSense will answer it in conversation. A year or a city on its own is one of the current runner's races listed in current_runner.races, never a runner to search for. Use recent_turns to resolve short follow-ups such as "yes", "that one" or "the prediction one". Only when the coach clearly asked for one of the controls above but a required detail is missing, use none and ask for that one detail in clarification, as a short question a person would ask aloud. Confidence is your estimate that intent and arguments match the actual words. Evidence must quote or closely identify words actually spoken. Never execute an action; local validation and the application do that separately."""
            signals = voice_signals or {}
            state = signals.get("ui_state") or {}
            current_runner = self.coach.runner(live.hud.runner_id)
            user = {
                "utterance": text,
                "current_runner": {
                    "runner_id": live.hud.runner_id,
                    "name": CoachService._summary(current_runner, False).get("name") or live.hud.runner_id,
                    "familiarity": live.engine.ctx.session.condition,
                    "races": [f"{r.get('event')} {r.get('year')}" for r in (live.hud.steer.races or [])],
                },
                "recent_turns": [{"role": m.get("role"), "content": str(m.get("content", ""))[:240]}
                                 for m in live.hud.history[-6:]],
                "current_ui_state": {
                    "temp_c": state.get("temp_c", 12.0),
                    "wind_kmh": state.get("wind_kmh", 0.0),
                    "profile": state.get("profile", "on_plan"),
                },
                "can_undo": bool(live.hud.steer._undo),
                "can_reveal": live.hud.truth_s is not None,
                "existing_runner_catalog": catalog,
            }
            result = ai.client.chat(model, [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ], temperature=0.0, json_mode=True)
            payload = parse_json_answer(result.text)
            verdict = validate_voice_intent(
                payload, rows, utterance=text,
                transcript_confidence=signals.get("confidence"),
                current_runner_id=live.hud.runner_id,
                current_runner_ids=[
                    live.hud.runner_id, live.hud.source_runner,
                    *(["demo"] if live.hud.source_runner == self.demo_runner else []),
                ],
                ui_state=state,
                can_undo=bool(live.hud.steer._undo),
                can_reveal=live.hud.truth_s is not None,
            )
            # The classifier found no screen control, or was unsure there was one. That is not a refusal
            # to act: mark it so the conversation path answers instead of a bare "please clarify".
            from ..hud.voice_actions import MIN_CONFIDENCE
            try:
                unsure = float(payload.get("confidence", 0.0)) < MIN_CONFIDENCE
            except (TypeError, ValueError):
                unsure = True
            if verdict.get("status") == "clarify" and not verdict.get("offer") and (
                    payload.get("intent") in (None, "", "none") or unsure):
                verdict = {**verdict, "unmapped": True}
            return verdict
        except Exception as exc:  # noqa: BLE001 - any provider or parsing failure must fall back to the safe rules
            log.warning("voice intent interpretation unavailable; using safe rule-based fallback: %s", exc)
            return None

    @staticmethod
    def _voice_route(text: str, source: str, live: Live, intent: dict) -> Routed:
        """Convert one validated intent into a structured Studio action; do not re-parse AI output."""
        name, args = intent["intent"], intent["arguments"]
        if name == "select_runner":
            # Familiarity belongs to the selected runner/session. Never carry the previous runner's
            # known/unknown choice across a voice-driven switch; ask again for a different runner.
            condition = (
                live.engine.ctx.session.condition
                if args["runner_id"] == live.hud.runner_id else "ask"
            )
            return Routed(
                reply=f"Opening {args['name']} from the existing runner database.",
                actions=[{"type": "runner", "id": args["runner_id"], "name": args["name"],
                          "condition": condition}],
                used=["ui"],
            )
        if name == "runner_database":
            return Routed(
                reply="Searching the existing runner database.",
                actions=[{"type": "runner_database", "query": args["query"],
                          "gender": args["gender"], "min_races": args["min_races"]}],
                used=["ui"],
            )
        if name == "set_familiarity":
            value = args["condition"]
            return Routed(
                reply="Updating how familiar you are with this existing runner.",
                actions=[{"type": "condition", "value": value}],
                used=["ui"],
            )
        if name in ("focus_panel", "close_panel"):
            action_type = "focus" if name == "focus_panel" else "close"
            verb = "Opening" if action_type == "focus" else "Closing"
            return Routed(
                reply=f"{verb} the {args['panel']} panel.",
                actions=[{"type": action_type, "panel": args["panel"]}],
                used=["ui"],
            )
        if name == "simulate":
            return Routed(
                reply="Running the race-day simulation with the validated settings.",
                actions=[{"type": "simulate", **args}],
                used=["ui"],
            )
        if name == "reveal":
            return route("reveal", live.hud, source)
        if name == "undo":
            return route("undo", live.hud, source)
        return Routed(reply="I couldn't map that request to a supported PaceSense action.", used=["prompt"])

    @staticmethod
    def _fallback_voice_simulation(text: str, voice_signals: dict | None) -> dict | None:
        """Safely handle a simple simulation command when Boson is unavailable.

        Only explicitly spoken weather/profile values are changed. Missing values are
        copied from the browser's current controls; adjectives never become guessed numbers.
        """
        actions, _, _ = ui_commands(text)
        if not any(action.get("type") == "simulate" for action in actions):
            return None
        if not re.search(
            r"\b(simulate|run\s+(?:the\s+)?(?:race|simulation)|play\s+the\s+race|"
            r"replay\s+the\s+race|what\s+if)\b",
            text, re.IGNORECASE,
        ):
            return None

        signals = voice_signals or {}
        supplied = signals.get("ui_state") or {}
        current = {
            "temp_c": supplied.get("temp_c", 12.0),
            "wind_kmh": supplied.get("wind_kmh", 0.0),
            "profile": supplied.get("profile", "on_plan"),
        }
        for key, default in (("temp_c", 12.0), ("wind_kmh", 0.0), ("profile", "on_plan")):
            if current[key] is None:
                current[key] = default

        temp_match = re.search(
            r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(?:°\s*c|degrees?(?:\s+celsius)?|celsius|c\b)",
            text, re.IGNORECASE,
        )
        wind_match = re.search(
            r"(?:(\d+(?:\.\d+)?)\s*(?:km/?h|kph|kilometres?\s+per\s+hour|kilometers?\s+per\s+hour)\s*(?:head)?wind|"
            r"(?:head)?wind\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:km/?h|kph)?)",
            text, re.IGNORECASE,
        )
        profile = current["profile"]
        if re.search(r"\b(fast start|starts too fast|start too fast)\b", text, re.IGNORECASE):
            profile = "fast_start"
        elif re.search(r"\b(late fade|fade late|hits the wall|hit the wall)\b", text, re.IGNORECASE):
            profile = "late_fade"

        payload = {
            "intent": "simulate",
            "arguments": {
                "temp_c": float(temp_match.group(1)) if temp_match else current["temp_c"],
                "wind_kmh": float(wind_match.group(1) or wind_match.group(2)) if wind_match else current["wind_kmh"],
                "profile": profile,
            },
            "confidence": 1.0,
            "evidence": [text],
        }
        from ..hud.voice_actions import validate_voice_intent
        return validate_voice_intent(
            payload, [], utterance=text,
            transcript_confidence=signals.get("confidence"),
            ui_state=current,
        )

    @staticmethod
    def _is_race_comment(text: str, live: Live) -> bool:
        """True when the utterance names one of the current runner's races and is not a UI command."""
        if ui_commands(text)[0]:
            return False
        lowered = text.casefold()
        for race in getattr(live.hud.steer, "races", ()) or ():
            event = str(race.get("event") or "").casefold()
            if event and event in lowered:
                return True
            year = race.get("year")
            if year is not None and re.search(rf"(?<!\d){int(year)}(?!\d)", text):
                return True
        return False

    # Words that can ask for a screen control. Deliberately broad: it only decides whether the control
    # classifier is worth a call, and the classifier and validator still decide what, if anything, to do.
    _UI_REQUEST_RE = re.compile(
        r"\b(open|show|display|view|see|bring\s+up|pull\s+up|look\s+up|search|find|filter|list|browse|"
        r"select|choose|pick|switch|load|go\s+to|take\s+me|close|hide|dismiss|minimi[sz]e|pin|focus|"
        r"simulat\w*|what\s+if|reveal|undo|revert|database|panel|dashboard|"
        r"familiar\w*|unfamiliar|know\s+(?:this|the|him|her)|coach\s+(?:this|the|him|her)|new\s+to\s+me|"
        r"known\s+to\s+me)\b", re.IGNORECASE)

    @staticmethod
    def _small_talk(text: str) -> str:
        """A reply for a word or two that is not a request ("wait", "hello"), without spending a model call."""
        words = re.sub(r"[^a-z' ]+", " ", text.casefold()).split()
        said = " ".join(words)
        if not words or len(words) > 4:
            return ""
        if re.fullmatch(r"(?:wait|hold on|hang on|one (?:sec|second|moment|minute)|just a (?:sec|second|moment|minute)|"
                        r"give me a (?:sec|second|moment|minute)|hmm+|um+|uh+|let me think)(?: please)?", said):
            return "Take your time. I'm listening."
        if re.fullmatch(r"(?:hello|hi|hey)(?: there| pacesense| pace sense)?|(?:are you|you) there|can you hear me", said):
            return "I'm here. Tell me about a race, or ask me about the prediction."
        if re.fullmatch(r"(?:thanks|thank you|thank you very much|cheers|great|perfect|got it)", said):
            return "You're welcome."
        return ""

    _OTHER_ASSISTANT_RE = re.compile(r"^\s*(?:ok(?:ay)?|hey|hi|hello)\s+(?:google|siri|alexa|cortana|bixby)\b|"
                                     r"^\s*(?:alexa|siri)\b", re.IGNORECASE)
    _ONE_WORD_REQUESTS = frozenset({"undo", "reveal", "reset", "why", "explain", "help", "repeat", "again",
                                    "simulate", "continue", "next", "back", "stop", "summary", "summarise",
                                    "summarize"})
    _GUIDE_RE = re.compile(r"^\s*(?:please\s+)?(?:guide\s+me|start\s+(?:the\s+)?(?:guided\s+)?interview|"
                           r"walk\s+me\s+through(?:\s+it)?|interview\s+me|guided\s+interview)(?:\s+please)?\s*$",
                           re.IGNORECASE)

    def _not_for_pacesense(self, text: str, live: Live) -> bool:
        """Speech the microphone caught that carries no request: another assistant's wake word, or a word
        or two that is not an answer, a command, a year or a race."""
        if self._OTHER_ASSISTANT_RE.search(text):
            return True
        words = [w for w in re.sub(r"[^a-z0-9' ]+", " ", text.casefold()).split()
                 if w not in {"um", "uh", "erm", "hmm", "so", "well", "like", "and", "the", "a", "oh"}]
        if len(words) > 2:
            return False
        if not words:
            return True
        from ..adaptive.answers import is_bare_answer
        if is_bare_answer(text) or self._small_talk(text) or ui_commands(text)[0]:
            return False
        if any(w.isdigit() for w in words) or set(words) & self._ONE_WORD_REQUESTS:
            return False
        names = {str(r.get("event") or "").casefold() for r in (live.hud.steer.races or [])}
        return not (set(words) & names)

    def _unheard(self, sid: str, live: Live, text: str, source: str, voice: dict | None) -> dict:
        """Acknowledge on screen, say nothing aloud, and keep it out of the conversation history."""
        reply = "I didn't catch a request in that, so I've left everything as it is. I'm still listening."
        out: dict[str, Any] = {"reply": reply, "actions": [], "card": None, "used": ["prompt"],
                               "simulation": None, "state": self.view(sid, live), "turn": None}
        data = {"applied": [], "turn": None, "estimate_s": None, "notes": [], "used": ["prompt"],
                "ui_only": True, "unheard": True, **({"voice": voice} if voice else {})}
        self._observe(sid, live, Event("utterance", text, source, data), out)
        out["adapt"]["speak"] = ""
        out["voice_action_pending"] = False
        return out

    def _guide(self, sid: str, live: Live, text: str) -> dict:
        """"Guide me", spoken or typed, starts the guided interview exactly like the button."""
        live.engine.ctx.session.state = state = self.view(sid, live)
        adapt = live.engine.start_interview()
        question = (adapt.get("prompt") or {}).get("message", "")
        reply = "Starting the guided interview." if question else "The guided interview has no more questions for this runner."
        adapt["speak"] = f"{reply} {question}{live.engine._choices_aloud(live.engine.pending) if live.engine.pending else ''}".strip()
        live.hud.remember_turn(text, f"{reply} {question}".strip())
        return {"reply": reply, "actions": [], "card": None, "used": ["prompt"], "simulation": None,
                "state": state, "turn": None, "adapt": adapt, "voice_action_pending": False}

    @classmethod
    def _wants_ui_action(cls, text: str) -> bool:
        return bool(cls._UI_REQUEST_RE.search(text) or ui_commands(text)[0])

    def _take_offer(self, sid: str, live: Live, text: str, source: str, voice: dict | None) -> dict | None:
        """"Yes" or "no" to a control PaceSense offered on the previous turn; None if `text` is neither."""
        offer = live.voice_offer
        if not offer:
            return None
        from ..adaptive.answers import NO, YES, _norm
        words = _norm(text)
        if len(words.split()) > 6 or not (YES.match(words) or NO.match(words)):
            return None
        live.voice_offer = None
        if NO.match(words):
            r = Routed(reply="Okay, I've left the screen as it is.", used=["prompt"])
        else:
            r = self._voice_route(text, source, live, offer)
        state = self.view(sid, live)
        out: dict[str, Any] = {"reply": r.reply, "actions": r.actions, "card": None, "used": r.used,
                               "simulation": None, "state": state, "turn": None}
        data = {"applied": [], "turn": None, "estimate_s": None, "notes": [], "used": r.used,
                "ui_only": True, "prompt_answer": "voice_offer", **({"voice": voice} if voice else {})}
        self._observe(sid, live, Event("utterance", text, source, data), out)
        out["voice_action_pending"] = source == "voice" and bool(r.actions)
        if out["voice_action_pending"]:
            out["adapt"]["speak"] = ""
        live.hud.remember_turn(text, r.reply)
        return out

    def _run_command(self, sid: str, live: Live, text: str, source: str, kind: str = "utterance",
                     extra: dict | None = None) -> dict:
        s = live.hud
        s.assistant_context = self._assistant_context(live)
        voice_signals = (extra or {}).get("voice") or {}
        # A spoken comment about one of this runner's races is handled by the validated steering
        # interpreter below, so skip the separate intent classifier for it and avoid two Boson calls
        # for one sentence. Everything else spoken (runner search, navigation, paraphrased controls)
        # still goes through the classifier; gating it on the fixed-phrase parser would switch the
        # classifier off for exactly the paraphrases it exists to understand.
        # The intent classifier only understands screen controls. Run it when the words could be asking
        # for one; a question or a comment goes straight to the conversation path, which has the history.
        voice_intent = (self._interpret_voice_intent(text, live, voice_signals)
                        if source == "voice" and self._wants_ui_action(text)
                        and not self._small_talk(text)
                        and not self._is_race_comment(text, live) else None)
        unmapped_clarification = ""
        if voice_intent and voice_intent.get("status") == "clarify" and voice_intent.get("unmapped"):
            # Not a control after all. Let the steering interpreter and the assistant try, and keep the
            # classifier's question only for the case where neither has anything to say.
            unmapped_clarification = voice_intent["message"]
            voice_intent = None
        live.voice_offer = None
        small_talk = self._small_talk(text) if source == "voice" else ""
        if small_talk:
            r = Routed(reply=small_talk, used=["prompt"])
        elif voice_intent and voice_intent.get("status") == "clarify":
            r = Routed(reply=voice_intent["message"], used=["prompt"])
            if voice_intent.get("offer"):
                live.voice_offer = {"status": "action", **voice_intent["offer"],
                                    "confidence": voice_intent.get("confidence", 0.0),
                                    "evidence": voice_intent.get("evidence", [])}
        elif voice_intent and voice_intent.get("status") == "action":
            r = self._voice_route(text, source, live, voice_intent)
        elif source == "voice":
            ui_actions, _, remainder = ui_commands(text)
            if len(ui_actions) > 1 or (ui_actions and remainder.strip()):
                r = Routed(reply="I heard more than one request. Which action should I do first?", used=["prompt"])
            elif ui_actions and ui_actions[0].get("type") == "simulate":
                fallback_intent = self._fallback_voice_simulation(text, voice_signals)
                if fallback_intent:
                    voice_intent = fallback_intent
                    if fallback_intent.get("status") == "clarify":
                        r = Routed(reply=fallback_intent["message"], used=["prompt"])
                    elif fallback_intent.get("status") == "action":
                        r = self._voice_route(text, source, live, fallback_intent)
                    else:
                        r = Routed(reply="Do you want me to run the race-day simulation?", used=["prompt"])
                else:
                    r = Routed(reply="Do you want me to run the race-day simulation?", used=["prompt"])
            elif any(action.get("type") not in {
                "runner_database", "condition", "focus", "close", "runner", "reveal"
            } for action in ui_actions):
                r = Routed(
                    reply="That control isn't available in PaceSense Studio yet. I haven't changed anything.",
                    used=["prompt"],
                )
            elif ui_actions and ui_actions[0].get("type") == "runner" and ui_actions[0].get("id") != "demo":
                target = str(ui_actions[0].get("id") or "")
                try:
                    self.coach.runner(target)
                except NotFound:
                    r = Routed(
                        reply="I couldn't find that exact runner ID in the existing database. Open the runner database to search.",
                        used=["prompt"],
                    )
                else:
                    r = route(text, s, source)
            else:
                r = route(text, s, source)
        else:
            r = route(text, s, source)
        if unmapped_clarification and not set(r.used) & {"steer", "assistant", "ui"}:
            r.reply = unmapped_clarification
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
                               "simulation": sim, "state": state, "turn": r.turn}
        if voice_intent:
            out["voice_intent"] = {key: voice_intent[key] for key in
                                   ("status", "intent", "arguments", "confidence", "evidence") if key in voice_intent}
        turn = r.turn or {}
        data = {"applied": turn.get("applied", []), "turn": turn.get("n"), "estimate_s": turn.get("estimate_s"),
                "notes": turn.get("notes", []),
                "used": r.used, "ui_only": "ui" in r.used and "steer" not in r.used, **(extra or {})}
        self._observe(sid, live, Event(kind, text, source, data), out)
        # AI-interpreted UI actions are confirmed by the browser only after the action succeeds.
        # Do not let the adaptive TTS speak the optimistic route reply before that confirmation.
        turn_changed = bool(turn) and (
            turn.get("source") == "undo" or bool(turn.get("applied"))
            or turn.get("estimate_s") is not None
        )
        ui_action_pending = any(action.get("type") != "refresh" for action in r.actions)
        out["voice_action_pending"] = source == "voice" and (ui_action_pending or turn_changed)
        if out["voice_action_pending"] or (voice_intent and voice_intent.get("status") == "action"):
            out["adapt"]["speak"] = ""
        # Keep the assistant's answer visible even when an adaptive loop also asks a follow-up.
        # The adaptation is rendered separately and included in the next AI context.
        if revealed:
            out["adapt"] = live.engine.process(self._reveal_event(live), state) | {"speak": out["adapt"]["speak"]}
        if kind == "utterance" and source in ("voice", "typed") and not (extra or {}).get("prompt_answer"):
            response = out.get("reply") or ""
            pending = live.engine.pending
            # Record a follow-up question once, on the turn it is raised. Copying the open question and its
            # button labels into every later turn taught the model to answer with them.
            if pending and live.engine.pending_new:
                response += "\nPaceSense then asked: " + str(pending.get("message", ""))
            if not (extra or {}).get("unheard"):
                s.remember_turn(text, response)
        return out

    def transcribe(self, sid: str, audio: bytes, fmt: str = "wav") -> dict:
        """The AI provider's transcript of one recorded utterance. Never raises for a provider problem: the
        browser keeps its own speech recognition as the fallback, so an empty text simply means "use that"."""
        live = self.sessions.entry(sid).value                     # 404 for an unknown or expired session
        if not self.ai_factory:
            return {"text": "", "available": False, "source": "none", "latency_s": 0.0}
        t0 = time.perf_counter()
        provider = "server"
        try:
            ai = self.ai_factory()
            if not getattr(ai, "online", False):
                return {"text": "", "available": False, "source": "none", "latency_s": 0.0}
            provider = str(getattr(ai, "provider", "server"))
            result = ai.transcribe(audio, fmt)
        except Exception as exc:  # noqa: BLE001 - any provider or decoding failure falls back to the browser transcript
            log.warning("server transcription unavailable; the browser transcript will be used: %s", exc)
            return {"text": "", "available": True, "source": "none", "error": str(exc)[:200],
                    "latency_s": round(time.perf_counter() - t0, 2)}
        text = " ".join(str(result.text or "").split())[:1000]
        text = self._repair_race_names(text, live)
        log.info("utterance transcribed", extra={"data": {
            "session": sid[:6], "provider": provider, "words": len(text.split()),
            "ms": round((time.perf_counter() - t0) * 1000)}})
        return {"text": text, "available": True, "source": provider,
                "latency_s": round(time.perf_counter() - t0, 2)}

    @staticmethod
    def _repair_race_names(text: str, live: Live) -> str:
        """Snap a near-miss spelling of one of this runner's race cities to the listed name ("Einthoven")."""
        import difflib
        names = {str(r.get("event") or "") for r in (live.hud.steer.races or [])}
        names = {n for n in names if len(n) >= 6 and " " not in n}
        if not names:
            return text

        def fix(match: re.Match) -> str:
            word = match.group(0)
            if len(word) < 6 or word.casefold() in {n.casefold() for n in names}:
                return word
            close = difflib.get_close_matches(word.casefold(), [n.casefold() for n in names], n=1, cutoff=0.8)
            return next(n for n in names if n.casefold() == close[0]) if close else word

        return re.sub(r"[A-Za-z]+", fix, text)

    def command(self, sid: str, text: str, source: str = "typed", voice: dict | None = None) -> dict:
        """What the coach said. If PaceSense has a question open and the words answer it ("yes", "the second
        one", "2010", "undo"), the answer is taken; otherwise it is a new sentence for the interpreter."""
        def run(live: Live):
            if source == "voice":
                log.info("voice turn", extra={"data": {
                    "session": sid[:6], "words": len(text.split()),
                    "transcriber": (voice or {}).get("transcriber") or "unreported",
                    "confidence": (voice or {}).get("confidence")}})
            taken = self._take_offer(sid, live, text, source, voice)
            if taken is not None:
                return taken
            if self._GUIDE_RE.match(text):
                return self._guide(sid, live, text)
            pend = live.engine.pending
            if pend:
                choices = pend["data"].get("choices") or []
                c = match_choice(text, choices)
                if c is None and pend.get("loop") == "clarify":
                    from ..adaptive.loops import CURRENT_NOTE, NOT_A_RACE_RE, NOW_RE
                    none = next((x for x in choices if x.get("action") == "dismiss"), None)
                    if none and (NOT_A_RACE_RE.search(text) or NOW_RE.search(text)):
                        # "No, it didn't happen during a race, it happened yesterday": close the question
                        # and keep what was said as context about the runner now.
                        out = self._take_choice(sid, live, none, text, source, pend, voice)
                        notes = live.engine.ctx.profile.setdefault("runner_notes", {}).setdefault(
                            live.hud.source_runner, [])
                        if not any(n.get("text") == text for n in notes):
                            notes.append({"t": time.time(), "text": text[:200]})
                            del notes[:-20]
                            live.engine._save()                   # the profile is reloaded on every event
                        out["reply"] = CURRENT_NOTE
                        if not live.engine.pending:
                            out["adapt"]["speak"] = CURRENT_NOTE
                        live.hud.remember_turn(text, CURRENT_NOTE)
                        return out
                if c is not None:
                    return self._take_choice(sid, live, c, text, source, pend, voice)
            if source == "voice" and self._not_for_pacesense(text, live):
                return self._unheard(sid, live, text, source, voice)
            ai_answers = bool(live.hud.ai is not None and getattr(live.hud.ai, "online", False) and live.hud.history)
            if not pend and is_bare_answer(text) and not ai_answers:   # "yes" with nothing to say yes to
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
        response = out.get("reply") or ""
        pending = live.engine.pending
        if pending and live.engine.pending_new:
            response += "\nPaceSense then asked: " + str(pending.get("message", ""))
        live.hud.remember_turn(said, response)
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
