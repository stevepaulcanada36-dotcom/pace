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
    voice_fragment: dict | None = None   # an unfinished spoken sentence, waiting for the rest of it


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
                    "transcribe_model": probe.config("transcribe")[0] if probe.online else None,
                    "transcribe_provider": getattr(probe, "transcribe_provider", None) if probe.online else None}})
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

    _HEDGES = ("maybe", "i hope", "i think", "i guess", "kind of", "sort of", "like", "you know", "probably")
    _UNSURE_RE = re.compile(r"\b(?:don'?t know|not sure|confus\w*|what do you (?:think|suggest|feel)|you decide|"
                            r"any suggestions?|what should i|help me|no idea)\b", re.IGNORECASE)
    _TERMS = ("data", "weight", "reduce", "increase", "decrease", "lower", "raise", "race", "prediction", "predict",
              "estimate", "accuracy", "percent", "percentage", "model", "target", "range", "representative",
              "precise", "accurate", "performance", "injury", "trust", "count")

    @staticmethod
    def _pull_text(race: dict) -> str:
        pull = (race.get("influence") or {}).get("pull_s")
        if pull is None or abs(float(pull)) < 1:
            return ""
        return (f", pulls the target {format_time(abs(float(pull)))[2:]} "
                f"{'slower' if float(pull) > 0 else 'faster'}")

    @staticmethod
    def _speaking_style(live: Live) -> str:
        """How this coach talks, measured from their own turns, so the reply can be given the same way."""
        said = [str(m.get("content", "")) for m in live.hud.history if m.get("role") == "user"][-8:]
        words = [w for t in said for w in re.findall(r"[a-z']+", t.casefold())]
        per_turn = (len(words) / len(said)) if said else 0.0
        joined = " ".join(said).casefold()
        hedges = sum(joined.count(h) for h in StudioService._HEDGES)
        unsure = sum(1 for t in said[-4:] if StudioService._UNSURE_RE.search(t))
        terms = [t for t in StudioService._TERMS if re.search(rf"\b{t}\w*\b", joined)][:8]
        try:
            persona = live.engine.ctx.persona.archetype()
        except Exception:  # noqa: BLE001 - the persona is optional colour for the reply style
            persona = {}
        reply = str((live.engine.ctx.session.voice_style or {}).get("reply") or "normal")
        limit = {"brief": "one sentence, at most 20 words", "full": "at most three short sentences"}.get(
            reply, "at most two short sentences, about 30 words")
        parts = [f"How to talk to this coach: reply in {limit}, in plain spoken English, answer first."]
        if persona.get("name"):
            parts.append(f"PaceSense's reading of how they work: {persona['name']}"
                         f"{' (' + persona['description'] + ')' if persona.get('description') else ''}.")
        if said:
            pace = ("short, quick phrases" if per_turn < 8 else "long, thinking-aloud sentences" if per_turn > 18
                    else "medium-length sentences")
            parts.append(f"They speak in {pace}"
                         + (", informally, with hedges like \"maybe\" and \"I hope\"" if hedges >= 3 else "")
                         + ". Match that register: everyday words, no jargon, no lecture.")
        if terms:
            parts.append("Use their own words for things where you can: " + ", ".join(terms) + ".")
        if unsure:
            parts.append("They have said they are unsure what to do, so lead: give one concrete recommendation "
                         "with one short reason, and if it is a change you can make, make it or offer exactly "
                         "that one thing. Do not list options.")
        parts.append("Give a reason only as one short clause after the answer. Never suggest adding a new runner, "
                     "opening another runner or leaving this session unless the coach asks for that.")
        return " ".join(parts)

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
                f"current weight {float(r.get('weight', 0.5)):.2f}"
                f"{StudioService._pull_text(r)}" for r in races))
        prediction = state.get("prediction") or {}
        if prediction.get("finish") and prediction.get("low_s") and prediction.get("high_s"):
            lines.append("Current model target for the next marathon: " + str(prediction["finish"]))
            lines.append("Current likely range: " + format_time(prediction["low_s"]) +
                         " to " + format_time(prediction["high_s"]))
        notes = (live.engine.ctx.profile.get("runner_notes") or {}).get(hud.source_runner, [])
        holdout = state.get("holdout") or {}
        if holdout.get("label"):
            if holdout.get("revealed") and holdout.get("truth_s"):
                check = (f"Accuracy check: this demo runner's real next marathon, {holdout['label']}, was hidden "
                         f"from the model and has now been revealed: {format_time(holdout['truth_s'])}. ")
                if holdout.get("error_pct") is not None:
                    check += (f"The current target is {float(holdout['error_pct']):.2f}% away from that result "
                              "(\"% off\" is the gap between target and real result as a share of the real "
                              "result; smaller is better). ")
                if holdout.get("equal_weights_error_pct") is not None:
                    check += (f"With equal weights it would have been "
                              f"{float(holdout['equal_weights_error_pct']):.2f}% off, so the coach's changes "
                              + ("helped." if float(holdout.get("error_pct") or 0) <
                                 float(holdout["equal_weights_error_pct"]) else "did not help."))
            else:
                check = (f"Accuracy check: this demo runner's real next marathon, {holdout['label']}, is hidden "
                         "from the model and from you. The target is the prediction for that race. The coach can "
                         "say \"reveal the real result\" to compare; you do not know the result.")
            lines.append(check + f" {holdout['label']} is therefore not in the race history above.")
        if prediction.get("initial_s") and prediction.get("finish_s"):
            moved = float(prediction["finish_s"]) - float(prediction["initial_s"])
            lines.append(f"With equal weights the target would be {format_time(prediction['initial_s'])}; the "
                         f"coach's changes have moved it {abs(moved):.0f} s "
                         f"{'slower' if moved > 0 else 'faster'}." if abs(moved) >= 1 else
                         "The weights are equal, so the coach has not moved the target yet.")
        lines.append(
            "What is on screen: a pacing chart (the plan against the past races), a similar-runners chart, a "
            "race-day simulator, and the likely-range bar. You cannot draw new charts; when the coach wants a "
            "visual, name the one that fits and tell them to say \"show the pacing chart\", \"show similar "
            "runners\" or \"open the simulator\".")
        lines.append(
            "When the coach asks what you think, asks you to interpret, or asks why: give one insight from the "
            "facts above that they could not read off the screen in a second, for example which race is pulling "
            "the target most and which way, a race that stands out from the others, or what their changes did "
            "to the target. Do not read the target and range back to them.")
        lines.append(StudioService._speaking_style(live))
        lines.append(
            "Acting on this session: you can change it only by calling set_race_weight, set_coach_estimate or "
            "undo_last_change. When the coach asks for a change and the race is clear, call the function in this "
            "turn without asking permission first; if no amount is given, move the weight by 0.2 from its "
            "current value. When the coach agrees to a change you proposed, call the function now. Never say a "
            "weight, the target or an estimate changed or was noted unless a function in this turn returned "
            "applied, and take any new numbers only from that result. You cannot edit a race result. Do not "
            "invent statistics: every percentage and time you state must appear in this context or in a "
            "function result.")
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
        r"\b(open|show|present|display|view|see|bring\s+up|pull\s+up|look\s+up|search|find|filter|list|browse|"
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

    _FAMILIARITY = (
        ("none", re.compile(
            r"\b(?:don'?t|do not|never|not)\b.{0,25}\b(?:know|met|seen|coach\w*|familiar|heard of)\b|"
            r"\b(?:not|n't)\s+(?:\w+\s+){0,2}(?:familiar|much)\b|\bunfamiliar\b|\bstranger\b|\bnew to me\b|"
            r"\bno idea who\b", re.IGNORECASE)),
        ("some", re.compile(
            r"\b(?:seen|met|watched)\s+(?:him|her|them|this runner|the runner)\b|\ba (?:little|bit)\b|\bsomewhat\b|"
            r"\bknow of\b|\bheard of\b|\bonce or twice\b|\bonly once\b|\bnot (?:very )?well\b|\bbarely\b",
            re.IGNORECASE)),
        ("coach", re.compile(
            r"\bI\s+(?:do\s+)?(?:coach|train)\s+(?:him|her|them|this runner|the runner)\b|\bmy (?:athlete|runner)\b|"
            r"\bknow (?:him|her|them|this runner) (?:very |really |quite )?well\b|\bfor years\b", re.IGNORECASE)),
    )

    @classmethod
    def _familiarity_answer(cls, text: str, choices: list[dict]) -> dict | None:
        """"I've seen him once, maybe in 2013" answers "How well do you know this runner?" without the
        coach having to say a button's exact words. Returns None unless exactly one reading fits."""
        by_key = {(c.get("interview") or [None, None])[1]: c for c in choices
                  if (c.get("interview") or [None])[0] == "familiarity"}
        about_the_runner = re.search(r"\b(?:runner|athlete|person|him|her|he|she|them|guy|man|woman|lady)\b",
                                     text, re.IGNORECASE)
        if not by_key or len(text.split()) > 30 or not about_the_runner:
            return None
        if cls._FAMILIARITY[1][1].search(text) and "some" in by_key:      # "seen him once" beats "not much"
            return by_key["some"]
        hits = [key for key, pattern in cls._FAMILIARITY if pattern.search(text) and key in by_key]
        return by_key[hits[0]] if len(hits) == 1 else None

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

    _TRAILING = frozenset({
        "but", "and", "so", "or", "because", "cause", "like", "the", "a", "an", "of", "to", "in", "on", "at",
        "with", "for", "from", "that", "which", "who", "was", "is", "are", "were", "i've", "i'm", "have", "has",
        "had", "seen", "than", "when", "if", "my", "maybe", "um", "uh", "well", "wait"})
    _HOLD_RE = re.compile(r"\b(?:wait|hold on|hang on|let me (?:complete|finish|think|say)(?: (?:it|this|that|"
                          r"the (?:statement|sentence)))?|one (?:second|moment|sec)|i'?m not (?:done|finished))\b[,.]?\s*",
                          re.IGNORECASE)
    _FRAGMENT_TTL_S = 20.0

    @classmethod
    def _is_unfinished(cls, text: str) -> bool:
        """The coach stopped mid-sentence ("...a pretty good run, but") or asked for a moment ("wait, let me
        complete it"): the rest is still coming, so this is not yet something to answer."""
        words = re.sub(r"[^a-z0-9' ]+", " ", text.casefold()).split()
        if not words or len(words) > 40:
            return False
        if cls._HOLD_RE.search(text) and len(cls._HOLD_RE.sub(" ", text).split()) <= 8:
            return True
        return len(words) >= 2 and words[-1] in cls._TRAILING

    @staticmethod
    def _is_noise(text: str) -> bool:
        """Laughter or a stammer: one word making up most of the utterance ("ha ha ha ha", "but, but, but")."""
        words = re.sub(r"[^a-z' ]+", " ", text.casefold()).split()
        if len(words) < 4:
            return False
        top = max(words.count(w) for w in set(words))
        return top / len(words) >= 0.6

    def _with_fragment(self, live: Live, text: str) -> str:
        """Put an unfinished sentence from a moment ago in front of what the coach says now."""
        frag = live.voice_fragment
        live.voice_fragment = None
        if not frag or time.time() - frag["t"] > self._FRAGMENT_TTL_S:
            return text
        joined = f"{frag['text']} {text}"
        joined = " ".join(self._HOLD_RE.sub(" ", joined).split()) or text
        return " ".join(joined.split()[-70:])

    def _keep_fragment(self, live: Live, text: str, n: int = 1) -> None:
        kept = " ".join(self._HOLD_RE.sub(" ", text).split())
        live.voice_fragment = ({"text": " ".join(kept.split()[-60:]), "t": time.time(), "n": n}
                               if kept and n <= 3 else None)

    def _unheard(self, sid: str, live: Live, text: str, source: str, voice: dict | None,
                 reply: str = "") -> dict:
        """Acknowledge on screen, say nothing aloud, and keep it out of the conversation history."""
        reply = reply or "I didn't catch a request in that, so I've left everything as it is. I'm still listening."
        out: dict[str, Any] = {"reply": reply, "actions": [], "card": None, "used": ["prompt"],
                               "simulation": None, "state": self.view(sid, live), "turn": None}
        data = {"applied": [], "turn": None, "estimate_s": None, "notes": [], "used": ["prompt"],
                "ui_only": True, "unheard": True, **({"voice": voice} if voice else {})}
        self._observe(sid, live, Event("utterance", text, source, data), out)
        out["adapt"]["speak"] = ""
        out["voice_action_pending"] = False
        out["unheard"] = True
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
        if offer.get("say"):
            if NO.match(words):
                reply = "Okay, nothing changed."
                live.hud.remember_turn(text, reply)
                return {"reply": reply, "actions": [], "card": None, "used": ["prompt"], "simulation": None,
                        "state": self.view(sid, live), "turn": None,
                        "adapt": live.engine.snapshot() | {"speak": reply}}
            # "Yes" to a change the assistant proposed: make it through the normal instruction path.
            return self._run_command(sid, live, offer["say"], source,
                                     extra={"prompt_answer": "voice_offer", **({"voice": voice} if voice else {})})
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
        s.reply_style = str((live.engine.ctx.session.voice_style or {}).get("reply") or "normal")
        voice_signals = (extra or {}).get("voice") or {}
        # A spoken comment about one of this runner's races is handled by the validated steering
        # interpreter below, so skip the separate intent classifier for it and avoid two Boson calls
        # for one sentence. Everything else spoken (runner search, navigation, paraphrased controls)
        # still goes through the classifier; gating it on the fixed-phrase parser would switch the
        # classifier off for exactly the paraphrases it exists to understand.
        # The intent classifier only understands screen controls. Run it when the words could be asking
        # for one; a question or a comment goes straight to the conversation path, which has the history.
        from ..hud.commands import chart_request
        voice_intent = (self._interpret_voice_intent(text, live, voice_signals)
                        if source == "voice" and self._wants_ui_action(text)
                        and not self._small_talk(text) and not chart_request(text)
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
        if getattr(r, "offer", None):
            live.voice_offer = dict(r.offer or {})
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
            provider = str(getattr(ai, "transcribe_provider", None) or getattr(ai, "provider", "server"))
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
            if source != "voice":
                return answer(live, text)
            if self._is_noise(text):
                return self._unheard(sid, live, text, source, voice)
            waited = int((live.voice_fragment or {}).get("n", 0))
            said = self._with_fragment(live, text)
            if waited < 3 and self._is_unfinished(said) and not is_bare_answer(text):
                # Mid-sentence: say nothing, keep the words, and answer when the rest arrives.
                self._keep_fragment(live, said, waited + 1)
                return self._unheard(sid, live, text, source, voice, reply="Go on, I'm listening.")
            out = answer(live, said)
            if "didn't catch" in str(out.get("reply", "")) and not out.get("unheard"):
                # Not understood yet. Do not say so aloud over a coach who is still talking; keep the words
                # in case the next sentence completes them.
                self._keep_fragment(live, said, waited + 1)
                out.setdefault("adapt", {})["speak"] = ""
            elif self._small_talk(text).startswith("Take your time"):
                out.setdefault("adapt", {})["speak"] = ""        # asked to wait: wait quietly
            return out

        def answer(live: Live, text: str):
            pend = live.engine.pending
            if pend:
                choices = pend["data"].get("choices") or []
                c = match_choice(text, choices)
                if c is None:
                    c = self._familiarity_answer(text, choices)
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
            # A bare "yes" goes to the assistant only if its last reply asked something; otherwise there
            # is nothing to agree to and answering would just repeat the last reply.
            last = next((m["content"] for m in reversed(live.hud.history) if m.get("role") == "assistant"), "")
            asked = "?" in str(last).split("\nPaceSense then asked:")[0][-160:]
            ai_answers = bool(live.hud.ai is not None and getattr(live.hud.ai, "online", False) and asked)
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
