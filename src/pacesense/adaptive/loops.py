"""N feedback loops, at four time scales, all running on every event.

The paper's system has one loop: the coach moves a slider, the model predicts again, and that is all. Here every
coach action is an `Event`; each loop below watches the events it cares about and may answer with an
`Adaptation` (rearrange the screen, ask a question, add an insight, update the persona). Loops are independent
and pluggable: add one by subclassing `FeedbackLoop` and listing it in `default_loops()`.

    instant        Data check · Clarify · Layout
    session        Bias guard · Explanations · Guided interview
    cross-session  Persona · Track record
    population     Shared learning (what other coaches' changes did to accuracy)

Nothing here changes the model's weights on its own: loops change what the coach sees and is asked. The coach
stays in control, which is the paper's premise; the loops make that control better informed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..domain import format_time
from .events import Adaptation, Event
from .interview import Question, questions
from .persona import Persona

CUE_CATEGORIES = {
    "heat": ("hot", "heat", "heatwave", "humid", "humidity", "warm"),
    "illness/injury": ("injur", "sick", "ill", "flu", "covid", "surgery", "pregnan", "postpartum", "cramp", "stomach"),
    "weather": ("wind", "rain", "cold", "freezing"),
    "pacing": ("too fast", "pacing", "faded", "hit the wall", "blew up"),
    "recency": ("recency", "recent", "long ago", "years ago", "outdated", "old race", "a while ago", "improved"),
    "excluded": ("ignore", "disregard", "exclude", "drop", "dnf", "fun run", "training run", "pacing a friend"),
    "best day": ("best", "pb", "peak", "perfect", "great race", "breakthrough", "level", "how she runs now",
                 "how he runs now"),
}
CUE_RE = re.compile(r"\b(hot|heat|injur\w*|sick|ill|flu|covid|wind\w*|rain\w*|cold|terrible|awful|bad race|disaster|"
                    r"off day|best|pb|peak|perfect|great race|slow\w*|tired|struggled|cramp\w*|too fast|ignore|"
                    r"representative|typical|fun run|dnf)\b", re.IGNORECASE)


def cue_category(cue: str) -> str:
    c = cue.lower()
    for cat, keys in CUE_CATEGORIES.items():
        if any(k in c for k in keys):
            return cat
    return "other"


@dataclass
class SessionContext:
    """What the loops know about the current session (rebuilt for every runner)."""
    state: dict = field(default_factory=dict)
    voice: bool = False
    layout: str = "standard"
    explain: str = "auto"
    guard_armed: bool = True
    against_kept: int = 0
    reactions: list[str] = field(default_factory=list)
    cues: list[dict] = field(default_factory=list)          # {cue, category, direction, race}
    revealed: bool = False
    asked: list[str] = field(default_factory=list)          # interview question ids already asked
    interview_on: bool = False
    spotlight: list[int] = field(default_factory=list)
    shared_counted: bool = False


@dataclass
class LoopContext:
    persona: Persona
    profile: dict                                           # persistent per coach (see engine.CoachProfile)
    population: dict                                        # persistent across coaches
    session: SessionContext


class FeedbackLoop:
    key = "loop"
    name = "Loop"
    timescale = "instant"
    answers = ""                                            # which open question from the paper it addresses

    def __init__(self) -> None:
        self.fired = 0
        self.last: str = ""

    def observe(self, ev: Event, ctx: LoopContext) -> list[Adaptation]:
        return []

    def adapt(self, kind: str, message: str, why: str = "", **data: Any) -> Adaptation:
        self.fired += 1
        self.last = message
        return Adaptation(self.key, kind, message, why, data)

    def status(self, ctx: LoopContext) -> dict:
        return {"key": self.key, "name": self.name, "timescale": self.timescale, "answers": self.answers,
                "fired": self.fired, "last": self.last}


# ---------------------------------------------------------------------------------------------- instant
class EvidenceLoop(FeedbackLoop):
    key, name, timescale = "evidence", "Data check", "instant"
    answers = "Every change is checked against the runner's own times, and a contradiction is raised, not hidden."

    def observe(self, ev, ctx):
        if ev.kind not in ("utterance", "slider", "interview_answer"):
            return []
        bad = [a for a in ev.applied if a.get("verdict") == "against_data"]
        if not bad:
            return []
        a = bad[0]
        fam = ctx.persona.value("familiarity")
        hint = (" You know this runner well, so it may well be right; it just isn't in the data."
                if fam >= 0.7 else " Say what you know that the times don't show, or undo it.")
        return [self.adapt("prompt", f"{a['race']}: {a.get('evidence', 'her own times disagree with this change.')}{hint}",
                           "a change went against the runner's own data",
                           choices=[{"label": "Keep it: I know something the data doesn't", "action": "keep",
                                     "turn": ev.data.get("turn")},
                                    {"label": "Undo", "action": "undo"},
                                    {"label": "Show similar runners", "action": "tab:similar"}])]


class ClarifyLoop(FeedbackLoop):
    key, name, timescale = "clarify", "Clarify", "instant"
    answers = "Talking is ambiguous; instead of guessing, PaceSense asks which race you meant."

    def observe(self, ev, ctx):
        if (ev.kind != "utterance" or ev.applied or ev.data.get("estimate_s") or ev.data.get("ui_only")
                or ev.data.get("notes")):                    # a race was found (it just didn't change)
            return []
        m = CUE_RE.search(ev.text or "")
        races = ctx.session.state.get("races") or []
        if not m or not races:
            return []
        cue = m.group(0)
        choices = [{"label": f"{r['event']} {r['year']}", "say": f"{r['event']} {r['year']}: {ev.text}"} for r in races]
        return [self.adapt("prompt", f'Which race was "{cue}"?', "a cue was heard but no race was named",
                           choices=choices + [{"label": "None of them", "action": "dismiss"}])]


class LayoutLoop(FeedbackLoop):
    key, name, timescale = "layout", "Layout", "instant"
    answers = "The screen follows the conversation: talking opens a voice stage, and what you mention comes forward."

    LAYOUTS: ClassVar[dict[str, str]] = {"analyst": "analysis", "intuitive": "races", "pragmatist": "summary", "explorer": "guided",
               "unknown": "standard"}

    def desired(self, ctx: LoopContext) -> str:
        lock = ctx.profile.get("layout_lock")
        if ctx.session.voice:
            return "voice"
        if lock:
            return lock
        return self.LAYOUTS.get(ctx.persona.archetype()["key"], "standard")

    def observe(self, ev, ctx):
        out = []
        if ev.kind in ("utterance", "slider", "interview_answer") and ev.applied:
            idx = [a["index"] for a in ev.applied if a.get("index") is not None]
            if idx and idx != ctx.session.spotlight:
                ctx.session.spotlight = idx
                out.append(self.adapt("ui", "", "brought the race you talked about forward", spotlight=idx))
        if ev.kind == "panel" and ev.data.get("panel"):
            usage = ctx.profile.setdefault("panel_usage", {})
            usage[ev.data["panel"]] = usage.get(ev.data["panel"], 0) + 1
        want = self.desired(ctx)
        if want != ctx.session.layout:
            why = {"voice": "you started talking: a calmer stage with the races you mention up front",
                   "analysis": "you tend to ask why: charts and similar runners first",
                   "races": "you work from what you know about the runner: races and notes first",
                   "summary": "you want the answer quickly: the summary and simulator first",
                   "guided": "you like options laid out: the guided interview leads",
                   "standard": "the default layout"}.get(want, "")
            old = ctx.session.layout
            ctx.session.layout = want
            if ev.kind != "session_start" or want != "standard":
                msg = "Voice stage" if want == "voice" else ("Voice stage closed: " if old == "voice" else "") + \
                    f"{want.capitalize()} view"
                out.append(self.adapt("ui", msg, why, layout=want))
        return out


# ---------------------------------------------------------------------------------------------- session
class BiasGuardLoop(FeedbackLoop):
    key, name, timescale = "bias", "Bias guard", "session"
    answers = ("Acceptance can be inflated when a coach steers the model toward their own prediction (paper, "
               "Sec. 6): when that happens PaceSense asks whether it is knowledge or expectation.")

    def threshold(self, ctx: LoopContext) -> float:
        t = 0.85
        if ctx.persona.value("reliance") < 0.4:
            t -= 0.1
        if ctx.persona.value("familiarity") >= 0.7:            # familiar runners: stronger opinions (RQ2)
            t -= 0.05
        return round(t, 2)

    def observe(self, ev, ctx):
        if ev.kind not in ("utterance", "slider", "interview_answer"):
            return []
        out = []
        if "against_data" in ev.verdicts:
            ctx.session.against_kept += 1
        stg = (ctx.session.state.get("steering") or {})
        idx, est = stg.get("index"), stg.get("coach_estimate_s")
        th = self.threshold(ctx)
        if idx is not None and est:
            if idx >= th and ctx.session.guard_armed:
                ctx.session.guard_armed = False
                out.append(self.adapt(
                    "prompt", f"The model is now {round(100 * idx)}% of the way to your own {format_time(est, False)}. "
                    "Is that something you know about her, or what you expected?",
                    f"steering index {idx:.2f} passed {th:.2f}",
                    choices=[{"label": "It's knowledge: keep it", "action": "ack"},
                             {"label": "Show what similar runners ran", "action": "tab:similar"},
                             {"label": "Back to the data", "say": "equal weights"}]))
            elif idx < th - 0.15:
                ctx.session.guard_armed = True
        if ctx.session.against_kept == 2 and "against_data" in ev.verdicts:
            out.append(self.adapt("insight", "Two of your changes now go against her own times. That can be right "
                                  "(you know her), but it is worth one more look before you commit.",
                                  "two changes kept against the data"))
        return out


class ExplanationLoop(FeedbackLoop):
    key, name, timescale = "explain", "Explanations", "session"
    answers = "Acceptance is not understanding: when you are unsure, explanations get fuller; when you agree, shorter."

    def observe(self, ev, ctx):
        if ev.kind != "reaction":
            return []
        v = ev.data.get("value")
        ctx.session.reactions.append(v)
        if v == "unsure":
            ctx.session.explain = "full"
            return [self.adapt("ui", "Showing full explanations", "you were unsure about a check", explain="full"),
                    self.adapt("prompt", "Want me to explain it?", "you were unsure",
                               choices=[{"label": "What is the likely range?", "say": "what does the likely range mean?"},
                                        {"label": "Why does the data disagree?", "say": "why does the data disagree?"},
                                        {"label": "No thanks", "action": "dismiss"}])]
        if ctx.session.reactions[-3:] == ["agree"] * 3 and ctx.session.explain != "brief":
            ctx.session.explain = "brief"
            return [self.adapt("ui", "Shorter explanations", "you agreed with the last three checks", explain="brief")]
        return []


class InterviewLoop(FeedbackLoop):
    key, name, timescale = "interview", "Guided interview", "session"
    answers = "An easier way in: answer a few questions by voice or tap; each answer becomes a checked change."

    def next_question(self, ctx: LoopContext) -> Question | None:
        if not ctx.session.state.get("races"):
            return None
        for q in questions(ctx.session.state):
            if q.id not in ctx.session.asked:
                return q
        return None

    def ask(self, ctx: LoopContext, why: str) -> list[Adaptation]:
        q = self.next_question(ctx)
        if q is None:
            ctx.session.interview_on = False
            return [self.adapt("insight", "That's the interview done. Everything you said is in the race weights; "
                               "press Reveal to see whether it helped.", "no questions left")]
        ctx.session.asked.append(q.id)
        return [self.adapt("prompt", q.text, q.why, question=q.to_dict(),
                           choices=[{"label": c["label"], "interview": [q.id, c["key"]]} for c in q.to_dict()["choices"]])]

    def observe(self, ev, ctx):
        if ev.kind == "session_start":
            arche = ctx.persona.archetype()["key"]
            if arche in ("explorer",) or ctx.profile.get("interview_default"):
                ctx.session.interview_on = True
                return self.ask(ctx, "you like to be guided")
            return []
        if ev.kind == "interview_answer":
            ctx.session.interview_on = True
            return self.ask(ctx, "next question")
        return []


# ---------------------------------------------------------------------------------------------- cross-session
class PersonaLoop(FeedbackLoop):
    key, name, timescale = "persona", "Persona", "cross-session"
    answers = "Different coaches need different interfaces; PaceSense learns how you work (and you can correct it)."

    def __init__(self) -> None:
        super().__init__()
        self._arch: str | None = None

    def observe(self, ev, ctx):
        ctx.persona.observe(ev)
        a = ctx.persona.archetype()
        if self._arch is None:
            self._arch = a["key"]
            return []
        if a["key"] != self._arch:
            self._arch = a["key"]
            if a["key"] == "unknown":
                return []
            cues = [c for d in ctx.persona.dims.values() for c in d.cues[:1]]
            return [self.adapt("persona", f"You work like {a['name']}: {a['description'].lower()} "
                               "The screen adapts; change it in your profile any time.",
                               "; ".join(cues[:3]), archetype=a)]
        return []


class TrackRecordLoop(FeedbackLoop):
    key, name, timescale = "record", "Track record", "cross-session"
    answers = ("The paper found coaches improved accuracy but did not learn from interacting (RQ4); this loop "
               "shows each coach which of their kinds of changes actually helped, over sessions.")

    def observe(self, ev, ctx):
        hist = ctx.profile.setdefault("outcomes", [])
        if ev.kind == "session_start" and hist:
            k = sum(1 for h in hist if h["improvement"] > 0.05)
            cats: dict[str, list[float]] = {}
            for h in hist:
                for c in set(h.get("categories", [])):
                    cats.setdefault(c, []).append(h["improvement"])
            good = sorted(((sum(v) / len(v), c) for c, v in cats.items() if len(v) >= 2), reverse=True)
            tail = ""
            if good:
                best, worst = good[0], good[-1]
                tail = f" Your {best[1]} changes helped most ({best[0]:+.2f} points on average)"
                if worst[1] != best[1] and worst[0] < 0:
                    tail += f"; your {worst[1]} changes tended to hurt ({worst[0]:+.2f})"
                tail += "."
            return [self.adapt("insight", f"Your track record: in {k} of {len(hist)} revealed sessions your changes "
                               f"made the prediction more accurate.{tail}", "your earlier sessions")]
        if ev.kind == "reveal" and not ctx.session.revealed:
            ctx.session.revealed = True
            err, eq = ev.data.get("error_pct"), ev.data.get("equal_weights_error_pct")
            if err is None or eq is None:
                return []
            imp = eq - err
            cats = sorted({c["category"] for c in ctx.session.cues})
            hist.append({"t": ev.t, "runner": ev.data.get("runner"), "improvement": round(imp, 4),
                         "categories": cats, "steering": (ctx.session.state.get("steering") or {}).get("index")})
            del hist[:-50]
            verdict = "more accurate" if imp > 0.05 else "less accurate" if imp < -0.05 else "about as accurate"
            return [self.adapt("insight", f"Your changes made the prediction {verdict} ({eq:.2f}% → {err:.2f}% off). "
                               f"That is session {len(hist)} in your track record.", "the hidden race was revealed",
                               improvement=imp)]
        return []


# ---------------------------------------------------------------------------------------------- population
class SharedLearningLoop(FeedbackLoop):
    key, name, timescale = "shared", "Shared learning", "population"
    answers = ("Expert knowledge is pooled, not lost: across all coaches, which kinds of change (heat, injury, "
               "recency…) made predictions more accurate. Shown as evidence; never applied automatically.")
    MIN_N = 3

    def observe(self, ev, ctx):
        stats = ctx.population.setdefault("cues", {})
        if ev.kind in ("utterance", "slider", "interview_answer"):
            for a in ev.applied:
                if a.get("verdict") in ("undo",) or a.get("cue") in ("slider", "undo"):
                    continue
                cat = cue_category(a.get("cue", ""))
                ctx.session.cues.append({"cue": a.get("cue"), "category": cat, "race": a.get("race"),
                                         "direction": "down" if a["new"] < a["old"] else "up"})
            out = []
            for cat in sorted({cue_category(a.get("cue", "")) for a in ev.applied if a.get("cue") not in ("slider", "undo")}):
                s = stats.get(cat)
                if s and s["n"] >= self.MIN_N:
                    out.append(self.adapt("insight", f"Across {s['n']} coach sessions, {cat} changes made the "
                                          f"prediction more accurate {round(100 * s['helped'] / s['n'])}% of the time.",
                                          "pooled outcomes from all coaches", category=cat))
            return out
        if ev.kind == "reveal" and ev.data.get("error_pct") is not None and not ctx.session.shared_counted:
            ctx.session.shared_counted = True
            imp = ev.data["equal_weights_error_pct"] - ev.data["error_pct"]
            for cat in {c["category"] for c in ctx.session.cues}:
                s = stats.setdefault(cat, {"n": 0, "helped": 0, "sum": 0.0})
                s["n"] += 1
                s["helped"] += int(imp > 0.05)
                s["sum"] = round(s["sum"] + imp, 4)
        return []


def default_loops() -> list[FeedbackLoop]:
    """Order matters only for display: persona first so the others see the updated persona."""
    return [PersonaLoop(), EvidenceLoop(), ClarifyLoop(), BiasGuardLoop(), ExplanationLoop(), InterviewLoop(),
            TrackRecordLoop(), SharedLearningLoop(), LayoutLoop()]


__all__ = ["FeedbackLoop", "LoopContext", "SessionContext", "default_loops", "cue_category"] + [
    c.__name__ for c in (EvidenceLoop, ClarifyLoop, LayoutLoop, BiasGuardLoop, ExplanationLoop, InterviewLoop,
                         PersonaLoop, TrackRecordLoop, SharedLearningLoop)]
