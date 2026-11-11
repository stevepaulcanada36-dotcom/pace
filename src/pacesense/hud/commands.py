"""One command bar for everything: "open the simulator", "2016 was a hot day", "why is the range so wide?".

`route(text, hud)` turns one utterance into
    reply    what PaceSense says back (shown, and optionally spoken)
    actions  what the interface should do: open/close/focus a panel, switch perspective, run the simulator, calm
             mode, open a runner, refresh after weight changes
    turn     the steering turn, when race weights changed (with its feedback card)

Order, so one sentence can do several things without misfiring:
    1. interface commands (panels, perspectives, simulator, calm mode, runner) are found and cut out of the text;
    2. what is left goes to PaceSense Live's interpreter: race weights, "reset", "undo", the coach's estimate.
       A cue carried over from an earlier sentence ("she was injured") is only applied when no interface command
       was found, so "simulate a hot day" never lowers a race;
    3. if neither applied, the question goes to the assistant (OpenRouter with a key, the offline helper without),
       which answers by calling PaceSense's own functions.
Every rule is plain Python and covered by tests/test_hud.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PANELS = {
    "races": ("races", "sliders", "weights", "past races"),
    "prediction": ("prediction", "target", "forecast", "likely range", "range"),
    "pacing": ("pacing", "pace plan", "splits", "pace band"),
    "similar": ("similar runners", "neighbours", "neighbors", "similar", "lookalikes"),
    "simulator": ("simulator", "simulation", "race day", "race-day", "course"),
    "feedback": ("feedback", "evidence", "checks", "data checks"),
    "steering": ("steering", "steering meter", "whose view", "meter"),
    "hologram": ("hologram", "runner model", "body", "avatar", "3d"),
    "chat": ("chat", "conversation", "assistant"),
    "log": ("change log", "log", "history of changes"),
    "record": ("track record", "results", "outcomes"),
}
PERSPECTIVES = {"coach": ("coach",), "runner": ("runner", "athlete"), "researcher": ("researcher", "research", "study")}
PROFILE_WORDS = {"fast_start": ("fast start", "goes out fast", "starts too fast", "start fast", "too fast"),
                 "late_fade": ("fade", "fades", "late fade", "hits the wall", "hit the wall", "blows up")}


@dataclass
class Routed:
    reply: str = ""
    actions: list[dict] = field(default_factory=list)
    turn: dict | None = None
    used: list[str] = field(default_factory=list)       # "ui", "steer", "assistant"
    steps: list[dict] = field(default_factory=list)


def _alts(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


PANEL_RE = re.compile(r"\b(show|open|bring up|display|hide|close|remove|minimi[sz]e|focus on|zoom (?:in )?on|pin)\s+"
                      r"(?:me\s+)?(?:the\s+|my\s+|her\s+|his\s+)?(" + _alts(w for ws in PANELS.values() for w in ws) +
                      r")(?:\s+panel)?\b", re.I)
PERSP_RE = re.compile(r"\b(?:switch|change|go)\s+(?:to\s+)?(?:the\s+)?(" + _alts(w for ws in PERSPECTIVES.values() for w in ws)
                      + r")\s+(?:view|mode|perspective)\b|\b(" + _alts(w for ws in PERSPECTIVES.values() for w in ws)
                      + r")\s+(?:view|mode|perspective)\b", re.I)
SIM_RE = re.compile(r"\b(simulate|simulation|run the race|play the race|replay the race|race[- ]day|what if)\b", re.I)
TEMP_RE = re.compile(r"(-?\d{1,2})\s*(?:°\s*c|degrees(?:\s+celsius)?|celsius|c\b)", re.I)
WIND_RE = re.compile(r"(\d{1,2})\s*(?:km/?h|kph|kilometres per hour|kilometers per hour)\s*(?:head)?wind|"
                     r"(?:head)?wind\s*(?:of\s*)?(\d{1,2})\s*(?:km/?h|kph)?", re.I)
CALM_RE = re.compile(r"\b(calm mode|focus mode|quiet mode|stop (?:rotating|spinning|moving)|pause the hologram|"
                     r"less motion|reduce motion)\b", re.I)
SPIN_RE = re.compile(r"\b(start (?:rotating|spinning)|rotate (?:her|him|the (?:body|hologram|runner))|resume)\b", re.I)
RUNNER_RE = re.compile(r"\b(?:open|load|switch to|show)\s+(?:runner\s+)?([RCK]-?[0-9A-Za-z]{4,12})\b")
DEMO_RE = re.compile(r"\b(demo runner|new demo|another runner|next runner)\b", re.I)
LAYOUT_RE = re.compile(r"\b(reset (?:the )?layout|tidy (?:up )?(?:the )?(?:screen|panels)|arrange (?:the )?panels)\b", re.I)
REVEAL_RE = re.compile(r"\b(reveal|show me the (?:real|actual|true) (?:best|result)|how (?:close|accurate) (?:am i|was i|is it))\b",
                       re.I)
# These phrases refer to existing database records, not adding new runners.
DATABASE_RE = re.compile(r"\b(?:open|show|view|browse|search)\s+(?:the\s+)?(?:runner\s+)?(?:database|directory|roster)\b|\bshow\s+(?:me\s+)?all\s+runners\b", re.I)
UNKNOWN_RE = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|dont)\s+know\s+(?:this\s+)?(?:the\s+)?runner|"
    r"i\s+have\s+not\s+coached\s+(?:this\s+)?runner|i\s+haven't\s+coached\s+(?:this\s+)?runner|"
    r"i\s+have\s+never\s+coached\s+(?:this\s+)?runner|not\s+familiar\s+with\s+(?:this\s+)?runner|"
    r"unfamiliar\s+(?:to\s+me|with\s+(?:this\s+)?runner)|unknown\s+to\s+me|"
    r"(?:this\s+)?runner\s+(?:is\s+)?new\s+to\s+me|new\s+runner\s+to\s+me|"
    r"mark\s+(?:this\s+)?runner\s+(?:as\s+)?(?:unknown|unfamiliar)(?:\s+to\s+me)?)\b", re.I)
KNOWN_RE = re.compile(
    r"\b(?:i\s+know\s+(?:this\s+)?runner|i\s+coach\s+(?:this\s+)?runner|"
    r"i\s+am\s+familiar\s+with\s+(?:this\s+)?runner|familiar\s+with\s+(?:this\s+)?runner|"
    r"(?:this\s+)?runner\s+is\s+known\s+to\s+me|known\s+to\s+me|mark\s+(?:this\s+)?runner\s+as\s+known)\b", re.I)


def _panel_key(word: str) -> str:
    w = word.lower()
    return next(k for k, ws in PANELS.items() if w in ws)


def ui_commands(text: str) -> tuple[list[dict], list[str], str]:
    """(actions, short descriptions, text with the commands removed)."""
    actions, said, rest = [], [], text
    # The runner is selected from the existing database; this command never creates a record.
    m = DATABASE_RE.search(rest)
    if m:
        actions.append({"type": "runner_database"})
        said.append("opened the existing runner database")
        rest = rest.replace(m.group(0), " ")
    m = UNKNOWN_RE.search(rest)
    if m:
        actions.append({"type": "condition", "value": "unknown"})
        said.append("marked this as an unfamiliar runner already in the database")
        rest = rest.replace(m.group(0), " ")
    else:
        m = KNOWN_RE.search(rest)
        if m:
            actions.append({"type": "condition", "value": "known"})
            said.append("marked this as a runner you know")
            rest = rest.replace(m.group(0), " ")
    for m in PANEL_RE.finditer(rest):
        verb, panel = m.group(1).lower(), _panel_key(m.group(2))
        kind = "close" if verb in ("hide", "close", "remove", "minimise", "minimize") else "focus"
        actions.append({"type": kind, "panel": panel})
        said.append(f"{'hid' if kind == 'close' else 'opened'} {panel}")
        rest = rest.replace(m.group(0), " ")
    m = PERSP_RE.search(text)
    if m:
        word = (m.group(1) or m.group(2)).lower()
        p = next(k for k, ws in PERSPECTIVES.items() if word in ws)
        actions.append({"type": "perspective", "value": p})
        said.append(f"switched to the {p} view")
        rest = rest.replace(m.group(0), " ")
    if SIM_RE.search(text):
        t = TEMP_RE.search(text)
        w = WIND_RE.search(text)
        temp = float(t.group(1)) if t else (27.0 if re.search(r"\b(hot|heat|warm|humid)\b", text, re.I) else
                                            5.0 if re.search(r"\b(cold|freezing)\b", text, re.I) else 12.0)
        wind = float(w.group(1) or w.group(2)) if w else (25.0 if re.search(r"\bwindy|strong wind", text, re.I) else 0.0)
        prof = next((k for k, ws in PROFILE_WORDS.items() if re.search(_alts(ws), text, re.I)), "on_plan")
        actions.append({"type": "simulate", "temp_c": temp, "wind_kmh": wind, "profile": prof})
        said.append(f"simulated the race at {temp:g} °C" + (f", {wind:g} km/h headwind" if wind else "")
                    + ("" if prof == "on_plan" else f", {prof.replace('_', ' ')}"))
        rest = ""                          # the whole sentence was about the simulation
    if CALM_RE.search(text):
        actions.append({"type": "calm", "value": True}); said.append("calm mode on"); rest = CALM_RE.sub(" ", rest)
    elif SPIN_RE.search(text):
        actions.append({"type": "calm", "value": False}); said.append("motion back on"); rest = SPIN_RE.sub(" ", rest)
    m = RUNNER_RE.search(text)
    if m:
        actions.append({"type": "runner", "id": m.group(1)}); said.append(f"opened runner {m.group(1)}")
        rest = rest.replace(m.group(0), " ")
    elif DEMO_RE.search(text):
        actions.append({"type": "runner", "id": "demo"}); said.append("opened a demo runner"); rest = DEMO_RE.sub(" ", rest)
    if LAYOUT_RE.search(text):
        actions.append({"type": "layout", "value": "reset"}); said.append("tidied the panels")
        rest = LAYOUT_RE.sub(" ", rest)
    if REVEAL_RE.search(text):
        actions.append({"type": "reveal"}); said.append("revealed the hidden best race"); rest = REVEAL_RE.sub(" ", rest)
    rest = re.sub(r"\b(and|then|also|please|can you|could you)\b", " ", rest, flags=re.I)
    return actions, said, " ".join(rest.split())


def _steering_candidate(text: str, hud: Any) -> bool:
    """Only spend an AI steering call when the utterance could describe a race or its relevance."""
    steer = getattr(hud, "steer", None)
    races = getattr(steer, "races", ()) or ()
    lowered = text.casefold()
    names_or_years = any(
        (str(r.get("event", "")).casefold() in lowered if r.get("event") else False)
        or re.search(rf"(?<!\d){int(r['year'])}(?!\d)", text)
        for r in races if r.get("year") is not None
    )
    has_context = getattr(steer, "context", None) is not None
    steering_language = re.search(
        r"\b(weight|count|matter|trust|rely|represent\w*|ignore|appeal\w*|relevan\w*|important|significant|"
        r"less\w*|more\w*|reduce|lower|raise|increase|keep|use|consider|level|worth|valid|"
        r"fast|faster|slow|slower|hot|cold|injur\w*|ill|sick|good|bad|"
        r"best|worst|unusual|typical|outlier|recent|precise|imprecise|accurate|inaccurate|reliable|unreliable|"
        r"correct|wrong|right|normal|fluke|misleading|reflect\w*|odd|strange|weird|anomal\w*|exception\w*)\b",
        text, re.I,
    )
    return bool(steering_language and (names_or_years or has_context))


QUESTION_RE = re.compile(
    r"^\s*(?:so\s+|and\s+|but\s+|ok(?:ay)?\s+|well\s+)*"
    r"(?:what|which|why|how|when|who|where|whose|is\s+(?:it|that|this|there)|are\s+(?:you|we|they|those)|"
    r"was\s+(?:it|that|this|there)|were\s+|does\s+|did\s+|do\s+you|tell\s+me|explain|"
    r"i\s+(?:just\s+)?(?:want|need|would\s+like|wanna)\s+to\s+(?:know|understand|check|see)|"
    r"i'?m\s+asking)\b"
    r"|\?\s*$"
    r"|\b(?:what\s+happened|what\s+(?:is|was|does|did|will|would|about)|which\s+\w+\s+(?:are\s+)?you(?:'re|\s+are)?\s+"
    r"(?:talking|referring)|is\s+it\s+not|isn'?t\s+it|"
    r"i\s+(?:just\s+)?(?:want|need)\s+to\s+know|(?:could|can|would)\s+you\s+(?:tell|explain|show)\s+me\s+(?:what|why|how|which))\b",
    re.IGNORECASE,
)
# A question that also spells out an instruction ("why don't you count 2019 less") is still an instruction.
DIRECTIVE_RE = re.compile(
    r"\b(?:count|weigh|weight|trust|rely\s+on|use|consider)\s+(?:\w+\s+){0,4}(?:less|more|fully|half|twice)\b|"
    r"\b(?:ignore|drop|exclude|discard|remove|reduce|lower|decrease|increase|raise|boost|set|reset)\s+"
    r"(?:the\s+|that\s+|this\s+|its\s+|her\s+|his\s+)?(?:weight|\d{4}|race)\b", re.IGNORECASE)


# Speech recognition often writes "race" as "raise", which the steering rules read as "increase the weight".
_RAISE_AS_RACE = (
    re.compile(r"\b(the|that|this|which|what|they|a|one|last|first|latest|recent|same|every|each|any|\d{4}|"
               r"amsterdam|rotterdam|eindhoven|marathon)\s+raise(s?)\b", re.IGNORECASE),
    re.compile(r"\braise(s?)\s+(which|that|in|on|from|of|was|is|were|are|happened|i'm|i am|you're|you are|"
               r"he|she|we|at|history|day|time|result|results)\b", re.IGNORECASE),
)


def repair_mishearings(text: str) -> str:
    """Fix the "race"/"raise" confusion where the grammar shows a race was meant ("which raise", "raise in 2019")."""
    text = _RAISE_AS_RACE[0].sub(lambda m: f"{m.group(1)} race{m.group(2)}", text)
    return _RAISE_AS_RACE[1].sub(lambda m: f"race{m.group(1)} {m.group(2)}", text)


def is_question(text: str) -> bool:
    """True when the coach is asking something, not telling PaceSense about a race."""
    return bool(QUESTION_RE.search(text)) and not DIRECTIVE_RE.search(text)


def route(text: str, hud: Any, source: str = "typed") -> Routed:
    """`hud` is a HudSession (web/hud_api.py): .steer (SteeringSession), .ask(question) -> (answer, steps)."""
    out = Routed()
    if source == "voice":
        text = repair_mishearings(text)
    actions, said, rest = ui_commands(text)
    if actions:
        out.actions += actions
        out.used.append("ui")
    steered = False
    # A question is answered, never applied: asking "what happened in 2019?" must not move a weight.
    if rest and len(rest) > 2 and not is_question(rest):
        from ..steer.intent import interpret
        it = interpret(rest, hud.steer.races, hud.steer.context, hud.steer.lexicon)
        strong = [a for a in it.actions if a.kind in ("reset", "undo", "recency") or a.confidence >= 0.8]
        should_interpret = bool(
            strong or it.estimate_s is not None or it.unknown_races or it.unresolved
            or (it.actions and not actions) or _steering_candidate(rest, hud)
        )
        if should_interpret:
            turn = hud.steer.hear(rest, source)
            meaningful_notes = [n for n in turn.notes if not n.startswith("Heard you, but nothing")]
            if (turn.applied or turn.estimate_s is not None or turn.source == "undo"
                    or meaningful_notes):
                out.turn = {**turn.to_dict(), "said": text}
                out.actions.append({"type": "refresh"})
                out.used.append("steer")
                steered = True
    if steered:
        out.reply = hud.feedback_text(out.turn)
    elif rest and len(rest) > 2 and not (actions and len(rest.split()) < 3):
        answer, steps = hud.ask(rest)
        out.reply, out.steps = answer, steps
        out.used.append("assistant")
    if said:
        lead = "Done: " + ", ".join(said) + "."
        out.reply = (lead + (" " + out.reply if out.reply else "")).strip()
    if not out.reply:
        out.reply = ("I heard you, but nothing in that changes a race or the screen. Try “2016 was a hot day”, "
                     "“simulate a hot day”, “switch to runner view” or ask a question.")
    return out


__all__ = ["route", "ui_commands", "is_question", "Routed", "PANELS", "PERSPECTIVES"]
