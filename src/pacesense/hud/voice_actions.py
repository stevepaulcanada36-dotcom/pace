"""Validate Boson interpretations of spoken PaceSense UI commands.

The model proposes an intent; deterministic checks bind it to the transcript,
the existing runner catalog and the current session/UI state before execution.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

MIN_CONFIDENCE = 0.78
PROFILES = {"on_plan", "fast_start", "late_fade"}
STUDIO_PANELS = {"races", "prediction", "pacing", "similar", "simulator", "feedback", "steering", "chat", "log", "record"}
INTENTS = {
    "runner_database", "select_runner", "set_familiarity",
    "focus_panel", "close_panel", "simulate", "reveal", "undo", "none",
}
PANEL_TERMS = {
    "races": ("race history", "races", "race", "weights", "past races"),
    "prediction": ("prediction", "forecast", "target", "likely range", "range"),
    "pacing": ("pacing plan", "pacing", "pace plan", "splits"),
    "similar": ("similar runners", "similar", "neighbours", "neighbors"),
    "simulator": ("simulator", "simulation", "race day", "race-day"),
    "feedback": ("feedback", "evidence", "checks", "data checks"),
    "steering": ("steering", "steering meter", "whose view", "meter"),
    "chat": ("chat", "conversation", "assistant", "coach notes"),
    "log": ("change log", "log", "history of changes"),
    "record": ("track record", "record", "results", "outcomes"),
}
QUERY_STOP_WORDS = {
    "a", "all", "an", "and", "are", "at", "browse", "by", "called", "can", "could",
    "current", "database", "directory", "do", "existing", "female", "filter", "find",
    "for", "from", "gender", "have", "has", "id", "in", "is", "least", "list", "male",
    "man", "men", "marathon", "marathons", "me", "minimum", "min", "more", "named",
    "name", "number", "of", "on", "only", "open", "or", "please", "previous", "race",
    "races", "record", "records", "roster", "run", "runner", "runners", "search", "see",
    "show", "the", "to", "view", "want", "we", "who", "with", "woman", "women", "you",
}
FAMILIARITY_FILTER_RE = re.compile(
    r"\b(?:familiar|unfamiliar|known|unknown)\s+runners?\b|"
    r"\brunners?\s+(?:(?:that|who)\s+)?(?:i\s+)?(?:know|don't know|do not know)\b|"
    r"\brunners?\s+(?:that\s+are\s+)?(?:familiar|unfamiliar|known|unknown)\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _text(value: Any, limit: int = 100) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _contains_phrase(text: str, phrase: str) -> bool:
    haystack, needle = _tokens(text), _tokens(phrase)
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _evidence_is_grounded(evidence: list[str], utterance: str) -> bool:
    heard = set(_tokens(utterance))
    ignored = {"a", "an", "and", "can", "could", "i", "me", "my", "please", "the", "to", "you"}
    for item in evidence:
        words = [word for word in _tokens(item) if word not in ignored]
        if words and sum(word in heard for word in words) / len(words) >= 0.75:
            return True
    return False


def _number_is_mentioned(text: str, number: int) -> bool:
    if re.search(rf"(?<!\w){number}(?!\w)", text):
        return True
    return any(word in _tokens(text) for word, value in NUMBER_WORDS.items() if value == number)


def _runner_search_terms(text: str) -> list[str]:
    """Words left after removing supported database commands and filters."""
    return [
        word for word in _tokens(text)
        if word not in QUERY_STOP_WORDS and word not in NUMBER_WORDS and not word.isdigit()
    ]


def _clarify(message: str, confidence: float = 0.0, evidence: list[str] | None = None) -> dict:
    return {"status": "clarify", "message": message, "confidence": confidence, "evidence": evidence or []}


def _state_value(ui_state: dict[str, Any] | None, key: str, default: Any) -> Any:
    value = (ui_state or {}).get(key, default)
    return default if value is None else value


def _exact_runner(target: str, runners: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in runners
        if target in {
            str(row.get("runner_id", "")).casefold(),
            str(row.get("name", "")).strip().casefold(),
        }
    ]



_ACTION_SPLIT_RE = re.compile(r"\s*(?:[;,]|\b(?:and\s+then|then|and|also|but)\b)\s*", re.IGNORECASE)
_PANEL_ACTION_RE = re.compile(r"\b(?:open|show|display|focus|bring\s+up|pin|close|hide|remove|minimi[sz]e)\b", re.IGNORECASE)
_PANEL_OPEN_RE = re.compile(r"\b(?:open|show|display|focus|bring\s+up|pin)\b", re.IGNORECASE)
_PANEL_CLOSE_RE = re.compile(r"\b(?:close|hide|remove|minimi[sz]e)\b", re.IGNORECASE)
_SIMULATION_ACTION_RE = re.compile(r"\b(?:simulate|simulation|run\s+(?:the\s+)?race|race[- ]day|what\s+if)\b", re.IGNORECASE)
_REVEAL_ACTION_RE = re.compile(
    r"\b(?:reveal|show\s+(?:me\s+)?(?:the\s+)?(?:real|actual|true)\s+(?:best\s+)?(?:result|finish)|"
    r"how\s+(?:close|accurate)\s+(?:am\s+i|was\s+i|is\s+it))\b",
    re.IGNORECASE,
)
_UNDO_ACTION_RE = re.compile(r"\bundo\b", re.IGNORECASE)
_QUESTION_ACTION_RE = re.compile(r"\b(?:ask|explain|tell me|what|why|how)\b", re.IGNORECASE)
_DATABASE_ACTION_RE = re.compile(
    r"\b(?:database|directory|roster)\b|\bshow\s+(?:me\s+)?(?:all|female|male)?\s+runners?\b",
    re.IGNORECASE,
)
_SEARCH_ACTION_RE = re.compile(r"\b(?:search|find|filter|list|browse)\b", re.IGNORECASE)
_SELECT_ACTION_RE = re.compile(
    r"\b(?:load|select)\s+(?:the\s+)?(?:runner\s+)?(?!database\b|directory\b|roster\b)\w+|"
    r"\b(?:open|switch(?:\s+to)?)\s+(?:the\s+)?(?:runner\s+)?"
    r"(?!database\b|directory\b|roster\b|panel\b|simulator\b|prediction\b|races?\b|"
    r"pacing\b|similar\b|feedback\b|steering\b|chat\b|log\b|record\b)\w+",
    re.IGNORECASE,
)
_FAMILIARITY_ACTION_RE = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|dont)\s+know\s+(?:this\s+|the\s+)?runner|"
    r"i\s+know\s+(?:this\s+|the\s+)?runner|i\s+coach\s+(?:this\s+)?runner|"
    r"i\s+am\s+familiar\s+with\s+(?:this\s+|the\s+)?runner|"
    r"not\s+familiar\s+with\s+(?:this\s+|the\s+)?runner|"
    r"unfamiliar\s+(?:to\s+me|with\s+(?:this\s+|the\s+)?runner)|"
    r"unknown\s+to\s+me|mark\s+(?:this\s+|the\s+)?runner\s+(?:as\s+)?(?:known|unknown|unfamiliar))\b",
    re.IGNORECASE,
)
_KNOWN_FAMILIARITY_RE = re.compile(
    r"\b(?:i\s+know\s+(?:this\s+|the\s+)?runner|i\s+coach\s+(?:this\s+)?runner|"
    r"i\s+am\s+familiar\s+with\s+(?:this\s+|the\s+)?runner|"
    r"familiar\s+with\s+(?:this\s+|the\s+)?runner|mark\s+(?:this\s+|the\s+)?runner\s+(?:as\s+)?known)\b",
    re.IGNORECASE,
)
_UNFAMILIAR_FAMILIARITY_RE = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|dont)\s+know\s+(?:this\s+|the\s+)?runner|"
    r"not\s+familiar\s+with\s+(?:this\s+|the\s+)?runner|"
    r"unfamiliar\s+(?:to\s+me|with\s+(?:this\s+|the\s+)?runner)|"
    r"unknown\s+to\s+me|mark\s+(?:this\s+|the\s+)?runner\s+(?:as\s+)?(?:unknown|unfamiliar))\b",
    re.IGNORECASE,
)
_COMPOUND_PANEL_TERMS = {
    "races": ("race history", "past races", "races panel", "race panel", "weights panel"),
    "prediction": ("prediction", "forecast", "target", "likely range", "range panel"),
    "pacing": ("pacing plan", "pacing panel", "pace plan", "splits"),
    "similar": ("similar runners", "similar panel", "neighbours", "neighbors"),
    "simulator": ("simulator", "race-day simulator", "race day simulator"),
    "feedback": ("feedback panel", "feedback loops", "evidence panel", "data checks"),
    "steering": ("steering meter", "steering panel", "whose view"),
    "chat": ("chat panel", "conversation panel", "coach notes"),
    "log": ("change log", "log panel", "history of changes"),
    "record": ("track record", "record panel", "results panel", "results"),
}


def _compound_action_conflict(utterance: str, intent: str) -> bool:
    """Reject a one-action model proposal when the transcript contains another action clause."""
    expected = {
        "runner_database": "runner_database",
        "select_runner": "select_runner",
        "set_familiarity": "set_familiarity",
        "focus_panel": "panel",
        "close_panel": "panel",
        "simulate": "simulate",
        "reveal": "reveal",
        "undo": "undo",
    }.get(intent)
    if expected is None:
        return False

    clauses = [part.strip() for part in _ACTION_SPLIT_RE.split(utterance) if part.strip()]
    if len(clauses) < 2:
        return False

    classified: list[tuple[set[str], set[str], str]] = []
    for clause in clauses:
        categories: set[str] = set()
        panels: set[str] = set()
        panel_terms = PANEL_TERMS if intent in ("focus_panel", "close_panel") else _COMPOUND_PANEL_TERMS
        if _PANEL_ACTION_RE.search(clause):
            panels = {
                key for key, terms in panel_terms.items()
                if any(_contains_phrase(clause, term) for term in terms)
            }
            if panels:
                categories.add("panel")

        if _FAMILIARITY_ACTION_RE.search(clause):
            categories.add("set_familiarity")
        if _SIMULATION_ACTION_RE.search(clause):
            categories.add("simulate")
        if _REVEAL_ACTION_RE.search(clause):
            categories.add("reveal")
        if _UNDO_ACTION_RE.search(clause):
            categories.add("undo")
        if _QUESTION_ACTION_RE.search(clause):
            categories.add("assistant_question")

        database_cue = bool(_DATABASE_ACTION_RE.search(clause))
        if _SEARCH_ACTION_RE.search(clause) and (
            intent == "runner_database" or re.search(r"\b(?:runner|runners)\b", clause, re.IGNORECASE)
        ):
            database_cue = True
        if intent == "runner_database" and re.search(
            r"\b(?:show|view|browse|search|find|filter|list)\b", clause, re.IGNORECASE
        ) and re.search(r"\b(?:runner|runners)\b", clause, re.IGNORECASE):
            database_cue = True
        if database_cue:
            categories.add("runner_database")

        if _SELECT_ACTION_RE.search(clause) and not panels and not database_cue:
            categories.add("select_runner")

        # "open Jamie Lee" has no explicit "runner" token, so classify it as a runner selection
        # only when it is not already a panel/database/familiarity/simulation action.
        if (not categories and re.search(r"\b(?:open|load|switch|select)\b", clause, re.IGNORECASE)
                and not _PANEL_ACTION_RE.search(clause)):
            categories.add("select_runner")

        if categories:
            classified.append((categories, panels, clause))

    if not classified:
        return False

    # A simulator panel opened as part of a simulation is the same user action, not a second one.
    for categories, panels, clause in classified:
        unexpected = categories - {expected}
        if intent == "simulate" and categories == {"panel"} and panels == {"simulator"}:
            if _PANEL_OPEN_RE.search(clause):
                continue
        if unexpected:
            return True

    if expected == "panel":
        all_panels = set().union(*(panels for _, panels, _ in classified))
        if len(all_panels) > 1:
            return True
        if _PANEL_OPEN_RE.search(utterance) and _PANEL_CLOSE_RE.search(utterance):
            return True
    elif expected == "select_runner":
        if sum("select_runner" in categories for categories, _, _ in classified) > 1:
            return True
    elif expected == "set_familiarity":
        if _KNOWN_FAMILIARITY_RE.search(utterance) and _UNFAMILIAR_FAMILIARITY_RE.search(utterance):
            return True
    elif expected == "simulate":
        if sum("simulate" in categories for categories, _, _ in classified) > 1:
            return True
        temperatures = re.findall(
            r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(?:°\s*c|degrees?(?:\s+celsius)?|celsius|c\b)",
            utterance, re.IGNORECASE,
        )
        if len({round(float(value), 2) for value in temperatures}) > 1:
            return True
    return False


def validate_voice_intent(
    payload: dict[str, Any],
    runners: Iterable[dict[str, Any]],
    *,
    utterance: str | None = None,
    transcript_confidence: float | None = None,
    current_runner_id: str | None = None,
    current_runner_ids: Iterable[str] | None = None,
    ui_state: dict[str, Any] | None = None,
    can_undo: bool = False,
    can_reveal: bool = False,
) -> dict:
    """Validate model JSON and resolve every target against real application state."""
    if not isinstance(payload, dict):
        return _clarify("I couldn't confidently understand that command. Could you say it another way?")
    raw_confidence = payload.get("confidence")
    if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
        return _clarify("I couldn't assess that command confidently. Could you rephrase it?")
    confidence = float(raw_confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return _clarify("I couldn't assess that command confidently. Could you rephrase it?")
    if transcript_confidence is not None:
        if (isinstance(transcript_confidence, bool)
                or not isinstance(transcript_confidence, (int, float))
                or not math.isfinite(float(transcript_confidence))
                or not 0.0 <= float(transcript_confidence) <= 1.0):
            return _clarify("The speech transcript was unclear. Please repeat the command.")
        confidence = min(confidence, float(transcript_confidence))
    evidence_raw = payload.get("evidence", [])
    evidence = [
        clean for item in evidence_raw[:4]
        if isinstance(item, str) and (clean := _text(item, 180))
    ] if isinstance(evidence_raw, list) else []
    if confidence < MIN_CONFIDENCE:
        return _clarify(
            _text(payload.get("clarification"), 240)
            or "I'm not certain what you want me to change. Could you clarify?",
            confidence, evidence,
        )
    if not evidence:
        return _clarify(
            "I heard the request, but I need one detail clarified before changing the screen.",
            confidence,
        )

    intent = payload.get("intent")
    args = payload.get("arguments")
    if not isinstance(intent, str) or intent not in INTENTS or not isinstance(args, dict):
        return _clarify(
            "That request doesn't map to a supported PaceSense action. Try opening a panel or searching the runner database.",
            confidence, evidence,
        )
    if utterance is not None and intent != "none" and _compound_action_conflict(utterance, intent):
        return _clarify(
            "I heard more than one action in that request. Which action should I do first?",
            confidence, evidence,
        )
    if utterance is not None and intent != "none" and not _evidence_is_grounded(evidence, utterance):
        return _clarify(
            "I couldn't match that interpretation to the words I heard. Please repeat or clarify the command.",
            confidence, evidence,
        )
    if intent == "none":
        clarification = _text(payload.get("clarification"), 240)
        return (
            _clarify(clarification, confidence, evidence) if clarification
            else {"status": "fallback", "confidence": confidence, "evidence": evidence}
        )

    if intent == "runner_database":
        query = _text(args.get("query"), 64)
        gender = args.get("gender") or ""
        if not isinstance(gender, str):
            return _clarify("Should I filter by female or male runners, or show everyone?", confidence, evidence)
        gender = {"female": "F", "male": "M", "all": ""}.get(gender.strip().lower(), gender.strip().upper())
        min_races = args.get("min_races", 0)
        if query and len(query) < 2:
            return _clarify("Please give me at least two letters or a runner ID to search for.", confidence, evidence)
        if gender not in ("", "F", "M"):
            return _clarify("Should I filter by female or male runners, or show everyone?", confidence, evidence)
        if (isinstance(min_races, bool) or not isinstance(min_races, (int, float))
                or not math.isfinite(float(min_races)) or not 0 <= min_races <= 20
                or int(min_races) != min_races):
            return _clarify("How many previous marathons should the runner have at minimum?", confidence, evidence)
        if utterance is not None:
            if not re.search(r"\b(open|show|view|browse|search|find|filter|list|see)\b", utterance, re.IGNORECASE):
                return _clarify("Do you want me to open or search the existing runner database?", confidence, evidence)
            if FAMILIARITY_FILTER_RE.search(utterance):
                return _clarify(
                    "Familiarity is a coach/session setting, not a runner-database filter. "
                    "Should I search by runner name or ID, gender, or minimum marathon count?",
                    confidence, evidence,
                )
            if query:
                query_words = _tokens(query)
                if (not _contains_phrase(utterance, query)
                        or any(word in QUERY_STOP_WORDS for word in query_words)):
                    return _clarify("I couldn't verify that runner search text in what you said. Which name or ID should I search?", confidence, evidence)
            search_terms = _runner_search_terms(utterance)
            if search_terms and not query:
                return _clarify(
                    "I heard a runner name or ID, but it wasn't included in the search. Which existing runner should I look for?",
                    confidence, evidence,
                )
            if query and search_terms and not set(search_terms).issubset(set(_tokens(query))):
                return _clarify(
                    "I couldn't include all the runner-name or ID words you said. Please repeat the exact name or ID.",
                    confidence, evidence,
                )
            heard_words = set(_tokens(utterance))
            heard_female = bool(heard_words.intersection({"female", "woman", "women"}))
            heard_male = bool(heard_words.intersection({"male", "man", "men"}))
            if heard_female and heard_male:
                if gender:
                    return _clarify("You asked for both female and male runners. Should I show everyone?", confidence, evidence)
            elif heard_female and gender != "F":
                return _clarify("I heard a female-runner filter, but it was missing from the search. Should I show female runners?", confidence, evidence)
            elif heard_male and gender != "M":
                return _clarify("I heard a male-runner filter, but it was missing from the search. Should I show male runners?", confidence, evidence)
            elif gender == "F" and not heard_female:
                return _clarify("I didn't hear a female-runner filter. Should I show female runners?", confidence, evidence)
            elif gender == "M" and not heard_male:
                return _clarify("I didn't hear a male-runner filter. Should I show male runners?", confidence, evidence)
            minimum_match = re.search(
                r"\b(?:at least|minimum(?: of)?|no fewer than|with)\s+"
                r"(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:previous\s+)?(?:marathons?|races?)\b",
                utterance, re.IGNORECASE,
            )
            if minimum_match:
                raw_minimum = minimum_match.group(1).lower()
                requested_minimum = int(raw_minimum) if raw_minimum.isdigit() else NUMBER_WORDS[raw_minimum]
                if int(min_races) != requested_minimum:
                    return _clarify(
                        f"I heard a minimum of {requested_minimum} previous marathons, but that filter wasn't applied. Should I use {requested_minimum}?",
                        confidence, evidence,
                    )
            if min_races and (
                not _number_is_mentioned(utterance, int(min_races))
                or not heard_words.intersection({"race", "races", "marathon", "marathons"})
            ):
                return _clarify("I couldn't verify that minimum-race filter. How many previous marathons should I require?", confidence, evidence)
        return {
            "status": "action", "intent": intent,
            "arguments": {"query": query, "gender": gender, "min_races": int(min_races)},
            "confidence": confidence, "evidence": evidence,
        }

    if intent == "select_runner":
        raw_target = _text(args.get("runner"), 100)
        target = raw_target.casefold()
        if not target:
            return _clarify("Which existing runner should I open? You can say their name or runner ID.", confidence, evidence)
        if utterance is not None:
            if not re.search(r"\b(open|load|switch|select)\b", utterance, re.IGNORECASE):
                return _clarify("Do you want me to open that runner, or are you asking a question about them?", confidence, evidence)
            if not _contains_phrase(utterance, raw_target):
                return _clarify("I couldn't verify that runner name or ID in what you said. Please repeat the name or ID.", confidence, evidence)
        rows = _exact_runner(target, runners)
        if len(rows) != 1:
            if len(rows) > 1:
                names = ", ".join(_text(row.get("name") or row.get("runner_id"), 48) for row in rows[:3])
                return _clarify(f"I found more than one matching runner: {names}. Which one do you mean?", confidence, evidence)
            return _clarify(
                f"I couldn't find an exact existing runner match for “{raw_target[:70]}”. Would you like me to open the runner database to search?",
                confidence, evidence,
            )
        row = rows[0]
        return {
            "status": "action", "intent": intent,
            "arguments": {
                "runner_id": str(row["runner_id"]),
                "name": _text(row.get("name") or row["runner_id"], 80),
            },
            "confidence": confidence, "evidence": evidence,
        }

    if intent == "set_familiarity":
        value = args.get("value")
        if isinstance(value, str):
            value = value.strip().lower()
            value = {"unknown": "unfamiliar", "familiar": "known"}.get(value, value)
        if value not in ("known", "unfamiliar"):
            return _clarify("Do you know this runner personally, or are they unfamiliar to you?", confidence, evidence)
        if utterance is not None:
            if value == "unfamiliar":
                cue = re.search(
                    r"\b(unfamiliar\s+(?:to me|with\s+(?:this\s+)?runner)|unknown\s+to\s+me|"
                    r"(?:this\s+)?runner\s+(?:is\s+)?new\s+to\s+me|new\s+runner\s+to\s+me|"
                    r"(?:i\s+)?(?:do\s+not|don't|dont)\s+know\s+(?:this\s+|the\s+)?runner|"
                    r"not\s+familiar\s+with\s+(?:this\s+)?runner|not\s+known\s+to\s+me|"
                    r"haven't\s+coached\s+(?:this\s+)?runner|have\s+not\s+coached\s+(?:this\s+)?runner|"
                    r"mark\s+(?:this\s+|the\s+)?runner\s+(?:as\s+)?(?:unknown|unfamiliar))\b",
                    utterance, re.IGNORECASE,
                )
            else:
                cue = re.search(
                    r"\b(i\s+know\s+(?:this\s+|the\s+)?runner|i\s+coach\s+(?:this\s+)?runner|"
                    r"i\s+am\s+familiar\s+with\s+(?:this\s+)?runner|familiar\s+with\s+(?:this\s+)?runner|"
                    r"runner\s+(?:is\s+)?known\s+to\s+me|known\s+to\s+me|"
                    r"mark\s+(?:this\s+|the\s+)?runner\s+(?:as\s+)?known)\b",
                    utterance, re.IGNORECASE,
                )
            if not cue:
                return _clarify("Do you know this runner personally, or are they unfamiliar to you?", confidence, evidence)
        target = _text(args.get("runner"), 100).casefold()
        if target:
            rows = _exact_runner(target, runners)
            if len(rows) != 1:
                return _clarify("I couldn't identify one existing runner for that familiarity change. Which runner do you mean?", confidence, evidence)
            row = rows[0]
            active_ids = {str(runner_id) for runner_id in (current_runner_ids or [])}
            if current_runner_id:
                active_ids.add(current_runner_id)
            if active_ids and str(row.get("runner_id")) not in active_ids:
                return _clarify(
                    f"Please open {_text(row.get('name') or row.get('runner_id'), 80)} first. I won't change the familiarity of a different runner.",
                    confidence, evidence,
                )
        return {
            "status": "action", "intent": intent,
            "arguments": {"condition": "known" if value == "known" else "unknown"},
            "confidence": confidence, "evidence": evidence,
        }

    if intent in ("focus_panel", "close_panel"):
        panel = _text(args.get("panel"), 32).lower()
        if panel not in STUDIO_PANELS:
            return _clarify("Which PaceSense panel should I open or close?", confidence, evidence)
        if utterance is not None:
            if intent == "focus_panel":
                verbs = r"\b(open|show|display|focus|bring up|pin)\b"
                prompt = "Do you want me to open a PaceSense panel?"
            else:
                verbs = r"\b(close|hide|remove|minimize|minimise)\b"
                prompt = "Do you want me to close a PaceSense panel?"
            if not re.search(verbs, utterance, re.IGNORECASE):
                return _clarify(prompt, confidence, evidence)
            if not any(_contains_phrase(utterance, term) for term in PANEL_TERMS[panel]):
                return _clarify("I couldn't verify which panel you meant. Please name the panel.", confidence, evidence)
        return {
            "status": "action", "intent": intent, "arguments": {"panel": panel},
            "confidence": confidence, "evidence": evidence,
        }

    if intent == "simulate":
        temp_default = _state_value(ui_state, "temp_c", 12.0)
        wind_default = _state_value(ui_state, "wind_kmh", 0.0)
        profile_default = _state_value(ui_state, "profile", "on_plan")
        temp, wind = args.get("temp_c", temp_default), args.get("wind_kmh", wind_default)
        profile = args.get("profile", profile_default)
        if (isinstance(temp, bool) or not isinstance(temp, (int, float))
                or not math.isfinite(float(temp)) or not 0 <= temp <= 35):
            return _clarify("What race-day temperature should I use? The Studio control supports 0 to 35 °C.", confidence, evidence)
        if (isinstance(wind, bool) or not isinstance(wind, (int, float))
                or not math.isfinite(float(wind)) or not 0 <= wind <= 40):
            return _clarify("What headwind should I use? The Studio control supports 0 to 40 km/h.", confidence, evidence)
        if not isinstance(profile, str) or profile not in PROFILES:
            return _clarify("Should I use the planned pace, a fast start, or a late fade?", confidence, evidence)
        if utterance is not None:
            if not re.search(r"\b(simulate|simulation|run the race|race[- ]day|what if)\b", utterance, re.IGNORECASE):
                return _clarify("Are you asking me to run the race-day simulation?", confidence, evidence)
            temp_match = re.search(
                r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(?:°\s*c|degrees?(?:\s+celsius)?|celsius|c\b)",
                utterance, re.IGNORECASE,
            )
            if temp_match:
                if abs(float(temp) - float(temp_match.group(1))) > 0.01:
                    return _clarify("I heard a different temperature. What temperature should I simulate?", confidence, evidence)
            elif re.search(r"\b(hot|heat|warm|humid|cold|freezing)\b", utterance, re.IGNORECASE):
                return _clarify("I heard a weather description but not a specific temperature. What temperature should I use?", confidence, evidence)
            elif abs(float(temp) - float(temp_default)) > 0.01:
                return _clarify("I didn't hear a new temperature, so I won't change the current setting. Should I keep it as it is?", confidence, evidence)

            wind_match = re.search(
                r"(?<!\w)(\d+(?:\.\d+)?)\s*(?:km/?h|kph|kilometres? per hour|kilometers? per hour)",
                utterance, re.IGNORECASE,
            )
            if wind_match:
                if abs(float(wind) - float(wind_match.group(1))) > 0.01:
                    return _clarify("I heard a different wind speed. What headwind should I use?", confidence, evidence)
            elif re.search(r"\b(wind|windy|headwind)\b", utterance, re.IGNORECASE):
                return _clarify("What headwind speed should I use in km/h?", confidence, evidence)
            elif abs(float(wind) - float(wind_default)) > 0.01:
                return _clarify("I didn't hear a new wind speed, so I'll keep the current setting. Is that right?", confidence, evidence)

            profile_cue = None
            if re.search(r"\b(fast start|starts too fast|start too fast)\b", utterance, re.IGNORECASE):
                profile_cue = "fast_start"
            elif re.search(r"\b(late fade|fade late|hits the wall|hit the wall)\b", utterance, re.IGNORECASE):
                profile_cue = "late_fade"
            if profile_cue and profile != profile_cue:
                return _clarify("I heard a different pacing profile. Should I use a fast start or a late fade?", confidence, evidence)
            if not profile_cue and profile != profile_default:
                return _clarify("I didn't hear a pacing change, so I'll keep the current profile. Is that right?", confidence, evidence)
        return {
            "status": "action", "intent": intent,
            "arguments": {"temp_c": float(temp), "wind_kmh": float(wind), "profile": profile},
            "confidence": confidence, "evidence": evidence,
        }

    if intent == "reveal":
        if utterance is not None and not re.search(r"\b(reveal|show|actual|real|true)\b", utterance, re.IGNORECASE):
            return _clarify("Do you want me to reveal the hidden demo result?", confidence, evidence)
        return (
            {"status": "action", "intent": intent, "arguments": {}, "confidence": confidence, "evidence": evidence}
            if can_reveal else _clarify("There is no hidden demo result to reveal in this session.", confidence, evidence)
        )

    if intent == "undo":
        if utterance is not None and not re.search(r"\bundo\b", utterance, re.IGNORECASE):
            return _clarify("Do you want me to undo the last change?", confidence, evidence)
        return (
            {"status": "action", "intent": intent, "arguments": {}, "confidence": confidence, "evidence": evidence}
            if can_undo else _clarify("There isn't a race-weight change to undo yet.", confidence, evidence)
        )

    return _clarify("I can't safely carry out that request yet.", confidence, evidence)


__all__ = ["MIN_CONFIDENCE", "STUDIO_PANELS", "validate_voice_intent"]
