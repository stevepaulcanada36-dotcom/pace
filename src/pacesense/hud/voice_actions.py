"""Validate Boson interpretations of spoken PaceSense UI commands.

The model proposes an intent; deterministic checks bind it to the transcript,
the existing runner catalog and the current session/UI state before execution.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

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
    "all", "browse", "database", "directory", "female", "find", "for", "in",
    "least", "male", "marathon", "marathons", "men", "minimum", "of", "previous",
    "race", "races", "runner", "runners", "search", "show", "the", "with", "women",
}
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
            if query:
                query_words = _tokens(query)
                if (not _contains_phrase(utterance, query)
                        or any(word in QUERY_STOP_WORDS for word in query_words)):
                    return _clarify("I couldn't verify that runner search text in what you said. Which name or ID should I search?", confidence, evidence)
            heard_words = set(_tokens(utterance))
            if gender == "F" and not heard_words.intersection({"female", "woman", "women"}):
                return _clarify("I didn't hear a female-runner filter. Should I show female runners?", confidence, evidence)
            if gender == "M" and not heard_words.intersection({"male", "man", "men"}):
                return _clarify("I didn't hear a male-runner filter. Should I show male runners?", confidence, evidence)
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
        value = value.strip().lower() if isinstance(value, str) else value
        value = {"unknown": "unfamiliar", "familiar": "known"}.get(value, value)
        if value not in ("known", "unfamiliar"):
            return _clarify("Do you know this runner personally, or are they unfamiliar to you?", confidence, evidence)
        if utterance is not None:
            if value == "unfamiliar":
                cue = re.search(
                    r"\b(unfamiliar|unknown runner|unknown to me|new to me|don't know|dont know|do not know|not familiar|not known to me|haven't coached|have not coached|mark (?:this|the) runner as unknown)\b",
                    utterance, re.IGNORECASE,
                )
            else:
                cue = re.search(r"\b(i know|know this runner|familiar|known to me|i coach|personally know|mark (?:this|the) runner as known)\b", utterance, re.IGNORECASE)
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
            verbs = r"\b(open|show|display|focus|bring up|pin|close|hide|remove|minimize|minimise)\b"
            if not re.search(verbs, utterance, re.IGNORECASE):
                return _clarify("Do you want me to open or close a PaceSense panel?", confidence, evidence)
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
