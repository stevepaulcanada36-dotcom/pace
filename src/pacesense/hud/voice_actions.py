"""Validate Boson interpretations of spoken PaceSense UI commands.

The model only proposes an intent. This module resolves every target against the
existing application state and rejects unsupported or unsafe actions.
"""
from __future__ import annotations

from typing import Any, Iterable


MIN_CONFIDENCE = 0.78
PROFILES = {"on_plan", "fast_start", "late_fade"}
STUDIO_PANELS = {"races", "prediction", "pacing", "similar", "simulator", "feedback", "steering", "chat", "log"}
INTENTS = {
    "runner_database", "select_runner", "set_familiarity",
    "focus_panel", "close_panel", "simulate", "reveal", "undo", "none",
}


def _text(value: Any, limit: int = 100) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _clarify(message: str, confidence: float = 0.0, evidence: list[str] | None = None) -> dict:
    return {"status": "clarify", "message": message, "confidence": confidence, "evidence": evidence or []}


def validate_voice_intent(
    payload: dict[str, Any],
    runners: Iterable[dict[str, Any]],
    *,
    can_undo: bool = False,
    can_reveal: bool = False,
) -> dict:
    """Validate model JSON and resolve runner targets to real database records."""
    if not isinstance(payload, dict):
        return _clarify("I couldn't confidently understand that command. Could you say it another way?")
    raw_conf = payload.get("confidence")
    if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
        return _clarify("I couldn't assess that command confidently. Could you rephrase it?")
    confidence = float(raw_conf)
    if not 0.0 <= confidence <= 1.0:
        return _clarify("I couldn't assess that command confidently. Could you rephrase it?")
    evidence_raw = payload.get("evidence", [])
    evidence = [clean for x in evidence_raw[:4] if isinstance(x, str) and (clean := _text(x, 180))] if isinstance(evidence_raw, list) else []
    if confidence < MIN_CONFIDENCE:
        return _clarify(_text(payload.get("clarification"), 240) or
                        "I'm not certain what you want me to change. Could you clarify?", confidence, evidence)
    if not evidence:
        return _clarify("I heard the request, but I need one detail clarified before changing the screen.", confidence)

    intent = payload.get("intent")
    args = payload.get("arguments")
    if not isinstance(intent, str) or intent not in INTENTS or not isinstance(args, dict):
        return _clarify("That request doesn't map to a supported PaceSense action. Try opening a panel or searching the runner database.", confidence, evidence)
    if intent == "none":
        clarification = _text(payload.get("clarification"), 240)
        return (_clarify(clarification, confidence, evidence) if clarification else
                {"status": "fallback", "confidence": confidence, "evidence": evidence})
    if intent == "runner_database":
        query = _text(args.get("query"), 64)
        gender = args.get("gender") or ""
        if not isinstance(gender, str):
            return _clarify("Should I filter by female or male runners, or show everyone?", confidence, evidence)
        gender = gender.strip()
        gender = {"female": "F", "male": "M", "all": ""}.get(gender.lower(), gender.upper())
        min_races = args.get("min_races", 0)
        if query and len(query) < 2:
            return _clarify("Please give me at least two letters or a runner ID to search for.", confidence, evidence)
        if gender not in ("", "F", "M"):
            return _clarify("Should I filter by female or male runners, or show everyone?", confidence, evidence)
        if (isinstance(min_races, bool) or not isinstance(min_races, (int, float))
                or not 0 <= min_races <= 20 or int(min_races) != min_races):
            return _clarify("How many previous marathons should the runner have at minimum?", confidence, evidence)
        min_races = int(min_races)
        return {"status": "action", "intent": intent,
                "arguments": {"query": query, "gender": gender, "min_races": min_races},
                "confidence": confidence, "evidence": evidence}
    if intent == "select_runner":
        target = _text(args.get("runner"), 100).casefold()
        if not target:
            return _clarify("Which existing runner should I open? You can say their name or runner ID.", confidence, evidence)
        rows = list(runners)
        matches = [r for r in rows if target in {
            str(r.get("runner_id", "")).casefold(), str(r.get("name", "")).strip().casefold()
        }]
        if len(matches) != 1:
            if len(matches) > 1:
                names = ", ".join(_text(r.get("name") or r.get("runner_id"), 48) for r in matches[:3])
                return _clarify(f"I found more than one matching runner: {names}. Which one do you mean?", confidence, evidence)
            return _clarify(f"I couldn't find an exact existing runner match for “{_text(args.get('runner'), 70)}”. Would you like me to open the runner database to search?", confidence, evidence)
        row = matches[0]
        return {"status": "action", "intent": intent,
                "arguments": {"runner_id": str(row["runner_id"]), "name": _text(row.get("name") or row["runner_id"], 80)},
                "confidence": confidence, "evidence": evidence}
    if intent == "set_familiarity":
        value = args.get("value")
        if value not in ("known", "unfamiliar"):
            return _clarify("Do you know this runner personally, or are they unfamiliar to you?", confidence, evidence)
        return {"status": "action", "intent": intent,
                "arguments": {"condition": "known" if value == "known" else "unknown"},
                "confidence": confidence, "evidence": evidence}
    if intent in ("focus_panel", "close_panel"):
        panel = _text(args.get("panel"), 32).lower()
        if panel not in STUDIO_PANELS:
            return _clarify("Which PaceSense panel should I open or close?", confidence, evidence)
        return {"status": "action", "intent": intent, "arguments": {"panel": panel},
                "confidence": confidence, "evidence": evidence}
    if intent == "simulate":
        temp, wind = args.get("temp_c", 12), args.get("wind_kmh", 0)
        profile = args.get("profile", "on_plan")
        if (isinstance(temp, bool) or not isinstance(temp, (int, float)) or not -10 <= temp <= 40):
            return _clarify("What race-day temperature should I use? It must be between -10 and 40 °C.", confidence, evidence)
        if (isinstance(wind, bool) or not isinstance(wind, (int, float)) or not 0 <= wind <= 60):
            return _clarify("What headwind should I use? It must be between 0 and 60 km/h.", confidence, evidence)
        if not isinstance(profile, str) or profile not in PROFILES:
            return _clarify("Should I use the planned pace, a fast start, or a late fade?", confidence, evidence)
        return {"status": "action", "intent": intent,
                "arguments": {"temp_c": float(temp), "wind_kmh": float(wind), "profile": profile},
                "confidence": confidence, "evidence": evidence}
    if intent == "reveal":
        return ({"status": "action", "intent": intent, "arguments": {},
                 "confidence": confidence, "evidence": evidence} if can_reveal else
                _clarify("There is no hidden demo result to reveal in this session.", confidence, evidence))
    if intent == "undo":
        return ({"status": "action", "intent": intent, "arguments": {},
                 "confidence": confidence, "evidence": evidence} if can_undo else
                _clarify("There isn't a race-weight change to undo yet.", confidence, evidence))
    return _clarify("I can't safely carry out that request yet.", confidence, evidence)


__all__ = ["MIN_CONFIDENCE", "STUDIO_PANELS", "validate_voice_intent"]
