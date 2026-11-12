"""A second opinion on the coach's changes: the case for, the case against, and what would settle it.

When a hosted model is available (OpenRouter or Boson), it is asked for a structured critique of the session
from a fact sheet that PaceSense writes. Every sentence it returns is checked against that sheet: one that
quotes a number the sheet does not contain is dropped, so the model can reason but cannot invent data. Without a
model, or if nothing survives the check, the same structure is filled from PaceSense's own counterweights.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..domain import format_time
from .critique import LIMITATIONS, consider, effective_races

log = logging.getLogger("pacesense.hud")

SECOND_OPINION_RE = re.compile(
    r"\b(?:second\s+opinion|devil'?s\s+advocate|argue\s+against|case\s+against|play\s+(?:the\s+)?(?:other|opposite)|"
    r"over\s*-?\s*fit\w*|am\s+i\s+(?:being\s+)?(?:biased|wrong|overconfident)|is\s+(?:this|that|it)\s+(?:biased|right)|"
    r"what\s+would\s+change\s+your\s+mind|challenge\s+(?:me|my)|critique|poke\s+holes|"
    r"what\s+do\s+you\s+(?:think|make)\s+(?:of|about)\s+my\s+(?:changes|weights|view|estimate)|"
    r"do\s+you\s+agree(?:\s+with\s+me)?|should\s+i\s+trust\s+(?:my|the)\s+\w+)\b", re.IGNORECASE)

_NUM_RE = re.compile(r"\d+:\d{2}(?::\d{2})?|\d+(?:\.\d+)?%?")


def fact_sheet(state: dict, *, said: list[str], changes: list[dict], analysis: dict) -> str:
    pred, steering = state.get("prediction") or {}, state.get("steering") or {}
    races = state.get("races") or []
    lines = [f"Target {format_time(pred['finish_s'])}; with equal weights {format_time(pred['initial_s'])}; "
             f"likely range {format_time(pred['low_s'])} to {format_time(pred['high_s'])}."
             if pred.get("finish_s") and pred.get("low_s") else ""]
    for r in races:
        ev = r.get("evidence") or {}
        pull = (r.get("influence") or {}).get("pull_s")
        lines.append(f"{r['event']} {r['year']}: finished {format_time(r['finish_s'])}, {ev.get('label', 'typical')}, "
                     f"weight {float(r.get('weight', 0.5)):.1f}, similar runners went on to run "
                     f"{format_time(r['prediction_s'])}"
                     + (f", pulls the target {round(abs(float(pull)))} s {'slower' if float(pull) > 0 else 'faster'}"
                        if pull and abs(float(pull)) >= 1 else "") + ".")
    eff = effective_races([float(r.get("weight", 0.5)) for r in races])
    if races:
        lines.append(f"The target rests on about {eff:.1f} of {len(races)} races' worth of evidence.")
    if steering.get("coach_estimate_s"):
        lines.append(f"The coach's own estimate: {format_time(steering['coach_estimate_s'])}.")
    for c in changes[-6:]:
        lines.append(f"Change: {c['race']} {c['old']:.1f} to {c['new']:.1f} ({str(c.get('verdict', '')).replace('_', ' ')})"
                     + (f" after the coach said: {c['cue']}" if c.get("cue") else "") + ".")
    for t in said[-6:]:
        lines.append(f"The coach said: {t[:160]}")
    for key in ("whose_view", "accuracy", "recency"):
        part = analysis.get(key) or {}
        if part.get("available"):
            lines.append(part["text"])
    lines.append("Limits: " + " ".join(LIMITATIONS))
    return "\n".join(x for x in lines if x)


def _allowed_numbers(sheet: str) -> set[str]:
    allowed = set(_NUM_RE.findall(sheet))
    for t in list(allowed):
        if t.count(":") == 2:
            allowed.add(t.rsplit(":", 1)[0])                     # 3:35:12 may be said as 3:35
    return allowed


def grounded(sentence: str, allowed: set[str]) -> bool:
    """False if the sentence quotes a number that is not on the fact sheet (small counts are fine)."""
    for tok in _NUM_RE.findall(sentence):
        if tok in allowed:
            continue
        if re.fullmatch(r"\d", tok):                          # "one race", "2 changes"
            continue
        return False
    return True


PROMPT = """You give a second opinion on a marathon coach's adjustments to a prediction model. The coach decides;
your job is to make the trade-off clear, not to agree and not to overrule. Use only the facts below. Do not
quote any number that is not in the facts. Plain spoken English, short sentences.

Return JSON only:
{"verdict": "one sentence: how well supported the coach's current view is",
 "for": ["at most two short points that support it"],
 "against": ["at most two short points against it, the strongest first"],
 "settle": "one sentence: what would settle it (the hidden race, similar runners, a reason the coach can state)"}

Facts:
"""


def _llm(ai: Any, sheet: str, question: str) -> dict | None:
    from ..llm.openrouter import parse_json_answer
    try:
        model, _ = ai.config("assistant")
        result = ai.client.chat(model, [{"role": "system", "content": PROMPT + sheet},
                                        {"role": "user", "content": question}], temperature=0.2)
        data = parse_json_answer(result.text)
    except Exception as exc:  # noqa: BLE001 - the deterministic opinion is the fallback
        log.warning("second opinion from the model unavailable: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    allowed = _allowed_numbers(sheet)

    def keep(items: Any, n: int) -> list[str]:
        items = items if isinstance(items, list) else [items]
        return [str(x).strip()[:220] for x in items if str(x).strip() and grounded(str(x), allowed)][:n]

    out = {"verdict": (keep(data.get("verdict"), 1) or [""])[0], "for": keep(data.get("for"), 2),
           "against": keep(data.get("against"), 2), "settle": (keep(data.get("settle"), 1) or [""])[0]}
    if not out["verdict"] or not (out["for"] or out["against"]):
        return None
    out["source"] = "model"
    return out


def _rules(state: dict, last_card: dict | None, analysis: dict) -> dict:
    points = consider(last_card, state) if last_card else []
    races = state.get("races") or []
    eff = effective_races([float(r.get("weight", 0.5)) for r in races])
    against = [p["text"] for p in points if p["kind"] == "against"]
    support = [p["text"] for p in points if p["kind"] == "for"]
    if not against and races and eff < len(races) - 0.5:
        against.append(f"Your weights concentrate the evidence: the target rests on about {eff:.1f} of {len(races)} races.")
    acc = analysis.get("accuracy") or {}
    if acc.get("available"):
        (support if "more accurate" in acc["text"] and "did not" not in acc["text"] else against).append(acc["text"])
    verdict = ("The data supports part of your view; weigh the points against before you keep it." if support and against
               else "The data mostly supports your view." if support
               else "Your view rests mainly on what you know rather than on the times." if against
               else "You have not changed much yet, so the model's view and the data still agree.")
    check = next((p["text"] for p in points if p["kind"] == "check"), "")
    return {"verdict": verdict, "for": support[:2], "against": against[:2],
            "settle": check or "Revealing the hidden race, or looking at similar runners, would test this.",
            "source": "rules"}


def second_opinion(ai: Any, state: dict, *, said: list[str], changes: list[dict], last_card: dict | None,
                   analysis: dict, question: str) -> dict:
    sheet = fact_sheet(state, said=said, changes=changes, analysis=analysis)
    online = ai is not None and getattr(ai, "online", False)
    return (_llm(ai, sheet, question) if online else None) or _rules(state, last_card, analysis)


def speak(opinion: dict) -> str:
    parts = [opinion["verdict"]]
    if opinion["against"]:
        parts.append(f"Against it: {opinion['against'][0]}")
    if opinion["for"]:
        parts.append(f"For it: {opinion['for'][0]}")
    return " ".join(p.rstrip(".") + "." for p in parts if p)


__all__ = ["SECOND_OPINION_RE", "fact_sheet", "grounded", "second_opinion", "speak"]
