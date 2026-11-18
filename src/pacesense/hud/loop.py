"""The reasoning loop: PaceSense thinks about the whole session, not just the last sentence.

Each change the coach makes is a *claim* ("Eindhoven 2019 was a hot day, count it less"). The loop:

1. records the claim with what it did (old → new weight, the target before and after, whether the runner's own
   times agree);
2. after every change, estimate or reveal, asks the hosted model to *reflect* on the session as a whole: the
   claims so far, which ones the data backs, how the target and the coach's own estimate have moved, and what the
   coach said in their own words. The model answers with a short chain of reasoning, a spoken reply in the
   coach's register, and one question or test that would move the session forward;
3. closes each claim when evidence arrives: when the hidden race is revealed, every claim is tested
   counterfactually (the same session with that one race put back to an equal weight), so the coach hears which
   of their calls helped and which did not.

Language models can reason here but cannot invent data: every sentence they return is checked against a fact
sheet PaceSense writes, and a sentence that quotes a number not on the sheet is dropped. Without a model, or if
nothing survives the check, the deterministic reply that PaceSense already built is used.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..domain import format_time

log = logging.getLogger("pacesense.hud")

NEUTRAL = 0.5
STATUS_WORDS = {"open": "not tested yet", "backed": "the runner's times agree", "knowledge": "rests on your knowledge",
                "questioned": "the runner's times disagree", "helped": "helped", "hurt": "hurt",
                "neutral": "made no difference", "undone": "undone"}


# ---- the claims ledger --------------------------------------------------------------------------------------
def record(claims: list[dict], card: dict | None, turn_n: int | None = None) -> list[dict]:
    """Add one claim per race the card changed; a later change to the same race supersedes the earlier one."""
    if not card:
        return claims
    for item in card.get("items") or []:
        for c in claims:
            if c["race"] == item["race"] and c["status"] not in ("undone", "superseded"):
                c["status"] = "superseded"
        verdict = str(item.get("verdict") or "")
        claims.append({
            "id": len(claims) + 1, "turn": turn_n if turn_n is not None else card.get("n"),
            "race": item["race"], "index": item.get("index"), "cue": _reason(item.get("cue")),
            "said": str(card.get("said") or "")[:200], "old": float(item["old"]), "new": float(item["new"]),
            "direction": "less" if float(item["new"]) < float(item["old"]) else "more",
            "verdict": verdict,
            "status": {"supported": "backed", "against_data": "questioned"}.get(verdict, "knowledge"),
            "target_before_s": card.get("target_before_s"),
        })
    del claims[:-40]
    return claims


def _reason(cue: object) -> str:
    """The coach's words behind a change, or "" when they were only a number ("to 0.1")."""
    text = str(cue or "").strip()[:160]
    return "" if re.fullmatch(r"(?:(?:set|to)\s*)*[\d.]+", text, re.IGNORECASE) else text


def active(claims: list[dict]) -> list[dict]:
    return [c for c in claims if c["status"] != "superseded"]


def _target(races: list[dict], weights: list[float]) -> float | None:
    """The target as PaceSense computes it: the weight-averaged finish that similar runners went on to run."""
    pairs = [(w, float(r["prediction_s"])) for r, w in zip(races, weights) if r.get("prediction_s") is not None]
    total = sum(w for w, _ in pairs)
    return sum(w * p for w, p in pairs) / total if total > 0 else None


def test(claims: list[dict], state: dict, truth_s: float | None) -> list[dict]:
    """Close each open claim against the hidden race: would the prediction have been closer without it?

    The counterfactual is the session as it is now with that one race put back to an equal weight, so each claim
    is judged by its own contribution, not by the order the changes were made in."""
    races = state.get("races") or []
    if truth_s is None or not races:
        return claims
    weights = [float(r.get("weight", NEUTRAL)) for r in races]
    now = _target(races, weights)
    if now is None:
        return claims
    err_now = 100 * abs(now - truth_s) / truth_s
    by_name = {f"{r['event']} {r['year']}": i for i, r in enumerate(races)}
    for c in active(claims):
        i = by_name.get(c["race"])
        if i is None:
            continue
        if abs(weights[i] - NEUTRAL) < 1e-9:
            c.update(status="undone", effect_pct=0.0)
            continue
        without = list(weights)
        without[i] = NEUTRAL
        alt = _target(races, without)
        if alt is None:
            continue
        err_without = 100 * abs(alt - truth_s) / truth_s
        effect = round(err_without - err_now, 2)            # positive: the claim made the prediction closer
        c["effect_pct"] = effect
        c["status"] = "helped" if effect > 0.01 else "hurt" if effect < -0.01 else "neutral"
    return claims


def ledger_lines(claims: list[dict]) -> list[str]:
    """One line per live claim, for the model and for the assistant's context."""
    out = []
    for c in active(claims):
        effect = c.get("effect_pct")
        tail = (f"; on the hidden race it {STATUS_WORDS[c['status']]} ({abs(effect):.2f} points of error)"
                if c["status"] in ("helped", "hurt") and effect is not None else f"; {STATUS_WORDS.get(c['status'], '')}")
        out.append(f"Claim {c['id']} (turn {c['turn']}): {c['race']} counted {c['direction']} "
                   f"({c['old']:.1f} to {c['new']:.1f})"
                   + (f" because the coach said “{c['cue']}”" if c["cue"] else "") + tail + ".")
    return out


def view(claims: list[dict]) -> list[dict]:
    """The claims the interface draws."""
    return [{"id": c["id"], "turn": c["turn"], "race": c["race"], "direction": c["direction"],
             "old": c["old"], "new": c["new"], "cue": c["cue"], "status": c["status"],
             "status_text": STATUS_WORDS.get(c["status"], c["status"]), "effect_pct": c.get("effect_pct")}
            for c in active(claims)]


def closure(claims: list[dict]) -> str:
    """A plain summary of what the reveal said about each claim (the deterministic fallback)."""
    tested = [c for c in active(claims) if c["status"] in ("helped", "hurt", "neutral")]
    if not tested:
        return ""
    helped = [c for c in tested if c["status"] == "helped"]
    hurt = [c for c in tested if c["status"] == "hurt"]
    parts = []
    if helped:
        best = max(helped, key=lambda c: c.get("effect_pct") or 0)
        parts.append(f"Your call on {best['race']} helped most" + (f" (“{best['cue']}”)" if best["cue"] else "") + ".")
    if hurt:
        worst = min(hurt, key=lambda c: c.get("effect_pct") or 0)
        parts.append(f"{worst['race']} pulled the prediction away from the real result; worth asking what you knew about it.")
    if not helped and not hurt:
        parts.append("None of your changes moved it much either way.")
    return " ".join(parts)


# ---- the model's reflection ---------------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d+:\d{2}(?::\d{2})?|\d+(?:\.\d+)?%?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _allowed(sheet: str) -> set[str]:
    allowed = set(_NUM_RE.findall(sheet))
    for t in list(allowed):
        if t.count(":") == 2:
            allowed.add(t.rsplit(":", 1)[0])
        if t.endswith("%"):
            allowed.add(t[:-1])
        elif "." in t:
            allowed.add(t + "%")
    return allowed


def grounded(sentence: str, allowed: set[str]) -> bool:
    """False if the sentence quotes a number that is not on the fact sheet (single digits are fine)."""
    return all(tok in allowed or re.fullmatch(r"\d", tok) for tok in _NUM_RE.findall(sentence))


def _keep(text: Any, allowed: set[str], limit: int = 320) -> str:
    sentences = [s.strip() for s in _SENT_RE.split(str(text or "").strip()) if s.strip()]
    return " ".join(s for s in sentences if grounded(s, allowed))[:limit].strip()


PROMPT = """You are PaceSense's reasoning partner for a marathon coach who is adjusting a case-based prediction.
The coach leads; you think alongside them. Your job each turn is to reason about the WHOLE session, not just the
last sentence, and keep the conversation moving.

How to reason (keep it to two or three short steps):
- What did the coach just do or say, and what claim does it make about the runner?
- What do the facts say about that claim: the runner's own times, what similar runners went on to run, how far it
  moved the target compared with the likely range, and how it fits the coach's earlier claims?
- What is still uncertain, and what single thing would test it?

How to reply (spoken aloud, so plain words, short sentences):
- React to what the coach did in their own words, then give one piece of reasoning that links cause to effect.
- Be honest both ways: agree when the data backs it, push back when the runner's times go the other way, and say
  plainly when it rests only on the coach's knowledge. Do not flatter and do not lecture.
- Connect to earlier turns when it helps ("earlier you counted 2017 less for heat too").
- End with exactly one question or one concrete test (something the coach can say or check next). Do not repeat a
  question you already asked. If nothing is uncertain, suggest the most useful next check instead.
- Never quote a number that is not in the facts. Never claim to have changed anything.

Return JSON only:
{"think": ["two or three short reasoning steps, each under 25 words"],
 "say": "the spoken reply, LENGTH",
 "ask": "one question or test to end on, under 20 words",
 "stance": "agree" | "partly" | "challenge" | "curious"}

Facts:
"""


def fact_sheet(state: dict, claims: list[dict], *, event: str, said: str, draft: str,
               coach_first_s: float | None = None, extra: str = "") -> str:
    pred, steering = state.get("prediction") or {}, state.get("steering") or {}
    races = state.get("races") or []
    holdout = state.get("holdout") or {}
    lines = []
    if pred.get("finish_s"):
        lines.append(f"Target now {format_time(pred['finish_s'])}; with equal weights it was "
                     f"{format_time(pred.get('initial_s') or pred['finish_s'])}"
                     + (f"; likely range {format_time(pred['low_s'])} to {format_time(pred['high_s'])}"
                        if pred.get("low_s") and pred.get("high_s") else "") + ".")
    for r in races:
        ev = r.get("evidence") or {}
        pull = (r.get("influence") or {}).get("pull_s")
        lines.append(f"{r['event']} {r['year']}: finished {format_time(r['finish_s'])} ({ev.get('label', 'typical')}), "
                     f"weight {float(r.get('weight', NEUTRAL)):.1f}; similar runners went on to run "
                     f"{format_time(r['prediction_s'])}"
                     + (f"; it pulls the target {round(abs(float(pull)))} s {'slower' if float(pull) > 0 else 'faster'}"
                        if pull and abs(float(pull)) >= 1 else "") + ".")
    if steering.get("coach_estimate_s"):
        lines.append(f"The coach's own estimate now: {format_time(steering['coach_estimate_s'])}"
                     + (f" (first said {format_time(coach_first_s)})" if coach_first_s
                        and abs(coach_first_s - float(steering['coach_estimate_s'])) > 1 else "") + ".")
    if holdout.get("label"):
        if holdout.get("revealed") and holdout.get("truth_s"):
            lines.append(f"Hidden race {holdout['label']} revealed: real time {format_time(holdout['truth_s'])}; the "
                         f"prediction is {float(holdout.get('error_pct') or 0):.2f}% off, against "
                         f"{float(holdout.get('equal_weights_error_pct') or 0):.2f}% with equal weights.")
        else:
            lines.append(f"The runner's best race ({holdout['label']}) is hidden; revealing it tests the changes.")
    lines.extend(ledger_lines(claims))
    lines.append(f"What just happened: {event}.")
    if said:
        lines.append(f"The coach's words: “{said[:240]}”")
    if draft:
        lines.append(f"PaceSense's own summary of the result: {draft[:400]}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


LENGTHS = {"brief": "one or two sentences, under 30 words", "full": "three or four sentences, under 70 words"}


def reflect(ai: Any, sheet: str, *, style: str = "normal", persona: str = "",
            recent: list[str] | None = None) -> dict | None:
    """The model's reflection, with every number checked, or None (then the caller keeps its own reply)."""
    if ai is None or not getattr(ai, "online", False):
        return None
    from ..llm.openrouter import parse_json_answer
    prompt = PROMPT.replace("LENGTH", LENGTHS.get(style, "two or three sentences, under 50 words"))
    if persona:
        prompt = prompt.replace("Facts:\n", f"About this coach: {persona}\n\nFacts:\n")
    avoid = [r for r in (recent or []) if r][-4:]
    messages = [{"role": "system", "content": prompt + sheet}]
    if avoid:
        messages.append({"role": "system", "content": "Your last replies (do not repeat their wording or "
                                                      "questions):\n- " + "\n- ".join(a[:200] for a in avoid)})
    messages.append({"role": "user", "content": "Reflect on this turn."})
    try:
        model, _ = ai.config("assistant")
        result = ai.client.chat(model, messages, temperature=0.55)
        data = parse_json_answer(result.text)
    except Exception as exc:  # noqa: BLE001 - PaceSense's own reply is the fallback
        log.warning("reflection unavailable; using PaceSense's own reply: %s", exc)
        return None
    allowed = _allowed(sheet)
    think_raw = data.get("think")
    think = [_keep(t, allowed, 200) for t in (think_raw if isinstance(think_raw, list) else [think_raw])]
    say, ask = _keep(data.get("say"), allowed), _keep(data.get("ask"), allowed, 160)
    stance = str(data.get("stance") or "")
    if not say:
        return None
    if ask and ask.rstrip("?.! ").casefold() in say.casefold():
        ask = ""                                          # the reply already ends on it
    return {"think": [t for t in think if t][:3], "say": say, "ask": ask,
            "stance": stance if stance in ("agree", "partly", "challenge", "curious") else "curious"}


def spoken(reflection: dict) -> str:
    """The reply to say: the reflection and its closing question, as one turn."""
    say, ask = reflection.get("say", "").strip(), reflection.get("ask", "").strip()
    return f"{say} {ask}".strip() if ask else say


__all__ = ["closure", "fact_sheet", "grounded", "ledger_lines", "record", "reflect", "spoken", "test", "view"]
