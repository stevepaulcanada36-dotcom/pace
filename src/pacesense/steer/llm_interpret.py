"""Optional AI reading of what the coach said, checked by code before any slider moves.

The rules in intent.py never move the wrong race, but miss paraphrases ("she ran it with a broken toe"). An AI
model reads paraphrases well but can invent things. So the model only *proposes* changes in a fixed JSON shape,
and code accepts a proposal only if:

* the race index exists,
* the `quote` it gives is really in what the coach said (no invented reasons), and
* the direction and strength are from a fixed list (no free numbers except an explicit "set").

`interpret_hybrid` uses the AI when it is available and falls back to the rules when it is not, when it fails,
or when it finds nothing the rules found. Which one answered is recorded on every turn.
Uses the "extract" task's model and temperature (Settings > AI models), so no new setting is needed.
"""
from __future__ import annotations

import json
from typing import Sequence

from ..llm.openrouter import LLMError, parse_json_answer
from ..domain import format_time
from .intent import Action, Interpretation, interpret

PROMPT = """You help a running coach adjust a marathon prediction model by voice.
The model combines the runner's past races; each race has a weight from 0 (ignore) to 1 (fully representative).
Read what the coach said and return ONLY this JSON:
{{"changes": [{{"race": <index>, "direction": "down"|"up"|"exclude"|"full"|"set", "value": <0-1, only for "set">,
   "strength": "slight"|"normal"|"strong", "quote": "<the coach's exact words that justify this>"}}],
  "command": null | "reset" | "undo" | "recency",
  "estimate": null | "<the coach's own finish-time estimate as h:mm:ss>"}}
Rules: only races the coach talks about; "quote" must be copied word for word from the coach's words;
"recency" means the coach wants recent races to count more in general; never guess.
The runner's races (index: event year, finish time):
{races}
The race being discussed before this: {context}"""

SHIFT = {"slight": 0.1, "normal": 0.2, "strong": 0.3}


def _race_list(races: Sequence[dict]) -> str:
    return "\n".join(f"{i}: {r.get('event')} {r.get('year')}"
                     + (f", {format_time(r['finish_s'])}" if r.get("finish_s") else "") for i, r in enumerate(races))


def validate_proposal(data: dict, text: str, races: Sequence[dict], context: int | None) -> Interpretation:
    """Turn the model's JSON into actions, dropping anything that fails the checks."""
    from ..assistant.intake import parse_duration
    out = Interpretation(text=text, context=context)
    said = " ".join(text.lower().split())
    cmd = data.get("command")
    if cmd in ("reset", "undo", "recency"):
        out.actions.append(Action(cmd, tuple(range(len(races))) if cmd == "recency" else (), None,
                                  f"AI: {cmd}", said, 0.7))
        if cmd in ("reset", "undo"):
            return out
    for ch in data.get("changes") or []:
        if not isinstance(ch, dict):
            continue
        i, quote = ch.get("race"), " ".join(str(ch.get("quote") or "").lower().split())
        if not isinstance(i, int) or not 0 <= i < len(races):
            out.unresolved.append(f"AI named a race that is not in the list ({i!r})")
            continue
        if not quote or quote not in said:
            out.unresolved.append(f"AI gave a reason that was not said ({quote!r}); ignored")
            continue
        d, strength = ch.get("direction"), SHIFT.get(ch.get("strength"), 0.2)
        if d == "exclude":
            a = Action("set", (i,), 0.0, quote, said, 0.7)
        elif d == "full":
            a = Action("set", (i,), 1.0, quote, said, 0.7)
        elif d == "set" and isinstance(ch.get("value"), (int, float)) and 0 <= ch["value"] <= 1:
            a = Action("set", (i,), round(float(ch["value"]), 1), quote, said, 0.7)
        elif d in ("down", "up"):
            a = Action("shift", (i,), strength if d == "up" else -strength, quote, said, 0.7)
        else:
            out.unresolved.append(f"AI proposed an unknown change ({d!r}); ignored")
            continue
        out.actions.append(a)
        out.context = i
    est = data.get("estimate")
    if est:
        s = parse_duration(str(est))
        if s is not None and 2 * 3600 <= s <= 7 * 3600:
            out.estimate_s = float(s)
    return out


def interpret_llm(ai, text: str, races: Sequence[dict], context: int | None = None) -> Interpretation:
    """Ask the 'extract' model. Raises LLMError when offline or when the answer cannot be used."""
    if not getattr(ai, "online", False):
        raise LLMError("AI is offline")
    model, temp = ai.config("extract")
    ctx = "none" if context is None else f"{context}: {races[context].get('event')} {races[context].get('year')}"
    r = ai.client.chat(model, [{"role": "system", "content": PROMPT.format(races=_race_list(races), context=ctx)},
                               {"role": "user", "content": text}], temperature=temp, json_mode=True)
    return validate_proposal(parse_json_answer(r.text), text, races, context)


def interpret_hybrid(ai, text: str, races: Sequence[dict], context: int | None = None) -> tuple[Interpretation, str]:
    """(interpretation, "ai" | "rules" | "rules (AI failed: ...)")."""
    rules = interpret(text, races, context)
    if ai is None or not getattr(ai, "online", False):
        return rules, "rules"
    try:
        llm = interpret_llm(ai, text, races, context)
    except (LLMError, json.JSONDecodeError, KeyError, TypeError) as e:
        return rules, f"rules (AI failed: {e})"
    if not llm.actions and rules.actions:
        return rules, "rules (AI found nothing)"
    if rules.estimate_s is not None and llm.estimate_s is None:
        llm.estimate_s = rules.estimate_s
    return llm, "ai"


__all__ = ["interpret_llm", "interpret_hybrid", "validate_proposal", "PROMPT"]
