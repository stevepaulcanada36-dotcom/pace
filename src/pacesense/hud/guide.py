"""Reasoning and guidance for the coach.

Two things the conversation was missing, both computed from PaceSense's own numbers (never by a language
model, so they cannot be invented):

* ``reasoning(card, races)``: why a change happened and what it did, as a short chain the coach can check:
  what was heard, what the runner's own data says about that race, and how the change moved the target.
* ``next_steps(...)``: what the coach could usefully do next, with a reason, chosen from what they have not
  tried yet in this session, so the session moves forward instead of repeating itself.
"""
from __future__ import annotations

from ..domain import format_time


def _gap_s(race: dict) -> float:
    """Seconds between this race and the runner's other races (positive = slower), or 0."""
    ev = race.get("evidence") or {}
    if not ev.get("direction") or race.get("finish_s") is None or ev.get("others_median_s") is None:
        return 0.0
    return float(race["finish_s"]) - float(ev["others_median_s"])


def _gap(race: dict) -> str:
    """'26 min slower than the runner's other races', or '' for a typical race."""
    gap = _gap_s(race)
    if abs(gap) < 60:
        return ""
    return f"{round(abs(gap) / 60)} min {'slower' if gap > 0 else 'faster'} than the runner's other races"


def _effect(race: dict, less: bool, target_before_s: float | None) -> str:
    """Why the target moves: it is the weighted average of what similar runners went on to run after each of
    this runner's races, so counting a race less moves the target away from that race's own prediction."""
    pred = race.get("prediction_s")
    if pred is None or target_before_s is None or abs(float(pred) - float(target_before_s)) < 1:
        return ""
    faster_than_target = float(pred) < float(target_before_s)
    moves = ("slower" if less else "faster") if faster_than_target else ("faster" if less else "slower")
    return (f"runners who ran a race like it went on to run {format_time(float(pred))}, "
            f"{'faster' if faster_than_target else 'slower'} than the {format_time(float(target_before_s))} target, "
            f"so counting it {'less' if less else 'more'} moves the target {moves}")


def reasoning(card: dict, races: list[dict]) -> list[str]:
    """A short reasoning chain per change: what was heard, what the data says, why the target moved."""
    steps: list[str] = []
    by_name = {f"{r['event']} {r['year']}": r for r in races}
    before = card.get("target_before_s")
    for item in card.get("items") or []:
        race = by_name.get(item["race"], {})
        less = item["new"] < item["old"]
        cue = str(item.get("cue") or "").strip()
        heard = f"Heard “{cue}” about {item['race']}" if cue else f"Changed {item['race']}"
        steps.append(f"{heard}: it now counts {'less' if less else 'more'} ({item['old']:.1f} → {item['new']:.1f}).")
        gap = _gap(race)
        if gap:                       # a typical race is already said on the card; an unusual one matters here
            steps.append(f"Its own time was {gap}.")
        effect = _effect(race, less, before)
        if effect:
            steps.append(effect[0].upper() + effect[1:] + ".")
    if steps and card.get("target_delta"):
        steps.append(f"Net effect: target {card['target']} ({card['target_delta']}).")
    return steps                      # the hidden-race check has its own line on the card


def because(item: dict, races: list[dict], target_before_s: float | None = None) -> str:
    """One spoken clause: why the change moved the target the way it did."""
    race = next((r for r in races if f"{r['event']} {r['year']}" == item["race"]), {})
    effect = _effect(race, item["new"] < item["old"], target_before_s)
    if effect:
        return effect
    gap = _gap(race)
    return f"its time was {gap}" if gap else "it is a typical race for this runner, so this rests on what you know"


def _outliers(races: list[dict]) -> list[dict]:
    """Races whose time stands out and whose weight the coach has not touched yet, largest gap first."""
    out = [r for r in races if (r.get("evidence") or {}).get("direction") and abs(float(r.get("weight", 0.5)) - 0.5) < 1e-9]
    return sorted(out, key=lambda r: -abs(_gap_s(r)))


def next_steps(state: dict, *, familiarity: str, used: set[str], shown: dict[str, int],
               limit: int = 3) -> list[dict]:
    """Up to ``limit`` suggestions {key, say, why}, the most useful first, preferring ones not yet shown.

    ``used`` names what the coach has already done this session ("changed", "against_data", "sim", "chart",
    "similar"); ``shown`` counts how often each suggestion has been offered."""
    races = state.get("races") or []
    steering = state.get("steering") or {}
    holdout = state.get("holdout") or {}
    prediction = state.get("prediction") or {}
    changed = any(abs(float(r.get("weight", 0.5)) - 0.5) > 1e-9 for r in races) or "changed" in used
    she = {"F": "she", "M": "he"}.get(str((state.get("runner") or {}).get("gender") or "").upper()[:1], "they")
    cands: list[dict] = []

    if familiarity == "ask":
        cands.append({"key": "familiarity", "say": "I don't know this runner",
                      "why": "PaceSense lets the data lead for a runner you don't know, and challenges you less "
                             "for one you coach."})
    for r in _outliers(races)[:2]:
        name, gap = f"{r['event']} {r['year']}", _gap(r)
        cands.append({"key": f"outlier:{name}", "say": f"{r['year']} was a hot day",
                      "why": f"{name} was {gap}. If you know why (heat, injury, a fall), say it and its weight "
                             f"follows; if not, the data treats it as normal." if gap else
                             f"{name} stands out. If you know why, say it and its weight follows."})
    if not changed and familiarity == "known":
        cands.append({"key": "recent", "say": "trust the recent races more",
                      "why": "You know this runner; recent form is often the best guide."})
    if changed and steering.get("coach_estimate_s") is None and prediction.get("finish_s"):
        guess = format_time(float(prediction["finish_s"]) - 180)
        cands.append({"key": "estimate", "say": f"I think {she} can run {guess}",
                      "why": "Your own estimate is kept beside the model's, so you can see whose view was "
                             "closer at the end."})
    if changed and holdout.get("label") and not holdout.get("revealed"):
        cands.append({"key": "reveal", "say": "reveal the real result",
                      "why": f"{holdout['label']} is hidden. Revealing it shows whether your changes moved the "
                             f"prediction closer."})
    if holdout.get("revealed") and changed:
        err, base = holdout.get("error_pct"), holdout.get("equal_weights_error_pct")
        if err is not None and base is not None and float(err) > float(base):
            cands.append({"key": "undo-after-reveal", "say": "show the race contributions",
                          "why": f"Your changes left it {float(err):.1f}% off versus {float(base):.1f}% with equal "
                                 f"weights; the chart shows which race pushed it away."})
    if "against_data" in used and "similar" not in used:
        cands.append({"key": "similar", "say": "show similar runners",
                      "why": "One of your changes goes against this runner's own times; similar runners show "
                             "what is typical."})
    if "chart" not in used and races:
        cands.append({"key": "chart", "say": "show the race contributions",
                      "why": "See which race is moving the target most, and which way."})
    if "sim" not in used and changed:
        cands.append({"key": "sim", "say": "simulate 25 degrees",
                      "why": "Test the plan against a hot race day before trusting it."})
    cands.append({"key": "range", "say": "what does the likely range mean?",
                  "why": "The range tells you how sure the model is, which matters as much as the target."})

    unique = list({c["say"]: c for c in reversed(cands)}.values())[::-1]     # first of each wording, in order
    ranked = sorted(enumerate(unique), key=lambda ic: (shown.get(ic[1]["key"], 0), ic[0]))
    return [c for _, c in ranked[:limit]]


def guidance_line(step: dict | None) -> str:
    """One short spoken line for the top suggestion."""
    if not step:
        return ""
    return f"Next, you could say “{step['say']}”. {step['why']}"


__all__ = ["because", "guidance_line", "next_steps", "reasoning"]

