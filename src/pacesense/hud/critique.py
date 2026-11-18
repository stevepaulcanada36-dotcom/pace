"""A fair counterweight to each change, so PaceSense neither defers to the coach nor overrides them.

Muijlwijk et al. (IUI '24) found that interaction helped accuracy, but also that a coach can steer the model to
their own expectation and still score full Acceptance (Sec. 6), that familiarity lowers reliance (RQ2), and
that the model cannot see injuries, training or race-day conditions. Each change therefore gets, from the
numbers alone:

* the case for it (what the coach knows, what the data agrees with),
* the case against it (it goes against the runner's own times, it is smaller than the model's uncertainty,
  it makes the target rest on one race, it came from a hope rather than a fact),
* what would settle it (the hidden race, similar runners, a reason the coach can state).

The coach still decides. PaceSense's job is to make the trade-off visible, not to agree.
"""
from __future__ import annotations

import re

HEDGE_RE = re.compile(r"\b(?:maybe|i hope|hopefully|i guess|i think|i feel|probably|perhaps|not sure|kind of|"
                      r"sort of|might|could be)\b", re.IGNORECASE)
FACT_RE = re.compile(r"\b(?:injur\w*|sick|ill|flu|hot|heat|wind\w*|rain\w*|fell|fall|cramp\w*|stomach|"
                     r"pregnan\w*|course|hill\w*|told me|i coach|i saw|i watched|i'?ve seen|i have seen|i'?ve watched|saw (?:her|him|it)|"
                     r"was there|trained|training)\b",
                     re.IGNORECASE)


def effective_races(weights: list[float]) -> float:
    """How many races the target effectively rests on (1 = one race carries it, n = all count equally)."""
    total, squares = sum(weights), sum(w * w for w in weights)
    return (total * total / squares) if squares else 0.0


def consider(card: dict, state: dict, said: str = "") -> list[dict]:
    """[{kind: for|against|check, text}] for one change card; empty when nothing changed."""
    items = card.get("items") or []
    if not items:
        return []
    races = state.get("races") or []
    pred = state.get("prediction") or {}
    out: list[dict] = []
    text = " ".join([said, *[str(i.get("cue") or "") for i in items]])

    # ---- for
    supported = [i for i in items if i.get("verdict") == "supported"]
    if supported:
        out.append({"kind": "for", "text": f"The data agrees with you about {supported[0]['race']}."})
    if FACT_RE.search(text):
        out.append({"kind": "for", "text": "You gave a concrete reason the times cannot show; that is exactly "
                                           "what the model is missing."})

    # ---- against
    against = [i for i in items if i.get("verdict") == "against_data"]
    if against:
        out.append({"kind": "against", "text": f"{against[0]['race']} goes the other way in her own times. "
                                               "Unless you know why, this may be expectation rather than knowledge."})
    if HEDGE_RE.search(text) and not FACT_RE.search(text):
        out.append({"kind": "against", "text": "You said it as a hope or a guess, not a fact. A guess can move "
                                               "the model just as far as knowledge can."})
    weights = [float(r.get("weight", 0.5)) for r in races]
    eff = effective_races(weights)
    if races and len(races) >= 3 and eff < 1.8:
        heavy = max(races, key=lambda r: float(r.get("weight", 0.5)))
        out.append({"kind": "against", "text": f"The target now rests mostly on {heavy['event']} {heavy['year']} "
                                               f"(about {eff:.1f} races' worth of evidence). One race is a thin "
                                               f"base; this is where overfitting starts."})
    before, after = card.get("target_before_s"), pred.get("finish_s")
    low, high = pred.get("low_s"), pred.get("high_s")
    if before is not None and after is not None and low and high:
        moved, half = abs(float(after) - float(before)), (float(high) - float(low)) / 2
        if moved and moved < 0.1 * half:
            out.append({"kind": "against", "text": f"This moved the target {round(moved)} s, well inside the "
                                                   f"model's uncertainty (±{int(half // 60)}:{int(half % 60):02d}). It barely "
                                                   f"matters to the prediction."})

    # ---- what would settle it
    holdout = state.get("holdout") or {}
    if holdout.get("label") and not holdout.get("revealed"):
        out.append({"kind": "check", "text": f"Revealing {holdout['label']} will show whether this helped. "
                                             f"One race cannot prove a method, but it is evidence."})
    elif against or eff < 1.8:
        out.append({"kind": "check", "text": "Similar runners show what is typical for a race like this one."})
    return out[:4]


def spoken(points: list[dict]) -> str:
    """One sentence to say after the change: the strongest point against, if there is one."""
    against = next((p for p in points if p["kind"] == "against"), None)
    return f"One thing to weigh: {against['text']}" if against else ""


# What the paper and this implementation cannot show, for the assistant and the second opinion.
LIMITATIONS = (
    ("The model sees only past marathon times; injuries, training, health and race-day conditions reach it only "
    "through the coach."),
    "The prediction is a target for a good day (a personal best that similar runners reached), not an average.",
    ("A coach can steer the model to their own expectation and still score full Acceptance, so agreement is not "
    "proof the model persuaded anyone (paper, Sec. 6)."),
    "Familiar runners invite strong opinions; the paper found coaches relied on the model less for them (RQ2).",
    "The hidden race is one result. One race checks a change; it does not prove a coach's method.",
    ("The study had few coaches and these runners are synthetic, so the numbers here illustrate the method rather "
    "than predict a real athlete."),
)

__all__ = ["LIMITATIONS", "consider", "effective_races", "spoken"]
