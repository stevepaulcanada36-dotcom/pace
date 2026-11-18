"""The paper's analysis, applied to one coaching session.

Muijlwijk et al. (IUI '24) measure Acceptance (weight of advice), Model adaptation, prediction accuracy against
a held-out race, how slider positions relate to race year, the kinds of knowledge coaches used, and how
familiarity changes reliance. This module computes the same things for the session on screen, in plain words,
so the coach can see what their interaction did. Everything is computed from PaceSense's numbers.
"""
from __future__ import annotations

import re

from ..domain import format_time
from ..measures import acceptance, model_adaptation

# Knowledge coaches used in the paper (Sec. 5.6), plus race-day conditions, which the coach notes often name.
KNOWLEDGE = {
    "Injuries and health": r"\b(?:injur\w*|hurt|knee|hamstring|calf|achilles|back|sick|ill|illness|flu|cold|"
                           r"pain|cramp\w*|recover\w*)\b",
    "Training load": r"\b(?:train\w*|mileage|long runs?|volume|workouts?|sessions?|works? out|tapering|taper)\b",
    "Mental strength": r"\b(?:mental\w*|confiden\w*|motivat\w*|nerv\w*|focus\w*|tough|determin\w*|mindset)\b",
    "Pacing consistency": r"\b(?:pac(?:e|ing|ed)|splits?|went out too fast|fast start|faded|negative split|even)\b",
    "Untapped potential": r"\b(?:potential|capable|better than (?:that|this|her|his)|faster than (?:that|this)|"
                          r"improv\w*|good runner|hard work|room to)\b",
    "Personal circumstances": r"\b(?:work|job|stress|family|baby|pregnan\w*|moved|travel\w*|exam\w*|life)\b",
    "Race-day conditions": r"\b(?:hot|heat|warm|wind\w*|rain\w*|cold day|weather|humid\w*|course|hill\w*)\b",
}
_KNOWLEDGE_RE = {k: re.compile(v, re.IGNORECASE) for k, v in KNOWLEDGE.items()}
PAPER_RECENCY_PER_YEAR = 0.042        # slider position rose with race year (paper, RQ4)


def _whose_view(model_initial: float, model_now: float, coach_pre: float | None, coach_post: float | None) -> dict:
    """The paper's four cases (slide 10): who moved toward whom."""
    if coach_pre is None:
        return {"available": False,
                "text": "Say your own estimate (“I think she can run 3:35”) to see whose view the final "
                        "prediction reflects."}
    post = coach_post if coach_post is not None else coach_pre
    adapt = model_adaptation(model_initial, model_now, coach_pre)
    acc = acceptance(model_now, coach_pre, post)
    toward = coach_pre - model_initial
    moved = model_now - model_initial
    beyond = abs(toward) > 1 and moved * toward > 0 and abs(moved) > abs(toward) + 60
    if beyond:
        case, label = 4, "Steered beyond your view"
        text = ("You moved the model past your own estimate. Acceptance reads as full, but the model did not "
                "persuade you; the paper flags exactly this case.")
    elif adapt is not None and adapt >= 0.75:
        case, label = 3, "Model steered to your view"
        text = ("The model now says what you already thought. Acceptance reads as full, but it reflects your "
                "steering rather than the model's advice. Is that knowledge, or expectation?")
    elif adapt is not None and adapt > 0.25:
        case, label = 2, "Met halfway"
        text = "You moved the model part of the way toward your view, and it holds the rest."
    elif acc is not None and acc >= 0.75:
        case, label = 1, "The model persuaded you"
        text = "Your estimate moved to the model's prediction while the model barely moved."
    elif acc is not None and acc > 0.25:
        case, label = 2, "Met halfway"
        text = "Your estimate moved part of the way toward the model."
    else:
        case, label = 0, "Two separate views"
        text = "Neither moved much toward the other; the model and you still disagree."
    return {"available": True, "case": case, "label": label, "text": text,
            "acceptance": None if acc is None else round(acc, 2),
            "adaptation": None if adapt is None else round(min(adapt, 9.99), 2),
            "points": {"model_initial_s": model_initial, "model_now_s": model_now,
                       "coach_first_s": coach_pre, "coach_now_s": post}}


def _accuracy(holdout: dict, model_initial: float, model_now: float, coach_pre: float | None,
              coach_post: float | None) -> dict:
    """RQ4: who was closer to the hidden race."""
    if not holdout.get("label"):
        return {"available": False, "text": "This runner has no hidden race to check against."}
    if not holdout.get("revealed") or not holdout.get("truth_s"):
        return {"available": False,
                "text": f"{holdout['label']} is hidden. Say “reveal the real result” to see whose prediction "
                        f"was closer."}
    truth = float(holdout["truth_s"])

    def pct(x: float | None) -> float | None:
        return None if x is None else round(100 * abs(x - truth) / truth, 2)

    rows = [("Equal weights", pct(model_initial)), ("Your model", pct(model_now))]
    if coach_pre is not None:
        rows.append(("Your first estimate", pct(coach_pre)))
    if coach_post is not None and coach_post != coach_pre:
        rows.append(("Your latest estimate", pct(coach_post)))
    scored = [(n, p) for n, p in rows if p is not None]
    best = min(scored, key=lambda r: r[1])
    equal, mine = float(rows[0][1] or 0), float(rows[1][1] or 0)
    helped = mine < equal - 0.005
    text = (f"Real result {format_time(truth)} ({holdout['label']}). Your changes "
            f"{'made the model more accurate' if helped else 'did not make the model more accurate'}: "
            f"{mine:.2f}% off versus {equal:.2f}% with equal weights. Closest: {best[0].lower()}. "
            f"In the paper coaches' changes cut the error from 3.14% to 2.33%.")
    return {"available": True, "truth_s": truth, "rows": [{"name": n, "pct": p} for n, p in rows], "text": text}


def _recency(races: list[dict]) -> dict:
    """Do the weights rise with race year, as they did for the paper's coaches?"""
    pts = [(int(r["year"]), float(r.get("weight", 0.5))) for r in races if r.get("year") is not None]
    if len(pts) < 2 or all(abs(w - 0.5) < 1e-9 for _, w in pts):
        return {"available": False, "text": "You have not changed any weights yet. Coaches in the paper "
                                            "mostly counted recent races more."}
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else 0.0
    if slope > 0.01:
        text = (f"Your weights rise with race year (about {slope:+.3f} per year), like the paper's coaches "
                f"({PAPER_RECENCY_PER_YEAR:+.3f}): you trust recent form more.")
    elif slope < -0.01:
        text = (f"Your weights fall with race year ({slope:+.3f} per year): you trust older races more, the "
                f"opposite of the paper's coaches. Worth saying why.")
    else:
        text = "Your weights do not depend on race year; you judged each race on something other than age."
    return {"available": True, "per_year": round(slope, 3), "text": text}


def _knowledge(said: list[str]) -> dict:
    used = {k: sum(1 for t in said if rx.search(t)) for k, rx in _KNOWLEDGE_RE.items()}
    used = {k: v for k, v in used.items() if v}
    if not used:
        return {"available": False, "categories": [],
                "text": "You have not described the runner yet. The paper's coaches used injuries, training load, "
                        "mental strength, pacing, untapped potential and personal circumstances."}
    names = sorted(used, key=lambda k: -used[k])
    return {"available": True, "categories": [{"name": k, "mentions": used[k]} for k in names],
            "text": "Knowledge you brought: " + ", ".join(n.lower() for n in names) + "."}


def _familiarity(condition: str) -> dict:
    if condition == "known":
        return {"available": True, "text": "You know this runner. In the paper coaches accepted the model less "
                                           "for their own runners (RQ2), so PaceSense checks changes that go "
                                           "against the data more closely."}
    if condition in ("unfamiliar", "unknown", "new"):
        return {"available": True, "text": "You don't know this runner, so the data leads. In the paper this is "
                                           "where coaches relied on the model most, and where their changes "
                                           "were tested for accuracy."}
    return {"available": False, "text": "Say how well you know this runner; the paper found it changes how "
                                        "much coaches rely on the model."}


def session_analysis(state: dict, *, coach_first_s: float | None, condition: str, said: list[str],
                     against_data: int = 0) -> dict:
    """The analysis for the session on screen, as sections with a sentence each (and numbers to draw)."""
    pred = state.get("prediction") or {}
    steering = state.get("steering") or {}
    model_now = float(pred.get("finish_s") or 0)
    model_initial = float(pred.get("initial_s") or model_now)
    coach_now = steering.get("coach_estimate_s")
    coach_now = float(coach_now) if coach_now else None
    first = coach_first_s if coach_first_s is not None else coach_now
    out = {
        "whose_view": _whose_view(model_initial, model_now, first, coach_now),
        "accuracy": _accuracy(state.get("holdout") or {}, model_initial, model_now, first, coach_now),
        "recency": _recency(state.get("races") or []),
        "knowledge": _knowledge(said),
        "familiarity": _familiarity(condition),
        "against_data": int(against_data),
    }
    if against_data:
        out["bias"] = {"available": True,
                       "text": f"{against_data} of your changes went against this runner's own times. Expert "
                               f"control can add knowledge the data lacks, or confirm what you expected; the "
                               f"paper asks how to tell the two apart."}
    return out


def summary(analysis: dict) -> str:
    """Two or three sentences for speech: the sections that have something to say, most telling first."""
    order = ("accuracy", "whose_view", "recency", "bias", "knowledge")
    lines = [analysis[k]["text"] for k in order if analysis.get(k, {}).get("available")]
    if not lines:
        return analysis["whose_view"]["text"]
    return " ".join(lines[:2])


__all__ = ["KNOWLEDGE", "session_analysis", "summary"]
