"""Short, spoken-length explanations built from the session's own numbers (no model, no guessing)."""
from __future__ import annotations

from ..domain import format_time


def race(state: dict, index: int) -> str:
    r = next((x for x in state.get("races", []) if x.get("index") == index), None)
    if r is None:
        return "I can't find that race any more."
    ev, name = r["evidence"], f"{r['event']} {r['year']}"
    med, z = ev.get("others_median_s"), ev.get("z")
    if med is None:
        return f"{name} is her only comparable race, so the data can't say much about it."
    gap = round(abs(r["finish_s"] - med) / 60)
    way = "slower" if r["finish_s"] > med else "faster"
    spread = f", about {abs(z):.1f} times her usual spread" if z is not None else ""
    tail = ("Races that far out usually say more about the day than about her current level."
            if z is not None and abs(z) >= 2 else "That's noticeable, but within what she sometimes does.")
    return (f"{name} took {r['finish']}. Her other recent races have a median of {format_time(med)}, "
            f"so it was {gap} minutes {way}{spread}. {tail}")


def likely_range(state: dict) -> str:
    p = state.get("prediction", {})
    if not p.get("low_s"):
        return "There aren't enough similar runners for a range."
    return (f"The likely range, {format_time(p['low_s'])} to {format_time(p['high_s'])}, is an uncertainty band "
            f"based on the 20th to 80th percentiles of similar runners' implied next-marathon finishes. "
            f"On held-out demo races, the actual result fell inside this range for about 80% of runners. "
            f"That is a validation result, not a guarantee for this runner. The target, "
            f"{p.get('finish', format_time(p['finish_s']))}, is the weighted middle.")


def data_check(state: dict) -> str:
    return ("Every change is compared with her own times. If you lower a race that was unusually slow, the data "
            "agrees. If you lower a typical or fast race, it doesn't, because nothing in her times marks it as "
            "unusual. That can still be right: you may know something the times don't show.")


def explain(key: str, state: dict, race_index: int | None = None) -> str:
    if key == "race" and race_index is not None:
        return race(state, race_index)
    if key == "range":
        return likely_range(state)
    return data_check(state)


__all__ = ["explain"]
