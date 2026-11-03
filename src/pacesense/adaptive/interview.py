"""The guided interview: instead of reading charts and moving sliders, the coach answers a few questions.

The questions come from the runner's own data (her slowest and fastest recent race, how recent the races are,
the model's target), so they are always about something the data cannot know. Each answer is either a sentence
for the same interpreter the coach could have said themselves (so the data checks still apply), or a note that
changes nothing. That keeps the interview a shortcut, not a second model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import format_time


@dataclass
class Choice:
    key: str
    label: str
    say: str | None = None              # sentence routed like a typed command
    familiarity: float | None = None    # sets the per-runner familiarity dimension

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "say": self.say}


@dataclass
class Question:
    id: str
    text: str
    why: str
    choices: list[Choice] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "why": self.why, "choices": [c.to_dict() for c in self.choices]}

    def choice(self, key: str) -> Choice:
        for c in self.choices:
            if c.key == key:
                return c
        raise KeyError(key)


def questions(state: dict) -> list[Question]:
    races = state["races"]
    pred = state["prediction"]
    pron = "she" if state.get("runner", {}).get("gender") != "M" else "he"
    poss = "her" if pron == "she" else "his"
    obj = "her" if pron == "she" else "him"
    qs = [Question("familiarity", "How well do you know this runner?",
                   "Coaches lean on the model more for runners they don't know (the paper's RQ2); "
                   "PaceSense adjusts how much it challenges you.",
                   [Choice("coach", f"I coach {obj}", familiarity=0.9),
                    Choice("some", "A little", familiarity=0.5),
                    Choice("none", "Not at all", familiarity=0.1)])]
    slow = sorted((r for r in races if r["evidence"]["direction"] > 0), key=lambda r: -(r["evidence"].get("z") or 0))
    if slow:
        r = slow[0]
        name = f"{r['event']} {r['year']}"
        med = r["evidence"].get("others_median_s")
        gap = "" if med is None else f" ({round((r['finish_s'] - med) / 60)} min slower than {poss} others)"
        qs.append(Question("slow", f"{name} was {poss} slowest recent race{gap}. What happened?",
                           "Only you know whether a slow race was bad luck or real form.",
                           [Choice("hot", "It was hot", f"{name} was a hot day"),
                            Choice("injured", f"{pron.capitalize()} was injured or ill", f"{name}: {pron} was injured"),
                            Choice("pacing", "Bad pacing", f"{name}: {pron} started too fast"),
                            Choice("typical", "That's really " + poss + " level", f"{name} is typical"),
                            Choice("skip", "Not sure")]))
    fast = sorted((r for r in races if r["evidence"]["direction"] < 0), key=lambda r: r["evidence"].get("z") or 0)
    best = fast[0] if fast else min(races, key=lambda r: r["finish_s"])
    name = f"{best['event']} {best['year']}"
    qs.append(Question("best", f"{name} ({best['finish']}) was {poss} fastest. Is that {poss} level now?",
                       "A best day can be form, or a one-off with perfect conditions.",
                       [Choice("level", f"Yes, that's {poss} level", f"{name} is exactly how {pron} runs now"),
                        Choice("oneoff", "It was a one-off", f"count {name} a bit less"),
                        Choice("skip", "Not sure")]))
    if len({r["year"] for r in races}) > 1:
        qs.append(Question("trend", f"Over the last few years, is {pron} getting faster, slower, or about the same?",
                           "Recent races can be weighted more when form has changed.",
                           [Choice("faster", "Getting faster", "trust the recent races more"),
                            Choice("slower", "Getting slower", "trust the recent races more"),
                            Choice("same", "About the same"),
                            Choice("skip", "Not sure")]))
    t = pred["finish_s"]
    opts = [t - 300, t, t + 300]
    qs.append(Question("estimate", f"What finish time do you expect for {poss} next marathon?",
                       "Your own estimate lets PaceSense show whose view the model is following (the steering meter).",
                       [Choice(f"e{i}", format_time(x, with_seconds=False), f"I think {pron} can run {format_time(x, with_seconds=False)}")
                        for i, x in enumerate(opts)] + [Choice("skip", "No idea")]))
    return qs


__all__ = ["Choice", "Question", "questions"]
