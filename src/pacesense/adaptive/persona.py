"""Coach persona: how this coach works, learned from what they say and do, never from who they are.

Four lasting dimensions, each a Beta(a, b) estimate in [0, 1] that every signal nudges, plus one per-runner
dimension (familiarity) that starts fresh with every runner:

    reliance     0 intuition-led   ...  1 data-led        (overrides the data checks vs. follows them)
    detail       0 just the answer ...  1 wants the reasons
    voice        0 clicks          ...  1 talks
    certainty    0 exploring       ...  1 decisive
    familiarity  0 new to runner   ...  1 knows the runner well        (this session only)

The persona is transparent: every nudge keeps the cue that caused it, the coach sees the profile and can set
any dimension by hand (an override wins over inference until reset). Grounding: the paper found that
familiarity changes reliance (RQ2) and that the same control can carry expertise or confirmation bias (Sec. 6),
so the system needs to know, per coach and per runner, which way a coach leans.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .events import Event

DIMENSIONS = {
    "reliance": ("Intuition-led", "Data-led"),
    "detail": ("Just the answer", "Wants the reasons"),
    "voice": ("Clicks", "Talks"),
    "certainty": ("Exploring", "Decisive"),
}
RUNNER_DIMENSIONS = {"familiarity": ("New to this runner", "Knows this runner")}

ARCHETYPES = {
    #            reliance detail certainty
    "analyst": ((0.75, 0.75, 0.55), "The Analyst", "Follows the data and wants to see why."),
    "intuitive": ((0.25, 0.40, 0.80), "The Intuitive Expert", "Trusts what they know about the runner over the numbers."),
    "pragmatist": ((0.55, 0.25, 0.70), "The Pragmatist", "Wants a sound answer quickly, with little fuss."),
    "explorer": ((0.50, 0.65, 0.25), "The Explorer", "Weighing it up; likes to be guided and shown options."),
}


def _words(*ws: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(ws, key=len, reverse=True)) + r")\b", re.IGNORECASE)


HEDGE = _words("maybe", "perhaps", "not sure", "i guess", "might", "possibly", "probably", "kind of", "sort of",
               "i wonder", "unsure", "hmm", "could be", "no idea", "don't know")
CERTAIN = _words("definitely", "certainly", "for sure", "absolutely", "clearly", "obviously", "always", "never",
                 "i'm sure", "i am sure", "no doubt", "without doubt", "trust me", "i know")
DATA = _words("data", "average", "median", "similar runners", "similar", "statistics", "stats", "range", "percent",
              "model", "evidence", "numbers", "typical", "trend", "compare", "probability")
WHY = _words("why", "explain", "how come", "what does", "what is", "show me", "reason", "how does", "tell me more")
INTUITION = _words("i know her", "i know him", "i know this runner", "trust me", "my gut", "gut feeling", "i feel",
                   "she always", "he always", "she never", "he never", "i've seen", "i have seen", "believe me")
RUNNER_KNOWLEDGE = _words("injured", "injury", "sick", "ill", "flu", "pregnant", "training", "trained", "work",
                          "stress", "knee", "hamstring", "calf", "illness", "her coach", "i coach", "my athlete",
                          "she told me", "he told me", "in training", "her form", "his form", "long runs")


@dataclass
class Dim:
    a: float = 2.0
    b: float = 2.0
    override: float | None = None
    cues: list[str] = field(default_factory=list)

    @property
    def inferred(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def value(self) -> float:
        return self.override if self.override is not None else self.inferred

    @property
    def n(self) -> float:
        return self.a + self.b - 4.0

    def nudge(self, toward_high: bool, weight: float, cue: str) -> None:
        if toward_high:
            self.a += weight
        else:
            self.b += weight
        self.cues = ([f"{'↑' if toward_high else '↓'} {cue}"] + self.cues)[:6]

    def to_dict(self) -> dict:
        return {"a": round(self.a, 3), "b": round(self.b, 3), "override": self.override, "cues": self.cues}

    @classmethod
    def from_dict(cls, d: dict | None) -> Dim:
        d = d or {}
        return cls(float(d.get("a", 2.0)), float(d.get("b", 2.0)), d.get("override"), list(d.get("cues", [])))


@dataclass
class Persona:
    dims: dict[str, Dim] = field(default_factory=lambda: {k: Dim() for k in DIMENSIONS})
    runner: dict[str, Dim] = field(default_factory=lambda: {k: Dim() for k in RUNNER_DIMENSIONS})
    signals: int = 0

    # ---- persistence ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"dims": {k: d.to_dict() for k, d in self.dims.items()}, "signals": self.signals}

    @classmethod
    def from_dict(cls, d: dict | None) -> Persona:
        p = cls()
        if d:
            for k in DIMENSIONS:
                p.dims[k] = Dim.from_dict((d.get("dims") or {}).get(k))
            p.signals = int(d.get("signals", 0))
        return p

    def new_runner(self, familiarity: float | None = None) -> None:
        self.runner = {k: Dim() for k in RUNNER_DIMENSIONS}
        if familiarity is not None:
            self.runner["familiarity"].override = familiarity

    # ---- reading it ---------------------------------------------------------------------------
    def value(self, key: str) -> float:
        return (self.dims.get(key) or self.runner[key]).value

    @property
    def confidence(self) -> float:
        """0 = just met, approaches 1 as evidence accumulates (about 0.6 after 8 signals)."""
        return 1 - math.exp(-self.signals / 9)

    def archetype(self) -> dict:
        if self.confidence < 0.25 and not any(d.override is not None for d in self.dims.values()):
            return {"key": "unknown", "name": "Getting to know you", "description":
                    "PaceSense adapts as you work. Nothing is assumed about you up front."}
        v = (self.value("reliance"), self.value("detail"), self.value("certainty"))
        key = min(ARCHETYPES, key=lambda k: sum((x - y) ** 2 for x, y in zip(v, ARCHETYPES[k][0])))
        _, name, desc = ARCHETYPES[key]
        return {"key": key, "name": name, "description": desc}

    def summary(self) -> dict:
        def row(k, d, labels):
            return {"key": k, "value": round(d.value, 3), "inferred": round(d.inferred, 3), "override": d.override,
                    "low": labels[0], "high": labels[1], "evidence": round(d.n, 1), "cues": d.cues[:4]}
        return {"archetype": self.archetype(), "confidence": round(self.confidence, 3), "signals": self.signals,
                "dimensions": [row(k, self.dims[k], DIMENSIONS[k]) for k in DIMENSIONS],
                "runner": [row(k, self.runner[k], RUNNER_DIMENSIONS[k]) for k in RUNNER_DIMENSIONS]}

    # ---- learning ----------------------------------------------------------------------------
    def _n(self, key: str, high: bool, w: float, cue: str) -> None:
        (self.dims.get(key) or self.runner[key]).nudge(high, w, cue)
        self.signals += 1

    def observe(self, ev: Event) -> list[str]:
        """Update from one event; returns the cues used (for the loop log)."""
        before = self.signals
        t = ev.text or ""
        if ev.kind in ("utterance", "interview_answer") and t:
            short = t if len(t) <= 40 else t[:37] + "…"
            if m := HEDGE.search(t):
                self._n("certainty", False, 1.0, f'said "{m.group(0)}"')
            if m := CERTAIN.search(t):
                self._n("certainty", True, 1.0, f'said "{m.group(0)}"')
            if m := INTUITION.search(t):
                self._n("reliance", False, 1.0, f'said "{m.group(0)}"')
            if m := DATA.search(t):
                self._n("reliance", True, 0.6, f'asked about "{m.group(0)}"')
            if m := WHY.search(t):
                self._n("detail", True, 1.0, f'asked "{m.group(0)}"')
            n_words = len(t.split())
            if n_words >= 14:
                self._n("detail", True, 0.5, "explained at length")
            elif n_words <= 4 and ev.kind == "utterance":
                self._n("detail", False, 0.4, f'kept it short ("{short}")')
            if m := RUNNER_KNOWLEDGE.search(t):
                self._n("familiarity", True, 1.0, f'knew "{m.group(0)}"')
            if ev.source == "voice":
                self._n("voice", True, 1.0, "spoke")
            elif ev.kind == "utterance":
                self._n("voice", False, 0.4, "typed")
            if ev.data.get("estimate_s"):
                self._n("certainty", True, 0.5, "gave an own estimate")
        elif ev.kind == "slider":
            self._n("voice", False, 0.8, "moved a slider")
        elif ev.kind == "voice_on":
            self._n("voice", True, 0.5, "turned voice on")
        elif ev.kind == "undo" and ev.data.get("undid_verdicts") and "against_data" in ev.data["undid_verdicts"]:
            self._n("reliance", True, 1.0, "undid a change the data disagreed with")
        elif ev.kind == "reaction":
            v, verdict = ev.data.get("value"), ev.data.get("verdict")
            if v == "unsure":
                self._n("detail", True, 1.0, "was unsure about a check")
                self._n("certainty", False, 0.5, "was unsure about a check")
            elif v == "agree" and verdict == "supported":
                self._n("reliance", True, 0.6, "agreed with a data check")
            elif v == "disagree" and verdict in ("supported", "against_data"):
                self._n("reliance", False, 1.0, "disagreed with a data check")
        elif ev.kind == "panel" and ev.data.get("panel") in ("similar", "insights", "pacing"):
            self._n("detail", True, 0.5, f"opened {ev.data['panel']}")
            self._n("reliance", True, 0.3, f"opened {ev.data['panel']}")
        if ev.kind in ("utterance", "slider") and "against_data" in ev.verdicts:
            self._n("reliance", False, 0.8, "made a change the data disagreed with")
        if ev.kind == "interview_answer" and ev.data.get("choice") == "skip":
            self._n("certainty", False, 0.5, "skipped a question")
        if ev.kind == "interview_answer" and ev.data.get("familiarity") is not None:
            self.runner["familiarity"].override = float(ev.data["familiarity"])
        return [] if self.signals == before else ["persona"]

    def edit(self, changes: dict[str, Any]) -> None:
        """Coach-set values (0..1) win over inference; None clears an override; 'reset' clears everything."""
        if changes.get("reset"):
            self.dims = {k: Dim() for k in DIMENSIONS}
            self.signals = 0
            return
        for k, v in changes.items():
            dim = self.dims.get(k) or self.runner.get(k)
            if dim is None:
                raise ValueError(f"unknown persona dimension {k!r}")
            dim.override = None if v is None else min(1.0, max(0.0, float(v)))


__all__ = ["ARCHETYPES", "DIMENSIONS", "RUNNER_DIMENSIONS", "Dim", "Persona"]
