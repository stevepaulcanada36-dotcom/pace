"""A spoken planning session: each utterance moves the sliders of a `PlanSession`, with reasons and evidence.

    s = SteeringSession(PlanSession(coach, "R00042"))
    turn = s.hear("2016 was a hot day and she was injured. I think she can run 3:25.")
    turn.applied[0].text      # 'Amsterdam 2016: 0.5 → 0.0 ("hot day + injured"). The data agrees: ...'
    s.state()                 # weights, prediction, likely range, influence, evidence, steering index

Everything goes through `PlanSession.set_weight`, so the dashboard's change log, the plan check and the saved
plan see voice changes exactly like slider changes. A turn can be undone ("scratch that"). Slider moves made by
hand can be logged with `manual()`, so the record shows which changes came with a spoken reason and which did not.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..domain import ValidationError, format_time
from ..planning import DEFAULT_WEIGHT, PlanSession
from .evidence import RaceEvidence, check_change, influence, race_evidence, steering_index
from .intent import Action, Interpretation, Lexicon, default_lexicon, interpret, recency_weights


@dataclass
class Applied:
    index: int
    race: str
    old: float
    new: float
    cue: str
    reasoned: bool
    verdict: str
    evidence: str

    @property
    def text(self) -> str:
        return f"{self.race}: {self.old:.1f} → {self.new:.1f} (\"{self.cue}\"). {self.evidence}"

    def to_dict(self) -> dict:
        return {"index": self.index, "race": self.race, "old": self.old, "new": self.new, "cue": self.cue,
                "reasoned": self.reasoned, "verdict": self.verdict, "evidence": self.evidence, "text": self.text}


@dataclass
class Turn:
    n: int
    text: str
    source: str                      # "voice", "typed", "manual", "undo"
    applied: list[Applied] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    finish_before_s: float = math.nan
    finish_after_s: float = math.nan
    estimate_s: float | None = None
    interpreter: str = "rules"
    at: float = field(default_factory=time.time)

    @property
    def delta_s(self) -> float:
        return self.finish_after_s - self.finish_before_s

    def summary(self) -> str:
        if not self.applied:
            return "" if self.notes else "Nothing to change."
        d = round(self.delta_s)
        move = "no change to the target" if d == 0 else f"target {format_time(self.finish_after_s)} ({'+' if d > 0 else '−'}{abs(d) // 60}:{abs(d) % 60:02d})"
        return f"{len(self.applied)} slider{'s' if len(self.applied) > 1 else ''} moved, {move}."

    def to_dict(self) -> dict:
        return {"n": self.n, "text": self.text, "source": self.source, "applied": [a.to_dict() for a in self.applied],
                "notes": self.notes, "finish_before_s": self.finish_before_s, "finish_after_s": self.finish_after_s,
                "delta_s": self.delta_s, "estimate_s": self.estimate_s, "interpreter": self.interpreter,
                "summary": self.summary()}


class SteeringSession:
    def __init__(self, plan: PlanSession, lexicon: Lexicon | None = None, half_life: float | None = None,
                 ai=None) -> None:
        """`ai`: an AIService to read paraphrases with an AI model (checked by code); None = rules only."""
        self.plan = plan
        self.ai = ai
        self.lexicon = lexicon or default_lexicon()
        self.half_life = half_life if half_life is not None else self.lexicon.half_life_years
        self.races: list[dict] = plan.overview["races"]
        self.evidence: list[RaceEvidence] = race_evidence([r["finish_s"] for r in self.races])
        self.initial_finish_s: float = plan.prediction["default_finish_s"]
        self.coach_estimate_s: float | None = None
        self.context: int | None = None
        self.turns: list[Turn] = []
        self._undo: list[list[float]] = []

    # ---- helpers ----------------------------------------------------------------------------
    def _name(self, i: int) -> str:
        r = self.races[i]
        return f"{r['event']} {r['year']}"

    def _set(self, turn: Turn, i: int, new: float, cue: str, reasoned: bool, say_if_unchanged: bool = False) -> None:
        new = round(min(1.0, max(0.0, new)), 1)
        old = self.plan.weights[i]
        if new == old:
            if say_if_unchanged:
                what = "counts for nothing" if old == 0 else "counts fully" if old == 1 else f"is at {old:.1f}"
                turn.notes.append(f"{self._name(i)} already {what} (heard \"{cue}\").")
            return
        try:
            change = self.plan.set_weight(i, new)
        except ValidationError as e:
            turn.notes.append(f"{self._name(i)} left at {old:.1f}: {e}.")
            return
        if cue == "undo":
            verdict, message = "undo", "Back to the value before the last change."
        else:
            chk = check_change(self.evidence[i], old, new, self._name(i))
            verdict, message = chk.verdict, chk.message
        if change is not None:
            said = "Set by hand." if cue == "slider" else "Undone." if cue == "undo" else f'You said: "{cue}".'
            change.note = (said + " " + change.note).strip()
        turn.applied.append(Applied(i, self._name(i), old, new, cue, reasoned, verdict, message))

    def _begin(self, text: str, source: str) -> Turn:
        self._undo.append(list(self.plan.weights))
        return Turn(len(self.turns) + 1, text, source, finish_before_s=self.plan.model_finish)

    def _end(self, turn: Turn) -> Turn:
        turn.finish_after_s = self.plan.model_finish
        if not turn.applied:
            self._undo.pop()
        self.turns.append(turn)
        return turn

    # ---- actions ------------------------------------------------------------------------------
    def hear(self, text: str, source: str = "voice") -> Turn:
        """Apply one utterance (spoken or typed)."""
        if self.ai is not None:
            from .llm_interpret import interpret_hybrid
            it, used = interpret_hybrid(self.ai, text, self.races, self.context)
        else:
            it, used = interpret(text, self.races, self.context, self.lexicon), "rules"
        self.context = it.context
        if any(a.kind == "undo" for a in it.actions):
            return self.undo(text)
        turn = self._begin(text, source)
        turn.interpreter = used
        if it.estimate_s is not None:
            self.coach_estimate_s = turn.estimate_s = it.estimate_s
            turn.notes.append(f"Your estimate: {format_time(it.estimate_s)}.")
        for a in it.actions:
            self._apply(turn, a)
        for u in it.unknown_races:
            turn.notes.append(f"{self.plan.name} has no race '{u}' among these races.")
        for u in it.unresolved:
            turn.notes.append(f"Heard \"{u}\" but not which race it is about. Name the race or the year.")
        if not it.understood and not it.unknown_races and not it.unresolved:
            turn.notes.append("Heard you, but nothing in that changes a race weight.")
        return self._end(turn)

    def _apply(self, turn: Turn, a: Action) -> None:
        if a.kind == "reset":
            for i in range(len(self.races)):
                self._set(turn, i, DEFAULT_WEIGHT, a.cue, True)
        elif a.kind == "recency":
            target = recency_weights([r["year"] for r in self.races], self.half_life, self.lexicon.recency_floor)
            for i, w in enumerate(target):
                current = self.plan.weights[i]
                if current < DEFAULT_WEIGHT and w > current:
                    continue        # a general "recent races more" never undoes a race the coach already lowered
                self._set(turn, i, w, a.cue, True)
        elif a.kind == "set":
            for i in a.races:
                self._set(turn, i, float(a.value), a.cue, reasoned=not a.cue.split()[0] in ("to", "at", "of", "=", "weight", "weighted"),
                          say_if_unchanged=True)
        elif a.kind == "shift":
            for i in a.races:
                self._set(turn, i, self.plan.weights[i] + float(a.value), a.cue, True, say_if_unchanged=True)

    def manual(self, i: int, weight: float) -> Turn:
        """A slider moved by hand (no reason given)."""
        turn = self._begin(f"{self._name(i)} slider", "manual")
        self._set(turn, i, weight, "slider", reasoned=False)
        return self._end(turn)

    def undo(self, text: str = "undo") -> Turn:
        turn = Turn(len(self.turns) + 1, text, "undo", finish_before_s=self.plan.model_finish)
        if not self._undo:
            turn.notes.append("Nothing to undo.")
        else:
            previous = self._undo.pop()
            for i, w in enumerate(previous):
                self._set(turn, i, w, "undo", True)
        turn.finish_after_s = self.plan.model_finish
        self.turns.append(turn)
        return turn

    # ---- read-only ---------------------------------------------------------------------------
    def reasoned_share(self) -> float | None:
        moves = [a for t in self.turns if t.source != "undo" for a in t.applied]
        return None if not moves else sum(a.reasoned for a in moves) / len(moves)

    def state(self) -> dict:
        pr = self.plan.prediction
        per = [None if p is None else p["finish_s"] for p in pr["per_race"]]
        infl = influence(per, self.plan.weights)
        steer = steering_index(self.initial_finish_s, self.plan.model_finish, self.coach_estimate_s)
        return {
            "runner": {"id": self.plan.runner_id, "name": self.plan.name},
            "races": [{"index": i, "event": r["event"], "year": r["year"], "finish_s": r["finish_s"],
                       "finish": r["finish"], "weight": self.plan.weights[i],
                       "prediction_s": per[i], "evidence": self.evidence[i].to_dict(),
                       "influence": infl[i].to_dict()} for i, r in enumerate(self.races)],
            "prediction": {"finish_s": self.plan.model_finish, "finish": format_time(self.plan.model_finish),
                           "low_s": pr["low_s"], "high_s": pr["high_s"], "initial_s": self.initial_finish_s,
                           "initial": format_time(self.initial_finish_s), "n_cases": pr["n_cases"]},
            "steering": steer.to_dict(),
            "reasoned_share": self.reasoned_share(),
            "half_life_years": None if math.isinf(self.half_life) else self.half_life,
            "turns": [t.to_dict() for t in self.turns[-20:]],
            "change_log": [c.text() for c in self.plan.changes[:20]],
        }

    def transcript(self) -> list[dict]:
        """Everything said, in order: enough to replay the session exactly (`replay`)."""
        out = []
        for t in self.turns:
            row = {"n": t.n, "text": t.text, "source": t.source}
            if t.source == "manual" and t.applied:          # a slider has no words; keep the move itself
                row.update(index=t.applied[0].index, weight=t.applied[0].new)
            out.append(row)
        return out


def replay(plan: PlanSession, transcript: Sequence[dict], **kw) -> SteeringSession:
    """Re-run a session from its transcript. Same words + same data + same lexicon = same weights."""
    s = SteeringSession(plan, **kw)
    for t in transcript:
        if t["source"] == "undo":
            s.undo(t["text"])
        elif t["source"] == "manual":
            if "index" in t and "weight" in t:              # older transcripts did not keep slider moves
                s.manual(int(t["index"]), float(t["weight"]))
        else:
            s.hear(t["text"], t["source"])
    return s


__all__ = ["SteeringSession", "Turn", "Applied", "replay"]
