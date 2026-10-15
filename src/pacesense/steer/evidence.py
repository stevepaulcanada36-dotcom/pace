"""The statistics behind PaceSense Live: what the data says about each race and about each change.

Four measures, all plain arithmetic on the model's own numbers (no AI, no randomness):

1. **Race evidence** (`race_evidence`). Is a race unusual compared with the runner's *other* races? A robust,
   leave-one-out z-score: (race time - median of the others) / (1.4826 x MAD of the others), with the scale
   floored at 2 % of the median so three similar races do not make a 1-minute difference look extreme.
2. **Evidence check** (`check_change`). When a coach lowers or raises a race, does the data point the same
   way? Lowering a race that was unusually slow is *supported*; lowering a typical race *rests on the coach's
   knowledge* (which may be exactly right: an injury does not show in a finish time); lowering one of the
   runner's fastest races goes *against the data* and makes the target slower.
3. **Influence** (`influence`). How much each race pulls the combined prediction, in seconds:
   pull_i = w_i (t_i - F) / sum(w), where t_i is the race's own prediction and F the combined one. The pulls
   sum to zero. Also the exact change if the race were left out.
4. **Steering index** (`steering_index`). The paper's Model Adaptation (Sec. 5.7):
   (Model_adapted - Model_initial) / (Coach_estimate - Model_initial). 0 = the model is where it started,
   1 = the coach has moved the model exactly onto their own estimate. Shown live, so the coach sees when the
   sliders are becoming a way to agree with themselves - the open question from the paper's discussion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Verdict = Literal["supported", "your_knowledge", "against_data", "no_change"]

Z_CLEAR = 1.0          # |z| at or above this: the race stands out from the runner's other races
SCALE_FLOOR = 0.02     # the spread used is at least 2 % of the median race time


@dataclass(frozen=True)
class RaceEvidence:
    index: int
    finish_s: float
    others_median_s: float | None
    z: float | None
    label: str             # "typical", "slower than her other races", "faster than her other races", "only race"

    @property
    def direction(self) -> int:
        """+1 slower than the others, -1 faster, 0 typical or unknown."""
        if self.z is None or abs(self.z) < Z_CLEAR:
            return 0
        return 1 if self.z > 0 else -1

    def to_dict(self) -> dict:
        return {"index": self.index, "z": None if self.z is None else round(self.z, 2), "label": self.label,
                "others_median_s": self.others_median_s, "direction": self.direction}


def race_evidence(finish_times: Sequence[float]) -> list[RaceEvidence]:
    t = np.asarray(finish_times, float)
    out = []
    for i, x in enumerate(t):
        others = np.delete(t, i)
        if len(others) == 0:
            out.append(RaceEvidence(i, float(x), None, None, "only race"))
            continue
        med = float(np.median(others))
        mad = float(np.median(np.abs(others - med))) * 1.4826
        scale = max(mad, SCALE_FLOOR * med)
        z = (x - med) / scale
        if abs(z) < Z_CLEAR:
            label = "typical for this runner"
        elif z > 0:
            label = "slower than this runner's other races"
        else:
            label = "faster than this runner's other races"
        out.append(RaceEvidence(i, float(x), med, float(z), label))
    return out


@dataclass(frozen=True)
class EvidenceCheck:
    verdict: Verdict
    message: str

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "message": self.message}


def _mins(s: float) -> str:
    m = abs(s) / 60
    return f"{m:.0f} min" if m >= 1.5 else f"{abs(s):.0f} s"


def check_change(ev: RaceEvidence, old: float, new: float, name: str = "this race") -> EvidenceCheck:
    if round(new - old, 1) == 0:
        return EvidenceCheck("no_change", "No change.")
    lowering = new < old
    gap = "" if ev.others_median_s is None else _mins(ev.finish_s - ev.others_median_s)
    if ev.direction == 0:
        return EvidenceCheck("your_knowledge",
                             f"The times do not single out {name} (within the usual spread of this runner's races), "
                             f"so this change rests on what you know about it.")
    slow = ev.direction > 0
    if lowering == slow:
        what = f"{gap} slower" if slow else f"{gap} faster"
        return EvidenceCheck("supported", f"The data agrees: {name} was {what} than this runner's other races.")
    if lowering:
        return EvidenceCheck("against_data", f"{name} was one of this runner's faster races ({gap} faster than the "
                                             f"others); counting it less makes the target slower.")
    return EvidenceCheck("against_data", f"{name} was one of this runner's slower races ({gap} slower than the "
                                         f"others); counting it more makes the target slower.")


@dataclass(frozen=True)
class Influence:
    index: int
    pull_s: float            # contribution to the combined prediction, seconds (sums to 0 over races)
    without_s: float | None  # combined prediction minus current if this race were left out
    per_step_s: float        # change in the prediction per +0.1 weight on this race

    def to_dict(self) -> dict:
        return {"index": self.index, "pull_s": self.pull_s, "without_s": self.without_s, "per_step_s": self.per_step_s}


def influence(per_race_finish: Sequence[float | None], weights: Sequence[float]) -> list[Influence]:
    t = np.array([np.nan if x is None else float(x) for x in per_race_finish])
    w = np.where(np.isnan(t), 0.0, np.asarray(weights, float))
    total = w.sum()
    if total <= 0:
        return [Influence(i, 0.0, None, 0.0) for i in range(len(t))]
    tt = np.nan_to_num(t)
    f = float(np.dot(w, tt) / total)
    out = []
    for i in range(len(t)):
        if np.isnan(t[i]):
            out.append(Influence(i, 0.0, None, 0.0))
            continue
        pull = float(w[i] * (t[i] - f) / total)
        rest = total - w[i]
        without = float(np.dot(np.delete(w, i), np.delete(tt, i)) / rest - f) if rest > 0 else None
        step = float(0.1 * (t[i] - f) / (total + 0.1)) if w[i] < 1 else 0.0
        out.append(Influence(i, pull, without, step))
    return out


@dataclass(frozen=True)
class Steering:
    index: float | None
    label: str
    coach_estimate_s: float | None

    def to_dict(self) -> dict:
        return {"index": self.index, "label": self.label, "coach_estimate_s": self.coach_estimate_s}


def steering_index(model_initial_s: float, model_adapted_s: float, coach_estimate_s: float | None,
                   min_gap_s: float = 60.0) -> Steering:
    if coach_estimate_s is None:
        return Steering(None, "Say your own estimate (\"I think she can run 3:25\") to see whose view the model "
                              "is moving towards.", None)
    gap = coach_estimate_s - model_initial_s
    if abs(gap) < min_gap_s:
        return Steering(None, "Your estimate and the model's agreed from the start.", coach_estimate_s)
    a = (model_adapted_s - model_initial_s) / gap
    if abs(a) < 0.15:
        label = "The model is where the data put it."
    elif a < 0:
        label = "The model has moved away from your estimate."
    elif a < 0.85:
        label = f"The model has moved {a:.0%} of the way towards your estimate."
    elif a <= 1.15:
        label = "The model now matches your own estimate. Is that the data, or the sliders agreeing with you?"
    else:
        label = "The model has been pushed beyond your own estimate."
    return Steering(round(float(a), 3), label, coach_estimate_s)


__all__ = ["RaceEvidence", "race_evidence", "EvidenceCheck", "check_change", "Influence", "influence",
           "Steering", "steering_index"]
