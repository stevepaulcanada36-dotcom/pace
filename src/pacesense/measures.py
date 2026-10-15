"""Outcome measures from Muijlwijk et al. (2024), Sections 4.2 and 5.7."""
from __future__ import annotations

from dataclasses import asdict, dataclass

ROUNDING_S = 300  # coaches rounded to 5 minutes; equal after rounding counts as full acceptance

ADAPTATION_CLASSES = {
    0: "Non-interactive condition",
    1: "No adaptation (<= 0.25)",
    2: "Some adaptation (0.25-0.75)",
    3: "Full adaptation (0.75-1.25)",
    4: "Over adaptation (> 1.25)",
}


def _round(s: float, step: int = ROUNDING_S) -> float:
    return round(s / step) * step


def acceptance(model_s: float, coach_pre_s: float, coach_post_s: float, rounding_s: int = ROUNDING_S) -> float | None:
    """Weight-of-advice style acceptance in [0, 1].

    ``model_s`` is Model_initial in the non-interactive condition and Model_adapted in
    the interactive condition. Returns ``None`` when undefined (coach_pre equals the model
    and the coach did not end on the model either).
    """
    if rounding_s and _round(coach_post_s, rounding_s) == _round(model_s, rounding_s):
        return 1.0
    denom = abs(model_s - coach_pre_s)
    if denom == 0:
        return None
    return max(0.0, 1.0 - abs(model_s - coach_post_s) / denom)


def model_adaptation(model_initial_s: float, model_adapted_s: float, coach_pre_s: float) -> float | None:
    """How far the coach moved the model towards their own initial prediction (0 = not, 1 = fully)."""
    denom = abs(coach_pre_s - model_initial_s)
    if denom == 0:
        return None
    return abs(model_adapted_s - model_initial_s) / denom


def adaptation_class(value: float | None, interactive: bool) -> int | None:
    if not interactive:
        return 0
    if value is None:
        return None
    if value <= 0.25:
        return 1
    if value <= 0.75:
        return 2
    if value <= 1.25:
        return 3
    return 4


def percent_error(truth_s: float | None, predicted_s: float | None) -> float | None:
    if truth_s is None or predicted_s is None:
        return None
    return 100.0 * abs(truth_s - predicted_s) / truth_s


@dataclass(frozen=True)
class TrialMeasures:
    acceptance: float | None
    acceptance_binary: int | None
    model_adaptation: float | None
    adaptation_class: int | None
    err_model_initial: float | None
    err_model_adapted: float | None
    err_coach_pre: float | None
    err_coach_post: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def compute_trial_measures(*, interactive: bool, coach_pre_s: float, model_initial_s: float,
                           model_adapted_s: float | None, coach_post_s: float,
                           truth_s: float | None = None) -> TrialMeasures:
    model_final = model_adapted_s if (interactive and model_adapted_s is not None) else model_initial_s
    acc = acceptance(model_final, coach_pre_s, coach_post_s)
    adapt = model_adaptation(model_initial_s, model_final, coach_pre_s) if interactive else None
    return TrialMeasures(
        acceptance=acc,
        acceptance_binary=None if acc is None else int(acc >= 0.5),
        model_adaptation=adapt,
        adaptation_class=adaptation_class(adapt, interactive),
        err_model_initial=percent_error(truth_s, model_initial_s),
        err_model_adapted=percent_error(truth_s, model_final) if interactive else None,
        err_coach_pre=percent_error(truth_s, coach_pre_s),
        err_coach_post=percent_error(truth_s, coach_post_s),
    )
