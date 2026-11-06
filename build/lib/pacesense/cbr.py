"""Case-based reasoning marathon model.

Follows Smyth & Cunningham (2017) as extended in Muijlwijk et al. (2024, Sec. 3.1):

* A *case* pairs a runner's non-personal-best race (nPB) with that runner's
  subsequent personal best (PB).
* For a query race, the k most similar nPB races (same gender; similar finish
  time, pacing profile and age) are retrieved. The predicted finish time is the
  query time scaled by the neighbours' similarity-weighted PB/nPB ratio, and the
  recommended pacing profile is the weighted mean of their PB profiles.
* A runner with several previous races gets one prediction per race; these are
  combined with per-race weights in [0, 1] (default 0.5 each, i.e. equal weights).
  In the interactive condition the coach sets these weights with sliders.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .domain import N_SEGMENTS, Race, Runner, ValidationError, even_profile

log = logging.getLogger(__name__)

DEFAULT_WEIGHT = 0.5


@dataclass(frozen=True)
class CBRConfig:
    k: int = 15
    time_window: float = 0.05          # candidate nPB times within +/- 5 % of the query
    w_time: float = 1.0                # distance per 1 % finish-time difference
    w_profile: float = 0.5             # distance per 1 % RMS pacing-profile difference
    w_age: float = 0.2                 # distance per 10 years of age difference
    missing_profile_distance: float = 0.02
    default_age: int = 40

    def __post_init__(self) -> None:
        if self.k < 1 or not 0 < self.time_window < 1:
            raise ValidationError("invalid CBR configuration")


@dataclass(frozen=True)
class RacePrediction:
    race: Race
    finish_s: float
    profile: np.ndarray | None
    n_neighbours: int
    mean_ratio: float
    case_finish_s: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)  # each analogue's implied finish
    case_weights: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)


# The "likely range" shown to coaches. The 20th-80th percentiles of the similar runners' implied finishes
# contained the held-out result for ~80 % of runners in leave-best-out validation on the demo data
# (78-82 % over two samples of 800). Re-check with `pacesense validate` on real data; it reports coverage.
RANGE_QUANTILES = (0.20, 0.80)
RANGE_LABEL = "80%"


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...]) -> list[float]:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return [float(np.interp(q, cum, v)) for q in qs]


@dataclass(frozen=True)
class Prediction:
    finish_s: float
    profile: np.ndarray
    weights: tuple[float, ...]
    per_race: tuple[RacePrediction | None, ...] = field(repr=False)
    low_s: float = float("nan")
    high_s: float = float("nan")

    @property
    def n_cases(self) -> int:
        return sum(p.n_neighbours for p, w in zip(self.per_race, self.weights) if p is not None and w > 0)


class _GenderIndex:
    """Columnar storage of the cases for one gender, sorted by nPB finish time."""

    def __init__(self, cases: list[tuple]) -> None:
        cases.sort(key=lambda c: c[1])
        n = len(cases)
        self.rid = np.array([c[0] for c in cases], dtype=object)
        self.n_time = np.array([c[1] for c in cases], dtype=float)
        self.age = np.array([c[2] for c in cases], dtype=float)
        self.pb_time = np.array([c[3] for c in cases], dtype=float)
        self.n_prof = np.full((n, N_SEGMENTS), np.nan)
        self.pb_prof = np.full((n, N_SEGMENTS), np.nan)
        for i, c in enumerate(cases):
            if c[4] is not None:
                self.n_prof[i] = c[4]
            if c[5] is not None:
                self.pb_prof[i] = c[5]
        self.size = n


class CaseBase:
    def __init__(self, runners: Sequence[Runner], config: CBRConfig | None = None) -> None:
        self.config = config or CBRConfig()
        buckets: dict[str, list[tuple]] = {"F": [], "M": []}
        for r in runners:
            if len(r.races) < 2:
                continue
            pb_idx = min(range(len(r.races)), key=lambda i: r.races[i].finish_s)
            if pb_idx == 0:
                continue
            pb = r.races[pb_idx]
            for race in r.races[:pb_idx]:
                age = race.age if race.age is not None else (r.age or self.config.default_age)
                buckets[r.gender].append((r.runner_id, race.finish_s, age, pb.finish_s, race.profile(), pb.profile()))
        self._index = {g: _GenderIndex(c) for g, c in buckets.items()}
        log.info("case base: %d female, %d male cases", self._index["F"].size, self._index["M"].size)

    def __len__(self) -> int:
        return self._index["F"].size + self._index["M"].size

    def predict_race(self, race: Race, gender: str, age: int | None = None,
                     exclude_runner: str | None = None) -> RacePrediction | None:
        cfg, idx = self.config, self._index[gender]
        if idx.size == 0:
            return None
        lo = np.searchsorted(idx.n_time, race.finish_s * (1 - cfg.time_window), "left")
        hi = np.searchsorted(idx.n_time, race.finish_s * (1 + cfg.time_window), "right")
        if hi <= lo:
            return None
        sl = slice(lo, hi)
        keep = idx.rid[sl] != exclude_runner if exclude_runner is not None else np.ones(hi - lo, bool)
        if not keep.any():
            return None
        n_time, cand_age = idx.n_time[sl][keep], idx.age[sl][keep]
        rel = np.abs(n_time - race.finish_s) / race.finish_s
        q_prof = race.profile()
        n_prof = idx.n_prof[sl][keep]
        if q_prof is None:
            prof_d = np.full(len(n_time), cfg.missing_profile_distance)
        else:
            prof_d = np.sqrt(np.mean((n_prof - q_prof) ** 2, axis=1))
            prof_d = np.where(np.isnan(prof_d), cfg.missing_profile_distance, prof_d)
        q_age = race.age if race.age is not None else (age or cfg.default_age)
        dist = cfg.w_time * rel / 0.01 + cfg.w_profile * prof_d / 0.01 + cfg.w_age * np.abs(cand_age - q_age) / 10
        k = min(cfg.k, len(dist))
        top = np.argpartition(dist, k - 1)[:k]
        w = 1.0 / (1.0 + dist[top])
        ratio = float(np.sum(w * idx.pb_time[sl][keep][top] / n_time[top]) / w.sum())
        pb_prof = idx.pb_prof[sl][keep][top]
        has = ~np.isnan(pb_prof).any(axis=1)
        profile = (np.sum(w[has, None] * pb_prof[has], axis=0) / w[has].sum()) if has.any() else None
        case_finish = race.finish_s * idx.pb_time[sl][keep][top] / n_time[top]
        return RacePrediction(race, race.finish_s * ratio, profile, k, ratio, case_finish, w / w.sum())


class MarathonModel:
    """Combines per-race CBR predictions with per-race weights."""

    def __init__(self, case_base: CaseBase) -> None:
        self.case_base = case_base

    def per_race(self, runner: Runner, races: Sequence[Race] | None = None) -> tuple[RacePrediction | None, ...]:
        races = list(races) if races is not None else runner.recent(5)
        return tuple(self.case_base.predict_race(r, runner.gender, runner.age, exclude_runner=runner.runner_id)
                     for r in races)

    @staticmethod
    def combine(per_race: Sequence[RacePrediction | None], weights: Sequence[float] | None = None) -> Prediction:
        if weights is None:
            weights = [DEFAULT_WEIGHT] * len(per_race)
        if len(weights) != len(per_race):
            raise ValidationError(f"expected {len(per_race)} weights, got {len(weights)}")
        w = np.asarray(weights, dtype=float)
        if np.any((w < 0) | (w > 1)) or np.any(np.isnan(w)):
            raise ValidationError("weights must be within [0, 1]")
        avail = np.array([p is not None for p in per_race])
        if not (w[avail] > 0).any():
            raise ValidationError("give at least one race with a prediction a weight above 0")
        times = np.array([p.finish_s if p else 0.0 for p in per_race])
        wa = np.where(avail, w, 0.0)
        finish = float(np.dot(wa, times) / wa.sum())
        prof_w, prof_sum = 0.0, np.zeros(N_SEGMENTS)
        for p, wi in zip(per_race, wa):
            if p is not None and p.profile is not None and wi > 0:
                prof_w += wi
                prof_sum += wi * p.profile
        profile = prof_sum / prof_w if prof_w > 0 else even_profile()
        vals, wts = [], []
        for p, wi in zip(per_race, wa):
            if p is not None and wi > 0 and len(p.case_finish_s):
                vals.append(p.case_finish_s)
                wts.append(p.case_weights * wi / wa.sum())
        low = high = float("nan")
        if vals:
            low, high = weighted_quantiles(np.concatenate(vals), np.concatenate(wts), RANGE_QUANTILES)
        return Prediction(finish, profile, tuple(float(x) for x in w), tuple(per_race), low, high)

    def predict(self, runner: Runner, races: Sequence[Race] | None = None,
                weights: Sequence[float] | None = None) -> Prediction:
        return self.combine(self.per_race(runner, races), weights)


def holdout_best(runner: Runner, recent: int = 5) -> tuple[Race, list[Race]]:
    """Paper's accuracy protocol: of the most recent races, hold out the fastest as ground truth."""
    races = runner.recent(recent)
    if len(races) < 2:
        raise ValidationError("need at least 2 races for a hold-out")
    best = min(range(len(races)), key=lambda i: races[i].finish_s)
    return races[best], [r for i, r in enumerate(races) if i != best]


@dataclass(frozen=True)
class ValidationResult:
    n: int
    mean_error_pct: float
    median_error_pct: float
    p90_error_pct: float
    range_coverage_pct: float = float("nan")   # share of held-out results inside the likely range


def validate_model(model: MarathonModel, runners: Sequence[Runner], n: int = 400, seed: int = 7,
                   min_races: int = 3) -> ValidationResult:
    """Leave-best-out validation with equal weights (the non-interactive model)."""
    rng = np.random.default_rng(seed)
    pool = [r for r in runners if len(r.races) >= min_races]
    if not pool:
        raise ValidationError("no runners with enough races to validate")
    chosen = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    errors, inside = [], []
    for i in chosen:
        runner = pool[i]
        truth, shown = holdout_best(runner)
        try:
            pred = model.predict(runner, shown)
        except ValidationError:
            continue
        errors.append(100 * abs(truth.finish_s - pred.finish_s) / truth.finish_s)
        if np.isfinite(pred.low_s):
            inside.append(pred.low_s <= truth.finish_s <= pred.high_s)
    e = np.asarray(errors)
    cov = 100 * float(np.mean(inside)) if inside else float("nan")
    return ValidationResult(len(e), float(e.mean()), float(np.median(e)), float(np.percentile(e, 90)), cov)
