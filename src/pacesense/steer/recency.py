"""Fit the "trust recent races more" profile on data instead of guessing it.

The paper found that coaches set more recent races as more representative (slider position rose with race
year, beta = 0.042, p < .001; Sec. 5.9). PaceSense Live turns "trust the recent races more" into weights
2 ** (-years before the latest race / half_life). `fit_half_life` picks half_life by the paper's own accuracy
protocol (hold out each runner's fastest recent race, predict it from the rest; Sec. 4.2.2) over a fixed grid
and a seeded sample, so the result is identical on every run and can be checked with `pacesense reproduce`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..cbr import MarathonModel, holdout_best
from ..domain import Runner, ValidationError
from .intent import recency_weights

DEFAULT_GRID = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, math.inf)


@dataclass(frozen=True)
class RecencyFit:
    best_half_life: float
    errors: dict[float, float]          # half-life -> mean percent error
    equal_weights_error: float
    n: int
    seed: int

    def to_dict(self) -> dict:
        fmt = lambda h: "equal" if math.isinf(h) else h
        return {"best_half_life": fmt(self.best_half_life), "n": self.n, "seed": self.seed,
                "equal_weights_error_pct": round(self.equal_weights_error, 4),
                "errors_pct": {str(fmt(h)): round(e, 4) for h, e in self.errors.items()}}


def fit_half_life(model: MarathonModel, runners: Sequence[Runner], grid: Sequence[float] = DEFAULT_GRID,
                  n: int = 300, seed: int = 20240318, min_races: int = 4, floor: float = 0.2) -> RecencyFit:
    rng = np.random.default_rng(seed)
    pool = [r for r in runners if len(r.recent(5)) >= min_races]
    if not pool:
        raise ValidationError("no runners with enough races to fit the recency profile")
    chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]
    errs: dict[float, list[float]] = {h: [] for h in grid}
    used = 0
    for runner in chosen:
        truth, shown = holdout_best(runner)
        per = model.per_race(runner, shown)
        if not any(p is not None for p in per):
            continue
        row = {}
        try:
            for h in grid:
                w = [0.5] * len(shown) if math.isinf(h) else recency_weights([r.year for r in shown], h, floor)
                pred = MarathonModel.combine(per, w)
                row[h] = 100 * abs(truth.finish_s - pred.finish_s) / truth.finish_s
        except ValidationError:
            continue
        for h, e in row.items():
            errs[h].append(e)
        used += 1
    means = {h: float(np.mean(v)) for h, v in errs.items() if v}
    best = min(means, key=lambda h: (round(means[h], 6), h))
    eq = means.get(math.inf, float("nan"))
    return RecencyFit(best, means, eq, used, seed)


__all__ = ["RecencyFit", "fit_half_life", "DEFAULT_GRID"]
