"""Tune the case-based model's settings on your data, without fooling yourself.

    pacesense tune --data data/boston.csv                # prints the best settings and the honest error
    pacesense tune --data data/boston.csv --write        # also saves them in config/pacesense.yaml (model:)

A grid of settings (number of similar runners k, the finish-time window, how much the pacing profile and age
count) is scored with the paper's protocol: hold out each runner's fastest recent race and predict it from the
rest (Sec. 4.2.2). Settings are *chosen* on one seeded sample of runners and *reported* on a different, disjoint
sample, so the reported error is not flattered by the search. Every run with the same data and seed gives the
same answer.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .cbr import CBRConfig, CaseBase, MarathonModel, holdout_best
from .domain import Runner, ValidationError

GRID = {"k": (5, 10, 15, 25), "time_window": (0.03, 0.05, 0.08), "w_profile": (0.0, 0.5, 1.0), "w_age": (0.0, 0.2)}


def _error(model: MarathonModel, runners: Sequence[Runner]) -> tuple[float, int]:
    errs = []
    for r in runners:
        truth, shown = holdout_best(r)
        try:
            p = model.predict(r, shown)
        except ValidationError:
            continue
        errs.append(100 * abs(truth.finish_s - p.finish_s) / truth.finish_s)
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


@dataclass
class TuneResult:
    best: dict
    default: dict
    tune_error_pct: float
    test_error_pct: float
    default_test_error_pct: float
    n_tune: int
    n_test: int
    seed: int
    grid: list[dict]
    seconds: float

    def text(self) -> str:
        b = ", ".join(f"{k}={v}" for k, v in self.best.items())
        gain = self.default_test_error_pct - self.test_error_pct
        verdict = ("no better than the defaults: keep them" if gain < 0.005 else
                   f"{gain:.2f} points better than the defaults")
        return (f"Best settings: {b}\n"
                f"Held-out test error: {self.test_error_pct:.2f}% vs {self.default_test_error_pct:.2f}% with the default "
                f"settings ({verdict}), on {self.n_test} runners not used for choosing.\n"
                f"(Chosen on {self.n_tune} other runners; seed {self.seed}; {self.seconds:.0f} s.)")

    @property
    def improves(self) -> bool:
        return self.default_test_error_pct - self.test_error_pct >= 0.005


def tune(runners: Sequence[Runner], n: int = 300, seed: int = 20240318, min_races: int = 3,
         grid: dict | None = None, log=lambda m: None) -> TuneResult:
    t0 = time.time()
    grid = grid or GRID
    pool = sorted((r for r in runners if len(r.recent(5)) >= min_races), key=lambda r: r.runner_id)
    if len(pool) < 40:
        raise ValidationError(f"need at least 40 runners with {min_races}+ races to tune (found {len(pool)})")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    half = min(n, len(pool) // 2)
    tune_set, test_set = [pool[i] for i in order[:half]], [pool[i] for i in order[half:2 * half]]
    rows = []
    best = None
    combos = list(itertools.product(*grid.values()))
    for j, values in enumerate(combos):
        cfg = CBRConfig(**dict(zip(grid.keys(), values)))
        err, used = _error(MarathonModel(CaseBase(runners, cfg)), tune_set)
        rows.append({**dict(zip(grid.keys(), values)), "tune_error_pct": round(err, 4), "n": used})
        if np.isfinite(err) and (best is None or err < best[0] - 1e-12):
            best = (err, cfg)
        if (j + 1) % 12 == 0:
            log(f"  {j + 1}/{len(combos)} settings tried")
    err_best, cfg = best
    test_err, n_test = _error(MarathonModel(CaseBase(runners, cfg)), test_set)
    default = CBRConfig()
    def_err, _ = _error(MarathonModel(CaseBase(runners, default)), test_set)
    pick = {k: getattr(cfg, k) for k in grid}
    return TuneResult(pick, {k: getattr(default, k) for k in grid}, err_best, test_err, def_err, len(tune_set), n_test,
                      seed, rows, time.time() - t0)


def write_model_config(settings: dict, path: Path) -> None:
    """Replace (or add) the `model:` section of config/pacesense.yaml, keeping everything else and its comments."""
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = "model:   # case-based model settings, chosen by `pacesense tune`\n" + "".join(
        f"  {k}: {v}\n" for k, v in settings.items())
    lines, out, skipping = text.splitlines(keepends=True), [], False
    for ln in lines:
        if ln.startswith("model:"):
            skipping = True
            continue
        if skipping and (ln.startswith(" ") or not ln.strip()):
            continue
        skipping = False
        out.append(ln)
    new = "".join(out).rstrip() + "\n\n" + block
    path.write_text(new, encoding="utf-8")


def configured_cbr() -> CBRConfig:
    """CBRConfig from the `model:` section of config/pacesense.yaml (defaults where absent)."""
    try:
        from .llm.config import load_config
        section = load_config().get("model") or {}
    except (ValueError, OSError):
        section = {}
    allowed = set(asdict(CBRConfig()))
    return CBRConfig(**{k: v for k, v in section.items() if k in allowed})


__all__ = ["tune", "TuneResult", "write_model_config", "configured_cbr", "GRID"]
