"""Race-day watch: what a runner would see on a watch while running a coach's plan.

A plan is nine paces (seconds per km), one per 5 km interval with a final 2.195 km. From it this module
derives kilometre splits (what a watch alerts on), the runner's status at any moment of the race (ahead or
behind, target pace, next split, projected finish and a short cue), a simulated race for demonstrations, and
exports for a pace band.

Everything here is plain arithmetic on the plan: nothing calls the prediction model.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

from .domain import MARATHON_KM, SEGMENT_KM, ValidationError, format_time

_EDGES = np.concatenate([[0.0], np.cumsum(SEGMENT_KM)])        # 0, 5, 10, ... 40, 42.195
ON_PLAN_S = 15          # within this many seconds of plan counts as "on plan"
SLIPPING_PCT = 0.02     # required pace this much faster than plan pace means the target is slipping


def _check(paces: Sequence[float]) -> np.ndarray:
    p = np.asarray(paces, dtype=float)
    if p.shape != (len(SEGMENT_KM),) or np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValidationError(f"a plan needs {len(SEGMENT_KM)} positive paces")
    return p


def pace_at(paces: Sequence[float], distance_km: float) -> float:
    """Planned pace (s/km) for the interval containing `distance_km`."""
    p = _check(paces)
    d = min(max(distance_km, 0.0), MARATHON_KM - 1e-9)
    return float(p[np.searchsorted(_EDGES, d, side="right") - 1])


def target_elapsed(paces: Sequence[float], distance_km: float) -> float:
    """Planned elapsed time (s) on reaching `distance_km`."""
    p = _check(paces)
    d = min(max(distance_km, 0.0), MARATHON_KM)
    seg_times = p * np.asarray(SEGMENT_KM)
    i = int(np.searchsorted(_EDGES, d, side="right") - 1)
    if i >= len(SEGMENT_KM):
        return float(seg_times.sum())
    return float(seg_times[:i].sum() + (d - _EDGES[i]) * p[i])


@dataclass(frozen=True)
class Split:
    marker: str            # "1 km" ... "42 km", "Finish"
    distance_km: float
    pace_s: float          # planned pace for this kilometre (s/km)
    split_s: float         # time for this kilometre (or the last 0.195 km)
    elapsed_s: float       # planned elapsed time at the marker


def km_splits(paces: Sequence[float]) -> list[Split]:
    """One split per kilometre plus the finish: what a watch's lap alerts follow."""
    _check(paces)
    marks = [float(k) for k in range(1, 43)] + [MARATHON_KM]
    out, prev = [], 0.0
    for d in marks:
        el = target_elapsed(paces, d)
        out.append(Split("Finish" if d == MARATHON_KM else f"{int(d)} km", d, pace_at(paces, d - 1e-6), el - prev, el))
        prev = el
    return out


Level = Literal["start", "on", "ahead", "behind", "slipping", "done"]


@dataclass(frozen=True)
class WatchStatus:
    distance_km: float
    elapsed_s: float
    target_elapsed_s: float
    delta_s: float              # actual minus plan: negative means ahead of plan
    target_pace_s: float        # planned pace for the current interval (s/km)
    next_marker: str
    next_marker_km: float
    next_marker_target_s: float
    projected_finish_s: float   # elapsed + the rest of the plan as written
    required_pace_s: float      # average pace needed from here to finish on the plan's time
    plan_finish_s: float
    level: Level
    cue: str

    @property
    def progress(self) -> float:
        return min(1.0, self.distance_km / MARATHON_KM)


def _clock(s: float) -> str:
    r = int(round(abs(s)))
    return f"{r // 3600}:{r % 3600 // 60:02d}:{r % 60:02d}" if r >= 3600 else f"{r // 60}:{r % 60:02d}"


def _pace(s_per_km: float, unit: str) -> str:
    r = int(round(s_per_km * (1.609344 if unit == "mi" else 1.0)))
    return f"{r // 60}:{r % 60:02d}/{unit}"


def status(paces: Sequence[float], distance_km: float, elapsed_s: float, unit: str = "km") -> WatchStatus:
    """The watch screen at a moment in the race."""
    p = _check(paces)
    if not 0 <= distance_km <= MARATHON_KM + 1e-9:
        raise ValidationError("distance must be between 0 and 42.195 km")
    if elapsed_s < 0:
        raise ValidationError("elapsed time cannot be negative")
    d = min(distance_km, MARATHON_KM)
    plan_finish = float(np.dot(p, SEGMENT_KM))
    target = target_elapsed(p, d)
    delta = elapsed_s - target
    remaining_km = MARATHON_KM - d
    remaining_plan = plan_finish - target
    projected = elapsed_s + remaining_plan
    required = (plan_finish - elapsed_s) / remaining_km if remaining_km > 1e-9 else float("nan")
    tp = pace_at(p, d)
    nxt = MARATHON_KM if d >= 42 else float(math.floor(d) + 1)
    nxt_label = "Finish" if nxt == MARATHON_KM else f"{int(nxt)} km"

    if d >= MARATHON_KM - 1e-9:
        level, cue = "done", f"Finished in {_clock(elapsed_s)}. Plan was {_clock(plan_finish)} ({_signed(delta)})."
    elif d < 0.05 and elapsed_s < 30:
        level, cue = "start", f"Start at {_pace(tp, unit)}. Target finish {_clock(plan_finish)}."
    elif abs(delta) <= ON_PLAN_S:
        level, cue = "on", f"On plan. Hold {_pace(tp, unit)}."
    elif delta < 0:
        if d < 30:
            level, cue = "ahead", f"{_clock(delta)} ahead of plan. Settle back to {_pace(tp, unit)}."
        else:
            level, cue = "ahead", f"{_clock(delta)} ahead with {remaining_km:.1f} km to go. Hold {_pace(tp, unit)}."
    elif remaining_km > 1e-9 and required < tp * (1 - SLIPPING_PCT):
        level, cue = ("slipping", f"{_clock(delta)} behind. Plan pace from here finishes in about "
                                  f"{_clock(projected)}; {_pace(required, unit)} needed for {_clock(plan_finish)}.")
    else:
        level, cue = "behind", f"{_clock(delta)} behind plan. {_pace(required, unit)} from here finishes on time."
    return WatchStatus(d, float(elapsed_s), target, delta, tp, nxt_label, nxt, target_elapsed(p, nxt), projected,
                       required, plan_finish, level, cue)


def _signed(delta: float) -> str:
    r = round(delta)
    if r == 0:
        return "on the dot"
    return f"{_clock(r)} {'faster' if r < 0 else 'slower'}"


# ---- simulation ----------------------------------------------------------------------------------
Profile = Literal["on_plan", "fast_start", "late_fade"]
PROFILES: dict[str, str] = {
    "on_plan": "Runs the plan, give or take a few seconds per km",
    "fast_start": "Goes out 3% fast for 10 km, then slows by 5% over the last 12 km",
    "late_fade": "On plan to 30 km, then slows by 6%",
}


def simulate(paces: Sequence[float], profile: str = "on_plan", seed: int = 7) -> list[tuple[float, float]]:
    """A made-up race for demonstrations: (distance km, elapsed s) every 0.1 km from start to finish."""
    p = _check(paces)
    if profile not in PROFILES:
        raise ValidationError(f"profile must be one of {sorted(PROFILES)}")
    rng = np.random.default_rng(seed)
    step = 0.1
    ds = np.round(np.arange(0, MARATHON_KM, step), 3)
    ds = np.append(ds, MARATHON_KM)
    track, t = [(0.0, 0.0)], 0.0
    km_noise = rng.normal(0, 0.008, 43)              # one wobble per kilometre
    for a, b in zip(ds[:-1], ds[1:]):
        base = pace_at(p, (a + b) / 2)
        f = 1.0 + km_noise[int(a)]
        if profile == "fast_start":
            f *= 0.97 if a < 10 else (1.05 if a >= 30 else 1.0)
        elif profile == "late_fade":
            f *= 1.06 if a >= 30 else 1.0
        t += base * f * (b - a)
        track.append((float(b), float(t)))
    return track


def position_at(track: Sequence[tuple[float, float]], race_clock_s: float) -> tuple[float, float]:
    """Where a simulated runner is after `race_clock_s` seconds: (distance km, elapsed s)."""
    if race_clock_s <= 0:
        return 0.0, 0.0
    times = [t for _, t in track]
    if race_clock_s >= times[-1]:
        return track[-1]
    i = int(np.searchsorted(times, race_clock_s))
    (d0, t0), (d1, t1) = track[i - 1], track[i]
    return d0 + (d1 - d0) * (race_clock_s - t0) / (t1 - t0), float(race_clock_s)


# ---- exports -------------------------------------------------------------------------------------
def pace_band_text(paces: Sequence[float], title: str = "", unit: str = "km", every_km: int = 1) -> str:
    """A printable pace band: target elapsed time at each kilometre (or every n km)."""
    rows = [s for s in km_splits(paces) if s.marker == "Finish" or int(s.distance_km) % every_km == 0]
    lines = [title or "Pace band", f"Target finish {format_time(rows[-1].elapsed_s)}", ""]
    lines += [f"{s.marker:>7}  {format_time(s.elapsed_s)}   {_pace(s.pace_s, unit)}" for s in rows]
    return "\n".join(lines) + "\n"


def splits_csv(paces: Sequence[float]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["marker", "distance_km", "pace_s_per_km", "split_s", "elapsed_s", "elapsed"])
    for s in km_splits(paces):
        w.writerow([s.marker, f"{s.distance_km:.3f}", f"{s.pace_s:.1f}", f"{s.split_s:.1f}", f"{s.elapsed_s:.1f}",
                    format_time(s.elapsed_s)])
    return buf.getvalue()


def plan_json(paces: Sequence[float], runner: str = "", race: str = "", race_date: str | None = None) -> str:
    """A generic JSON description of the plan for a watch app. Not a specific watch brand's format."""
    splits = km_splits(paces)
    return json.dumps({"format": "pacesense-plan/1", "runner": runner, "race": race, "race_date": race_date,
                       "target_finish_s": round(splits[-1].elapsed_s, 1),
                       "intervals": [{"from_km": float(a), "to_km": float(b), "pace_s_per_km": round(float(x), 1)}
                                     for a, b, x in zip(_EDGES[:-1], _EDGES[1:], _check(paces))],
                       "km_splits": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(s).items()}
                                     for s in splits]}, indent=2)
