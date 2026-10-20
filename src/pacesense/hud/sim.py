"""Race-day simulator for the command center: replay a plan under conditions, with an uncertainty band.

    simulate_race(paces, low_s, high_s, temp_c=24, wind_kmh=15, profile="late_fade")

returns frames every 0.5 km (planned and simulated elapsed time, the 80 % band, the watch cue) and a summary.

Condition effects are ASSUMPTIONS, stated here and shown in the interface, not fitted on data:

* Heat: no effect up to 11 °C; above that, the runner slows by HEAT_PCT_PER_C per °C on average (0.3 %/°C,
  so 25 °C ≈ 4 % slower), growing through the race (half the effect at the start, 1.5x at the finish) and
  larger for slower runners (scaled by sqrt(plan finish / 3:30)). The direction and rough size follow
  Ely et al. (2007, Med Sci Sports Exerc) on marathon performance and temperature; the exact numbers are ours.
* Wind: a headwind component slows by WIND_PCT * v^2 (v in m/s), halved because a course is not all into the
  wind: 18 km/h (5 m/s) ≈ 0.6 % slower.
* The band: the plan scaled to the model's likely range (20th-80th percentile of similar runners), i.e. where
  8 in 10 runners like this would be at each point if they ran this shape of race.
Both constants live in config/pacesense.yaml under `simulator:` so they can be changed and are hashed by
`pacesense reproduce`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .. import racewatch
from ..domain import MARATHON_KM, SEGMENT_KM, ValidationError

HEAT_START_C = 11.0
HEAT_PCT_PER_C = 0.30
WIND_PCT = 0.05


def _settings() -> tuple[float, float, float]:
    try:
        from ..llm.config import load_config
        s = load_config().get("simulator") or {}
    except (ValueError, OSError):
        s = {}
    return (float(s.get("heat_start_c", HEAT_START_C)), float(s.get("heat_pct_per_c", HEAT_PCT_PER_C)),
            float(s.get("wind_pct", WIND_PCT)))


def condition_factor(distance_km: float, temp_c: float, wind_kmh: float, plan_finish_s: float) -> float:
    """Multiplier (>= 1) on the time taken around this point of the race."""
    start, per_c, wind_pct = _settings()
    heat = max(0.0, temp_c - start) * per_c / 100
    heat *= (0.5 + distance_km / MARATHON_KM) * math.sqrt(max(plan_finish_s, 7200) / 12600)
    v = max(0.0, wind_kmh) / 3.6
    wind = wind_pct / 100 * v * v * 0.5
    return 1.0 + heat + wind


@dataclass(frozen=True)
class Frame:
    km: float
    plan_s: float
    sim_s: float
    low_s: float
    high_s: float
    delta_s: float
    pace_s: float
    level: str
    cue: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def simulate_race(paces: Sequence[float], low_s: float | None, high_s: float | None, temp_c: float = 12.0,
                  wind_kmh: float = 0.0, profile: str = "on_plan", seed: int = 7, step_km: float = 0.5) -> dict:
    if not -10 <= temp_c <= 40:
        raise ValidationError("temperature must be between -10 and 40 °C")
    if not 0 <= wind_kmh <= 60:
        raise ValidationError("headwind must be between 0 and 60 km/h")
    p = np.asarray(paces, float)
    plan_finish = float(np.dot(p, SEGMENT_KM))
    trace = racewatch.simulate(list(p), profile, seed)          # (km, elapsed) every 0.1 km, behaviour only
    ds = np.array([d for d, _ in trace] + [MARATHON_KM])
    el = np.array([e for _, e in trace] + [trace[-1][1] + (MARATHON_KM - trace[-1][0]) * p[-1]])
    dt = np.diff(el)
    mid = (ds[:-1] + ds[1:]) / 2
    f = np.array([condition_factor(m, temp_c, wind_kmh, plan_finish) for m in mid])
    sim = np.concatenate([[0.0], np.cumsum(dt * f)])
    lo_k = (low_s / plan_finish) if low_s else 1.0
    hi_k = (high_s / plan_finish) if high_s else 1.0
    frames = []
    for km in list(np.arange(0, MARATHON_KM, step_km)) + [MARATHON_KM]:
        s = float(np.interp(km, ds, sim))
        plan = racewatch.target_elapsed(list(p), float(km))
        st = racewatch.status(list(p), float(km), s)
        j = min(int(np.searchsorted(ds, km, side="right")) - 1, len(dt) - 1)
        pace = float(dt[j] * f[j] / max(ds[j + 1] - ds[j], 1e-9))
        frames.append(Frame(round(float(km), 3), plan, s, plan * lo_k, plan * hi_k, s - plan, pace, st.level, st.cue))
    finish = frames[-1].sim_s
    first_behind = next((fr.km for fr in frames if fr.delta_s > 60), None)
    return {
        "frames": [fr.to_dict() for fr in frames],
        "summary": {"plan_finish_s": plan_finish, "sim_finish_s": finish, "delta_s": finish - plan_finish,
                    "low_s": low_s, "high_s": high_s,
                    "inside_range": bool(low_s is not None and high_s is not None and low_s <= finish <= high_s),
                    "first_minute_behind_km": first_behind, "profile": profile, "profile_text":
                    racewatch.PROFILES[profile], "temp_c": temp_c, "wind_kmh": wind_kmh,
                    "conditions_pct": round(100 * (condition_factor(21.1, temp_c, wind_kmh, plan_finish) - 1), 2)},
        "assumptions": "Heat and wind effects are stated assumptions (see hud/sim.py), not fitted on data.",
    }


__all__ = ["simulate_race", "condition_factor"]
