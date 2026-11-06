"""Core domain objects: races with 5 km split segments, and runners."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

MARATHON_KM = 42.195
SEGMENT_KM: tuple[float, ...] = (5.0,) * 8 + (2.195,)
SEGMENT_LABELS: tuple[str, ...] = ("0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-42.2")
N_SEGMENTS = len(SEGMENT_KM)
_SEG = np.asarray(SEGMENT_KM)

MIN_FINISH_S = 2 * 3600
MAX_FINISH_S = 7 * 3600

Gender = Literal["F", "M"]


class ValidationError(ValueError):
    """Raised when input data is invalid."""


def parse_time(value: object) -> float | None:
    """Parse ``h:mm:ss``, ``h:mm`` or plain seconds. Returns seconds or ``None``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if (isinstance(value, float) and math.isnan(value)) else float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" not in text:
            return float(text)
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 3600 + parts[1] * 60
    return None


def format_time(seconds: float | None, with_seconds: bool = True) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-:--:--" if with_seconds else "-:--"
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if with_seconds else f"{h}:{m:02d}"


def is_valid_finish(seconds: float | None) -> bool:
    return seconds is not None and MIN_FINISH_S <= seconds <= MAX_FINISH_S


@dataclass(frozen=True, slots=True)
class Race:
    """One marathon result. ``segments_s`` holds the 9 segment durations (5 km x 8 + 2.195 km)."""

    event: str
    year: int
    finish_s: float
    age: int | None = None
    segments_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not is_valid_finish(self.finish_s):
            raise ValidationError(f"finish time {self.finish_s!r} outside {MIN_FINISH_S}-{MAX_FINISH_S} s")
        if self.segments_s is not None:
            if len(self.segments_s) != N_SEGMENTS or any(not (s > 0) for s in self.segments_s):
                raise ValidationError("segments_s must be 9 positive durations")

    @classmethod
    def from_cumulative_splits(cls, event: str, year: int, finish_s: float, splits_s: Sequence[float] | None,
                               age: int | None = None) -> "Race":
        """Build a race from the elapsed times at 5, 10, ... 40 km (8 values)."""
        segments = None
        if splits_s is not None:
            if len(splits_s) != 8:
                raise ValidationError("expected 8 cumulative splits (5 to 40 km)")
            cum = [*splits_s, finish_s]
            segments = tuple(cum[i] - (cum[i - 1] if i else 0.0) for i in range(N_SEGMENTS))
        return cls(event=event, year=year, finish_s=float(finish_s), age=age, segments_s=segments)

    def profile(self) -> np.ndarray | None:
        """Relative pace per segment (segment pace / average pace); ``None`` without splits."""
        if self.segments_s is None:
            return None
        avg_pace = self.finish_s / MARATHON_KM
        return (np.asarray(self.segments_s) / _SEG) / avg_pace


@dataclass(slots=True)
class Runner:
    runner_id: str
    gender: Gender
    races: list[Race] = field(default_factory=list)
    name: str = ""
    age: int | None = None

    def __post_init__(self) -> None:
        if self.gender not in ("F", "M"):
            raise ValidationError(f"gender must be 'F' or 'M', got {self.gender!r}")
        self.races.sort(key=lambda r: (r.year, r.finish_s))
        if self.age is None:
            ages = [r.age for r in self.races if r.age is not None]
            self.age = ages[-1] if ages else None

    @property
    def display_name(self) -> str:
        return self.name or self.runner_id

    def recent(self, n: int = 5) -> list[Race]:
        return self.races[-n:]


def pace_per_km(finish_s: float, profile: np.ndarray) -> np.ndarray:
    """Seconds per km for each segment given a finish time and a relative profile."""
    return profile * (finish_s / MARATHON_KM)


def even_profile() -> np.ndarray:
    return np.ones(N_SEGMENTS)


def linear_pacing_profile(strategy: float) -> np.ndarray:
    """Coach pacing slider: -1 = end fast (negative split), 0 = even, +1 = start fast."""
    if not -1.0 <= strategy <= 1.0:
        raise ValidationError("pacing strategy must be within [-1, 1]")
    j = np.arange(N_SEGMENTS)
    p = 1 - strategy * 0.06 * (1 - 2 * j / 8)
    return p / (np.dot(p, _SEG) / MARATHON_KM)
