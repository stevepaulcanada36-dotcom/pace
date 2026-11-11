"""Request bodies for the Studio API. Pydantic validates every request before it reaches the model, and the
same classes generate the OpenAPI document served at /api/v1/openapi.json."""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


COACH_ID = r"^[A-Za-z0-9_\-]{3,64}$"


class CreateSession(_Body):
    runner_id: str = Field("demo", max_length=64, description='A runner id, or "demo" for the built-in example.')
    coach_id: str = Field("anonymous", pattern=COACH_ID,
                          description="A stable, opaque id for the coach (the app keeps one per browser). Persona, "
                                      "track record and layout preferences are stored under it.")
    condition: Literal["known", "unknown", "ask"] = Field(
        "ask", description="Does the coach know this runner? (the paper's familiar / unfamiliar conditions). "
                           "'ask' lets PaceSense ask first.")
    holdout: bool | None = Field(None, description="Hide the runner's best race to measure accuracy "
                                                   "(the paper's protocol). Defaults to true for the demo.")


class VoiceUIState(_Body):
    """Current simulator controls, supplied only as context for a voice command."""
    temp_c: float = Field(12.0, ge=0, le=35)
    wind_kmh: float = Field(0.0, ge=0, le=40)
    profile: Literal["on_plan", "fast_start", "late_fade"] = "on_plan"


class VoiceSignals(_Body):
    """How an utterance was spoken, measured in the browser (all optional)."""
    wpm: float | None = Field(None, ge=0, le=600, description="Words per minute")
    duration_ms: float | None = Field(None, ge=0, le=120000)
    latency_ms: float | None = Field(None, ge=0, le=120000, description="Pause after PaceSense finished speaking")
    barge_in: bool = Field(False, description="The coach started talking while PaceSense was speaking")
    fillers: int = Field(0, ge=0, le=100)
    confidence: float | None = Field(None, ge=0, le=1, description="Speech recognition confidence")
    transcriber: str | None = Field(None, max_length=40, pattern=r"^[A-Za-z0-9:_\- ]*$",
                                    description="Where the words came from: the server provider, or browser:<why>")
    ui_state: VoiceUIState | None = None

    @field_validator("wpm", "duration_ms", "latency_ms", "fillers", "confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any, info: Any) -> Any:
        """These are browser measurements of how something was said. An odd reading (a two-word burst
        measured at 900 words a minute) must not make the server refuse what was said."""
        limits = {"wpm": 600, "duration_ms": 120000, "latency_ms": 120000, "fillers": 100, "confidence": 1}
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        if not math.isfinite(value):
            return None
        return min(limits[info.field_name], value)             # a negative reading is still refused below


class Command(_Body):
    text: str = Field(..., min_length=1, max_length=1000, examples=["2016 was a hot day and she was injured"])
    source: Literal["typed", "voice"] = "typed"
    voice: VoiceSignals | None = None


class SetWeight(_Body):
    index: int = Field(..., ge=0, le=20)
    weight: float = Field(..., ge=0.0, le=1.0)


class Simulate(_Body):
    temp_c: float = Field(12.0, ge=-10, le=40, description="Air temperature on race day, °C")
    wind_kmh: float = Field(0.0, ge=0, le=60, description="Headwind, km/h")
    profile: Literal["on_plan", "fast_start", "late_fade"] = "on_plan"


class React(_Body):
    turn: int = Field(..., ge=1)
    value: Literal["agree", "unsure", "disagree"]
    note: str = Field("", max_length=500)


class Estimate(_Body):
    finish: str = Field(..., pattern=r"^\d:\d{2}(:\d{2})?$", examples=["3:25"],
                        description="The coach's own finish estimate, h:mm or h:mm:ss")


class SessionEvent(_Body):
    kind: Literal["voice_on", "voice_off", "panel", "dismiss"]
    panel: Literal["races", "pacing", "similar", "insights", "simulator", "notes", "profile", "loops"] | None = None


class Condition(_Body):
    condition: Literal["known", "unknown"]


class Choose(_Body):
    index: int = Field(..., ge=0, le=20)


class InterviewAnswer(_Body):
    question: str = Field(..., max_length=32)
    choice: str = Field(..., max_length=32)


class ProfileEdit(_Body):
    """Set a persona dimension (0-1), clear it with null, or reset everything. Optional session id applies
    the change to an open session immediately (the screen re-arranges)."""
    reliance: float | None = Field(None, ge=0, le=1)
    detail: float | None = Field(None, ge=0, le=1)
    voice: float | None = Field(None, ge=0, le=1)
    certainty: float | None = Field(None, ge=0, le=1)
    familiarity: float | None = Field(None, ge=0, le=1)
    layout_lock: Literal["standard", "voice", "analysis", "races", "summary", "guided"] | None = None
    interview_default: bool | None = None
    tempo: float | None = Field(None, ge=0, le=1)
    voice_preset: Literal["quick", "balanced", "calm", "handsfree"] | None = None
    reset: bool = False
    session_id: str | None = Field(None, max_length=64)


BODIES: dict[str, type[BaseModel]] = {"CreateSession": CreateSession, "Command": Command, "SetWeight": SetWeight, "Simulate": Simulate,
          "React": React, "Estimate": Estimate, "SessionEvent": SessionEvent,
          "InterviewAnswer": InterviewAnswer, "ProfileEdit": ProfileEdit, "Condition": Condition,
          "Choose": Choose, "VoiceSignals": VoiceSignals, "VoiceUIState": VoiceUIState}

__all__ = ["BODIES", "COACH_ID", "Choose", "Command", "Condition", "CreateSession", "Estimate", "InterviewAnswer",
           "ProfileEdit", "React", "SessionEvent", "SetWeight", "Simulate", "VoiceSignals", "VoiceUIState"]
