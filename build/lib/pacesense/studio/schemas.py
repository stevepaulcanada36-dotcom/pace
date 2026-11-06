"""Request bodies for the Studio API. Pydantic validates every request before it reaches the model, and the
same classes generate the OpenAPI document served at /api/v1/openapi.json."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CreateSession(_Body):
    runner_id: str = Field("demo", max_length=64, description='A runner id, or "demo" for the built-in example.')
    holdout: bool | None = Field(None, description="Hide the runner's best race to measure accuracy "
                                                   "(the paper's protocol). Defaults to true for the demo.")


class Command(_Body):
    text: str = Field(..., min_length=1, max_length=1000, examples=["2016 was a hot day and she was injured"])
    source: Literal["typed", "voice"] = "typed"


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


BODIES = {"CreateSession": CreateSession, "Command": Command, "SetWeight": SetWeight, "Simulate": Simulate,
          "React": React, "Estimate": Estimate}

__all__ = ["BODIES", "Command", "CreateSession", "Estimate", "React", "SetWeight", "Simulate"]
