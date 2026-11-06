"""What the adaptive layer observes: one `Event` per thing the coach does, and one `Adaptation` per thing the
system changes in response. Both are plain data so they can be logged, replayed and shown to the coach."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

EVENT_KINDS = ("session_start", "utterance", "slider", "undo", "reaction", "reveal", "simulate", "voice_on",
               "voice_off", "panel", "interview_answer", "persona_edit", "condition", "prompt_answer")


@dataclass
class Event:
    kind: str
    text: str = ""
    source: str = "typed"                    # typed | voice | manual | system
    data: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")

    # convenient views on `data`
    @property
    def applied(self) -> list[dict]:
        return list(self.data.get("applied") or [])

    @property
    def verdicts(self) -> list[str]:
        return [a.get("verdict", "") for a in self.applied]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Adaptation:
    """One change the system makes, and why. `kind` says where it lands:

    ui       the screen rearranges (layout, spotlight, explanation depth)
    prompt   PaceSense asks the coach something (with choices to tap or say)
    insight  a short message in the coach notes
    persona  the coach profile changed
    """
    loop: str
    kind: str
    message: str
    why: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["EVENT_KINDS", "Adaptation", "Event"]
