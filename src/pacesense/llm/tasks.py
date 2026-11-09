"""The AI tasks in PaceSense, and which Boson model each one uses.

Each task has its own model and temperature, so a cheap fast model can fill fields while a stronger model
answers questions, and the Model lab can compare models task by task. Where a setting comes from, first
match wins:

    1. Settings page (saved in the database)       "settings"
    2. Environment, e.g. PACESENSE_MODEL_REFINE     "environment"
    3. The default below                            "default"
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..domain import ValidationError
from ..storage import Repository
from .config import load_config


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    what: str
    needs: str                 # what the model must support: "tools", "audio" or "text"
    default_model: str
    default_temperature: float
    suggestions: tuple[str, ...]   # starting points for the Model lab; check the live list for current models

    @property
    def env_model(self) -> str:
        return f"PACESENSE_MODEL_{self.key.upper()}"

    @property
    def env_temperature(self) -> str:
        return f"PACESENSE_TEMPERATURE_{self.key.upper()}"


TASKS: dict[str, Task] = {t.key: t for t in (
    Task("assistant", "Ask PaceSense (chat)",
         "Answers questions about runners and plans by calling PaceSense functions. The model must support "
         "function calling (tools).", "tools", "higgs-realtime", 0.2,
         ("higgs-realtime",)),
    Task("transcribe", "Speech to text",
         "Turns a voice recording into text. Use a transcription model (Whisper-style) or a chat model that "
         "accepts audio.", "audio", "higgs-realtime", 0.0,
         ("higgs-realtime",)),
    Task("extract", "Fill fields",
         "Reads what was said or typed and fills the fields of a form (runner details, plan details, a result). "
         "Answers in JSON.", "text", "higgs-realtime", 0.0,
         ("higgs-realtime",)),
    Task("refine", "Suggest a refined version",
         "Rewrites the input as a clearer, more precise version without changing its facts, and lists what is "
         "unclear. You choose whether to use it.", "text", "higgs-realtime", 0.3,
         ("higgs-realtime",)),
)}


class ModelSettings:
    """Per-task model and temperature, saved in the database under the setting "llm_models"."""

    KEY = "llm_models"

    def __init__(self, repo: Repository, env: dict[str, str] | None = None):
        self.repo = repo
        self.env = dict(os.environ if env is None else env)
        self.config = load_config()

    def _saved(self) -> dict[str, dict]:
        return self.repo.get_setting(self.KEY, {}) or {}

    def get(self, task: str) -> dict[str, Any]:
        """{"model", "temperature", "source"} for one task."""
        t = TASKS[task]
        saved = self._saved().get(task, {})
        if saved.get("model"):
            model, source = saved["model"], "settings"
        elif self.env.get(t.env_model):
            model, source = self.env[t.env_model], "environment"
        else:
            configured = self.config.get("models", {}).get(task, {})
            model = str(configured.get("model") or t.default_model)
            source = "YAML config" if model != t.default_model else "default"
        temp = saved.get("temperature")
        if temp is None:
            try:
                configured = self.config.get("models", {}).get(task, {})
                temp = (float(self.env[t.env_temperature]) if t.env_temperature in self.env
                        else float(configured.get("temperature", t.default_temperature)))
            except ValueError:
                temp = t.default_temperature
        return {"model": model, "temperature": float(temp), "source": source}

    def all(self) -> dict[str, dict[str, Any]]:
        return {k: self.get(k) for k in TASKS}

    def update(self, task: str, model: str | None = None, temperature: float | None = None) -> dict[str, Any]:
        if task not in TASKS:
            raise ValidationError(f"unknown task {task!r}; tasks are {sorted(TASKS)}")
        saved = self._saved()
        entry = dict(saved.get(task, {}))
        if model is not None:
            model = model.strip()
            if model and ("/" not in model or " " in model):
                raise ValidationError("an Boson model id looks like provider/model, e.g. openai/gpt-4o-mini")
            entry["model"] = model or None
        if temperature is not None:
            if not 0 <= float(temperature) <= 2:
                raise ValidationError("temperature must be between 0 and 2")
            entry["temperature"] = round(float(temperature), 2)
        saved[task] = entry
        self.repo.set_setting(self.KEY, saved)
        return self.get(task)

    def has_saved(self, task: str) -> bool:
        return bool(self._saved().get(task))

    def reset(self, task: str) -> dict[str, Any]:
        saved = self._saved()
        saved.pop(task, None)
        self.repo.set_setting(self.KEY, saved)
        return self.get(task)
