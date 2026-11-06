"""AIService: every AI feature in PaceSense goes through here, each task with its own OpenRouter model.

    voice recording ──transcribe──> text ──extract──> form fields ──check──> fields shown to the coach
                                      └──refine──> clearer version + what changed + questions (coach decides)

Without an OpenRouter key (or with PACESENSE_AI=offline) the text tasks fall back to simple rules and chat
to the offline helper. Speech-to-text defaults to local Whisper, which is free to run but must be installed.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..storage import Repository
from .forms import FORMS, FormCheck, fact_text, facts
from .config import load_config, openrouter_api_key
from .openrouter import LLMError, OpenRouterClient, Transcript, Usage, parse_json_answer
from .tasks import TASKS, ModelSettings


class AIUnavailable(LLMError):
    """The feature needs an OpenRouter key."""


@dataclass
class StepInfo:
    """One AI call, shown under results and recorded by the Model lab."""
    task: str
    model: str
    latency_s: float = 0.0
    usage: Usage = field(default_factory=Usage)


@dataclass
class Extraction:
    check: FormCheck
    step: StepInfo

    @property
    def fields(self) -> dict[str, Any]:
        return self.check.fields


@dataclass
class Refinement:
    text: str
    changes: list[str]
    questions: list[str]
    added: list[str]      # numbers or times in the refined version that were not in the input
    dropped: list[str]    # numbers or times in the input that the refined version left out
    step: StepInfo

    @property
    def keeps_facts(self) -> bool:
        return not self.added and not self.dropped


@dataclass
class VoiceResult:
    transcript: Transcript
    extraction: Extraction | None       # None when that step failed (see errors); the transcript is kept
    refinement: Refinement | None
    errors: list[str] = field(default_factory=list)


EXTRACT_PROMPT = """You fill in a form from what a marathon coach said or typed.
Form: {title}. Return only a JSON object of this shape:
{schema}
Rules: use only what the text says; null (or an empty list) for anything not said; never guess a number.
Keep times as the coach said them if unsure of the format. Today is {today}."""

REFINE_PROMPT = """You help a marathon coach write precise input for PaceSense. The input is {purpose}.
It may be a voice transcript with filler words or recognition mistakes.
Rewrite it as a clearer, more precise version. Rules:
- Keep every fact: names, numbers, times, dates and units. Do not add facts or numbers that are not there.
- Write times as h:mm:ss and dates as YYYY-MM-DD when the input gives them.
- Remove filler words and repetition; keep the coach's meaning and language.
- If something needed seems missing or ambiguous, do not fill it in: ask about it in "questions".
Return only JSON: {{"refined": "the refined text", "changes": ["what you changed, briefly"], "questions": ["..."]}}"""


class AIService:
    def __init__(self, repo: Repository, api_key: str | None = None, env: dict[str, str] | None = None,
                 client: OpenRouterClient | None = None):
        self.repo = repo
        self.env = dict(os.environ if env is None else env)
        self.settings = ModelSettings(repo, self.env)
        self.offline_forced = self.env.get("PACESENSE_AI", "auto") == "offline"
        config = load_config()
        configured_key, configured_source = openrouter_api_key()
        key = api_key or self.env.get("OPENROUTER_API_KEY") or configured_key
        self.key_source = ("browser session" if api_key else
                           "environment" if self.env.get("OPENROUTER_API_KEY") else configured_source)
        provider = config.get("openrouter", {})
        if self.offline_forced:
            self.client = None
        elif client is not None:
            self.client = client
        else:
            self.client = (OpenRouterClient(
                key,
                base_url=str(provider.get("base_url") or "https://openrouter.ai/api/v1"),
                app_name=str(provider.get("app_name") or "PaceSense"),
                app_url=self.env.get("PACESENSE_APP_URL") or provider.get("app_url"),
            ) if key else None)
        self.language = self.env.get("PACESENSE_LANGUAGE") or config.get("voice", {}).get("language") or None

    @property
    def online(self) -> bool:
        return self.client is not None

    def why_offline(self) -> str | None:
        if self.online:
            return None
        if self.offline_forced:
            return "AI is set to offline (PACESENSE_AI=offline)."
        return "No OpenRouter key: chat and form tasks use offline rules; local Whisper can transcribe voice if installed."

    def config(self, task: str, model: str | None = None) -> tuple[str, float]:
        c = self.settings.get(task)
        return model or c["model"], c["temperature"]

    def label(self, task: str) -> str:
        return f"{self.config(task)[0]} via OpenRouter" if self.online else "offline rules (no AI)"

    def _need(self, what: str) -> OpenRouterClient:
        if self.client is None:
            raise AIUnavailable(f"{what} needs an OpenRouter key (OPENROUTER_API_KEY, or paste one in Settings)")
        return self.client

    # ---- tasks ---------------------------------------------------------------------------------
    def chat_model(self, model: str | None = None):
        """The model for Ask PaceSense: OpenRouter with the assistant task's settings, or the offline helper."""
        from ..assistant.chat import OpenRouterModel
        from ..assistant.offline import OfflineModel
        if not self.online:
            return OfflineModel()
        m, temp = self.config("assistant", model)
        return OpenRouterModel(self.client, m, temp)

    @property
    def transcription_backend(self) -> str:
        """The dashboard defaults to local Whisper; Model lab can pass a model to test OpenRouter."""
        return str(self.repo.get_setting("transcription_backend", self.env.get("PACESENSE_TRANSCRIBE_BACKEND", "local")))

    def transcribe(self, audio: bytes, fmt: str = "wav", model: str | None = None) -> Transcript:
        # An explicit model is used by the Model lab to compare OpenRouter models. Ordinary voice input
        # follows the selected backend and defaults to local transcription to avoid API charges.
        if model is None and self.transcription_backend == "local":
            from .local_whisper import transcribe as local_transcribe
            return local_transcribe(audio, fmt, self.language)
        m, _ = self.config("transcribe", model)
        return self._need("Speech to text").transcribe(m, audio, fmt, self.language)

    def extract(self, form: str, text: str, model: str | None = None) -> Extraction:
        f = FORMS[form]
        if not self.online:
            return Extraction(f.check(f.offline(text)), StepInfo("extract", "offline rules"))
        if form == "question":           # nothing to extract: the text is the question
            return Extraction(f.check({"question": text}), StepInfo("extract", "not needed"))
        import datetime as dt
        m, temp = self.config("extract", model)
        r = self.client.chat(m, [
            {"role": "system", "content": EXTRACT_PROMPT.format(title=f.title, schema=f.schema,
                                                                today=dt.date.today().isoformat())},
            {"role": "user", "content": text}], temperature=temp, json_mode=True)
        return Extraction(f.check(parse_json_answer(r.text)), StepInfo("extract", m, r.latency_s, r.usage))

    def refine(self, form: str, text: str, model: str | None = None) -> Refinement:
        f = FORMS[form]
        if not text.strip():
            raise LLMError("there is nothing to refine")
        if not self.online:
            refined = _tidy(text)
            return self._refinement(text, refined, ["Offline: tidied spacing and punctuation only."], [],
                                    StepInfo("refine", "offline rules"))
        m, temp = self.config("refine", model)
        r = self.client.chat(m, [{"role": "system", "content": REFINE_PROMPT.format(purpose=f.purpose)},
                                 {"role": "user", "content": text}], temperature=temp, json_mode=True)
        data = parse_json_answer(r.text)
        refined = str(data.get("refined") or "").strip()
        if not refined:
            raise LLMError("the model returned no refined text")
        return self._refinement(text, refined, [str(x) for x in data.get("changes") or []],
                                [str(x) for x in data.get("questions") or []],
                                StepInfo("refine", m, r.latency_s, r.usage))

    @staticmethod
    def _refinement(original: str, refined: str, changes: list[str], questions: list[str],
                    step: StepInfo) -> Refinement:
        a, b = facts(original), facts(refined)
        return Refinement(refined, changes, questions, sorted(fact_text(x) for x in b - a),
                          sorted(fact_text(x) for x in a - b), step)

    def voice(self, form: str, audio: bytes, fmt: str = "wav") -> VoiceResult:
        """Record → transcribe → fill the form's fields → suggest a refined version.

        Only a failed transcription raises. If filling or refining fails, the transcript (already paid for)
        is still returned, with the error, so the page can show it and offer to try that step again."""
        t = self.transcribe(audio, fmt)
        if not t.text:
            raise LLMError("no speech was recognised in the recording")
        return self.process_text(form, t)

    def process_text(self, form: str, t: Transcript) -> VoiceResult:
        """Fill and refine from a transcript (also used to retry after an error, or after editing it)."""
        out = VoiceResult(t, None, None)
        for name, fn in (("Fill fields", lambda: self.extract(form, t.text)),
                         ("Refine", lambda: self.refine(form, t.text))):
            try:
                value = fn()
            except LLMError as e:
                out.errors.append(f"{name}: {e}")
                continue
            if name == "Fill fields":
                out.extraction = value
            else:
                out.refinement = value
        return out


def _tidy(text: str) -> str:
    t = " ".join(text.split())
    t = re.sub(r"\b(um+|uh+|erm|you know|like,)\s*", "", t, flags=re.I).strip()
    t = re.sub(r"\s+([,.;:?!])", r"\1", t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t and t[-1] not in ".?!":
        t += "."
    return t


def describe_usage(u: Usage) -> str:
    bits = [f"{u.prompt_tokens + u.completion_tokens} tokens"] if (u.prompt_tokens or u.completion_tokens) else []
    if u.audio_seconds:
        bits.append(f"{u.audio_seconds:.0f} s audio")
    if u.cost is not None:
        bits.append(f"${u.cost:.5f}")
    return ", ".join(bits)


__all__ = ["AIService", "AIUnavailable", "Extraction", "Refinement", "StepInfo", "VoiceResult", "TASKS",
           "describe_usage"]
