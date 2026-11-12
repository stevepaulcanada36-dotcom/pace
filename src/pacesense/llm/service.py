"""AIService: every AI feature in PaceSense goes through here, each task through Boson Higgs Realtime.

    voice recording ──transcribe──> text ──extract──> form fields ──check──> fields shown to the coach
                                      └──refine──> clearer version + what changed + questions (coach decides)

Without a Boson API key (or with PACESENSE_AI=offline) the text tasks fall back to simple rules and chat
to the offline helper. Speech-to-text uses Boson by default; local Whisper remains an optional fallback.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..storage import Repository
from .forms import FORMS, FormCheck, fact_text, facts
from .config import boson_api_key, load_config, openrouter_api_key
from .openrouter import LLMError, OpenRouterClient, Transcript, Usage, parse_json_answer
from .boson import BosonClient
from .tasks import TASKS, ModelSettings


class AIUnavailable(LLMError):
    """The feature needs a Boson API key."""


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


log = logging.getLogger("pacesense.llm")

PROVIDER_NAMES = {"openrouter": "OpenRouter", "boson": "Boson"}
# The models PaceSense used with OpenRouter before the Boson integration. Override per deployment with
# PACESENSE_OPENROUTER_MODEL (chat, interpretation, forms) and PACESENSE_OPENROUTER_TRANSCRIBE_MODEL.
OPENROUTER_CHAT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_TRANSCRIBE_MODEL = "openai/whisper-1"
BOSON_MODEL = "higgs-realtime"
# After a provider rejects its key or runs out of credit, skip it for a while instead of failing every turn.
_PROVIDER_COOLDOWN_S = 300.0
_provider_down_until: dict[str, float] = {}


def _is_account_failure(error: LLMError) -> bool:
    return getattr(error, "status", None) in (401, 402, 403)


class FailoverClient:
    """Use the primary provider; if a call fails, answer that call with the backup provider.

    The two providers name their models differently, so each call is given the model for whichever
    provider handles it. A key or credit failure parks the primary for a few minutes.
    """

    def __init__(self, primary: Any, primary_name: str, backup: Any, backup_name: str,
                 backup_model: str, backup_transcribe_model: str) -> None:
        self.primary, self.primary_name = primary, primary_name
        self.backup, self.backup_name = backup, backup_name
        self.backup_model, self.backup_transcribe_model = backup_model, backup_transcribe_model

    def _primary_available(self) -> bool:
        return time.monotonic() >= _provider_down_until.get(self.primary_name, 0.0)

    def _failed(self, what: str, error: LLMError) -> None:
        if _is_account_failure(error):
            _provider_down_until[self.primary_name] = time.monotonic() + _PROVIDER_COOLDOWN_S
        log.warning("%s %s failed (%s); using %s for this request", PROVIDER_NAMES[self.primary_name], what,
                    error, PROVIDER_NAMES[self.backup_name])

    def chat(self, model: str, messages: list[dict], **kwargs: Any):
        if self._primary_available():
            try:
                return self.primary.chat(model, messages, **kwargs)
            except LLMError as error:
                self._failed("chat", error)
        return self.backup.chat(self.backup_model, messages, **kwargs)

    def transcribe(self, model: str, audio: bytes, fmt: str = "wav", language: str | None = None):
        if self._primary_available():
            try:
                return self.primary.transcribe(model, audio, fmt, language)
            except LLMError as error:
                self._failed("transcription", error)
        return self.backup.transcribe(self.backup_transcribe_model, audio, fmt, language)

    def __getattr__(self, name: str) -> Any:            # models(), and anything else, from the primary
        return getattr(self.primary, name)


class AIService:
    def __init__(self, repo: Repository, api_key: str | None = None, env: dict[str, str] | None = None,
                 client: Any | None = None):
        self.repo = repo
        self.env = dict(os.environ if env is None else env)
        self.settings = ModelSettings(repo, self.env)
        self.offline_forced = self.env.get("PACESENSE_AI", "auto") == "offline"
        config = load_config()
        session_key = api_key.strip() if isinstance(api_key, str) else ""
        boson_key, boson_source = boson_api_key(self.env)
        boson_key = session_key or boson_key
        openrouter_key, openrouter_source = openrouter_api_key(self.env)
        wanted = (self.env.get("PACESENSE_AI_PROVIDER") or "auto").strip().lower()
        if wanted not in ("auto", "openrouter", "boson"):
            wanted = "auto"
        # OpenRouter (the original PaceSense architecture) leads when its key is configured; Boson is the
        # other provider. Whichever is not leading becomes the backup if its key is present too.
        if wanted == "boson" or (wanted == "auto" and not openrouter_key):
            self.provider = "boson"
        else:
            self.provider = "openrouter"
        self.key_source = ("browser session" if session_key and self.provider == "boson" else
                           boson_source if self.provider == "boson" else openrouter_source)
        self.backup_provider: str | None = None
        self.transcribe_provider: str = self.provider
        self._stt_client: Any = None
        if self.offline_forced:
            self.client = None
        elif client is not None:
            self.client = client
        else:
            provider_config = config.get("openrouter", {})
            clients: dict[str, Any] = {}
            if openrouter_key:
                clients["openrouter"] = OpenRouterClient(
                    openrouter_key,
                    base_url=str(provider_config.get("base_url") or "https://openrouter.ai/api/v1"))
            if boson_key:
                clients["boson"] = BosonClient(boson_key)
            primary = clients.get(self.provider)
            other = "boson" if self.provider == "openrouter" else "openrouter"
            if primary is None and other in clients:        # the requested provider has no key
                self.provider, other = other, self.provider
                primary = clients[self.provider]
                self.key_source = boson_source if self.provider == "boson" else openrouter_source
            if primary is not None and other in clients:
                self.backup_provider = other
                self.client = FailoverClient(
                    primary, self.provider, clients[other], other,
                    BOSON_MODEL if other == "boson" else self._openrouter_model("assistant"),
                    BOSON_MODEL if other == "boson" else self._openrouter_model("transcribe"))
            else:
                self.client = primary
            # Speech to text can use a different provider from the analysis. Boson's transcription is
            # fast and already proven in this app, so it keeps that job when its key is present, while
            # OpenRouter (when configured) does the reasoning. PACESENSE_TRANSCRIBE_PROVIDER overrides.
            stt = (self.env.get("PACESENSE_TRANSCRIBE_PROVIDER") or "").strip().lower()
            if stt not in clients:
                stt = "boson" if "boson" in clients else self.provider
            if stt in clients and stt != self.provider:
                self.transcribe_provider = stt
                self._stt_client = FailoverClient(
                    clients[stt], stt, clients[self.provider], self.provider,
                    BOSON_MODEL if self.provider == "boson" else self._openrouter_model("assistant"),
                    BOSON_MODEL if self.provider == "boson" else self._openrouter_model("transcribe"))
        self.language = self.env.get("PACESENSE_LANGUAGE") or config.get("voice", {}).get("language") or None

    def _openrouter_model(self, task: str) -> str:
        if task == "transcribe":
            return (self.env.get("PACESENSE_OPENROUTER_TRANSCRIBE_MODEL") or "").strip() or OPENROUTER_TRANSCRIBE_MODEL
        return (self.env.get("PACESENSE_OPENROUTER_MODEL") or "").strip() or OPENROUTER_CHAT_MODEL

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAMES[self.provider]

    @property
    def online(self) -> bool:
        return self.client is not None

    def why_offline(self) -> str | None:
        if self.online:
            return None
        if self.offline_forced:
            return "AI is set to offline (PACESENSE_AI=offline)."
        return "No BOSON_API_KEY: PaceSense uses offline rules. Set BOSON_API_KEY in the server environment to enable Boson."

    def config(self, task: str, model: str | None = None) -> tuple[str, float]:
        c = self.settings.get(task)
        if model:
            return model, c["temperature"]
        if self.provider == "openrouter":
            return self._openrouter_model(task), c["temperature"]
        return c["model"], c["temperature"]

    def label(self, task: str) -> str:
        return f"{self.config(task)[0]} via {self.provider_name}" if self.online else "offline rules (no AI)"

    def _need(self, what: str) -> Any:
        if self.client is None:
            raise AIUnavailable(f"{what} needs a Boson API key (set BOSON_API_KEY in the server environment)")
        return self.client

    # ---- tasks ---------------------------------------------------------------------------------
    def chat_model(self, model: str | None = None):
        """The model for Ask PaceSense: Boson Higgs Realtime, or the offline helper."""
        from ..assistant.chat import BosonModel
        from ..assistant.offline import OfflineModel
        if not self.online:
            return OfflineModel()
        m, temp = self.config("assistant", model)
        return BosonModel(self.client, m, temp, provider=self.provider_name)

    @property
    def transcription_backend(self) -> str:
        """Boson handles voice by default; local Whisper is an explicit offline fallback."""
        return str(self.repo.get_setting("transcription_backend", self.env.get("PACESENSE_TRANSCRIBE_BACKEND", "boson")))

    def transcribe(self, audio: bytes, fmt: str = "wav", model: str | None = None) -> Transcript:
        # Voice uses Boson by default. Local Whisper is available only when explicitly selected as the backend.
        if model is None and self.transcription_backend == "local":
            from .local_whisper import transcribe as local_transcribe
            return local_transcribe(audio, fmt, self.language)
        if model is None and self._stt_client is not None and self.online:
            m = BOSON_MODEL if self.transcribe_provider == "boson" else self._openrouter_model("transcribe")
            return self._stt_client.transcribe(m, audio, fmt, self.language)
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
