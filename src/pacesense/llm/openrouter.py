"""A small OpenRouter client, using only the Python standard library.

OpenRouter (https://openrouter.ai) gives one API key and one OpenAI-style API for models from many providers,
so the same code can call `openai/gpt-4o-mini`, `anthropic/...`, `google/gemini-2.5-flash` and others, and
the Model lab can compare them. Three calls are used:

    POST /chat/completions        chat, function calling, JSON answers, audio in a message
    POST /audio/transcriptions    speech to text (JSON body with base64 audio)
    GET  /models                  the model list with prices and what each model accepts

Every result carries the time taken and the usage OpenRouter reports (tokens and cost in US dollars), which
the Model lab records. The network is reached through a `transport` function, so tests replace it.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

BASE_URL = "https://openrouter.ai/api/v1"
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Guard against accidental huge uploads and unexpectedly large API bills.

# transport(method, url, headers, body or None, timeout) -> (HTTP status, parsed JSON or text)
Transport = Callable[[str, str, dict, "bytes | None", float], "tuple[int, Any]"]


class LLMError(RuntimeError):
    """OpenRouter or the model provider refused or failed the request (bad key, no credit, unknown model…)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None          # US dollars, as OpenRouter reports it
    audio_seconds: float | None = None

    @classmethod
    def from_json(cls, u: dict | None) -> "Usage":
        u = u or {}
        cost = u.get("cost")
        return cls(int(u.get("prompt_tokens") or u.get("input_tokens") or 0),
                   int(u.get("completion_tokens") or u.get("output_tokens") or 0),
                   float(cost) if cost is not None else None,
                   float(u["seconds"]) if u.get("seconds") is not None else None)

    def __add__(self, other: "Usage") -> "Usage":
        cost = None if self.cost is None and other.cost is None else (self.cost or 0) + (other.cost or 0)
        secs = None if self.audio_seconds is None and other.audio_seconds is None else \
            (self.audio_seconds or 0) + (other.audio_seconds or 0)
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens,
                     cost, secs)


@dataclass
class ChatResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)    # [{"id", "name", "arguments": dict}]
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""


@dataclass
class Transcript:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""
    route: str = "transcriptions"       # or "chat" when the model only takes audio inside a chat message


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"could not reach OpenRouter: {getattr(e, 'reason', e)}") from e
    try:
        return status, json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, raw.decode("utf-8", "replace")


def _error_text(status: int, data: Any) -> str:
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    return str(data)[:300] or f"HTTP {status}"


class OpenRouterClient:
    def __init__(self, api_key: str, *, base_url: str = BASE_URL, app_name: str = "PaceSense",
                 app_url: str | None = None, transport: Transport | None = None, timeout: float = 90):
        if not api_key:
            raise LLMError("an OpenRouter API key is needed")
        self.base_url = base_url.rstrip("/")
        self.transport = transport or urllib_transport
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                        "X-Title": app_name}          # app attribution shown on openrouter.ai; optional
        if app_url:
            self.headers["HTTP-Referer"] = app_url
        self._models: list[dict] | None = None

    # ---- plumbing ----------------------------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        status, data = self.transport(method, f"{self.base_url}{path}", dict(self.headers), body, self.timeout)
        if status >= 400:
            raise LLMError(f"{_error_text(status, data)} (HTTP {status})", status)
        if not isinstance(data, dict):
            raise LLMError(f"unexpected answer from OpenRouter: {str(data)[:200]}", status)
        if "error" in data and not data.get("choices") and "text" not in data and "data" not in data:
            raise LLMError(_error_text(status, data), status)     # some provider errors arrive with HTTP 200
        return data

    # ---- chat --------------------------------------------------------------------------------
    def chat(self, model: str, messages: list[dict], *, tools: list[dict] | None = None, temperature: float = 0.2,
             max_tokens: int | None = None, json_mode: bool = False) -> ChatResult:
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                   "usage": {"include": True}}      # ask OpenRouter to report the cost
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        data = self._request("POST", "/chat/completions", payload)
        latency = time.perf_counter() - t0
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"the model returned no answer: {str(data)[:200]}") from e
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_invalid_json": raw}
            calls.append({"id": c.get("id") or f"call_{len(calls)}", "name": fn.get("name", ""), "arguments": args})
        content = msg.get("content")
        if isinstance(content, list):          # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return ChatResult(content or "", calls, Usage.from_json(data.get("usage")), latency,
                          data.get("model") or model)

    # ---- speech to text ----------------------------------------------------------------------
    def transcribe(self, model: str, audio: bytes, fmt: str = "wav", language: str | None = None) -> Transcript:
        """Speech to text. Uses the transcription endpoint (Whisper-style models); when the model is a chat
        model that accepts audio (for example google/gemini-2.5-flash), sends the audio in a chat message."""
        if not audio:
            raise LLMError("the recording is empty")
        if len(audio) > MAX_AUDIO_BYTES:
            raise LLMError("the recording is larger than 25 MB; trim it or export it in a smaller audio format")
        b64 = base64.b64encode(audio).decode("ascii")
        payload: dict[str, Any] = {"model": model, "input_audio": {"data": b64, "format": fmt}}
        if language:
            payload["language"] = language
        t0 = time.perf_counter()
        try:
            data = self._request("POST", "/audio/transcriptions", payload)
            return Transcript(str(data.get("text") or "").strip(), Usage.from_json(data.get("usage")),
                              time.perf_counter() - t0, model, "transcriptions")
        except LLMError as e:
            if e.status not in (400, 404, 422):
                raise
            first = e
        # Not a transcription model: try it as a chat model that listens.
        prompt = ("Transcribe this recording word for word. Reply with the transcript only, no comments."
                  + (f" The language is {language}." if language else ""))
        try:
            r = self.chat(model, [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}]}], temperature=0)
        except LLMError as e:
            raise LLMError(f"{model} could not transcribe: {first} / {e}", e.status) from e
        return Transcript(r.text.strip(), r.usage, time.perf_counter() - t0, model, "chat")

    # ---- models ------------------------------------------------------------------------------
    def models(self, refresh: bool = False) -> list[dict]:
        """[{id, name, prompt_usd_per_mtok, completion_usd_per_mtok, context, inputs, tools}] sorted by id."""
        if self._models is None or refresh:
            # The default catalog omits speech-to-text-only models (including Whisper on some catalog
            # versions). Query that collection separately so the UI can discover the models it needs.
            catalogues = [self._request("GET", "/models")]
            try:
                catalogues.append(self._request("GET", "/models?output_modalities=transcription"))
            except LLMError:
                # Keep ordinary model discovery useful if the transcription filter is unavailable.
                pass
            by_id: dict[str, dict] = {}
            for data in catalogues:
                for m in data.get("data") or []:
                    model_id = m.get("id", "")
                    if not model_id:
                        continue
                    pricing = m.get("pricing") or {}
                    arch = m.get("architecture") or {}

                    def per_mtok(v):
                        try:
                            return round(float(v) * 1e6, 4)
                        except (TypeError, ValueError):
                            return None
                    row = {"id": model_id, "name": m.get("name", ""),
                           "prompt_usd_per_mtok": per_mtok(pricing.get("prompt")),
                           "completion_usd_per_mtok": per_mtok(pricing.get("completion")),
                           "context": m.get("context_length"),
                           "inputs": list(arch.get("input_modalities") or []),
                           "tools": "tools" in (m.get("supported_parameters") or [])}
                    # Prefer a richer default-catalog entry if an id appears in both responses.
                    if model_id not in by_id or len(row["inputs"]) > len(by_id[model_id]["inputs"]):
                        by_id[model_id] = row
            self._models = sorted(by_id.values(), key=lambda x: x["id"])
        return self._models


def parse_json_answer(text: str) -> dict:
    """The JSON object in a model's answer, tolerating ```json fences and words around it."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("the model did not answer with JSON")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"the model's JSON could not be read: {e}") from e
    if not isinstance(data, dict):
        raise LLMError("the model's JSON was not an object")
    return data
