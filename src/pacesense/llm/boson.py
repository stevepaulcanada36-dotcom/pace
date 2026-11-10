"""Boson Higgs Realtime client used by PaceSense for all hosted AI tasks.

The permanent API key stays server-side. Each request opens a short-lived WebSocket
connection to Boson's Realtime API; browser code never receives the key.
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

from .openrouter import MAX_AUDIO_BYTES, ChatResult, LLMError, Transcript, Usage

REALTIME_URL = "wss://api.boson.ai/v1/realtime"
MODEL = "higgs-realtime"
SUPPORTED_AUDIO_FORMATS = {"wav", "wave", "mp3", "m4a", "ogg", "webm", "flac"}
AUDIO_CHUNK_BYTES = 5 * 1024 * 1024
MAX_AUDIO_DURATION_S = 60 * 60
PCM_SAMPLE_RATE = 24000


class BosonClient:
    def __init__(self, api_key: str, *, timeout: float = 90):
        normalized_key = api_key.strip() if isinstance(api_key, str) else ""
        if not normalized_key:
            raise LLMError("a Boson API key is required")
        self.api_key = normalized_key
        self.timeout = timeout

    def _connect(self):
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise LLMError("Boson support requires the websockets package; install the PaceSense studio dependencies") from exc
        try:
            return connect(
                REALTIME_URL,
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                open_timeout=15,
                close_timeout=5,
                max_size=16 * 1024 * 1024,
            )
        except Exception as exc:
            raise LLMError(f"could not connect to Boson Higgs Realtime: {exc}") from exc

    @staticmethod
    def _send(ws, event: dict[str, Any]) -> None:
        ws.send(json.dumps(event, ensure_ascii=False))

    def _await_session_updated(self, ws) -> dict[str, Any]:
        """Wait until Boson confirms the requested Realtime session configuration."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            # A websocket receive timeout is only a polling interval, not proof that
            # Boson failed to update the session. Keep waiting until the overall
            # handshake deadline instead of aborting after the first 10 seconds.
            try:
                raw = ws.recv(timeout=max(1, min(10, deadline - time.monotonic())))
            except TimeoutError:
                continue
            event = json.loads(raw)
            kind = event.get("type", "")
            if kind == "error":
                err = event.get("error", {})
                raise LLMError(str(err.get("message") or err))
            if kind == "session.updated":
                return event.get("session", {})
        raise LLMError("Boson Realtime session configuration timed out")

    @staticmethod
    def _tool_specs(tools: list[dict] | None) -> list[dict]:
        out = []
        for tool in tools or []:
            fn = tool.get("function", tool)
            out.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return out

    @staticmethod
    def _conversation_item(message: dict) -> dict | None:
        role = message.get("role")
        if role in ("system", "developer"):
            return None
        if role == "assistant" and message.get("tool_calls"):
            return None
        if role in ("user", "assistant"):
            text = message.get("content")
            if isinstance(text, list):
                text = " ".join(str(x.get("text", "")) for x in text if isinstance(x, dict))
            return {"type": "message", "role": role,
                    "content": [{"type": "input_text" if role == "user" else "text",
                                 "text": str(text or "")}]}
        if role == "tool":
            return {"type": "function_call_output", "call_id": message.get("tool_call_id"),
                    "output": str(message.get("content", ""))}
        return None

    @staticmethod
    def _require_completed(response: dict, operation: str) -> None:
        """Do not treat cancelled, incomplete, failed, or unknown response states as success."""
        status = response.get("status")
        if status == "completed":
            return
        details = response.get("status_details") or {}
        error = details.get("error") or {}
        message = error.get("message") or details.get("reason")
        suffix = f": {message}" if message else ""
        raise LLMError(f"Boson {operation} did not complete (status: {status or 'unknown'}){suffix}")

    @staticmethod
    def _response_text(response: dict) -> str:
        parts = []
        for item in response.get("output", []):
            for part in item.get("content", []):
                if part.get("type") in ("text", "output_text") and part.get("text"):
                    parts.append(part["text"])
        return "".join(parts)

    def chat(self, model: str, messages: list[dict], *, tools: list[dict] | None = None,
             temperature: float = 0.2, max_tokens: int | None = None,
             json_mode: bool = False) -> ChatResult:
        if model and model not in (MODEL, "boson/higgs-realtime"):
            raise LLMError(f"unsupported Boson model {model!r}; PaceSense uses {MODEL}")
        system = "\n\n".join(str(m.get("content", "")) for m in messages
                               if m.get("role") in ("system", "developer"))
        session: dict[str, Any] = {
            "type": "realtime",
            "model": MODEL,
            "output_modalities": ["text"],
            "instructions": system,
            "tools": self._tool_specs(tools),
            "tool_choice": "auto" if tools else "none",
        }
        # Higgs Realtime documents temperature as a session setting. Keeping it here
        # avoids sending an unsupported per-response field to compatible Realtime APIs.
        if temperature is not None:
            session["temperature"] = max(0.0, min(2.0, float(temperature)))
        if json_mode:
            session["instructions"] += "\nReturn only one valid JSON object. Do not use Markdown fences."
        t0 = time.perf_counter()
        calls: dict[str, dict] = {}
        output = []
        usage_data: dict[str, Any] = {}
        try:
            with self._connect() as ws:
                self._send(ws, {"type": "session.update", "session": session})
                self._await_session_updated(ws)
                for message in messages:
                    if message.get("role") == "assistant" and message.get("tool_calls"):
                        for call in message["tool_calls"]:
                            fn = call.get("function", {})
                            self._send(ws, {"type": "conversation.item.create", "item": {
                                "type": "function_call", "call_id": call.get("id"),
                                "name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")}})
                    else:
                        item = self._conversation_item(message)
                        if item:
                            self._send(ws, {"type": "conversation.item.create", "item": item})
                response_options: dict[str, Any] = {}
                if max_tokens:
                    response_options["max_output_tokens"] = int(max_tokens)
                self._send(ws, {"type": "response.create", "response": response_options})
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    raw = ws.recv(timeout=max(1, deadline - time.monotonic()))
                    event = json.loads(raw)
                    kind = event.get("type", "")
                    if kind == "error":
                        err = event.get("error", {})
                        raise LLMError(str(err.get("message") or err))
                    if kind in ("response.output_text.delta", "response.text.delta"):
                        output.append(event.get("delta", ""))
                    elif kind in ("response.output_text.done", "response.text.done"):
                        if not output and event.get("text"):
                            output.append(event["text"])
                    elif kind == "response.output_item.done":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            cid = item.get("call_id") or item.get("id")
                            calls[cid] = {"id": cid, "name": item.get("name", ""),
                                          "arguments": item.get("arguments", "{}")}
                    elif kind == "response.function_call_arguments.done":
                        cid = event.get("call_id") or event.get("item_id")
                        if cid:
                            calls[cid] = {"id": cid, "name": event.get("name", ""),
                                          "arguments": event.get("arguments", "{}")}
                    elif kind == "response.done":
                        response = event.get("response", {})
                        self._require_completed(response, "chat response")
                        usage_data = response.get("usage") or {}
                        if not output:
                            output.append(self._response_text(response))
                        for item in response.get("output", []):
                            if item.get("type") == "function_call":
                                cid = item.get("call_id") or item.get("id")
                                calls[cid] = {"id": cid, "name": item.get("name", ""),
                                              "arguments": item.get("arguments", "{}")}
                        break
                else:
                    raise LLMError("Boson chat response timed out")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Boson Realtime request failed: {exc}") from exc
        parsed_calls = []
        for call in calls.values():
            try:
                args = json.loads(call["arguments"] or "{}")
            except (TypeError, json.JSONDecodeError):
                args = {"_invalid_json": call["arguments"]}
            parsed_calls.append({"id": call["id"], "name": call["name"], "arguments": args})
        return ChatResult("".join(output), parsed_calls, Usage.from_json(usage_data), time.perf_counter() - t0, MODEL)

    @staticmethod
    def _audio_to_pcm24k(audio: bytes, fmt: str = "wav") -> bytes:
        """Decode supported audio containers and resample to mono, signed 16-bit, 24 kHz PCM."""
        try:
            import av
        except ImportError as exc:
            raise LLMError("audio conversion requires PyAV; install the PaceSense dashboard or studio dependencies") from exc
        try:
            chunks: list[bytes] = []
            total_bytes = 0
            max_pcm_bytes = PCM_SAMPLE_RATE * 2 * MAX_AUDIO_DURATION_S
            resampler = av.AudioResampler(format="s16", layout="mono", rate=PCM_SAMPLE_RATE)

            def collect(frames) -> None:
                nonlocal total_bytes
                for frame in frames:
                    chunk = frame.to_ndarray().tobytes()
                    total_bytes += len(chunk)
                    if total_bytes > max_pcm_bytes:
                        raise LLMError("the decoded recording is longer than the 60-minute limit")
                    chunks.append(chunk)

            with av.open(io.BytesIO(audio), mode="r") as container:
                if not container.streams.audio:
                    raise LLMError("the recording contains no audio stream")
                for frame in container.decode(audio=0):
                    collect(resampler.resample(frame))
                collect(resampler.resample(None))
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"could not decode the {fmt.upper()} recording; check that it is a supported, valid audio file") from exc
        pcm = b"".join(chunks)
        if not pcm:
            raise LLMError("the recording contains no decodable audio")
        return pcm

    # Backwards-compatible helper retained for callers and existing tests.
    @staticmethod
    def _wav_to_pcm24k(audio: bytes) -> bytes:
        return BosonClient._audio_to_pcm24k(audio, "wav")

    def transcribe(self, model: str, audio: bytes, fmt: str = "wav",
                   language: str | None = None) -> Transcript:
        if not audio:
            raise LLMError("the recording is empty")
        if len(audio) > MAX_AUDIO_BYTES:
            raise LLMError("the recording is larger than 25 MB")
        normalized_fmt = fmt.lower().lstrip(".")
        if normalized_fmt not in SUPPORTED_AUDIO_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS - {"wave"}))
            raise LLMError(f"unsupported audio format {fmt!r}; supported formats: {supported}")
        if model and model not in (MODEL, "boson/higgs-realtime"):
            raise LLMError(f"unsupported Boson model {model!r}; PaceSense uses {MODEL}")
        pcm = self._audio_to_pcm24k(audio, normalized_fmt)
        t0 = time.perf_counter()
        instructions = "Transcribe the supplied speech exactly. Return only the recognized words, without commentary."
        if language:
            instructions += f" The spoken language is {language}."
        try:
            with self._connect() as ws:
                self._send(ws, {"type": "session.update", "session": {
                    "type": "realtime", "model": MODEL, "output_modalities": ["text"],
                    "instructions": instructions,
                    "audio": {"input": {
                        "format": {"type": "audio/pcm", "rate": PCM_SAMPLE_RATE},
                        "turn_detection": None,
                    }},
                }})
                self._await_session_updated(ws)
                for offset in range(0, len(pcm), AUDIO_CHUNK_BYTES):
                    chunk = pcm[offset:offset + AUDIO_CHUNK_BYTES]
                    self._send(ws, {"type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(chunk).decode("ascii")})
                self._send(ws, {"type": "input_audio_buffer.commit"})
                self._send(ws, {"type": "response.create"})
                output = []
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    event = json.loads(ws.recv(timeout=max(1, min(10, deadline - time.monotonic()))))
                    kind = event.get("type", "")
                    if kind == "error":
                        err = event.get("error", {})
                        raise LLMError(str(err.get("message") or err))
                    if kind in ("response.output_text.delta", "response.text.delta"):
                        output.append(event.get("delta", ""))
                    elif kind in ("response.output_text.done", "response.text.done") and not output:
                        output.append(event.get("text", ""))
                    elif kind == "response.done":
                        response = event.get("response", {})
                        self._require_completed(response, "transcription")
                        if not output:
                            output.append(self._response_text(response))
                        break
                else:
                    raise LLMError("Boson transcription timed out")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Boson voice transcription failed: {exc}") from exc
        return Transcript("".join(output).strip(),
                          Usage(audio_seconds=len(pcm) / (PCM_SAMPLE_RATE * 2)),
                          time.perf_counter() - t0, MODEL, "realtime")


__all__ = ["MODEL", "REALTIME_URL", "SUPPORTED_AUDIO_FORMATS", "BosonClient"]
