"""Boson Higgs Realtime client used by PaceSense for all hosted AI tasks.

The permanent API key stays server-side. Each request opens a short-lived WebSocket
connection to Boson's Realtime API; browser code never receives the key.
"""
from __future__ import annotations

import base64
import io
import json
import time
import wave
from typing import Any

from .openrouter import ChatResult, LLMError, Transcript, Usage, MAX_AUDIO_BYTES

REALTIME_URL = "wss://api.boson.ai/v1/realtime"
MODEL = "higgs-realtime"


class BosonClient:
    def __init__(self, api_key: str, *, timeout: float = 90):
        if not api_key:
            raise LLMError("a Boson API key is required")
        self.api_key = api_key.strip()
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
            item_role = role
            return {"type": "message", "role": item_role,
                    "content": [{"type": "input_text" if role == "user" else "text",
                                 "text": str(text or "")}]}
        if role == "tool":
            return {"type": "function_call_output", "call_id": message.get("tool_call_id"),
                    "output": str(message.get("content", ""))}
        return None

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
        if json_mode:
            session["instructions"] += "\nReturn only one valid JSON object. Do not use Markdown fences."
        t0 = time.perf_counter()
        calls: dict[str, dict] = {}
        output = []
        usage_data: dict[str, Any] = {}
        try:
            with self._connect() as ws:
                self._send(ws, {"type": "session.update", "session": session})
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
                if temperature is not None:
                    response_options["temperature"] = max(0.0, min(2.0, float(temperature)))
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
                            calls[item.get("call_id") or item.get("id")] = {
                                "id": item.get("call_id") or item.get("id"),
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            }
                    elif kind == "response.function_call_arguments.done":
                        cid = event.get("call_id") or event.get("item_id")
                        calls[cid] = {"id": cid, "name": event.get("name", ""),
                                      "arguments": event.get("arguments", "{}")}
                    elif kind == "response.done":
                        response = event.get("response", {})
                        usage_data = response.get("usage") or {}
                        if response.get("status") == "failed":
                            details = response.get("status_details") or {}
                            error = details.get("error") or {}
                            raise LLMError(str(error.get("message") or "Boson response failed"))
                        for item in response.get("output", []):
                            if item.get("type") == "function_call":
                                cid = item.get("call_id") or item.get("id")
                                calls[cid] = {"id": cid, "name": item.get("name", ""),
                                              "arguments": item.get("arguments", "{}")}
                            for part in item.get("content", []):
                                if part.get("type") in ("text", "output_text") and not output:
                                    output.append(part.get("text", ""))
                        break
                else:
                    raise LLMError("Boson response timed out")
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
    def _wav_to_pcm24k(audio: bytes) -> bytes:
        try:
            import numpy as np
            with wave.open(io.BytesIO(audio), "rb") as wav:
                channels, width, rate, frames = (wav.getnchannels(), wav.getsampwidth(),
                                                 wav.getframerate(), wav.getnframes())
                raw = wav.readframes(frames)
        except Exception as exc:
            raise LLMError("Boson voice input currently expects a PCM WAV recording") from exc
        if width == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        elif width == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            vals = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
            vals = (vals ^ 0x800000) - 0x800000
            samples = vals.astype(np.float32) / 8388608
        elif width == 4:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
        else:
            raise LLMError("unsupported WAV sample width")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if rate != 24000 and len(samples):
            new_len = max(1, round(len(samples) * 24000 / rate))
            old_x = np.arange(len(samples), dtype=np.float64)
            new_x = np.linspace(0, max(0, len(samples) - 1), new_len)
            samples = np.interp(new_x, old_x, samples)
        return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()

    def transcribe(self, model: str, audio: bytes, fmt: str = "wav",
                   language: str | None = None) -> Transcript:
        if not audio:
            raise LLMError("the recording is empty")
        if len(audio) > MAX_AUDIO_BYTES:
            raise LLMError("the recording is larger than 25 MB")
        if fmt.lower() not in ("wav", "wave"):
            raise LLMError("Boson voice transcription requires WAV audio; convert the recording to PCM WAV first")
        pcm = self._wav_to_pcm24k(audio)
        t0 = time.perf_counter()
        instructions = "Transcribe the supplied speech exactly. Return only the recognized words, without commentary."
        if language:
            instructions += f" The spoken language is {language}."
        try:
            with self._connect() as ws:
                self._send(ws, {"type": "session.update", "session": {
                    "type": "realtime", "model": MODEL, "output_modalities": ["text"],
                    "instructions": instructions, "input_audio_format": "pcm16"}})
                self._send(ws, {"type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm).decode("ascii")})
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
                        break
                else:
                    raise LLMError("Boson transcription timed out")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Boson voice transcription failed: {exc}") from exc
        return Transcript("".join(output).strip(), Usage(audio_seconds=len(pcm) / 48000), time.perf_counter() - t0, MODEL, "realtime")
