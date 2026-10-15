"""A stand-in for OpenRouter's HTTP API, plugged into OpenRouterClient as its transport.

It answers like the real service: /audio/transcriptions returns {"text", "usage"}, /chat/completions returns
choices with a message (text or tool calls) and usage with a cost, /models returns a model list. Tests script
the answers, and every request is kept so tests can check what was sent.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from pacesense.llm.openrouter import OpenRouterClient


def chat_reply(content: str | None = None, tool_calls: list[tuple[str, dict]] = (), cost: float = 0.0001,
               model: str = "test/model") -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(tool_calls)]
    return {"model": model, "choices": [{"message": msg}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost}}


class FakeOpenRouter:
    def __init__(self):
        self.requests: list[dict] = []
        self.transcripts: list[Any] = []          # text, or (status, error) tuples
        self.chats: list[Any] = []                # chat_reply(...) dicts, or (status, error) tuples
        self.chat_handler: Callable[[dict], dict] | None = None   # used when self.chats is empty
        self.model_list = [
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini", "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
             "context_length": 128000, "architecture": {"input_modalities": ["text", "image"]},
             "supported_parameters": ["tools", "temperature"]},
        ]
        # OpenRouter exposes transcription-only models through a separate catalog filter.
        self.transcription_models = [
            {"id": "openai/whisper-1", "name": "Whisper", "pricing": {"prompt": "0", "completion": "0"},
             "context_length": None, "architecture": {"input_modalities": ["audio"]}, "supported_parameters": []},
        ]

    def client(self) -> OpenRouterClient:
        return OpenRouterClient("sk-or-test", transport=self)

    def __call__(self, method: str, url: str, headers: dict, body: bytes | None, timeout: float):
        req = {"method": method, "path": url.split("/api/v1", 1)[-1], "headers": headers,
               "body": json.loads(body) if body else None}
        self.requests.append(req)
        if req["path"] == "/models":
            return 200, {"data": self.model_list}
        if req["path"] == "/models?output_modalities=transcription":
            return 200, {"data": self.transcription_models}
        if req["path"] == "/audio/transcriptions":
            r = self.transcripts.pop(0) if self.transcripts else "hello"
            if isinstance(r, tuple):
                return r[0], {"error": {"message": r[1]}}
            return 200, {"text": r, "usage": {"seconds": 3.0, "cost": 0.0003}}
        if req["path"] == "/chat/completions":
            if self.chats:
                r = self.chats.pop(0)
                if isinstance(r, tuple):
                    return r[0], {"error": {"message": r[1]}}
                return 200, r
            if self.chat_handler:
                return 200, self.chat_handler(req["body"])
            return 200, chat_reply("ok")
        return 404, {"error": {"message": "not found"}}

    def bodies(self, path: str) -> list[dict]:
        return [r["body"] for r in self.requests if r["path"] == path]


def smart_handler(body: dict) -> dict:
    """Answers extract and refine requests sensibly, so pages can be exercised end to end."""
    system = body["messages"][0]["content"] if body["messages"][0]["role"] == "system" else ""
    user = body["messages"][-1]["content"]
    if "fill in a form" in system:
        if "A new runner" in system:
            return chat_reply(json.dumps({"name": "Anna de Vries", "gender": "female", "age": 34, "races": [
                {"event": "Amsterdam", "year": 2023, "finish": "3:35:48"},
                {"event": "Rotterdam", "year": 2024, "finish": "3h33m11"}]}))
        if "Plan details" in system:
            return chat_reply(json.dumps({"target_race": "Toronto Waterfront Marathon", "race_date": "2026-10-18",
                                          "notes": "Calf strain in August; start conservatively."}))
        if "race result" in system:
            return chat_reply(json.dumps({"finish": "3:21:04"}))
        return chat_reply(json.dumps({"question": user}))
    if "clearer, more precise" in system:
        return chat_reply("```json\n" + json.dumps({"refined": user.strip().capitalize().rstrip(".") + ".",
                                                    "changes": ["Removed filler words"], "questions": []}) + "\n```")
    return chat_reply("I can help with that.")
