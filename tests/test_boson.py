"""Focused unit tests for the Boson Higgs Realtime adapter."""
import io
import json
import unittest
import wave

from pacesense.llm.boson import (
    AUDIO_CHUNK_BYTES,
    BosonClient,
    MODEL,
    REALTIME_URL,
)
from pacesense.llm.openrouter import LLMError


class FakeSocket:
    def __init__(self, events):
        self.events = list(events)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self, timeout=None):
        if not self.events:
            raise TimeoutError("no more fake events")
        return json.dumps(self.events.pop(0))


def wav_bytes(rate=24000, channels=1, frames=240):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * frames * channels)
    return buffer.getvalue()


class BosonAdapterTests(unittest.TestCase):
    def test_requires_server_key(self):
        with self.assertRaises(LLMError):
            BosonClient("")

    def test_model_and_endpoint(self):
        self.assertEqual(MODEL, "higgs-realtime")
        self.assertEqual(REALTIME_URL, "wss://api.boson.ai/v1/realtime")

    def test_tool_specs_convert_chat_completions_shape_to_realtime_shape(self):
        specs = BosonClient._tool_specs([{
            "type": "function",
            "function": {
                "name": "predict",
                "description": "Predict a runner",
                "parameters": {"type": "object", "properties": {"runner_id": {"type": "string"}}},
            },
        }])
        self.assertEqual(specs[0]["type"], "function")
        self.assertEqual(specs[0]["name"], "predict")
        self.assertEqual(specs[0]["parameters"]["type"], "object")

    def test_conversation_items_preserve_tool_outputs(self):
        item = BosonClient._conversation_item({
            "role": "tool", "tool_call_id": "call-1", "content": '{"ok": true}',
        })
        self.assertEqual(item, {
            "type": "function_call_output", "call_id": "call-1", "output": '{"ok": true}',
        })
        self.assertIsNone(BosonClient._conversation_item({
            "role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}],
        }))

    def test_chat_uses_realtime_and_reads_text_response(self):
        socket = FakeSocket([
            {"type": "session.updated", "session": {"type": "realtime"}},
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.done", "response": {
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}],
            }},
        ])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        result = client.chat("higgs-realtime", [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Say hello"},
        ])
        self.assertEqual(result.text, "Hello")
        self.assertEqual((result.usage.prompt_tokens, result.usage.completion_tokens), (3, 2))
        self.assertEqual(socket.sent[0]["type"], "session.update")
        self.assertEqual(socket.sent[0]["session"]["temperature"], 0.2)
        self.assertEqual(socket.sent[-1]["type"], "response.create")
        self.assertNotIn("temperature", socket.sent[-1].get("response", {}))

    def test_chat_clamps_temperature_in_session_configuration(self):
        socket = FakeSocket([
            {"type": "session.updated", "session": {"type": "realtime"}},
            {"type": "response.done", "response": {"status": "completed", "output": []}},
        ])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        client.chat("higgs-realtime", [{"role": "user", "content": "hello"}], temperature=2.5)
        self.assertEqual(socket.sent[0]["session"]["temperature"], 2.0)
        self.assertNotIn("temperature", socket.sent[-1].get("response", {}))


    def test_chat_rejects_session_configuration_error(self):
        socket = FakeSocket([{
            "type": "error",
            "error": {"message": "invalid session configuration"},
        }])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        with self.assertRaisesRegex(LLMError, "invalid session configuration"):
            client.chat("higgs-realtime", [{"role": "user", "content": "hello"}])

    def test_chat_parses_function_call(self):
        socket = FakeSocket([
            {"type": "session.updated", "session": {"type": "realtime"}},
            {
                "type": "response.done",
                "response": {"status": "completed", "output": [{
                    "type": "function_call", "call_id": "call-1", "name": "predict",
                    "arguments": '{"runner_id": "R1"}',
                }]},
            },
        ])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        result = client.chat("higgs-realtime", [{"role": "user", "content": "Predict"}],
                             tools=[{"type": "function", "function": {"name": "predict"}}])
        self.assertEqual(result.tool_calls, [{"id": "call-1", "name": "predict", "arguments": {"runner_id": "R1"}}])

    def test_chat_rejects_incomplete_response(self):
        socket = FakeSocket([{"type": "session.updated", "session": {"type": "realtime"}}, {"type": "response.done", "response": {
            "status": "incomplete", "status_details": {"reason": "max_output_tokens"},
        }}])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        with self.assertRaisesRegex(LLMError, "incomplete"):
            client.chat("higgs-realtime", [{"role": "user", "content": "hello"}])

    def test_chat_rejects_cancelled_response(self):
        socket = FakeSocket([{"type": "session.updated", "session": {"type": "realtime"}}, {"type": "response.done", "response": {"status": "cancelled"}}])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        with self.assertRaisesRegex(LLMError, "cancelled"):
            client.chat("higgs-realtime", [{"role": "user", "content": "hello"}])

    def test_audio_is_resampled_to_mono_24khz_pcm(self):
        # 10 ms of 48 kHz stereo becomes 10 ms of 24 kHz mono: 240 samples / 480 bytes.
        pcm = BosonClient._audio_to_pcm24k(wav_bytes(rate=48000, channels=2, frames=480), "wav")
        self.assertEqual(len(pcm), 480)

    def test_invalid_audio_is_reported_as_llm_error(self):
        with self.assertRaisesRegex(LLMError, "decode"):
            BosonClient._audio_to_pcm24k(b"not audio", "wav")

    def test_transcription_uses_documented_realtime_audio_configuration(self):
        socket = FakeSocket([
            {"type": "session.updated", "session": {"type": "realtime"}},
            {"type": "response.output_text.delta", "delta": "runner transcript"},
            {"type": "response.done", "response": {"status": "completed"}},
        ])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        client._audio_to_pcm24k = lambda audio, fmt: b"\x00\x00" * 240
        result = client.transcribe("higgs-realtime", b"audio", "mp3")
        session = socket.sent[0]["session"]
        self.assertEqual(session["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertIsNone(session["audio"]["input"]["turn_detection"])
        self.assertNotIn("input_audio_format", session)
        self.assertNotIn("turn_detection", session)
        self.assertEqual(result.text, "runner transcript")
        self.assertEqual(result.usage.audio_seconds, 0.01)

    def test_transcription_rejects_incomplete_response(self):
        socket = FakeSocket([{"type": "session.updated", "session": {"type": "realtime"}}, {"type": "response.done", "response": {
            "status": "incomplete", "status_details": {"reason": "content_filter"},
        }}])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        client._audio_to_pcm24k = lambda audio, fmt: b"\x00\x00" * 240
        with self.assertRaisesRegex(LLMError, "incomplete"):
            client.transcribe("higgs-realtime", b"audio", "wav")

    def test_transcription_rejects_unknown_upload_format(self):
        client = BosonClient("test-key")
        with self.assertRaisesRegex(LLMError, "unsupported audio format"):
            client.transcribe("higgs-realtime", b"audio", "aac")

    def test_transcription_sends_audio_in_api_sized_chunks(self):
        socket = FakeSocket([
            {"type": "session.updated", "session": {"type": "realtime"}},
            {"type": "response.done", "response": {"status": "completed"}},
        ])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        client._audio_to_pcm24k = lambda audio, fmt: b"\x00" * (AUDIO_CHUNK_BYTES + 10)
        client.transcribe("higgs-realtime", b"audio", "wav")
        appends = [event for event in socket.sent if event["type"] == "input_audio_buffer.append"]
        self.assertEqual(len(appends), 2)
        self.assertEqual(len(__import__("base64").b64decode(appends[0]["audio"])), AUDIO_CHUNK_BYTES)
        self.assertEqual(len(__import__("base64").b64decode(appends[1]["audio"])), 10)


if __name__ == "__main__":
    unittest.main()
