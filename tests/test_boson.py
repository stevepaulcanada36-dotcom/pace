"""Focused unit tests for the Boson Higgs Realtime adapter."""
import io
import json
import unittest
import wave

from pacesense.llm.boson import BosonClient, MODEL
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


class BosonAdapterTests(unittest.TestCase):
    def test_requires_server_key(self):
        with self.assertRaises(LLMError):
            BosonClient("")

    def test_model_is_higgs_realtime(self):
        self.assertEqual(MODEL, "higgs-realtime")

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
        self.assertEqual(socket.sent[-1]["type"], "response.create")

    def test_chat_parses_function_call(self):
        socket = FakeSocket([{
            "type": "response.done",
            "response": {"status": "completed", "output": [{
                "type": "function_call", "call_id": "call-1", "name": "predict",
                "arguments": "{\\"runner_id\\": \\"R1\\"}",
            }]},
        }])
        client = BosonClient("test-key")
        client._connect = lambda: socket
        result = client.chat("higgs-realtime", [{"role": "user", "content": "Predict"}],
                             tools=[{"type": "function", "function": {"name": "predict"}}])
        self.assertEqual(result.tool_calls, [{"id": "call-1", "name": "predict", "arguments": {"runner_id": "R1"}}])

    def test_wav_audio_is_converted_to_pcm_24khz(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)
        pcm = BosonClient._wav_to_pcm24k(buffer.getvalue())
        self.assertEqual(len(pcm), 480)


if __name__ == "__main__":
    unittest.main()
