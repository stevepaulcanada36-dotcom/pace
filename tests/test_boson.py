"""Focused unit tests for the Boson Higgs Realtime adapter."""
import io
import wave
import unittest

from pacesense.llm.boson import BosonClient, MODEL
from pacesense.llm.openrouter import LLMError


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
