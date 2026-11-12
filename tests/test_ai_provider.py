"""Which hosted AI provider PaceSense uses, and what happens when the leading one fails."""
import unittest

from pacesense.llm import service as llm_service
from pacesense.llm.boson import BosonClient
from pacesense.llm.openrouter import (
    ChatResult,
    LLMError,
    OpenRouterClient,
    Transcript,
    Usage,
)
from pacesense.llm.service import AIService, FailoverClient
from pacesense.storage import Repository


class Recorder:
    def __init__(self, name, error=None):
        self.name, self.error, self.calls = name, error, []

    def chat(self, model, messages, **kwargs):
        self.calls.append(("chat", model))
        if self.error:
            raise self.error
        return ChatResult(self.name, [], Usage(), 0.0, model)

    def transcribe(self, model, audio, fmt="wav", language=None):
        self.calls.append(("transcribe", model))
        if self.error:
            raise self.error
        return Transcript(self.name, Usage(), 0.0, model, "test")


class ProviderSelectionTests(unittest.TestCase):
    def ai(self, **env):
        return AIService(Repository(":memory:"), env=env)

    def test_no_key_is_offline(self):
        self.assertFalse(self.ai().online)

    def test_boson_alone(self):
        ai = self.ai(BOSON_API_KEY="b")
        self.assertEqual((ai.provider, ai.backup_provider), ("boson", None))
        self.assertIsInstance(ai.client, BosonClient)
        self.assertEqual(ai.config("assistant")[0], "higgs-realtime")
        self.assertIn("via Boson", ai.label("assistant"))

    def test_openrouter_alone_uses_the_original_models(self):
        ai = self.ai(OPENROUTER_API_KEY="o")
        self.assertEqual((ai.provider, ai.backup_provider), ("openrouter", None))
        self.assertIsInstance(ai.client, OpenRouterClient)
        self.assertEqual(ai.config("assistant")[0], "openai/gpt-4o-mini")
        self.assertEqual(ai.config("extract")[0], "openai/gpt-4o-mini")
        self.assertEqual(ai.config("transcribe")[0], "openai/whisper-1")
        self.assertIn("via OpenRouter", ai.label("assistant"))
        self.assertIn("via OpenRouter", ai.chat_model().label)

    def test_openrouter_leads_and_boson_backs_it_up_when_both_keys_are_set(self):
        ai = self.ai(OPENROUTER_API_KEY="o", BOSON_API_KEY="b")
        self.assertEqual((ai.provider, ai.backup_provider), ("openrouter", "boson"))
        self.assertIsInstance(ai.client, FailoverClient)
        self.assertEqual(ai.client.backup_model, "higgs-realtime")

    def test_analysis_and_transcription_can_use_different_providers(self):
        ai = self.ai(OPENROUTER_API_KEY="o", BOSON_API_KEY="b")
        self.assertEqual((ai.provider, ai.transcribe_provider), ("openrouter", "boson"))
        heard = Recorder("boson-stt")
        ai._stt_client.primary = heard
        self.assertEqual(ai.transcribe(b"audio", "wav").text, "boson-stt")
        self.assertEqual(heard.calls, [("transcribe", "higgs-realtime")])
        ai = self.ai(OPENROUTER_API_KEY="o", BOSON_API_KEY="b", PACESENSE_TRANSCRIBE_PROVIDER="openrouter")
        self.assertEqual(ai.transcribe_provider, "openrouter")
        self.assertEqual(self.ai(BOSON_API_KEY="b").transcribe_provider, "boson")

    def test_provider_can_be_pinned_and_models_overridden(self):
        ai = self.ai(OPENROUTER_API_KEY="o", BOSON_API_KEY="b", PACESENSE_AI_PROVIDER="boson")
        self.assertEqual((ai.provider, ai.backup_provider), ("boson", "openrouter"))
        self.assertEqual(ai.config("assistant")[0], "higgs-realtime")
        ai = self.ai(OPENROUTER_API_KEY="o", PACESENSE_OPENROUTER_MODEL="google/gemini-2.5-flash",
                     PACESENSE_OPENROUTER_TRANSCRIBE_MODEL="openai/gpt-4o-transcribe")
        self.assertEqual(ai.config("assistant")[0], "google/gemini-2.5-flash")
        self.assertEqual(ai.config("transcribe")[0], "openai/gpt-4o-transcribe")

    def test_a_pinned_provider_without_a_key_falls_back_to_the_one_that_has_one(self):
        ai = self.ai(BOSON_API_KEY="b", PACESENSE_AI_PROVIDER="openrouter")
        self.assertEqual(ai.provider, "boson")
        self.assertTrue(ai.online)

    def test_offline_switch_wins(self):
        self.assertFalse(self.ai(OPENROUTER_API_KEY="o", PACESENSE_AI="offline").online)


class FailoverTests(unittest.TestCase):
    def setUp(self):
        llm_service._provider_down_until.clear()
        self.addCleanup(llm_service._provider_down_until.clear)

    def client(self, error=None):
        primary, backup = Recorder("primary", error), Recorder("backup")
        return FailoverClient(primary, "openrouter", backup, "boson", "higgs-realtime", "higgs-realtime"), primary, backup

    def test_healthy_primary_answers(self):
        client, _primary, backup = self.client()
        self.assertEqual(client.chat("openai/gpt-4o-mini", []).text, "primary")
        self.assertEqual(backup.calls, [])

    def test_a_failed_call_is_answered_by_the_backup_with_its_own_model(self):
        client, primary, backup = self.client(LLMError("upstream timeout", 504))
        self.assertEqual(client.chat("openai/gpt-4o-mini", [], temperature=0).text, "backup")
        self.assertEqual(backup.calls, [("chat", "higgs-realtime")])
        self.assertEqual(client.transcribe("openai/whisper-1", b"x").text, "backup")
        self.assertEqual(len(primary.calls), 2)                  # a transient failure is retried next time

    def test_a_rejected_key_parks_the_primary(self):
        client, primary, backup = self.client(LLMError("invalid key", 401))
        client.chat("openai/gpt-4o-mini", [])
        client.chat("openai/gpt-4o-mini", [])
        self.assertEqual(len(primary.calls), 1)                  # not tried again while parked
        self.assertEqual(len(backup.calls), 2)


if __name__ == "__main__":
    unittest.main()
