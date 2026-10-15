"""Local transcription behavior without downloading or loading a real Whisper model."""
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pacesense.llm.local_whisper import transcribe
from pacesense.llm.openrouter import LLMError


class Segment:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self):
        self.path = None
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        assert Path(path).read_bytes() == b"audio-bytes"
        return iter([Segment(" Plan a marathon "), Segment("in four hours. ")]), object()


class LocalWhisperTests(unittest.TestCase):
    def test_transcribes_locally_and_deletes_temporary_audio(self):
        model = FakeModel()
        with patch("pacesense.llm.local_whisper._load_model", return_value=model):
            result = transcribe(b"audio-bytes", "wav", "en")
        self.assertEqual(result.text, "Plan a marathon in four hours.")
        self.assertEqual(result.model, "faster-whisper/base")
        self.assertEqual(result.route, "local")
        self.assertEqual(model.kwargs, {"language": "en", "vad_filter": True})
        self.assertFalse(Path(model.path).exists())

    def test_rejects_empty_and_oversized_recordings(self):
        with self.assertRaisesRegex(LLMError, "recording is empty"):
            transcribe(b"")
        with self.assertRaisesRegex(LLMError, "larger than 25 MB"):
            transcribe(b"x" * (25 * 1024 * 1024 + 1))


if __name__ == "__main__":
    unittest.main()
