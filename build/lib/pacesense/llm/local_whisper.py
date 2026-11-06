"""Local, no-per-request-cost speech transcription using faster-whisper.

The model is downloaded on first use and cached on the machine. Audio is written to a temporary
file because faster-whisper/FFmpeg decodes common browser recording formats from paths.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

from .openrouter import LLMError, Transcript

_MODEL = None
_MODEL_NAME = None
_MODEL_LOCK = threading.Lock()


def transcribe(audio: bytes, fmt: str = "wav", language: str | None = None) -> Transcript:
    if not audio:
        raise LLMError("recording is empty")
    if len(audio) > 25 * 1024 * 1024:
        raise LLMError("recording is larger than 25 MB; trim or compress it and retry")

    model_name = os.environ.get("PACESENSE_LOCAL_WHISPER_MODEL", "base").strip() or "base"
    try:
        model = _load_model(model_name)
    except ImportError as exc:
        raise LLMError(
            'Local transcription is not installed. From the PaceSense project folder run: '
            'pip install -e ".[local-stt]"'
        ) from exc
    except Exception as exc:
        raise LLMError(f"could not load local Whisper model {model_name!r}: {exc}") from exc

    suffix = "." + (fmt.lower().lstrip(".") or "wav")
    start = time.perf_counter()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="pacesense-audio-", suffix=suffix, delete=False) as handle:
            handle.write(audio)
            temp_path = handle.name
        segments, _info = model.transcribe(temp_path, language=language, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    except Exception as exc:
        raise LLMError(f"local Whisper could not decode or transcribe this recording: {exc}") from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass
    return Transcript(text=text, latency_s=time.perf_counter() - start,
                      model=f"faster-whisper/{model_name}", route="local")


def _load_model(model_name: str):
    global _MODEL, _MODEL_NAME
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_NAME == model_name:
            return _MODEL
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise
        device = os.environ.get("PACESENSE_LOCAL_WHISPER_DEVICE", "cpu")
        compute_type = os.environ.get("PACESENSE_LOCAL_WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
        _MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
        _MODEL_NAME = model_name
        return _MODEL
