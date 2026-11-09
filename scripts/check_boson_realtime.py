"""Run live Boson Higgs Realtime smoke tests without exposing the API key.

From the repository root:
    python scripts/check_boson_realtime.py
    python scripts/check_boson_realtime.py --audio boson_test.mp3
"""
from __future__ import annotations

import argparse
import base64
import getpass
import time
import wave
from pathlib import Path

from pacesense.llm.boson import MODEL, PCM_SAMPLE_RATE, BosonClient
from pacesense.llm.openrouter import LLMError


def test_realtime_audio_output(client: BosonClient, output_path: Path) -> None:
    """Request spoken output over Realtime and save the returned PCM as a WAV file."""
    chunks: list[bytes] = []
    deadline = time.monotonic() + client.timeout
    with client._connect() as ws:
        client._send(ws, {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": MODEL,
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcm", "rate": PCM_SAMPLE_RATE},
                        "voice": "default",
                    },
                },
            },
        })
        client._await_session_updated(ws)
        client._send(ws, {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say: Boson Realtime audio output is working."}],
            },
        })
        client._send(ws, {"type": "response.create"})
        while time.monotonic() < deadline:
            import json

            event = json.loads(ws.recv(timeout=max(1, min(10, deadline - time.monotonic()))))
            kind = event.get("type", "")
            if kind == "error":
                error = event.get("error", {})
                raise LLMError(str(error.get("message") or error))
            if kind == "response.output_audio.delta":
                chunks.append(base64.b64decode(event.get("delta", "")))
            elif kind == "response.done":
                client._require_completed(event.get("response", {}), "audio response")
                break
        else:
            raise LLMError("Boson Realtime audio response timed out")

    pcm = b"".join(chunks)
    if not pcm:
        raise LLMError("Boson Realtime completed without returning audio chunks.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(PCM_SAMPLE_RATE)
        wav.writeframes(pcm)
    print("PASS: Realtime audio output.")
    print(f"Saved {len(pcm)} bytes of PCM audio to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional WAV, MP3, M4A, OGG, WebM or FLAC file to test Boson audio transcription.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boson_realtime_test.wav"),
        help="WAV destination for Realtime speech output (default: boson_realtime_test.wav).",
    )
    args = parser.parse_args()

    api_key = getpass.getpass("Boson API key (input hidden): ").strip()
    if not api_key:
        parser.error("A Boson API key is required.")

    client = BosonClient(api_key, timeout=60)
    print(f"Connecting to Boson Higgs Realtime ({MODEL})...")
    result = client.chat(
        MODEL,
        [
            {
                "role": "system",
                "content": "You are an API connectivity test. Reply in one short sentence.",
            },
            {
                "role": "user",
                "content": "Reply with: Boson Realtime chat is working.",
            },
        ],
        temperature=0.0,
    )
    if not result.text.strip():
        raise RuntimeError("The Realtime connection completed but returned no text.")
    print("PASS: Realtime WebSocket session and text response.")
    print("Model response:", result.text.strip())
    print(f"Latency: {result.latency_s:.2f}s")

    test_realtime_audio_output(client, args.output)

    if args.audio:
        if not args.audio.is_file():
            parser.error(f"Audio file does not exist: {args.audio}")
        transcript = client.transcribe(
            MODEL,
            args.audio.read_bytes(),
            args.audio.suffix.lstrip(".") or "wav",
        )
        if not transcript.text.strip():
            raise RuntimeError("Boson accepted the audio but returned an empty transcript.")
        print("PASS: Realtime audio input and transcription.")
        print("Transcript:", transcript.text.strip())
        print(f"Audio duration: {transcript.usage.audio_seconds:.2f}s")
    else:
        print("SKIP: Audio transcription (pass --audio path/to/recording.mp3 to test it).")

    print("Boson Realtime smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
