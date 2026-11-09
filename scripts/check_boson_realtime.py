"""Run a live Boson Higgs Realtime smoke test without exposing the API key.

From the repository root:
    python scripts/check_boson_realtime.py
    python scripts/check_boson_realtime.py --audio boson_test.mp3
"""
from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from pacesense.llm.boson import MODEL, BosonClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional WAV, MP3, M4A, OGG, WebM or FLAC file to test Boson audio transcription.",
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

    if args.audio:
        if not args.audio.is_file():
            parser.error(f"Audio file does not exist: {args.audio}")
        audio = args.audio.read_bytes()
        transcript = client.transcribe(
            MODEL,
            audio,
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
