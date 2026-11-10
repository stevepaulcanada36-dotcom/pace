# Boson Higgs Realtime verification

PaceSense uses the Boson Higgs Realtime WebSocket API from the Python server. The
permanent `BOSON_API_KEY` must remain server-side; do not put it in browser
JavaScript, HTML, a committed `.env` file, logs, screenshots, or chat messages.

## Automated checks

Pushes, pull requests, and manual workflow dispatches run the Python unit-test
matrix (Python 3.10, 3.12, and 3.13 on Ubuntu and Windows), followed by the
quality job (Ruff, mypy, JavaScript syntax checks, and a server smoke test), then
the Docker build/startup smoke test. Matrix jobs run one at a time and dependent
jobs run sequentially to limit simultaneous hosted-runner demand. New runs on the
same branch cancel older in-progress runs.

The optional live Boson test runs in the quality job only when the GitHub Actions
repository secret `BOSON_API_KEY` is configured. It checks a real Realtime
WebSocket text response and audio response. It writes the returned audio to a
temporary WAV file and does not print the key. If the secret is absent, the step
reports a clear skip rather than failing the rest of the workflow.

## Run the live test locally

Install the project and development dependencies:

```bash
python -m pip install -e ".[studio,dev]"
```

Set `BOSON_API_KEY` in the process environment using your normal secret manager,
then run:

```bash
python scripts/check_boson_realtime.py --output /tmp/boson-realtime-test.wav
```

To additionally test audio input/transcription, pass a local recording:

```bash
python scripts/check_boson_realtime.py --audio ./sample.wav --output /tmp/boson-realtime-test.wav
```

The script prompts with hidden input if the environment variable is absent. Never
paste a production key into a terminal transcript that is being recorded.

## Current browser voice path

The Studio voice control uses two things together:

1. The browser's `SpeechRecognition` / `webkitSpeechRecognition` interface paces the turn (it
   detects when the coach starts and stops speaking) and produces a provisional transcript.
2. On desktop browsers, when a Boson key is configured, the microphone is also captured as PCM.
   When the coach pauses, that utterance is sent as a 16 kHz mono WAV to
   `POST /api/v1/sessions/{sid}/transcribe`. The server asks Boson (`higgs-stt-3.1` input
   transcription inside a Higgs Realtime session) for the transcript, and the browser uses it in
   place of its own text.

Browser recognition is the fallback: if Boson does not answer within 7 seconds, returns nothing,
or fails twice in a row, the browser's transcript is used and Boson transcription is switched off
for that page. Speech playback is still the browser's speech synthesis. The permanent API key
never reaches the browser. This is per-utterance upload, not a live microphone stream to Boson.

## Protocol notes (Higgs Realtime differs from OpenAI Realtime)

- Boson sends nothing on connect. It acknowledges the first `session.update` with
  `session.created`; the adapter accepts `session.created` or `session.updated`.
- `response.create` is sent bare. A `response` body makes the model ignore the conversation.
- `tool_choice` is only sent as `"auto"`, together with tools.
- `input_audio_buffer.append` accepts at most 1 MiB of base64 per event, so audio is sent in
  512 KiB chunks.

## Release gate

Do not call the release verified until all of the following are true:

- The full GitHub Actions workflow has completed successfully for the candidate
  commit, including all six Python/OS matrix combinations and the Docker job.
- The live Boson test has run with the repository secret configured and passed.
- Direct microphone streaming has been implemented and exercised in a real
  browser session, if it is part of the release acceptance criteria.
- The deployed service has been checked against the intended commit and its
  readiness endpoint returns HTTP 200.
- Any previously exposed API credentials have been revoked and replaced.

A successful mocked unit test or a successful REST text-to-speech request does
not prove that the Realtime WebSocket, audio input, or browser microphone path
works end to end.
