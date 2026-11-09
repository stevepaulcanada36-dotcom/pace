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

The Studio browser voice control currently uses the browser's
`SpeechRecognition` / `webkitSpeechRecognition` interface to produce text and
then submits that text to the PaceSense command API. Speech playback is handled
by the browser. This is browser speech recognition, not a direct microphone
audio stream to Boson Realtime.

The Python Boson adapter supports server-side WebSocket chat and uploaded-audio
transcription. A direct browser microphone stream requires a separately designed
authenticated, bounded-lifetime audio relay (or a documented Boson client-secret
flow). Do not expose the permanent API key to the browser. Do not describe the
current browser voice path as direct Boson microphone streaming.

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
