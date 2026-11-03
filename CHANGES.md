# Changes

## 3.0.0: the adaptive studio (N feedback loops, persona, voice stage)
- `pacesense.adaptive`: events, a transparent coach persona (reliance, detail, modality, certainty, plus
  per-runner familiarity; four archetypes; editable and deletable) and nine pluggable feedback loops at four time
  scales: Persona, Data check, Clarify, Bias guard, Explanations, Guided interview, Track record, Shared learning,
  Layout. Loops never change the model; they change what the coach sees and is asked.
- Studio: voice stage (live transcript, spotlight race cards, large choices, spoken replies with the mic paused),
  guided interview, persona chip and profile dialog, feedback-loops bar, "why did the screen change?" log,
  layouts chosen by the persona (analysis / races / summary / guided) with a lock.
- API: session events, interview, coach profile (GET/PUT/DELETE), loop registry; `adapt` block on every response;
  exports include the adaptations and their reasons.
- docs/FUTURE_SCOPE.md: from the paper's open questions to each loop, plus an evaluation plan. 21 new tests (230).

## 2.5.0: PaceSense Studio, the production web app
- `pacesense studio`: a clean, light (and dark) dashboard for coaches: prediction summary with the hidden-race
  accuracy check and steering meter, race table with evidence and pull per race, coach notes with feedback cards
  and reactions (typed or continuous voice), pacing / similar-runners / insights charts, race-day simulator,
  replayable export. No 3D, no framework, no build step; works on phones.
- Backend on Starlette + Uvicorn with Pydantic validation, a versioned API (`/api/v1`), OpenAPI document,
  RFC 9457 errors with request ids, bounded session store (idle expiry, size cap, per-session locks),
  health/readiness probes, Prometheus metrics, JSON logs, CSP and security headers, settings from the environment.
- Slider moves are now kept in the transcript, so `steer.replay` reproduces sessions that mix talking and sliders.
- Docker image (multi-stage, non-root, health check), docker compose, Makefile, CI jobs for lint, types,
  a live-server smoke test and the container. 11 new tests (209 in total).

## 2.4.1: light command center and a version that needs no server
- The command center now uses a white, light design (navy and teal wireframe; colour only for meaning).
- `docs/command-center-demo.html` runs the whole command center in any browser with no install: the in-browser
  engine (`resources/hud_local.js`) is a port of the Python session and matches it to 4 decimals on weights,
  target, likely range, pacing, influence and the simulator. Rebuild with `python scripts/build_hud_demo.py`.

## 2.4.0: command center, real data, tuning

**Command center** (`pacesense hud`, `src/pacesense/hud/`, `resources/hud.html`)
- A holographic runner (drawn on a canvas with PaceSense's own 3D projection: no libraries, works offline) runs at
  the plan's pace inside a ring that shows the pacing plan; drag to rotate; calm background and ☾ calm mode.
- One command bar for voice (continuous in Chrome/Edge) or typing: race talk moves the sliders, interface commands
  change the screen ("show the simulator", "runner view"), questions go to the assistant. Replies can be spoken.
- Panels you can drag, hide and pin; Coach / Runner / Researcher views, each remembering its own layout.
- Feedback loop: a card for every change (what moved, what the data says, effect on target and range, next steps,
  "makes sense / not sure / disagree"); demo mode hides the runner's best race so each change says whether it moved
  the prediction closer to her real result; track record for saved plans with results.
- Race-day simulator: heat, headwind and runner behaviour, with the likely-range band; assumptions in config.

**Real data and tuning**
- `pacesense import`: Kaggle-style results files (Boston 2015-2017 recognised; others by column names or `--map`),
  linking the same runner across years by name, gender and birth year; a saved import report.
- `pacesense tune`: held-out grid search over the model's settings; `--write` saves only settings that beat the
  defaults; every command reads them from `model:` in `config/pacesense.yaml`.

Tests: 175 → 195.

## 2.3.0: PaceSense Live (proposal) and reproducibility

**PaceSense Live: steer the model by talking** (`src/pacesense/steer/`)
- `pacesense live` opens a page that listens continuously (browser speech, or private local Whisper with
  pause detection) and moves the race sliders from what the coach says; typing and a demo script also work.
- Interpreter (`intent.py`): deterministic rules in `resources/steer_lexicon.yaml`: race references (years,
  events, "the last race", "it"), cues ("hot day", "ignore", "her best race"), modifiers and negation, explicit
  numbers ("set 2014 to 0.3"), commands (reset, undo, "trust the recent races more") and the coach's own estimate.
- Statistics (`evidence.py`): robust leave-one-out race evidence, an evidence check per change (data agrees /
  your knowledge / against the data), each race's influence on the prediction, and the paper's Model Adaptation
  as a live steering index.
- `recency.py` fits the "recent races count more" half-life with the paper's hold-out protocol.
- `session.py`: every change goes through `PlanSession.set_weight`, carries its spoken reason, can be undone, and
  a session replays exactly from its transcript. Manual slider moves are logged as changes without a reason.
- Optional checked AI reader (`llm_interpret.py`): an AI model may only propose changes in a fixed JSON form;
  code drops proposals whose race does not exist or whose quoted reason was not said.
- Benchmark (`resources/steer_benchmark.yaml`): 18 dev + 24 held-out sentences; rules score 18/18 and 14/24
  (race precision 1.00, recall 0.50). Held-out failures are kept, not tuned away.
- HTTP: `/live` and `/api/steer/*`; Streamlit: a "Talk to the sliders" box on Plan a race, step 2.

**Reproducibility** (`src/pacesense/repro.py`)
- `pacesense reproduce` rebuilds every reported number and writes `results/manifest.json` (seed, parameters,
  input and output SHA-256, environment, git commit); `--check` re-runs and compares.
- One seed for everything random: `reproducibility.seed` in `config/pacesense.yaml`.
- CI (`.github/workflows/ci.yml`): tests plus two independent `reproduce` runs on Linux and Windows.
- `constraints-verified.txt`; cleaner `requirements.txt` (no forced PyAV); the Docker image now includes `config/`.
- Package version unified (was 1.0.0 in `__init__`, 2.2.0 in `pyproject.toml`).

**Proposal deck**: `docs/proposal/PaceSense_Live_Proposal.html` (open in Chrome; works offline except fonts).

Tests: 141 → 175.
