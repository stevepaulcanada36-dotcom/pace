# Changes

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
