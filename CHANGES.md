# Changes

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
