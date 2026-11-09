# PaceSense: interactive marathon prediction, all in Python

See [PaceSense-Guidebook.pdf](PaceSense-Guidebook.pdf) for the project workflow, architecture, Boson setup,
voice pipeline, model evaluation, testing, privacy boundaries and deployment notes.

PaceSense implements the system and study from:

> Muijlwijk, H., Willemsen, M. C., Smyth, B., & IJsselsteijn, W. A. (2024). *Benefits of
> Human-AI Interaction for Expert Users Interacting with Prediction Models: a Study on
> Marathon Running.* IUI '24, 245-258. https://doi.org/10.1145/3640543.3645205

Everything is Python: the model, the services, the analysis, and a **Streamlit dashboard** that coaches use.

| Part | What it is | From the paper? |
|---|---|---|
| Model (`cbr.py`) | Case-based reasoning: a challenging yet realistic finish time and pacing plan from runners who ran a similar race and later ran their best, combining several past races with a weight per race | Yes (Sec. 3.1) |
| Research study (dashboard page) | The paper's procedure: four runners per coach, interactive or non-interactive condition, questionnaire | Yes (Sec. 3.2, 4.1) |
| Analysis (`analysis.py`, `stats.py`) | Acceptance, trust, model adaptation and accuracy, with the paper's models and tests | Yes (Sec. 4.2, 5) |
| Plan a race (dashboard page) | The same model as a planning tool: race weights, likely range, editable pacing, strategies, change log | Built on the paper |
| **Race-day watch** (dashboard page) | A saved plan as a watch screen during the race: target pace, ahead or behind, next split, cue; live race simulation; pace band | PaceSense addition |
| **Ask PaceSense** (dashboard page) | An AI assistant that analyses runners and plans by calling PaceSense's own functions, and checks and corrects runner details before they are added | PaceSense addition |
| **Voice input** (several pages) | Speak instead of typing: transcribed, fields filled in, a refined version suggested for review | PaceSense addition |
| **PaceSense Live** (`steer/`, `pacesense live`, Plan step 2) | *Proposal.* Talk instead of dragging: what the coach says moves the race sliders, continuously; each move is checked against the data, and a live steering index shows whose view the model is taking | Proposed extension (answers the paper's open question on steering) |
| **Reproducibility** (`repro.py`, `pacesense reproduce`) | One command rebuilds every reported number and writes a manifest of input and output hashes; `--check` proves a re-run matches | PaceSense addition |
| **AI models per task** (`llm/`, Settings, Model lab) | Chat, voice transcription, field filling and refining use Boson Higgs Realtime; local Whisper is an optional transcription fallback | PaceSense addition |

## PaceSense Studio (start here)

**New in 3.1: a voice assistant for known and unknown runners.** Talk to PaceSense; the sliders move as you
speak (and stay draggable, in the conversation too). For an unknown runner it suggests changes from the data and
you answer yes / no / why; for a runner you know, it captures what you say about them for next time. Twelve
loops, with different ones on for each situation; questions answered by voice; a persona that learns from how you
speak (speed, cutting in, pauses) as well as what you talk about; turn-taking presets; talk over it any time.

**New in 3.0: it adapts to the coach.** Nine feedback loops instead of one, a transparent coach persona, a
voice stage that re-arranges the screen as you talk, and a guided interview. See
[docs/FUTURE_SCOPE.md](docs/FUTURE_SCOPE.md) for the reasoning from the paper, loop by loop.

| Adaptive feature | What the coach sees |
|---|---|
| **Feedback loops bar** | Persona · Data check · Clarify · Bias guard · Explanations · Guided interview · Track record · Shared learning · Layout. Each lights up when it acts; "Why did the screen change?" lists every adaptation and its reason |
| **Persona chip** | "You work like The Intuitive Expert" with a confidence ring. Opens the profile: four dimensions plus familiarity with this runner, the cues behind each, sliders to correct them, a layout lock, the track record chart and "Forget me" |
| **Talk** (V) | The voice stage: live transcript, big target, race cards that light up when mentioned, PaceSense's question with large choices to tap or say; replies are spoken (the mic pauses meanwhile) |
| **Guide me** (G) | Five questions built from the runner's data; each answer becomes a checked change |
| **PaceSense asks** | Clarify ("which race was hot?"), Data check (keep / undo / show similar runners), Bias guard ("knowledge or expectation?") |

Adaptive API: `POST /sessions` takes `condition` (known / unknown / ask) · `PUT /sessions/{id}/condition` ·
`POST /sessions/{id}/choice` · `POST /sessions/{id}/commands` takes `voice` signals (wpm, barge_in, latency_ms,
fillers, confidence) and answers the open question when the words fit it · `POST /sessions/{id}/events` (voice_on, voice_off, panel, dismiss) · `POST /sessions/{id}/interview/start`
· `POST /sessions/{id}/interview` · `GET|PUT|DELETE /coaches/{coach_id}/profile` · `GET /loops`. Every session
response carries an `adapt` block (`ui`, `persona`, `loops`, `prompt`, `adaptations`, `speak`).

The production web app: a clean dashboard for coaches on top of a versioned, validated JSON API.

```bash
pip install -e ".[studio]"
pacesense studio                         # http://127.0.0.1:8000  (or: python -m pacesense.studio)
pacesense studio --data data/races.csv   # your own results (see "Real data" below)
```

| On screen | What it does |
|---|---|
| **Target, likely range, accuracy, whose view** | The prediction and its weighted 20th–80th percentile range; in the demo the runner's best race is hidden, so every change says whether it moved the prediction *closer to what really happened* (Reveal at the end); the steering index shows whether the model is following the data or your own estimate |
| **Race history** | One row per race: what her own times say about it (typical / slower / faster than her other races), its weight slider, and how many seconds it pulls the target |
| **Coach notes** | Type or talk (continuous listening in Chrome/Edge): "2016 was a hot day and she was injured", "trust the recent races more", "I think she can run 3:25", "simulate 26 degrees and she starts too fast". Each change gets a card: what moved, whether the data agrees, the new target, and the hidden-race check; react with *makes sense / not sure / disagree* |
| **Pacing plan · Similar runners · Insights** | Plan pace per 5 km against her past races; for each race, what runners who ran the same time ran next; plan checks and alternative strategies |
| **Race-day simulator** | Temperature, headwind and pacing mistakes (fast start, late fade) → finish, gap to plan along the course, and whether it stays in the likely range |
| **Export** | A JSON record of the session that `pacesense.steer.replay` turns back into exactly the same weights |

Keyboard: `/` type a note · `Ctrl+Z` undo · `?` help. Light and dark themes; works on a phone.

**Engineering.** Starlette + Uvicorn (ASGI); Pydantic validates every request (`extra="forbid"`, ranges, formats)
and generates the OpenAPI document at `/api/v1/openapi.json`; errors are RFC 9457 problem documents carrying a
request id; sessions live in a bounded store with idle expiry and a lock per session (one worker, many threads);
`/healthz`, `/readyz` and Prometheus `/metrics` (per-route latency histograms, session gauges); JSON logs with
`PACESENSE_LOG_FORMAT=json`; strict Content-Security-Policy and no third-party scripts. Every setting is an
environment variable with a safe default (`.env.example`, `src/pacesense/studio/settings.py`); bad values stop the
server at start-up with a readable message.

```bash
docker compose up --build                # or: docker build -t pacesense . && docker run -p 8000:8000 pacesense
make check                               # lint (ruff) + types (mypy) + tests, as in CI
```

API (all JSON, under `/api/v1`):

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/runners?q=` | | runner picker (the demo first) |
| POST | `/sessions` | `{runner_id: "demo", holdout?}` | 201 `{id, state}` |
| GET / DELETE | `/sessions/{id}` | | state / 204 |
| POST | `/sessions/{id}/commands` | `{text, source: typed\|voice}` | `{reply, card, actions, simulation, state}` |
| PUT | `/sessions/{id}/weights` | `{index, weight 0–1}` | `{card, state}` |
| POST | `/sessions/{id}/undo` · `/reveal` | | `{card, state}` · state |
| POST | `/sessions/{id}/estimate` | `{finish: "3:25"}` | as commands |
| POST | `/sessions/{id}/simulations` | `{temp_c, wind_kmh, profile}` | simulation |
| POST | `/sessions/{id}/reactions` | `{turn, value: agree\|unsure\|disagree}` | counts |
| GET | `/sessions/{id}/export` | | replayable transcript + final state |

The earlier interfaces (Streamlit dashboard, PaceSense Live, the 3D command center) still work and share the same
model and services.

## Quick start

```bash
pip install -e ".[dashboard]"          # or: pip install -r requirements.txt && pip install -e .
pacesense dashboard --data data/races.csv
```

Open http://localhost:8501. On Windows PowerShell, inside your activated virtual environment, the
commands are the same.

Without `--data`, a synthetic demo database is generated. The dashboard stores plans, runners, settings and
study answers in `pacesense.sqlite3` (change it with `--db`). To use the Study results page, add
`--admin-token CHOOSE_A_TOKEN`.

The command-line tools still work without Streamlit:

```bash
pacesense validate --data data/races.csv               # model accuracy, and how often the likely range holds
pacesense predict  --data data/races.csv R00042 --weights 0.2,1.0
pacesense simulate --data data/races.csv --coaches 71 --db study.sqlite3   # simulated coaches (testing only)
pacesense analyze  --db study.sqlite3 --out report.md  # the paper's Section 5 analyses
```

## Command center (experimental 3D view)

```bash
pacesense hud --data data/races.csv          # opens http://127.0.0.1:8000/hud
```

No install needed to look around: open `docs/command-center-demo.html` in Chrome or Edge. It runs the same demo
in the browser (typing, panels, views, simulator, feedback loop; voice and the AI assistant need the app). Rebuild it
with `python scripts/build_hud_demo.py` after changing the interface.

A holographic runner turns slowly in the middle of the screen, running at the pace of the current plan; the ring
around her is the pacing plan (warmer sections are slower). Drag her to look around. Glass panels sit around her:
past races and their weights, the prediction and likely range, the steering meter, the feedback loop, the pacing
plan, runners like her, the race-day simulator, the change log, the track record and the conversation.

**One command bar, by voice or typing.** Chrome and Edge listen continuously (they send audio to their own speech
service); typing works everywhere; "Speak" reads replies aloud and pauses the microphone meanwhile.

| You say | What happens |
|---|---|
| "2014 was a hot day and she was injured", "ignore the first race", "I think she can run 3:30", "scratch that" | race weights move (PaceSense Live's interpreter), with a feedback card |
| "show the simulator", "hide the change log", "switch to runner view", "calm mode", "reset the layout" | the screen changes |
| "simulate 26 degrees with a 15 km/h headwind", "what if she starts too fast" | the race is replayed on the ring and the chart |
| "why is the likely range so wide?", "is my plan realistic?" | the assistant answers with PaceSense's own functions (Boson with a key, the offline helper without) |

The hologram reacts to what you say: legs flash for an injury, the body warms for heat, the chest glows for a best
race. Rules: `hud/commands.py`; "simulate a hot day" never lowers a race, because interface commands are taken out
of the sentence before the race interpreter reads it.

**Views.** *Coach* (races, prediction, steering, feedback, simulator), *Runner* (prediction, pacing, simulator:
no statistics), *Researcher* (races, runners like her, prediction, track record, steering, feedback). Drag panels
by their title, hide them with ×, or pick them under ▦ Panels; each view remembers its own arrangement in this
browser. ☾ Calm slows everything down (and is the default if your system asks for reduced motion).

**The feedback loop.** The paper's sliders gave a number back, never a reason. Here every change gets a card: what
you said, what moved, whether the runner's own races agree (*data agrees* / *your knowledge* / *against the
data*), what it did to the target and range, two or three next steps, and buttons to say whether it makes sense
(reactions are counted in the steering panel). In demo mode the runner's fastest recent race is hidden, as in the
paper's accuracy test, so each card also says whether the change moved the prediction **closer to her real
result**; "reveal" shows it. With real runners, the track record panel compares saved plans with recorded results.

**Race-day simulator.** Heat (no effect up to 11 °C, then about 0.3 % per °C, growing through the race and larger
for slower runners), headwind and three runner behaviours; the band is the likely range. The heat and wind numbers
are stated assumptions, not fitted, and can be changed under `simulator:` in `config/pacesense.yaml`.

## Real data (Kaggle) and tuning

```bash
pacesense import marathon_results_2015.csv marathon_results_2016.csv marathon_results_2017.csv --out data/boston.csv
pacesense tune --data data/boston.csv            # add --write to use the chosen settings everywhere
pacesense hud --data data/boston.csv
```

`import` reads Kaggle-style results files (Boston 2015-2017 is recognised by its columns; other files by column
names such as `name`, `gender`/`sex`, `age`, `5k`...`40k`, `finish`/`net time`; or `--map finish="Net Time"`).
Results files have no runner id, so the same runner is linked across years by normalised name, gender and birth
year (year − age, ±1); runners who share a name, gender and birth year in one race are left unlinked rather than
guessed. Everything dropped is counted, and the report is saved next to the output (`…csv.import.json`).

`tune` searches the model's settings (similar runners k, time window, pacing and age weights) with the paper's
hold-out protocol: settings are chosen on one seeded sample of runners and the error is reported on a different
one, so the number is not flattered by the search. `--write` only saves settings that beat the defaults on the
held-out runners.

## PaceSense Live: talk to the sliders (proposal)

```bash
pacesense live --data data/races.csv        # opens http://127.0.0.1:8000/live
```

Press **Demo runner**, then the microphone, and talk the way a coach talks about a runner:

> "2014 was a hot day and she was injured. I think she can run 3:29. Trust the recent races more."

The sliders move as you speak. Every move shows **why** (the words it came from) and **what the data says**:

| Tag | Meaning |
|---|---|
| ✅ data agrees | you lowered a race that was clearly slower than her others (or raised a faster one) |
| 🧠 your knowledge | the times do not single this race out; the change rests on what you know (an injury does not show in a finish time) |
| ⚠️ against the data | you lowered one of her faster races (or raised a slow one): the target gets slower |

The right-hand panel shows the prediction, its likely range and the **steering index**: the paper's Model
Adaptation, (model now − model at equal weights) / (your estimate − model at equal weights), live. At 1.0 the
sliders have moved the model exactly onto your own estimate. "Scratch that" undoes the last thing you said.
"Play demo script" runs a scripted conversation for presenting without a microphone.

**Listening.** *Browser speech* (Chrome, Edge) listens continuously; those browsers send the audio to their own
speech service. *Private: local Whisper* keeps audio on the computer: it detects pauses, cuts the speech into
phrases and transcribes each with faster-whisper (`pip install -e ".[local-stt]"`). Typing works everywhere.
On **Plan a race**, step 2 has the same as a box: type, or record a phrase.

**How words become weights** (`src/pacesense/steer/`):

```
"2014 was a hot day and she was injured"
   │ intent.py      race references ("2014", "Amsterdam", "the last race", "it") + cues from
   │                resources/steer_lexicon.yaml ("hot day" −0.3, "injured" −0.3, "ignore" → 0, "best race" +0.3,
   │                "a bit" ×0.5, "not" flips); descriptions attach to the race before them, commands
   │                ("ignore 2016") to the race after; no race → the race talked about last; never guessed
   ▼
Amsterdam 2014 −0.6 ("hot day + injured")
   │ session.py     PlanSession.set_weight(…) → same change log, plan check and saved plan as the sliders
   │ evidence.py    robust leave-one-out z-score of the race vs her other races → ✅ / 🧠 / ⚠️;
   │                influence of each race (seconds it pulls the prediction); steering index
   ▼
"Amsterdam 2014: 0.5 → 0.0. The data agrees: Amsterdam 2014 was 19 min slower than her other races."
```

The rules are deterministic, so a session can be replayed exactly from its transcript (`steer.replay`). With an
Boson API key and *AI reads paraphrases* ticked, an AI model reads what the rules miss; it may only propose
changes in a fixed JSON form, and code drops any proposal whose race does not exist or whose quoted reason was not
actually said (`steer/llm_interpret.py`).

**How well it understands** (`resources/steer_benchmark.yaml`, scored by `pacesense reproduce`): 18/18 on the
sentences the rules were written with; **14/24 on held-out sentences** written before the rules saw them, with
race precision 1.00 (it never moved a race the coach did not mean) and recall 0.50 (it misses paraphrases such as
"she ran it with a broken toe"). The held-out failures are kept, not tuned away, so this number stays honest; the
AI reader is the proposed fix, to be scored on the same set.

API for other programs: `POST /api/steer/sessions`, `…/utterances`, `…/audio`, `…/manual`, `…/undo`,
`GET …/transcript` (see `web/steer_api.py`). Live sessions are kept in memory and are never written to the study
database: the research study stays exactly as the paper ran it.

## Reproducibility

```bash
pacesense reproduce --out results                    # about 15 s; --quick for a smaller run
pacesense reproduce --check results/manifest.json    # re-run in a temporary folder and compare
```

`reproduce` rebuilds every number PaceSense reports: the synthetic data (or your `--data`), the model's
leave-best-out accuracy, the fitted "recent races" profile, the interpreter benchmark, and the simulated study
with its Section 5 analysis. `results/manifest.json` records:

- **inputs**: seed and parameters, SHA-256 of the data, `config/pacesense.yaml`, the lexicon, the benchmark and the
  questionnaire;
- **outputs**: SHA-256 of `validation.json`, `recency_fit.json`, `steer_benchmark.json`, `study_report.md`
  (numbers rounded to 6 significant digits before hashing, so last-bit floating-point differences between machines
  do not count; real changes do);
- **environment**: Python, platform, package versions and the git commit (marked `-dirty` with uncommitted edits).

`--check` prints `REPRODUCED` or lists what differs (and whether the environment changed), and exits 1 on a
mismatch, so it can gate CI (`.github/workflows/ci.yml` runs the tests and two independent `reproduce` runs on
Linux and Windows, Python 3.10 to 3.13).

**One seed.** Everything random (synthetic data, validation samples, the recency fit, study assignment, simulated
coaches) uses `reproducibility.seed` in `config/pacesense.yaml` unless `--seed` is given.

**Freezing the environment.** `constraints-verified.txt` lists the numerical library versions the pipeline was
last verified with (`pip install -e ".[dashboard]" -c constraints-verified.txt`). After a run you are happy with,
freeze your own environment next to the manifest:

```bash
pip freeze --exclude-editable > requirements-lock.txt     # then: pip install -r requirements-lock.txt
```

or use the Docker image, which copies the config and pins the Python version. The study database
(`study.sqlite3`) is not hashed: it holds random session ids and clock times by design; its analysis is.

## One central YAML configuration (recommended)

PaceSense keeps the Boson model and task temperatures in `config/pacesense.yaml`. The hosted model is fixed to `higgs-realtime`; task-specific temperatures can be tuned in Settings.

For local development, you can configure your Boson key in a private local config file:

1. Copy `config/local.example.yaml` to `config/local.yaml`.
2. Put your key in `config/local.yaml` under `boson.api_key`.
3. Restart PaceSense.

`config/local.yaml` is ignored by Git and is not included as a real secrets file in the ZIP. **Do not commit or share it**. The key lookup order is `BOSON_API_KEY` environment variable, then `config/local.yaml`. Render should use the environment variable. You can instead set `PACESENSE_CONFIG_DIR` to point at a different config folder. The public defaults contain no secrets.

Boson is the default voice transcription method. Local Whisper is an optional fallback that can be selected in Settings.

## The dashboard

The sidebar has three groups.

**Coach**

- **Plan a race.** On the first visit you choose *Guide me step by step* or *Show me everything*; switch any
  time with the layout toggle or in Settings. Guided steps show one step at a time, with Back and Next:
  1. *Runner*: search your runners or the race database, or try a demo runner.
  2. *Past races*: the paper's sliders (0 to 1, starting at 0.5). *Reset to equal weights* returns to the
     model on its own, the paper's non-interactive condition.
  3. *Prediction*: the target, its likely range ("about 8 in 10 runners like this finished between…"), and
     past races against the model's target.
  4. *Pacing*: pick a strategy, compare with past races on the chart, and type a new pace for any interval;
     the finish time follows.
  5. *Check and save*: realistic, ambitious or cautious, with warnings; save, or download as text or CSV.

  A summary bar on every step shows the runner, the model's time, your plan's time and the check, and
  explains the last change, e.g. "Amsterdam 2010 weight changed from 0.5 to 0.1. Model 1:22 faster
  (3:11:06)". Step 5 lists every change, which makes the "steering" the paper reports in Table 4 visible.
- **Race-day watch.** Choose a saved plan (or the plan you are working on). *A moment in the race*: set the
  distance and how far ahead or behind the runner is, and see the watch screen and cue. *Simulate a race*:
  a simulated runner (on plan, fast start, or late fade) runs the plan at 60x to 300x speed; the watch,
  the cue and an ahead/behind chart update live. Download a pace band (text), kilometre splits (CSV) or a
  generic JSON plan. The JSON is not any watch brand's file format.
- **Ask PaceSense.** A chat about your runners and plans: "Is my plan realistic?", "Which past race pulls
  the prediction most?", "At 21.1 km in 1:45:00, am I on plan?", "Explain the likely range". You can also
  describe a runner in your own words ("Add runner Anna de Vries, female, 34: Amsterdam 2023 3:35:48;
  Rotterdam 2024 3h33m11"): the assistant reads times written in different ways, finds mistakes (a year in
  the future, splits that do not add up, splits given per 5 km instead of elapsed) and asks for what is
  missing; you add the runner with a button. You can also ask by voice. See
  [Ask PaceSense](#ask-pacesense-the-ai-assistant).
- **Runners**, **Saved plans** (record the actual result after a race), **Track record** (how plans and the
  model compared with results).

**Research**

- **Research study.** The study as the paper ran it (see below). Unchanged by the coach pages.
- **Study results.** The Section 5 analyses and CSV exports, locked behind the admin token.
- **Boson evaluation lab.** Score the Boson model against repeatable test cases; see [AI models](#ai-models-through-boson).

**Help**

- **How it works.** Every function, where it is, and whether it comes from the paper; key terms; how the code
  is organised. Every heading and number also has a "?" tooltip with the same explanation.
- **Settings.** Layout, explanations on or off, minutes per km or per mile, past-race overlay, and **AI models**:
  the Boson connection status and temperature per AI task. The key stays in the server environment, so settings
  survive a reload.

## How the code is organised

```
src/pacesense/
  dashboard/            Streamlit app: what people see
    app.py              page setup and navigation (entry point)
    ui.py               shared helpers: services, preferences, summary bar, explanations
    charts.py           matplotlib charts: pacing, race bars, likely range, watch face, ahead/behind
    services.py         builds the model and services once per server process
    voice.py            voice input and "suggest a clearer version", used on several pages
    pages/              one file per page: plan, watch, assistant, lab, runners, saved, record, study, results, howitworks, settings
  assistant/            Ask PaceSense: input checks, functions for the AI, the chat loop, offline helper
  llm/                  Boson client, AI tasks and their models, forms voice can fill, AIService, evals
  steer/                PaceSense Live: intent.py (words → weight changes), evidence.py (data checks, influence,
                        steering index), recency.py, session.py (SteeringSession, undo, replay),
                        llm_interpret.py (checked AI reader), benchmark.py
  hud/                  command center: session.py (feedback loop, demo hold-out), commands.py (one command bar),
                        sim.py (race-day simulator); the page is resources/hud.html
  importers.py          `pacesense import`: Kaggle-style results → race CSV, linking runners across years
  tuning.py             `pacesense tune`: held-out grid search; model settings from config (`model:`)
  repro.py              `pacesense reproduce`: the pipeline, the manifest and --check
  planning.py           PlanSession: weights, strategies, pace edits, change log (no UI code)
  racewatch.py          race-day watch: km splits, status and cues, race simulation, pace band exports
  explain.py            every explanation and the list of functions
  coach.py              CoachService: predictions, plan checks, saved plans, preferences
  study.py              StudyService: the paper's study procedure
  cbr.py                the case-based reasoning model and the likely range
  measures.py, analysis.py, stats.py   the paper's measures and Section 5 analyses
  data.py, storage.py   race CSV and synthetic data; SQLite storage
  web/                  JSON API over the same services, for other programs
  cli.py                the `pacesense` command
```

Each layer only calls the layer below it: pages call `PlanSession`, `racewatch`, `CoachService` and
`StudyService`; those call the model; the model reads the data. So the logic can be used and tested from
plain Python, and a Streamlit page reads as a description of what is on screen. For example:

```python
from pacesense.dashboard.services import build_services
from pacesense.planning import PlanSession
from pacesense import racewatch

s = build_services(":memory:", "data/races.csv")
plan = PlanSession(s.coach, "R00042")
print(plan.set_weight(0, 0.1).text())        # what the change did to the model and the plan
print(racewatch.status(plan.paces, 21.1, 6000).cue)
```

## Ask PaceSense (the AI assistant)

```
coach's question -> AI model -> "call predict(R00042)" -> PaceSense runs it -> result -> AI model -> answer
```

The AI does not calculate finish times, ranges or paces. It chooses which PaceSense function to call,
PaceSense runs it with the same code as the dashboard, and the AI words the answer from the result. Under
every answer, *What PaceSense calculated* lists each call and its result, so every number can be checked.

| Function the AI may call | What it returns |
|---|---|
| `find_runners`, `runner_details` | runners matching a search; a runner's races, paces and past model targets |
| `predict` | target, likely range, similar runners, prediction per race, strategies, insights (with any weights) |
| `check_pacing_plan` | finish, realistic/ambitious/cautious, warnings for a set of paces |
| `current_plan` | the plan open on Plan a race, including your list of changes |
| `watch_moment` | the race-day watch at a distance and time |
| `saved_plans` | saved plans, results and the track record |
| `check_runner_details`, `propose_runner` | checked and corrected runner details; a runner ready for you to add |
| `explain` | PaceSense's own explanation of a term, and whether it is from the paper |

Without a Boson API key (or with `--offline-ai`), an **offline helper** answers instead: no AI and no
network, just word rules that call the same functions. It understands the common questions above, one at a
time.

Limits, on purpose: the assistant cannot read the research study's data (the study must stay as the paper
ran it, and participants' answers are not sent to an outside service), and it cannot save anything. With
a key, questions and the runner data looked up for them are sent to Boson and the provider of the
model you chose.

Code: `assistant/intake.py` (checks and corrections, no AI), `assistant/tools.py` (the functions and their
descriptions for the AI), `assistant/chat.py` (the conversation loop and the Boson chat adapter),
`assistant/offline.py` (the offline helper), `dashboard/views/assistant.py` (the page).

## AI models through Boson

All hosted AI features use **Boson Higgs Realtime** through Boson's Realtime WebSocket API. PaceSense uses one
hosted model for assistant chat and function calling, speech transcription, form extraction and refinement.
Task-specific temperatures remain configurable; the model itself is fixed to `higgs-realtime`.

| Task | Used for | Model |
|---|---|---|
| Ask PaceSense (chat) | Answering questions with PaceSense functions | `higgs-realtime` |
| Speech to text | Transcribing voice recordings | `higgs-realtime` (or optional local Whisper fallback) |
| Fill fields | Turning text into checked form fields | `higgs-realtime` |
| Suggest a refined version | Rewriting input without changing facts | `higgs-realtime` |

### Configure the Boson key

Set `BOSON_API_KEY` on the server. The key must never be committed to Git or entered into browser-side code.

**Render:** open your PaceSense web service → **Environment** → add `BOSON_API_KEY` with your Boson key as its secret value. The Render Blueprint declares this variable as a secret placeholder. The deployment also sets `PACESENSE_AI=1` to enable the assistant.

**Windows PowerShell (local development):**

```powershell
pip install -e ".[dashboard,local-stt]"
$env:BOSON_API_KEY="your-boson-api-key"
pacesense dashboard --data data\races.csv
```

**macOS/Linux:**

```bash
pip install -e ".[dashboard,local-stt]"
export BOSON_API_KEY="your-boson-api-key"
pacesense dashboard --data data/races.csv
```

The Boson client uses the Realtime WebSocket endpoint and the `higgs-realtime` model. The server keeps the
permanent key; it is not exposed to the browser. If the key is absent, PaceSense falls back to its offline
assistant and rule-based form handling.

### Voice input

On **Ask PaceSense**, **Runners**, **Plan a race** and **Saved plans**, voice recordings can be transcribed,
then used to fill and refine form fields. Boson voice transcription currently accepts PCM WAV recordings.
The optional local Whisper fallback can be selected in Settings and requires
`pip install -e ".[local-stt]"`.

### Boson evaluation lab

The evaluation lab runs PaceSense's repeatable test cases against the single Boson model and scores the results:

| Task | Built-in cases | Score |
|---|---|---|
| Ask PaceSense | 6 questions (prediction, weights, similar runners, a past race, the open plan, the watch) | right function called (0.4) + right value in the answer (0.6); halved if it states a time no function returned |
| Fill fields | 6 dictations (runners, plan details, results) | share of expected fields filled correctly |
| Refine | 4 inputs | keeps every number and time without adding any (0.6) + keeps the key words (0.4) |
| Speech to text | your own recording + what you said | 1 − word error rate |

A case passes at 0.8. You can add your own case for each task. Results show score, cases passed, errors and
response time. Provider cost is not currently reported by this integration. Code: `llm/evals.py`,
`dashboard/views/lab.py`.

## What the coach sees in the study (Sec. 4.1)

1. Informed consent, then a short introduction.
2. **Select your runners**: search the database for two of their own runners, add a runner's marathons by
   hand, or *continue without my own runners* (then all four runners are unfamiliar).
3. Four runners, starting at random with the two familiar or the two unfamiliar ones. For each:
   - **Introduction**: gender, age, the five most recent marathons with 5 km pace charts. The coach gives a
     finish time and pacing strategy (**Coach_pre**). For their own runners they also rate how well they
     know the runner (1-5) and the runner's motivations (nine types).
   - **Model page** (Figure 1): the model's target time (**Model_initial**), an expandable explanation of
     the model, the race charts, and in the interactive condition a slider under each race (0-1 in steps of
     0.1, default 0.5) that updates the output (**Model_adapted**). Then the new finish time
     (**Coach_post**), the new pacing strategy and a motivation.
4. Questionnaire: trust, self-efficacy, computer self-efficacy, experience with models, demographics.

For unfamiliar runners the fastest of their five most recent races is hidden and kept as ground truth
(Sec. 4.2.2). Unfamiliar runners are the runners previous coaches selected as their own, matched on the
number of marathons shown; before any exist, they come from the database (Sec. 4.1).

## Where each part of the paper lives

| Paper | Code |
|---|---|
| CBR model: nPB to PB cases, similar runners, recommended pacing (3.1) | `cbr.CaseBase` |
| Several previous races combined with adjustable weights (3.1, 3.2) | `cbr.MarathonModel` |
| Interactive vs non-interactive condition, equal weights 0.5 (3.2) | `study.StudyService`, `dashboard/views/study.py` |
| Procedure, runner selection and assignment (4.1) | `study.StudyService` |
| Study pages and Figure 1 layout | `dashboard/views/study.py` |
| Acceptance, with the 0 cap and the 5-minute rounding rule (4.2.1) | `measures.acceptance` |
| Trust questionnaire and covariates (4.2.1, 4.2.3) | `resources/questionnaire.json` |
| Percent error against the held-out best race (4.2.2) | `measures.percent_error`, `cbr.holdout_best` |
| Tables 1, 2 and 4; Sections 5.3, 5.7-5.10 | `analysis` |

## What the paper does not specify, and what this implementation assumes

- **CBR parameters.** The paper refers to Smyth and Cunningham (2017). Here: 15 nearest cases of the same
  gender, within 5% of the finish time, similar on finish time, 5 km pacing profile and age; prediction =
  race time x weighted mean PB/nPB ratio of the neighbours (`cbr.CBRConfig`).
- **Combining races.** A weighted average of the per-race predictions, using the slider weights.
- **Likely range.** The 20th to 80th percentile of similar runners' results; in validation it contains the
  actual result for about 80% of runners. `pacesense validate` reports this for your data.
- **Questionnaire wording.** The paper prints only a few items; those are used word for word. The rest of
  `questionnaire.json` are placeholders (marked `"source": "adapted"`); replace them with the original
  items (trust: McKnight et al., 2002; computer self-efficacy: Marakas et al., 2007) before a real study.
- **Pacing strategy slider.** In the study it maps to a linear profile from "end fast" to "start fast".
- **Coaching experience.** First principal component of the four experience questions (the paper used a
  factor score).
- **Binary acceptance check.** A linear probability mixed model (the paper used a multilevel logistic model).
- **AI features.** Not in the paper. Default models are inexpensive starting points, not a recommendation;
  the Model lab's built-in cases are few and simple, so add cases that match your own use before choosing.
- **Race-day watch.** Not in the paper. Cues use simple rules: within 15 s is on plan; "slipping" means
  the pace needed to finish on time is more than 2% faster than the plan's pace.

## Data

The paper used 63,000 race records of 24,000 runners from the Amsterdam, Eindhoven and Rotterdam marathons,
2008-2019. That dataset is not included; `data/races.csv` is a synthetic stand-in (`pacesense generate-data`
makes another). Real results load with `--data`:

| column | required | notes |
|---|---|---|
| `runner_id` | yes | stable id |
| `finish` | yes | `h:mm:ss` or seconds |
| `gender` | recommended | `F` / `M` |
| `name`, `age`, `event`, `year` | optional | `name` lets coaches find their own runners |
| `s5` ... `s40` | optional | elapsed time at each 5 km point, for pace charts and pacing similarity |

## JSON API (for other programs)

People use the dashboard. Other programs can use the same services over HTTP: `pacesense serve` (or
gunicorn, below), then `GET /` lists every endpoint. Main groups: `/api/coach/*` (runners, predictions with
`weights` and `unit`, plan checks, saved plans, results, preferences), `/api/sessions/*` (the study
procedure) and `/api/admin/*` (report and CSV exports, `Authorization: Bearer <token>`). Paces are always in
seconds per km.

## Deployment

```bash
docker build -t pacesense .
docker run -p 8000:8000 -v pacesense-data:/data -e PACESENSE_AI=1 -e BOSON_API_KEY pacesense
```

or without Docker:

```bash
export PACESENSE_DB=/var/lib/pacesense/pacesense.sqlite3 PACESENSE_DATA=/srv/races.csv
export PACESENSE_ADMIN_TOKEN=$(openssl rand -hex 24)
export PACESENSE_AI=1
export BOSON_API_KEY=your-boson-api-key            # enable Boson Higgs Realtime
streamlit run src/pacesense/dashboard/app.py --server.port 8501
```

The dashboard has no login of its own: run it on your own computer, or put it behind an authenticating
proxy (or Streamlit's own authentication) if you host it. The JSON API's coach routes also have no authentication;
keep `pacesense serve` bound to localhost or place any public deployment behind authentication and HTTPS. Only
`/api/admin/*` uses `PACESENSE_ADMIN_TOKEN`. Use one server process per SQLite file and back up the database file.

### Render (password-protected demo)

The repository includes a Render Blueprint in `render.yaml`. To deploy it, create a new Blueprint in Render,
connect this GitHub repository, and select the `deploy/render` branch. Render will read the Blueprint and ask you
to provide `PACESENSE_ACCESS_USER`, `PACESENSE_ACCESS_PASSWORD` and `BOSON_API_KEY` as secret environment values. Use a unique
password; do not commit credentials to Git. The Studio app then requires HTTP Basic Authentication for the UI
and API. Render health checks at `/readyz` and `/healthz` remain accessible to the platform.

The Blueprint uses the Free web-service plan for a low-cost demonstration. **Do not use this configuration for
important or long-lived user data:** the SQLite database is stored on the free instance's ephemeral filesystem
and may be lost after a restart, spin-down, or deploy. Free web services also sleep after 15 minutes without
traffic. For durable data, move persistence to a supported database or use a paid service with a persistent disk.
The free Render Postgres tier also expires after 30 days, so it is not a permanent free database option.

## Tests

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -t .     # the full test suite
```

They cover PaceSense Live (interpreter, evidence statistics, sessions, replay, benchmark floors, the checked AI
reader, the HTTP routes and the Plan-page box), `reproduce` (two runs give identical hashes; a changed seed is
detected), the model, measures and statistics, the study procedure, planning sessions, the race-day watch,
the assistant, legacy HTTP-client compatibility tests, voice input and the Boson adapter (including
`tests/test_boson.py`, which runs without a key or network), every chart, and every
dashboard page. Pages are run against `tests/fake_streamlit.py`, a stand-in that takes
Streamlit's own parameter names, rejects anything else, and plays back clicks and typed values. It checks
the page logic and Streamlit calls, not how the page looks; for that, run the dashboard.

`pacesense simulate` runs simulated coaches through the real study code. It exists to test the pipeline and
plan sample sizes; its behaviour is invented, so its results say nothing about real coaches.
