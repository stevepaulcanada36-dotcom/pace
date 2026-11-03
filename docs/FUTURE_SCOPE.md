# PaceSense 3.0: from one feedback loop to many

This document sets out the product's future direction, starting from Muijlwijk, Willemsen, Smyth &
IJsselsteijn, *Benefits of Human-AI Interaction for Expert Users Interacting with Prediction Models: a Study on
Marathon Running* (IUI '24), and shows where each idea is implemented in this code.

## Where the paper leaves us

| The paper found | What stayed open |
|---|---|
| Sliders raised **acceptance** (β = 0.266, p < .001) and **perceived competence** (β = 0.780, p < .001) | Acceptance is not trust or understanding. Coaches who steer the model toward their own estimate score high on acceptance but may simply be confirming themselves (Sec. 6). |
| Coaches relied on the model more for **unfamiliar** runners (RQ2) | The same interface was used for everyone. Nothing adapted to who the coach is or how well they know the runner. |
| Coaches **improved accuracy** (3.14% → 2.33% error, p = .018, RQ4) | Interacting did not make coaches *learn*: nobody saw which of their changes actually helped. |
| One interaction loop: move a slider, see a new prediction | The loop ends there: there is no follow-up, no question back to the coach, and no memory across sessions or across coaches. |
| Think-aloud with 7 coaches surfaced rich knowledge (injuries, heat, form) | That knowledge was spoken, but the interface only took slider positions. |

## The product: PaceSense Studio 3.0 (adaptive)

PaceSense Studio 3.0 has three ideas. Each is implemented and tested.

### 1. N feedback loops instead of one (`src/pacesense/adaptive/loops.py`)

Every coach action becomes an `Event`. Nine independent loops watch the events, at four time scales, and may
respond with an `Adaptation`. A loop can rearrange the screen, ask the coach a question, add an insight, or
update the coach's persona. **No loop changes the model on its own; the coach stays in control.**

| # | Loop | Time scale | What it does | Open question it answers |
|---|---|---|---|---|
| 1 | Persona | across sessions | Learns how the coach works, from what they say, ask and change | One interface for every expert |
| 2 | Data check | now | Raises a change that contradicts the runner's own times, and offers keep / undo / show similar runners | Interaction can help *or* harm |
| 3 | Clarify | now | When a cue such as "hot" or "injured" is heard without a race, asks which race was meant | Talking is ambiguous |
| 4 | Bias guard | this session | When the model is ≥ 85% of the way to the coach's own estimate, asks "knowledge or expectation?". It fires once, re-arms, and is stricter for familiar runners and intuition-led coaches | Acceptance inflated by steering (Sec. 6) |
| 5 | Explanations | this session | "Not sure" reactions switch to full explanations; three "makes sense" reactions in a row switch to short ones | Acceptance ≠ understanding |
| 6 | Guided interview | this session | Asks five questions built from the data (her slowest race, her fastest, trend, own estimate) | An easier way in for experts |
| 7 | Track record | across sessions | After every reveal, records whether the coach's changes helped, and by kind of change; opens the next session with it | Coaches did not learn (RQ4) |
| 8 | Shared learning | all coaches | Pools, across coaches, how often each kind of change (heat, injury, recency…) made predictions more accurate. It is shown as evidence and never applied automatically | Expert knowledge is lost after each session |
| 9 | Layout | now | Voice opens a voice stage; the race being talked about comes forward; the layout follows the persona unless the coach locks it | The UI should follow the conversation |

### 2. Persona understanding (`src/pacesense/adaptive/persona.py`)

- **Four lasting dimensions**, each a Beta estimate in [0, 1]:
  - reliance (intuition-led ↔ data-led);
  - detail (just the answer ↔ wants the reasons);
  - modality (clicks ↔ talks);
  - certainty (exploring ↔ decisive).
- **One per-runner dimension:** familiarity, reset with every runner; this follows RQ2 of the paper.
- **Archetypes:** the nearest of four is shown as a name the coach can recognise: The Analyst, The Intuitive
  Expert, The Pragmatist and The Explorer.
- **Transparent:**
  - every nudge keeps the cue that caused it ("Because you said 'trust me'");
  - the coach sees the profile and can set any dimension, and their setting wins until reset;
  - "Forget me" deletes the profile.
- **Never inferred from who the coach is:** only from how they work in this app.
- **What the persona changes:**
  - the layout: analysis-first, races-first, summary-first or guided;
  - how much is explained;
  - whether replies are spoken, and how long they are;
  - how quickly the bias guard speaks up.

### 3. Voice-first, self-arranging interface (`src/pacesense/studio/static/adaptive.js`)

- **Talk** (or the mic, or the V key) turns the screen into a **voice stage**:
  - a live transcript;
  - the target in large type;
  - one card per race, where the race you mention lights up;
  - PaceSense's current question with large choices you can tap or say.
- **Replies are spoken** and the microphone pauses while PaceSense speaks. With no microphone, typing works the same way.
- **Guide me** (G) starts the interview: the expert answers questions instead of operating controls. Every
  answer is turned into a sentence for the same interpreter, so the data checks still apply.
- **The feedback-loops bar** shows all nine loops and which one just fired; "Why did the screen change?" lists every
  adaptation with its reason.

## Try it

```bash
pip install -e ".[studio]"
pacesense studio
```

1. Say or type "I know her, 2010 she was injured, definitely". The persona moves toward *Intuitive Expert*
   and the layout becomes races-first.
2. Type "it was a hot day" without naming a race. The Clarify loop asks which race you meant.
3. Press **Talk**. The voice stage opens; mention a race and its card lights up.
4. Press **Guide me** and answer the questions.
5. Press **Reveal**. The Track record loop scores your changes, and the next session opens with your record.

## How it could be evaluated (the next study)

- **Design:** the paper's 2×2 design (interactive vs. non-interactive, familiar vs. unfamiliar runner) plus a third
  arm, *adaptive*: all loops on.
- **Measures:**
  - the paper's measures: acceptance, perceived competence, willingness to depend, and accuracy against the hidden race;
  - **calibration:** do coaches accept the model more when it is right?
  - **learning:** does a coach's improvement in accuracy rise across sessions?
- **Separating steering from trust:** the Bias guard prompts and their answers, together with the steering index, give a
  direct measure of steering, which the paper could not separate from trust.
- **Data:** every session export already contains the transcript, the adaptations and their reasons.

## Limits and safeguards

- **No automatic changes:** the loops never change weights; Shared learning is evidence, not a rule.
- **Personas are hypotheses about how someone works, not labels for who they are:** they are visible, editable and deletable.
- **Interpreting speech is rule-based:** the AI reader is optional and checked by code, and the loops ask when unsure
  instead of guessing.
- **Heat and wind in the simulator** are stated assumptions, not effects fitted on data.
