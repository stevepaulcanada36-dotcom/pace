"""How it works: the idea, every function and where it comes from, key terms, and how the code is organised."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ...explain import FUNCTIONS, HELP
from .. import ui

ARCHITECTURE = """
```
 Streamlit dashboard (pacesense.dashboard)                        JSON API (pacesense.web)
   pages: plan · watch · ask · runners · saved · lab · settings …   for other programs
   voice.py: record → transcribe → fill fields → suggest refined
        │                                   │
        │                                   ▼
        │                AI (pacesense.llm): AIService, one OpenRouter model per task
        │                  transcribe · extract · refine · assistant  ──►  OpenRouter API
        │                  assistant may only call assistant/tools.py
        ▼                                   ▼
 Services ─ PlanSession (planning.py) · racewatch.py · CoachService (coach.py) · StudyService (study.py)
        │
        ▼
 Model ─ CaseBase + MarathonModel (cbr.py): similar runners, race weights, likely range
        │
        ▼
 Data ─ race CSV (data.py) · SQLite: runners, plans, results, settings, model choices, lab runs (storage.py)
```
"""


def render() -> None:
    ui.page_header("How PaceSense works", "Every function, where to find it, and whether it comes from the research paper.")
    if st.button("Start the guided steps", key="tour", type="primary"):
        ui.save_prefs(view="guided")
        st.session_state["step"] = 2 if ui.plan() else 1
        ui.go("plan")
    st.subheader("The idea in one paragraph")
    st.markdown("A model predicts a challenging yet realistic marathon time from runners who ran a similar race and "
                "later ran their best. The coach knows things the race data doesn't, so the coach can tell the model "
                "how representative each past race is and watch the prediction change. The research paper (Muijlwijk "
                "et al., IUI '24) tested whether that interaction makes coaches accept and trust the model more, and "
                "whether it makes the model more accurate. The dashboard is a planning tool built on the same model; "
                "the Research study page runs the study as the paper did. The Race-day watch shows how a plan could "
                "be used during the race itself. Ask PaceSense lets you ask about runners and plans in your own words: "
                "an AI picks which PaceSense functions to run and explains their results, without calculating "
                "anything itself.")
    st.subheader("All functions")
    st.dataframe(pd.DataFrame([{"Function": f.name, "Where": f.where, "What it does": f.what,
                                "Comes from": "The paper" if f.source == "Paper" else "PaceSense", "Note": f.note}
                               for f in FUNCTIONS]), hide_index=True)
    st.subheader("Key terms")
    for k in ("prediction", "target", "range", "weights", "equal", "pacing", "intervals", "strategies", "verdict",
              "changes", "watch", "assistant", "voice", "lab"):
        with st.expander(HELP[k].title):
            st.markdown(HELP[k].text)
            st.caption(HELP[k].paper)
    st.subheader("How the code is organised")
    st.markdown(ARCHITECTURE)
    st.caption("Each layer only calls the layer below it, so the model and services work from Python or a notebook "
               "without the dashboard.")
