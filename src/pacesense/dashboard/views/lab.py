"""Model lab: run the same test cases through several OpenRouter models and choose one per task."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ...llm import evals
from ...llm.forms import FORMS
from ...llm.tasks import TASKS
from .. import ui


def render() -> None:
    ui.page_header("Model lab", "Run the same test cases through several OpenRouter models, compare score, speed "
                                "and cost, and choose the model for each task.")
    ui.show_toasts()
    ai, coach = ui.ai(), ui.services().coach
    if not ai.online:
        st.info("The Model lab calls OpenRouter models, so it needs an OpenRouter key. Add it under Settings › AI "
                "models. Saved results from earlier runs are still shown below.")
        ui.link("settings", "Open Settings")

    task_key = st.selectbox("Task", list(TASKS), format_func=lambda k: TASKS[k].title, key="lab_task")
    task = TASKS[task_key]
    current = ai.settings.get(task_key)
    st.caption(f"{task.what} Now using **{current['model']}** (from {current['source']}).")

    models_key = f"lab_models_{task_key}"
    st.session_state.setdefault(models_key, "\n".join(dict.fromkeys([current["model"], *task.suggestions])))
    models_text = st.text_area("Models to compare: one OpenRouter model id per line", key=models_key, height=120,
                               help="Find ids with 'Load the model list' under Settings › AI models, or on "
                                    "openrouter.ai/models. The suggestions are starting points; models change.")
    models = [m.strip() for m in models_text.splitlines() if m.strip()]

    built_in = evals.cases(task_key, coach)
    labels = {c.id: f"{c.id}: {c.input[:70]}{'…' if len(c.input) > 70 else ''}" for c in built_in}
    picked = st.multiselect("Test cases", list(labels), default=list(labels), format_func=labels.get,
                            key=f"lab_cases_{task_key}")
    chosen = [c for c in built_in if c.id in picked]
    own = own_case(task_key)
    if own is not None:
        chosen.append(own)
    st.caption(f"{evals.SCORING[task_key]} A case passes at {evals.PASS:.0%} or more.")

    n = len(models) * len(chosen)
    if st.button(f"Run {n} test{'s' if n != 1 else ''}", key="lab_run", type="primary",
                 disabled=not (ai.online and n)):
        bar = st.progress(0.0, text="Starting…")
        results = evals.run_lab(ai, task_key, models, chosen, coach, progress=lambda f, t: bar.progress(f, text=t))
        evals.save_results(ui.services().repo, results)
        st.session_state["lab_last"] = (task_key, results)

    last = st.session_state.get("lab_last")
    if last and last[0] == task_key:
        show_results(task_key, last[1])
    history(task_key)


def own_case(task_key: str) -> evals.Case | None:
    """An extra case of your own: needed for speech to text (a recording plus what was said)."""
    if task_key == "assistant":
        return None
    with st.container(border=True):
        st.markdown("**Your own case**" + (" (needed for speech to text)" if task_key == "transcribe" else " (optional)"))
        if task_key == "transcribe":
            rec = st.audio_input("Record a test sentence", key="lab_audio")
            ref = st.text_input("What you said, word for word (the reference)", key="lab_reference")
            if rec is not None and ref.strip():
                return evals.Case("your-recording", "transcribe", "(recording)", {"reference": ref},
                                  audio=rec.getvalue(), audio_format="wav")
            return None
        form = st.selectbox("Input type", [k for k in FORMS if task_key == "refine" or k != "question"],
                            format_func=lambda k: FORMS[k].title, key=f"lab_own_form_{task_key}")
        text = st.text_area("Input text", key=f"lab_own_text_{task_key}")
        if task_key == "refine":
            keep = st.text_input("Words the refined version must keep (comma separated)", key="lab_own_keep")
            if text.strip():
                return evals.Case("your-text", "refine", text, {"keep": [k.strip() for k in keep.split(",") if k.strip()]},
                                  form=form)
            return None
        expected = st.text_area("Expected fields as JSON", key="lab_own_expected",
                                placeholder='{"finish": "3:21:04"}  ·  runner: {"name": …, "gender": "F", "age": 34, '
                                            '"races": [["Amsterdam", 2023, "3:35:48"]]}')
        if text.strip() and expected.strip():
            try:
                return evals.Case("your-text", "extract", text, json.loads(expected), form=form)
            except json.JSONDecodeError as e:
                st.error(f"The expected fields are not valid JSON: {e}")
        return None


def show_results(task_key: str, results: list) -> None:
    st.subheader("Results")
    table = evals.summary(results)
    st.dataframe(table, hide_index=True)
    st.caption("Best first: highest average score, then fastest. Cost is what OpenRouter reported for these runs.")
    rows = pd.DataFrame([{"Model": r.model, "Case": r.case_id, "Score": r.score, "Passed": "yes" if r.passed else "",
                          "Seconds": r.latency_s, "Cost (USD)": r.cost_usd, "Notes": r.error or "; ".join(r.details),
                          "Output": r.output[:300]} for r in results])
    st.dataframe(rows, hide_index=True)
    if not table.empty:
        a, b = st.columns([3, 1])
        best = a.selectbox(f"Use this model for {TASKS[task_key].title}", list(table["Model"]), key="lab_pick")
        b.write("")
        if b.button("Use it", key="lab_use", type="primary"):
            ui.ai().settings.update(task_key, model=best)
            st.success(f"{TASKS[task_key].title} now uses {best}. Change it any time under Settings › AI models.")


def history(task_key: str) -> None:
    df = evals.saved_results(ui.services().repo)
    if df.empty:
        return
    st.subheader("All saved runs")
    st.caption("Every run is kept in the database, so you can compare models across days. Download it for your report.")
    mine = df[df["task"] == task_key]
    if not mine.empty:
        by_model = (mine.groupby("model").agg(runs=("score", "size"), score=("score", "mean"),
                                              seconds=("latency_s", "mean")).round(3)
                    .sort_values("score", ascending=False).reset_index())
        st.dataframe(by_model, hide_index=True)
    st.download_button("Download all runs (CSV)", df.to_csv(index=False), file_name="pacesense_model_lab.csv",
                       mime="text/csv", key="lab_csv")
