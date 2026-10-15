"""Saved plans: reopen a plan, see it on the watch, and record the actual result after the race."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ...domain import ValidationError, format_time, parse_time
from .. import ui
from ..voice import voice_panel


def render() -> None:
    ui.page_header("Saved plans", "After a race, record the result to see how close the plan and the model were.")
    coach = ui.services().coach
    plans = coach.list_plans()
    if not plans:
        st.info("No saved plans yet. Plan a race and save it; it will appear here.")
        ui.link("plan", "Plan a race")
        return
    st.dataframe(pd.DataFrame([{"Runner": p["runner_name"], "Target race": p["target_race"], "Date": p["race_date"] or "",
                                "Your plan": p["finish"], "Model": format_time(p["model_finish_s"]),
                                "Result": p["result"] or "",
                                "Plan off by": f"{p['errors_pct']['plan']:.1f}%" if p["result"] else ""} for p in plans]),
                 hide_index=True)
    labels = {p["id"]: f"{p['runner_name']} · {p['target_race']} · {p['finish']}" for p in plans}
    pid = st.selectbox("Plan", list(labels), format_func=labels.get, key="saved_pick")
    chosen = next(p for p in plans if p["id"] == pid)
    a, b = st.columns(2)
    if a.button("Open in Plan a race", key="saved_open"):
        ui.open_saved_plan(chosen)
        ui.go("plan")
    if b.button("Show on the race-day watch", key="saved_watch"):
        st.session_state["watch_plan"] = pid
        ui.go("watch")

    result_key = f"result_{pid}"
    st.session_state.setdefault(result_key, chosen["result"] or "")

    def apply(fields: dict, text: str) -> None:
        if fields.get("finish"):
            st.session_state[result_key] = fields["finish"]

    voice_panel("result", f"result_voice_{pid}", apply, label="Say the finish time")
    with st.form("record_result"):
        st.markdown(f"**Record the result** for {chosen['runner_name']}, {chosen['target_race']}")
        finish = st.text_input("Actual finish time (h:mm:ss)", key=result_key)
        if st.form_submit_button("Record result", type="primary"):
            try:
                seconds = parse_time(finish)
                if seconds is None:
                    raise ValidationError("enter the finish time as h:mm:ss, for example 3:31:20")
                done = coach.record_result(pid, seconds)
                st.success(f"Recorded. The plan was off by {done['errors_pct']['plan']:.1f}%, the model with equal "
                           f"weights by {done['errors_pct']['model_default']:.1f}%.")
            except ValidationError as e:
                st.error(str(e))
