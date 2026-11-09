"""Settings: how the dashboard looks for you. Saved in the database, so they survive a reload."""
from __future__ import annotations

import streamlit as st

from ...domain import ValidationError
from ...llm.openrouter import LLMError
from ...llm.tasks import TASKS
from .. import ui


def render() -> None:
    ui.page_header("Settings", "Choose how the dashboard looks for you. Changes are saved straight away.")
    ui.show_toasts()
    p = ui.prefs()
    view = st.radio("Layout", ["guided", "full"], index=0 if p["view"] == "guided" else 1, key=f"set_view_{p['view']}",
                    format_func={"guided": "Guided steps: one step at a time, with Back and Next",
                                 "full": "All controls: everything on one page"}.get)
    explanations = st.toggle("Show a short explanation under each step", value=p["explanations"],
                             key=f"set_expl_{p['explanations']}", help="The '?' tooltips always work, whatever you choose.")
    unit = st.radio("Pace unit", ["km", "mi"], index=0 if p["unit"] == "km" else 1, key=f"set_unit_{p['unit']}",
                    format_func={"km": "Minutes per kilometre", "mi": "Minutes per mile (intervals stay 5 km)"}.get)
    overlay = st.toggle("Show the runner's most recent race on the pacing chart", value=p["overlay_last_race"],
                        key=f"set_overlay_{p['overlay_last_race']}")
    changes = {k: v for k, v in (("view", view), ("explanations", explanations), ("unit", unit),
                                 ("overlay_last_race", overlay)) if v != p[k]}
    if changes:
        ui.save_prefs(**changes)
        st.session_state.setdefault("toasts", []).append("Saved.")
        st.rerun()
    st.divider()
    st.markdown("**Welcome**")
    st.caption("Show the 'How would you like to use PaceSense?' choice again on Plan a race.")
    if st.button("Show the welcome again", key="set_welcome"):
        ui.save_prefs(experience=None)
        ui.go("plan")
    st.divider()
    ai_models()


def ai_models() -> None:
    """Boson provider status and per-task temperature settings."""
    import os
    st.subheader("Boson AI and voice input")
    st.caption("PaceSense uses Boson Higgs Realtime for hosted chat, speech transcription, field extraction and refinement. "
               "The API key is configured on the server and is never entered or stored in this dashboard.")
    repo = ui.services().repo
    backend = str(repo.get_setting("transcription_backend",
                                   os.environ.get("PACESENSE_TRANSCRIBE_BACKEND", "boson")))
    selected = st.radio("Speech-to-text method", ["boson", "local"],
                        index=0 if backend != "local" else 1,
                        format_func={"boson": "Boson Higgs Realtime", "local": "Local Whisper — offline fallback"}.get,
                        key=f"transcription_backend_{backend}", horizontal=False)
    if selected != backend:
        repo.set_setting("transcription_backend", selected)
        st.session_state.setdefault("toasts", []).append("Speech-to-text method saved.")
        st.rerun()
    if selected == "local":
        st.caption('Local fallback uses faster-whisper. Install with: pip install -e ".[local-stt]".')
    ai = ui.ai()
    if ai.online:
        st.success(f"Boson API key loaded from {ai.key_source or 'server configuration'}. Model: higgs-realtime.")
    else:
        st.warning("Boson AI is offline. Add BOSON_API_KEY to the server environment (Render → your service → Environment), then redeploy.")
        st.caption(ai.why_offline() or "")
    for key, task in TASKS.items():
        cur = ai.settings.get(key)
        with st.container(border=True):
            st.markdown(f"**{task.title}**  \\n{task.what}")
            st.caption(f"Model: higgs-realtime · settings source: {cur['source']}")
            a, b = st.columns([2, 1])
            temp = a.number_input("Temperature", min_value=0.0, max_value=2.0,
                                  value=float(cur["temperature"]), step=0.1,
                                  key=f"boson_temp_{key}_{cur['temperature']}",
                                  help="Lower values are more consistent; higher values allow more variation.")
            b.write("")
            b.button("Default", key=f"boson_reset_{key}", on_click=ai.settings.reset, args=(key,),
                     disabled=not ai.settings.has_saved(key))
            new_temp = float(temp) if abs(float(temp) - cur["temperature"]) > 1e-9 else None
            if new_temp is not None:
                try:
                    ai.settings.update(key, temperature=new_temp)
                    st.session_state.setdefault("toasts", []).append(f"{task.title}: temperature saved.")
                    st.rerun()
                except ValidationError as e:
                    st.error(str(e))
