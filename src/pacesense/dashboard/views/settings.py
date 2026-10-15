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
    """One OpenRouter model (and temperature) per AI task. Saved in the database."""
    import os
    st.subheader("AI models and voice input")
    st.caption("Chat, field filling and refinement use the per-task OpenRouter models below. Voice transcription can "
               "run locally for no API fee, or use OpenRouter (which may require credits).")
    repo = ui.services().repo
    backend = str(repo.get_setting("transcription_backend", os.environ.get("PACESENSE_TRANSCRIBE_BACKEND", "local")))
    selected = st.radio("Speech-to-text method", ["local", "openrouter"], index=0 if backend == "local" else 1,
                        format_func={"local": "Local Whisper — no per-recording API charge",
                                     "openrouter": "OpenRouter — may cost money"}.get,
                        key=f"transcription_backend_{backend}", horizontal=False,
                        help="Local mode needs the optional faster-whisper package and downloads a model on first use.")
    if selected != backend:
        repo.set_setting("transcription_backend", selected)
        st.session_state.setdefault("toasts", []).append("Speech-to-text method saved.")
        st.rerun()
    if backend == "local":
        st.caption('Local model: faster-whisper/base by default. Install with: pip install -e ".[local-stt]". '
                   "Set PACESENSE_LOCAL_WHISPER_MODEL=small for better accuracy on a capable computer.")
    else:
        st.caption("OpenRouter transcription uses the Speech to text model below and may be blocked by balance limits.")
    ai = ui.ai()
    if ai.online:
        source = ai.key_source or "configuration"
        st.success(f"OpenRouter key loaded from {source}.")
        if source == "browser session":
            a, b = st.columns([3, 1])
            a.caption("Key is held for this browser session only.")
            b.button("Forget the key", key="ai_forget_key", on_click=lambda: st.session_state.pop("openrouter_key", None))
    else:
        st.caption("Preferred setup: put the key in config/local.yaml (copy config/local.example.yaml), or set OPENROUTER_API_KEY.")
        # Kept as a convenience fallback; not required when the local YAML config has a key.
        st.text_input("Temporary OpenRouter API key (this browser session only)", key="ai_key_input",
                      type="password", placeholder="sk-or-…", on_change=_store_key,
                      help="Create a key at openrouter.ai/keys. A local config file is the easiest persistent setup.")
        st.info(ai.why_offline())
    settings = ai.settings
    for key, task in TASKS.items():
        cur = settings.get(key)
        with st.container(border=True):
            st.markdown(f"**{task.title}**  \n{task.what}")
            a, b, c = st.columns([3, 1.3, 1])
            model = a.text_input("Model", value=cur["model"], key=f"ai_model_{key}_{cur['model']}",
                                 help="Starting points: " + ", ".join(task.suggestions))
            temp = b.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(cur["temperature"]),
                                  step=0.1, key=f"ai_temp_{key}_{cur['temperature']}",
                                  help="0 = the same answer every time; higher = more varied wording.")
            c.write("")
            c.button("Default", key=f"ai_reset_{key}", on_click=settings.reset, args=(key,),
                     disabled=not settings.has_saved(key), help=f"Back to {task.default_model} (or the environment).")
            st.caption(f"Now: {cur['model']} · from {cur['source']}")
            new_model = model.strip() if model.strip() != cur["model"] else None      # save only what changed
            new_temp = float(temp) if abs(float(temp) - cur["temperature"]) > 1e-9 else None
            if new_model is not None or new_temp is not None:
                try:
                    settings.update(key, model=new_model, temperature=new_temp)
                    st.session_state.setdefault("toasts", []).append(f"{task.title}: {model.strip() or 'default'}")
                    st.rerun()
                except ValidationError as e:
                    st.error(str(e))
    model_list(ai)


def _store_key() -> None:
    key = (st.session_state.get("ai_key_input") or "").strip()
    if key:
        st.session_state["openrouter_key"] = key


def model_list(ai) -> None:
    if not ai.online:
        return
    if st.button("Load the model list from OpenRouter", key="ai_load_models"):
        try:
            st.session_state["or_models"] = ai.client.models()
        except LLMError as e:
            st.error(f"Could not load the model list: {e}")
    models = st.session_state.get("or_models")
    if models:
        import pandas as pd
        q = st.text_input("Filter models", key="ai_model_filter", placeholder="e.g. gemini, transcribe, llama")
        kind = st.radio("Show", ["all", "function calling (for chat)", "audio input (for speech to text)"],
                        horizontal=True, key="ai_model_kind")
        rows = [m for m in models if q.lower() in (m["id"] + " " + m["name"]).lower()
                and (kind == "all" or (m["tools"] if kind.startswith("function") else "audio" in m["inputs"]
                                       or "transcribe" in m["id"] or "whisper" in m["id"]))]
        st.dataframe(pd.DataFrame([{"Model id": m["id"], "Name": m["name"], "$ per M input tokens": m["prompt_usd_per_mtok"],
                                    "$ per M output tokens": m["completion_usd_per_mtok"], "Context": m["context"],
                                    "Inputs": ", ".join(m["inputs"]), "Tools": "yes" if m["tools"] else ""}
                                   for m in rows[:300]]), hide_index=True)
        shown = min(300, len(rows))
        st.caption(f"Showing {shown} of {len(rows)} matching models ({len(models)} catalog entries). Copy an id "
                   "into a task above or into the Model lab.")
