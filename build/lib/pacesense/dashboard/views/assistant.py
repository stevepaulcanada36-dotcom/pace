"""Ask PaceSense: chat with an AI that analyses runners and plans using PaceSense's own functions."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import racewatch
from ...assistant import Assistant, AssistantError, AssistantTools
from ...coach import runner_payload_races
from ...domain import ValidationError, format_time
from .. import ui
from ..voice import refine_box, voice_panel
from .runners import time_field


def _tools() -> AssistantTools:
    return AssistantTools(ui.services().coach, plan=ui.plan, unit=ui.unit(),
                          pending_runner=st.session_state.get("assistant_pending"))


def suggestions() -> list[str]:
    p = ui.plan()
    out = []
    if p is not None:
        half = format_time(racewatch.target_elapsed(p.paces, 21.1) + 60)
        out += [f"Is my plan for {p.name} realistic?", "Which past race pulls the prediction the most?",
                f"At 21.1 km in {half}, am I on plan?"]
    else:
        out += ["Show my saved plans"]
    out += ["Explain the likely range",
            "Add runner Anna de Vries, female, 34: Amsterdam 2023 3:35:48; Rotterdam 2024 3h33m11"]
    return out


def render() -> None:
    ui.page_header("Ask PaceSense", "Ask about a runner, a plan or the data, or describe a runner to add. Every "
                                    "number in an answer comes from PaceSense's own calculations.")
    ui.show_toasts()
    chat = st.session_state.setdefault("chat", [])
    ai = ui.ai()
    model = ai.chat_model()

    top = st.columns([3, 1])
    top[0].caption(f"Answering: **{model.label}**" + (f" · talking about **{ui.plan().name}**, the plan open on "
                                                       "Plan a race" if ui.plan() else ""))
    top[1].button("Clear conversation", key="assistant_clear", on_click=_clear, disabled=not chat)
    if not ai.online:
        st.info("No OpenRouter key, so chat uses the offline helper (common questions only). Voice input can use "
                "local Whisper; see Settings › AI models and voice input for setup. Add an OpenRouter key for "
                "the full AI chat and form-filling features.")
    with st.expander("Privacy"):
        st.markdown(
            "- With an OpenRouter key, your questions and the runner data PaceSense looks up for them are sent to "
            "OpenRouter and the model provider you chose. Without a key, the offline helper answers on this computer.\n"
            "- The assistant never sees the research study's data, and cannot save anything: a runner it prepares "
            "is only added when you press the button.\n"
            "- Which model answers is set under Settings › AI models (task: Ask PaceSense).")

    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander(f"What PaceSense calculated ({len(msg['steps'])})"):
                    for s in msg["steps"]:
                        args = ", ".join(f"{k}={v}" for k, v in s["arguments"].items())
                        st.markdown(f"**{s['name']}**({args})")
                        st.json(s["result"], expanded=False)
            if msg.get("meta"):
                st.caption(msg["meta"])

    pending_card()

    question = None
    if not chat:
        st.caption("Try one of these:")
        for i, text in enumerate(suggestions()):
            if st.button(text, key=f"assistant_suggest_{i}"):
                question = text
    question = composer() or question
    typed = st.chat_input("Ask about a runner, a plan or the data", key="assistant_input")
    question = typed or question
    if question:
        ask(question, model)
        st.rerun()


def composer() -> str | None:
    """Speak a question (or type a longer one), get a clearer version suggested, then send it."""
    with st.container(border=True):
        voice_panel("question", "ask_voice", _apply_question, label="Ask by voice")
        if st.session_state.pop("ask_clear_draft", False):    # sent last run: start empty
            st.session_state["ask_draft"] = ""
        st.session_state.setdefault("ask_draft", "")
        draft = st.text_area("Your question", key="ask_draft", height=80,
                             placeholder="Spoken questions appear here. You can edit them before sending.")
        a, b = st.columns(2)
        send = a.button("Send", key="ask_send", type="primary", disabled=not draft.strip())
        with b:
            refine_box("question", draft, "ask_refine", _apply_question)
    if send:
        # Hide the old voice review and suggestion, but remember the recording so it is not sent again.
        old = st.session_state.get("ask_voice_voice") or {}
        st.session_state["ask_voice_voice"] = {"digest": old.get("digest")}
        st.session_state.pop("ask_refine_refine", None)
        st.session_state["ask_clear_draft"] = True
        return draft.strip()
    return None


def _apply_question(fields: dict, text: str) -> None:
    st.session_state["ask_draft"] = fields.get("question") or text


def ask(question: str, model) -> None:
    chat = st.session_state.setdefault("chat", [])
    history = [{"role": m["role"], "content": m["content"]} for m in chat]
    chat.append({"role": "user", "content": question})
    tools = _tools()
    try:
        with st.spinner("Working it out…"):
            answer = Assistant(model, tools).ask(question, history)
    except AssistantError as e:
        chat.append({"role": "assistant", "content": f"The AI service did not answer: {e}\n\nCheck the key and "
                                                     "the internet connection, or remove the key to use the offline helper."})
        return
    meta = f"{model.label} · {answer.latency_s:.1f} s"
    if answer.usage.cost is not None:
        meta += f" · ${answer.usage.cost:.5f}"
    chat.append({"role": "assistant", "content": answer.text, "model": model.label, "meta": meta,
                 "steps": [{"name": s.name, "arguments": s.arguments, "result": s.result} for s in answer.steps]})
    st.session_state["assistant_pending"] = tools.pending_runner


def pending_card() -> None:
    draft = st.session_state.get("assistant_pending")
    if not draft:
        return
    with st.container(border=True):
        st.markdown(f"**Ready to add: {draft['name']}** ({'female' if draft['gender'] == 'F' else 'male'}"
                    + (f", {draft['age']}" if draft.get("age") else "") + ")")
        st.dataframe(pd.DataFrame([{"Marathon": r["event"], "Year": r["year"], "Finish": r["finish"],
                                    "Splits": "yes" if r.get("splits") else ""} for r in draft["races"]]),
                     hide_index=True)
        a, b = st.columns(2)
        add = a.button("Add this runner and plan a race", key="assistant_add", type="primary")
        b.button("Discard", key="assistant_discard", on_click=lambda: st.session_state.pop("assistant_pending", None))
    if add:
        try:
            races = runner_payload_races(draft["races"], draft.get("age"), time_field)
            rid = ui.services().coach.add_runner(draft["name"], draft["gender"], draft.get("age"), races)["runner_id"]
        except ValidationError as e:
            st.error(str(e))
            return
        st.session_state.pop("assistant_pending", None)
        st.session_state.setdefault("toasts", []).append(f"{draft['name']} added.")
        ui.open_runner(rid)
        ui.go("plan")


def _clear() -> None:
    st.session_state["chat"] = []
    st.session_state.pop("assistant_pending", None)
