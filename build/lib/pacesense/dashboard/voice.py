"""Voice input and "suggest a clearer version", reusable on any page.

    voice_panel(form, key, apply)  record or upload → transcribed automatically → apply(fields, text) fills the
                                   page's inputs → a refined version is suggested; "Use the refined version"
                                   calls apply() again with fields read from the refined text.
    refine_box(form, text, key, apply)  the same suggestion for typed text.

`apply` is the page's own function that writes the fields into its widgets' session state. It runs before
those widgets are drawn (straight after processing, or in a button callback), as Streamlit requires.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import streamlit as st

from ..llm.openrouter import LLMError
from ..llm.service import Refinement, describe_usage
from . import ui

Apply = Callable[[dict[str, Any], str], None]
AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "webm", "flac"]


def _audio(key: str) -> tuple[bytes, str] | None:
    """A recording from the microphone, or an uploaded file. Returns (bytes, format)."""
    rec_col, up_col = st.columns(2)
    rec = rec_col.audio_input("Record", key=f"{key}_rec", help="Press the microphone, speak, press stop. "
                                                                "It is transcribed straight away.")
    up = up_col.file_uploader("…or upload a recording", type=AUDIO_TYPES, key=f"{key}_file")
    if rec is not None:
        if up is not None:
            st.caption("Using the recording. Clear it (✕) to use the uploaded file instead.")
        return rec.getvalue(), "wav"
    if up is not None:
        ext = (up.name.rsplit(".", 1)[-1] if "." in up.name else "wav").lower()
        return up.getvalue(), ext
    return None


def voice_panel(form: str, key: str, apply: Apply, label: str = "Speak instead of typing") -> None:
    ai = ui.ai()
    st.markdown(f"**🎙️ {label}**")
    if ai.transcription_backend == "openrouter" and not ai.online:
        st.caption("OpenRouter transcription needs an API key. Or choose Local Whisper under Settings for no API fee.")
        ui.link("settings", "Open Settings")
        return
    got = _audio(key)
    state = st.session_state.setdefault(f"{key}_voice", {})
    if got is not None:
        audio, fmt = got
        digest = hashlib.sha1(audio).hexdigest()
        if state.get("digest") != digest:              # a new recording: process it once, automatically
            state.clear()
            state["digest"] = digest
            try:
                with st.spinner("Transcribing and filling in…"):
                    t = ai.transcribe(audio, fmt)
                    if not t.text:
                        raise LLMError("no speech was recognised in the recording")
                    r = ai.process_text(form, t)
            except LLMError as e:
                state["error"] = f"Could not transcribe the recording: {e}"
            else:
                _store(state, r)
                apply(r.extraction.fields if r.extraction else {}, t.text)
                st.session_state.setdefault("toasts", []).append(
                    "Transcribed and filled in. Check the suggestion below." if not r.errors else
                    "Transcribed, but a step failed: see below.")
                st.rerun()
    if state.get("error"):
        st.error(state["error"])
        st.button("Try again", key=f"{key}_retry", on_click=state.pop, args=("digest", None))
    if state.get("transcript"):
        _review(form, key, state, apply)


def _store(state: dict, r) -> None:
    """Keep what the page needs from a VoiceResult between reruns."""
    state.update(transcript=r.transcript.text, errors=r.errors, refinement=r.refinement,
                 check=_check_dict(r.extraction.check) if r.extraction else None,
                 steps=[f"Speech to text: {r.transcript.model} ({r.transcript.latency_s:.1f} s"
                        + (f", {describe_usage(r.transcript.usage)}" if describe_usage(r.transcript.usage) else "") + ")",
                        _step_text("Fill fields", r.extraction.step) if r.extraction else "",
                        _step_text("Refine", r.refinement.step) if r.refinement else ""],
                 _transcript=r.transcript)


def _retry_steps(form: str, key: str, apply: Apply) -> None:
    """Fill and refine again from the stored transcript, without transcribing (and paying) again."""
    state = st.session_state.get(f"{key}_voice") or {}
    if not state.get("_transcript"):
        return
    r = ui.ai().process_text(form, state["_transcript"])
    _store(state, r)
    if r.extraction:
        apply(r.extraction.fields, r.transcript.text)


def _check_dict(c) -> dict:
    return {"corrections": c.corrections, "problems": c.problems, "missing": c.missing}


def _step_text(name: str, step) -> str:
    u = describe_usage(step.usage)
    return f"{name}: {step.model} ({step.latency_s:.1f} s{', ' + u if u else ''})"


def _review(form: str, key: str, state: dict, apply: Apply) -> None:
    with st.container(border=True):
        st.caption("What you said")
        st.markdown(f"> {state['transcript']}")
        chk = state.get("check")
        if chk:
            for c in chk["corrections"]:
                st.caption(f"Corrected: {c}")
            for p in chk["problems"]:
                st.warning(p)
            if chk["missing"]:
                st.info("Not mentioned: " + "; ".join(chk["missing"]))
        if state.get("errors"):
            for e in state["errors"]:
                st.error(e)
            st.button("Try these steps again", key=f"{key}_retry_steps", on_click=_retry_steps,
                      args=(form, key, apply), help="Uses the same transcript; the recording is not sent again.")
        r: Refinement | None = state.get("refinement")
        if r is not None:
            _suggestion(form, key, r, apply, extract=True)
        st.caption(" · ".join(s for s in state.get("steps", []) if s))


def _suggestion(form: str, key: str, r: Refinement, apply: Apply, extract: bool = True) -> None:
    st.markdown("**Suggested clearer version**")
    st.info(r.text)
    for c in r.changes[:5]:
        st.caption(f"• {c}")
    for q in r.questions[:3]:
        st.caption(f"❓ {q}")
    if r.added:
        st.warning("The suggestion has numbers that you did not say: " + ", ".join(r.added) + ". Check them.")
    if r.dropped:
        st.warning("The suggestion leaves out: " + ", ".join(r.dropped) + ".")
    a, b = st.columns(2)
    a.button("Use the refined version", key=f"{key}_use_refined", type="primary",
             on_click=_use_refined, args=(form, key, r.text, apply, extract))
    b.button("Keep my words", key=f"{key}_keep", on_click=_dismiss, args=(key,))


def _use_refined(form: str, key: str, text: str, apply: Apply, extract: bool) -> None:
    """Fill the page with the refined version: its fields are read again (extract) when the page has fields."""
    fields: dict[str, Any] = {}
    if extract:
        try:
            fields = ui.ai().extract(form, text).fields
        except LLMError as e:
            st.session_state.setdefault("toasts", []).append(f"Could not read the refined version: {e}")
            return
    apply(fields, text)
    _dismiss(key)
    st.session_state.setdefault("toasts", []).append("Refined version filled in.")


def _dismiss(key: str) -> None:
    voice = st.session_state.get(f"{key}_voice")
    if voice:
        voice.pop("refinement", None)
    st.session_state.pop(f"{key}_refine", None)


def refine_box(form: str, text: str, key: str, apply: Apply, label: str = "Suggest a clearer version",
               extract: bool = False) -> None:
    """A button that asks the refine model for a clearer version of typed text, and shows it for review.
    With extract=True, using it also re-reads the form's fields from the refined text."""
    if st.button(label, key=f"{key}_ask", disabled=not (text or "").strip(),
                 help=f"Uses {ui.ai().label('refine')}. Nothing changes until you choose to use it."):
        try:
            with st.spinner("Thinking…"):
                st.session_state[f"{key}_refine"] = ui.ai().refine(form, text)
        except LLMError as e:
            st.error(f"No suggestion: {e}")
    r = st.session_state.get(f"{key}_refine")
    if r is not None:
        with st.container(border=True):
            _suggestion(form, key, r, apply, extract=extract)
            st.caption(_step_text("Refine", r.step))


__all__ = ["voice_panel", "refine_box"]
