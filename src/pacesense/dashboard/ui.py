"""Shared Streamlit helpers: services, preferences, the plan session, the summary bar and explanations.

Pages import `st` from here indirectly; everything Streamlit-specific that more than one page needs lives
in this module, so each page file reads as a plain description of what is on screen.
"""
from __future__ import annotations

from typing import Any, Callable

import streamlit as st
from matplotlib.figure import Figure

from ..domain import format_time
from ..explain import HELP
from ..llm.service import AIService
from ..planning import Change, PlanSession, change_text
from .services import Services, services_from_env

# Filled in by app.py: page key -> st.Page, so pages can link to each other without importing each other.
PAGES: dict[str, Any] = {}
VERDICT = {"realistic": ("Realistic", "green"), "ambitious": ("Ambitious", "orange"), "cautious": ("Cautious", "blue")}


@st.cache_resource(show_spinner="Loading the race database and finding similar runners…")
def services() -> Services:
    """One copy per server process, shared by every browser session."""
    return services_from_env()


def ai() -> AIService:
    """The AI features, with a model per task. A key pasted in Settings is kept for this browser session only."""
    return AIService(services().repo)


# ---- preferences (saved in the database) ------------------------------------------------------
def prefs() -> dict:
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = services().coach.preferences()
    return st.session_state["prefs"]


def save_prefs(**changes: Any) -> None:
    st.session_state["prefs"] = services().coach.update_preferences(changes)
    plan = st.session_state.get("plan")
    if plan is not None and "unit" in changes:
        plan.set_unit(changes["unit"])


def unit() -> str:
    return prefs()["unit"]


# ---- the current planning session --------------------------------------------------------------
def plan() -> PlanSession | None:
    return st.session_state.get("plan")


# Widget values that must survive moving between guided steps or pages. Streamlit drops a widget's value in
# any run where the widget is not drawn; writing it back to session state at the top of each run keeps it.
KEEP = ("target_race", "target_date", "target_notes", "ask_draft", "new_name", "new_gender", "new_age")


def keep_widget_values() -> None:
    for k in [*KEEP, *(k for k in st.session_state if str(k).startswith("lab_models_"))]:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def _forget_voice(*keys: str) -> None:
    for k in keys:
        st.session_state.pop(f"{k}_voice", None)
        st.session_state.pop(f"{k}_refine", None)


def open_runner(runner_id: str) -> None:
    _forget_voice("plan_voice", "plan_notes")
    for k in ("target_race", "target_date", "target_notes"):
        st.session_state.pop(k, None)
    st.session_state["plan"] = PlanSession(services().coach, runner_id, unit())
    st.session_state["step"] = 2


def open_saved_plan(saved: dict) -> None:
    _forget_voice("plan_voice", "plan_notes")
    st.session_state["plan"] = PlanSession.from_saved_plan(services().coach, saved, unit())
    st.session_state["target_race"] = saved["target_race"]
    st.session_state["target_date"] = _date(saved.get("race_date"))
    st.session_state["target_notes"] = saved.get("notes") or ""
    st.session_state["step"] = 5


def announce(change: Change | None) -> None:
    """Queue a message saying what a change did. Queued rather than shown, because changes often happen in
    widget callbacks or right before st.rerun(), where drawing is unreliable; pages call show_toasts()."""
    if change is not None:
        st.session_state.setdefault("toasts", []).append(change.text())


def show_toasts() -> None:
    for message in st.session_state.pop("toasts", []):
        st.toast(message)


def _date(value: str | None):
    import datetime as dt
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:
        return None


# ---- navigation ---------------------------------------------------------------------------------
def go(page_key: str) -> None:
    st.switch_page(PAGES[page_key])


def link(page_key: str, label: str) -> None:
    if page_key in PAGES:
        st.page_link(PAGES[page_key], label=label)


# ---- display helpers ----------------------------------------------------------------------------
def show(fig: Figure) -> None:
    st.pyplot(fig, clear_figure=True)


def why(text: str) -> None:
    """A one-line explanation under a heading, shown when explanations are turned on in Settings."""
    if prefs()["explanations"]:
        st.caption(text)


def tip(key: str) -> str:
    """Tooltip text for a widget's `help=` (the small "?" Streamlit draws next to it)."""
    return HELP[key].tooltip()


def explain(key: str) -> None:
    h = HELP[key]
    with st.expander(f"What is this? {h.title}"):
        st.markdown(h.text)
        st.caption(h.paper)


def delta(seconds: float) -> str | None:
    """A metric delta: '-1:22' (faster, shown green with delta_color='inverse') or '+0:40'."""
    r = round(seconds)
    if r == 0:
        return None
    text = change_text(r).split()[0]
    return f"-{text}" if r < 0 else f"+{text}"


def summary_bar(p: PlanSession) -> None:
    """Runner, model, plan and verdict, with the last change explained. Visible on every step."""
    last = p.changes[0] if p.changes else None
    label, colour = VERDICT[p.check["verdict"]]
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1.6])
        c1.metric("Runner", p.name)
        c2.metric("Model", format_time(p.model_finish), delta(last.model_delta) if last else None,
                  delta_color="inverse", help=tip("prediction"))
        c3.metric("Your plan", format_time(p.plan_finish), delta(last.plan_delta) if last else None,
                  delta_color="inverse")
        c4.markdown(f"**Check**  \n:{colour}[**{label}**]", help=tip("verdict"))
        if last:
            st.caption(f"**Last change:** {last.text()}")
        else:
            st.caption("No changes yet. Adjust the race weights or the pacing and each change will be explained here.")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def button_row(*buttons: tuple[str, str, Callable[[], None] | None]) -> None:
    """Render buttons side by side: (label, key, on_click)."""
    cols = st.columns(len(buttons))
    for col, (label, key, fn) in zip(cols, buttons):
        col.button(label, key=key, on_click=fn)
