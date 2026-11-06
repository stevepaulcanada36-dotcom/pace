"""Runners: search your own runners and the race database, and add a runner by hand."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ...coach import runner_payload_races
from ...domain import ValidationError, parse_time
from .. import ui
from ..voice import voice_panel

BLANK_RACES = pd.DataFrame({"Marathon": ["", ""], "Year": [None, None], "Finish (h:mm:ss)": ["", ""],
                            "Splits at 5…40 km (optional)": ["", ""]})


def render() -> None:
    ui.page_header("Runners", "Your runners, and runners from the race database.")
    coach = ui.services().coach
    q = st.text_input("Search", key="runners_query", placeholder="Name or runner number, for example R00042")
    results = coach.search(q, limit=25)
    if results:
        st.dataframe(pd.DataFrame([{"Runner": r["name"], "Yours": "yes" if r["own"] else "", "Gender": r["gender"],
                                    "Age": r["age"], "Marathons": r["n_races"], "Best": r["best"],
                                    "Best race": r["best_race"]} for r in results]), hide_index=True)
        labels = {r["runner_id"]: f"{r['name']} · best {r['best']}" for r in results}
        a, b = st.columns([3, 1])
        rid = a.selectbox("Plan a race for", list(labels), format_func=labels.get, key="runners_pick")
        b.write("")
        if b.button("Plan a race", key="runners_plan", type="primary"):
            ui.open_runner(rid)
            ui.go("plan")
    elif q.strip():
        st.info("No runners match. Try another name or runner number, or add the runner below.")
    else:
        st.caption("You have no runners of your own yet. Search the race database by runner number, or add one below.")

    st.divider()
    add_runner_form()


def apply_runner(fields: dict, text: str) -> None:
    """Voice (or its refined version) fills the add-runner form. Runs before the form is drawn."""
    if fields.get("name"):
        st.session_state["new_name"] = fields["name"]
    if fields.get("gender") in ("F", "M"):
        st.session_state["new_gender"] = fields["gender"]
    if isinstance(fields.get("age"), int) and 10 <= fields["age"] <= 100:     # the age box only takes 10 to 100
        st.session_state["new_age"] = fields["age"]
    races = [r for r in fields.get("races") or [] if r.get("finish") or r.get("event")]
    if races:
        st.session_state["new_races"] = pd.DataFrame({
            "Marathon": [r.get("event") or "" for r in races],
            "Year": [r.get("year") or None for r in races],
            "Finish (h:mm:ss)": [r.get("finish") or "" for r in races],
            "Splits at 5…40 km (optional)": [", ".join(r["splits"]) if r.get("splits") else "" for r in races]})
        st.session_state["new_races_version"] = st.session_state.get("new_races_version", 0) + 1


def add_runner_form() -> None:
    st.subheader("Add a runner")
    voice_panel("runner", "runner_voice", apply_runner,
                label="Describe the runner by voice: name, gender, age, and past marathons with year and time")
    st.caption("Enter past marathons, oldest first. Splits are the elapsed times at 5, 10 … 40 km, separated by "
               "commas; with splits the model also matches pacing.")
    for k, default in (("new_name", ""), ("new_gender", "F"), ("new_age", None)):
        st.session_state.setdefault(k, default)
    table = st.session_state.get("new_races", BLANK_RACES)
    version = st.session_state.get("new_races_version", 0)
    with st.form("add_runner", clear_on_submit=False):
        a, b, c = st.columns([2, 1, 1])
        name = a.text_input("Name", key="new_name")
        gender = b.selectbox("Gender", ["F", "M"], format_func={"F": "Female", "M": "Male"}.get, key="new_gender")
        age = c.number_input("Age", min_value=10, max_value=100, value=None, step=1, key="new_age")
        races = st.data_editor(table, num_rows="dynamic", hide_index=True,
                               key="add_runner_races" if not version else f"add_runner_races_{version}",
                               column_config={"Year": st.column_config.NumberColumn(min_value=1980, max_value=2100, step=1)})
        submitted = st.form_submit_button("Add runner", type="primary")
    if not submitted:
        return
    try:
        rid = add_runner(name, gender, int(age) if age is not None else None, races)
    except ValidationError as e:
        st.error(str(e))
        return
    st.success(f"{name.strip()} added.")
    for k in ("runner_voice_voice", "new_races", "new_races_version"):
        st.session_state.pop(k, None)
    ui.open_runner(rid)
    ui.go("plan")


def add_runner(name: str, gender: str, age: int | None, races: pd.DataFrame) -> str:
    """Validate the form rows and store the runner. Returns the new runner id."""
    items = []
    for _, row in races.iterrows():
        event = str(row.get("Marathon") or "").strip()
        finish = str(row.get("Finish (h:mm:ss)") or "").strip()
        if not event and not finish:
            continue
        splits_text = str(row.get("Splits at 5…40 km (optional)") or "").strip()
        splits = [x.strip() for x in splits_text.replace(";", ",").split(",")] if splits_text else None
        if splits is not None and len(splits) != 8:
            raise ValidationError("enter 8 split times (5 to 40 km), or leave splits empty")
        year = row.get("Year")
        items.append({"event": event or "Marathon", "year": int(year) if pd.notna(year) and year != "" else 0,
                      "finish": finish, "splits": splits})
    return ui.services().coach.add_runner(name, gender, age, runner_payload_races(items, age, time_field))["runner_id"]


def time_field(value, name: str) -> float:
    t = parse_time(value)
    if t is None:
        raise ValidationError(f"{name} must be a time like 3:31:20")
    return t
