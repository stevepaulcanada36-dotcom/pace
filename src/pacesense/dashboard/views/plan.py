"""Plan a race: choose a runner, weight the past races, read the prediction, shape the pacing, save.

Guided steps show one section at a time with Back and Next; "All controls" shows every section on one
page. Both layouts call the same section functions, so they never differ in what they can do.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from ... import racewatch
from ...domain import ValidationError, format_time
from ...planning import PlanSession, change_text, pace_text
from .. import charts, ui
from ..voice import refine_box, voice_panel

STEPS = ("Runner", "Past races", "Prediction", "Pacing", "Check and save")
NEXT = {1: "Next: weight the past races", 2: "Next: see the prediction", 3: "Next: shape the pacing",
        4: "Next: check and save"}


def render() -> None:
    prefs = ui.prefs()
    guided = prefs["view"] == "guided"
    ui.page_header("Plan a race", "One step at a time: choose a runner, adjust the model, shape the pacing, then save."
                   if guided else "Every control on one page. Switch to guided steps any time.")
    # The key follows the saved preference, so a change made elsewhere (welcome, Settings) is never undone here.
    layout = st.radio("Layout", ["Guided steps", "All controls"], index=0 if guided else 1, horizontal=True,
                      key=f"layout_{prefs['view']}", label_visibility="collapsed")
    if (layout == "Guided steps") != guided:
        ui.save_prefs(view="guided" if layout == "Guided steps" else "full")
        st.rerun()
    ui.show_toasts()
    if "flash_error" in st.session_state:
        st.error(st.session_state.pop("flash_error"))
    if prefs["experience"] is None:
        welcome()
    p = ui.plan()
    step = st.session_state.setdefault("step", 1)
    if p is None:
        step = st.session_state["step"] = 1
    if guided:
        stepper(step, has_runner=p is not None)
    if p is not None:
        ui.summary_bar(p)

    if guided:
        {1: runner_section, 2: weights_section, 3: prediction_section, 4: pacing_section, 5: save_section}[step](p)
        step_buttons(step)
    else:
        runner_section(p, compact=True)
        if p is None:
            return
        left, right = st.columns([3, 2], gap="large")
        with left:
            weights_section(p)
            pacing_section(p)
            save_section(p)
        with right:
            prediction_section(p)


# ---- frame -------------------------------------------------------------------------------------
def welcome() -> None:
    with st.container(border=True):
        st.subheader("How would you like to use PaceSense?")
        st.caption("You can change this later in Settings.")
        a, b = st.columns(2)
        if a.button("Guide me step by step", key="welcome_new", type="primary",
                    help="One step at a time, with a short explanation of each part of the model."):
            ui.save_prefs(experience="new", view="guided", explanations=True)
            st.rerun()
        if b.button("Show me everything", key="welcome_expert",
                    help="All controls on one page, without explanations. For coaches who know the tool."):
            ui.save_prefs(experience="experienced", view="full", explanations=False)
            st.rerun()


def stepper(step: int, has_runner: bool) -> None:
    cols = st.columns(len(STEPS))
    for n, (col, name) in enumerate(zip(cols, STEPS), start=1):
        mark = "✓ " if n < step else ""
        if col.button(f"{mark}{n}. {name}", key=f"stepper_{n}", type="primary" if n == step else "secondary",
                      disabled=(not has_runner and n > 1)):
            st.session_state["step"] = n
            st.rerun()


def step_buttons(step: int) -> None:
    if ui.plan() is None:
        return
    back, _, nxt = st.columns([1, 2, 1.4])
    if step > 1 and back.button("Back", key="step_back"):
        st.session_state["step"] = step - 1
        st.rerun()
    if step < 5 and nxt.button(NEXT[step], key="step_next", type="primary"):
        st.session_state["step"] = step + 1
        st.rerun()
    if step == 5 and nxt.button("Plan another race", key="step_restart"):
        st.session_state["step"] = 1
        st.rerun()


# ---- 1. runner -----------------------------------------------------------------------------------
def runner_section(p: PlanSession | None, compact: bool = False) -> None:
    coach = ui.services().coach
    with st.container(border=True):
        if not compact:
            st.subheader("Choose a runner")
            ui.why("Pick the runner you are planning a race for. The model uses their last five marathons.")
        elif p is not None:
            st.markdown(f"**{p.name}** · {'Female' if p.overview['gender'] == 'F' else 'Male'}"
                        f"{', ' + str(p.overview['age']) if p.overview['age'] else ''} · best {p.overview['best']} "
                        f"({p.overview['best_race']}) · {p.overview['n_races']} marathons")
        q = st.text_input("Search your runners or the race database", key="runner_query",
                          placeholder="Name or runner number, for example R00042")
        results = coach.search(q)
        if results:
            labels = {r["runner_id"]: f"{r['name']} · {'F' if r['gender'] == 'F' else 'M'}"
                      f"{', ' + str(r['age']) if r['age'] else ''} · best {r['best']} · {r['n_races']} marathons"
                      f"{' · your runner' if r['own'] else ''}" for r in results}
            choice = st.selectbox("Runner", list(labels), format_func=labels.get, key="runner_choice")
            a, b, c = st.columns([1.2, 1.4, 2])
            if a.button("Use this runner", key="runner_use", type="primary"):
                ui.open_runner(choice)
                st.rerun()
        else:
            if q.strip():
                st.info("No runners match. Try a runner number such as R00042, or add the runner on the Runners page.")
            a, b, c = st.columns([1.2, 1.4, 2])
        if b.button("Try a demo runner", key="runner_demo"):
            demo = next((r for r in coach.search("R00") if r["n_races"] >= 4), None)
            if demo is None:
                st.warning("The race database has no demo runners.")
            else:
                ui.open_runner(demo["runner_id"])
                st.rerun()
        with c:
            ui.link("runners", "Add a runner with their past marathons")
        if p is not None and not compact:
            st.caption(f"Currently planning for **{p.name}**.")


# ---- 2. race weights -----------------------------------------------------------------------------
def weights_section(p: PlanSession) -> None:
    with st.container(border=True):
        st.subheader("Weight the past races", help=ui.tip("weights"))
        ui.why(f"How well does each race show what {p.name} can do now? Lower a race that went wrong for reasons that "
               "no longer apply. Not sure? Leave them all at 0.5.")
        races = p.overview["races"]
        cols = st.columns(min(3, len(races)) or 1)
        for i, race in enumerate(races):
            with cols[i % len(cols)].container(border=True):
                pr = p.prediction["per_race"][i]
                st.markdown(f"**{race['event']} {race['year']}**")
                st.caption(f"Ran {race['finish']} · {'predicts ' + format_time(pr['finish_s']) if pr else 'no similar runners'}",
                           help="The model's prediction from this race alone.")
                ui.show(charts.sparkline(race["paces"]))
                key = f"weight_{p.runner_id}_{i}_{p.version}"
                st.slider("How representative?", 0.0, 1.0, value=float(p.weights[i]), step=0.1, key=key,
                          help="0 = not representative, 1 = very representative",
                          on_change=_weight_changed, args=(i, key))
        if st.button("Reset to equal weights", key="weights_reset", help=ui.tip("equal")):
            change = p.reset_weights()
            if change is None:
                st.session_state.setdefault("toasts", []).append("All races already count equally.")
            ui.announce(change)
            st.rerun()


def _weight_changed(i: int, key: str) -> None:
    p = ui.plan()
    try:
        ui.announce(p.set_weight(i, st.session_state[key]))
    except ValidationError as e:
        st.session_state["flash_error"] = str(e)


# ---- 3. prediction -------------------------------------------------------------------------------
def prediction_section(p: PlanSession) -> None:
    pr = p.prediction
    with st.container(border=True):
        st.subheader("Model prediction", help=ui.tip("prediction"))
        d = pr["finish_s"] - pr["default_finish_s"]
        st.metric("Challenging yet realistic target", format_time(pr["finish_s"]),
                  ui.delta(d) and f"{ui.delta(d)} vs equal weights", delta_color="inverse", help=ui.tip("target"))
        if pr["low_s"] is not None:
            ui.show(charts.range_chart(pr["low_s"], pr["high_s"], pr["finish_s"], p.plan_finish))
            st.markdown(f"About 8 in 10 runners like {p.name} went on to finish between **{format_time(pr['low_s'])}** "
                        f"and **{format_time(pr['high_s'])}**.", help=ui.tip("range"))
        a, b = st.columns(2)
        a.metric("Model with equal weights", format_time(pr["default_finish_s"]), help=ui.tip("equal"))
        b.metric("Similar runners used", pr["n_cases"], help=ui.tip("cases"))
        with st.expander("More detail: what the data says and past races"):
            for line in pr["insights"]:
                st.markdown(f"- {line}")
            hist = p.overview["history"]
            if hist:
                st.markdown("**Past races against the model's target**", help=ui.tip("history"))
                st.dataframe(pd.DataFrame([{"Race": f"{h['event']} {h['year']}", "Result": h["finish"],
                                            "Target": h["model"], "Result vs target": change_text(h["difference_s"])}
                                           for h in hist]), hide_index=True)
                st.caption("The model aims for a personal best, so most results come in slower than its target.")


# ---- 4. pacing -----------------------------------------------------------------------------------
def pacing_section(p: PlanSession) -> None:
    unit = p.unit
    with st.container(border=True):
        st.subheader("Shape the pacing", help=ui.tip("pacing"))
        ui.why("Pick a starting point, then change any interval's pace in the table. The finish time follows.")
        options = [s["key"] for s in p.strategies] + (["custom"] if p.source == "custom" else [])
        finish = {s["key"]: s["finish"] for s in p.strategies} | {"custom": format_time(p.plan_finish)}
        desc = {s["key"]: s["description"] for s in p.strategies} | {"custom": "the plan as you edited it"}
        key = f"strategy_{p.runner_id}_{p.version}"
        st.radio("Pacing strategy", options, index=options.index(p.source), key=key, help=ui.tip("strategies"),
                 format_func=lambda k: f"{p.strategy_name(k)} · {finish[k]} ({desc[k]})",
                 on_change=_strategy_changed, args=(key,))
        with_splits = [r for r in p.overview["races"] if r["paces"]]
        names = [f"{r['event']} {r['year']}" for r in with_splits][-3:]
        default = names[-1:] if ui.prefs()["overlay_last_race"] else []
        shown = st.multiselect("Compare with past races", names, default=default, key=f"overlay_{p.runner_id}")
        overlays = [(n, r["paces"]) for n, r in zip(names, with_splits[-3:]) if n in shown]
        ui.show(charts.pace_chart(p.paces, p.prediction["paces"], overlays, unit))

        rows = p.interval_rows()
        table = pd.DataFrame({"Interval": [r["interval"] for r in rows], f"Pace (min/{unit})": [r["pace"] for r in rows],
                              "Time": [r["time"] for r in rows], "Elapsed": [r["elapsed"] for r in rows]})
        edited = st.data_editor(
            table, hide_index=True, key=f"paces_{p.runner_id}_{p.version}",
            disabled=["Interval", "Time", "Elapsed"],
            column_config={f"Pace (min/{unit})": st.column_config.TextColumn(
                help=f"Type a new pace as m:ss per {unit}", validate=r"^\d{1,2}:[0-5]\d$")})
        new = list(edited[f"Pace (min/{unit})"])
        if new != list(table[f"Pace (min/{unit})"]):
            try:
                for change in p.set_pace_texts(new):
                    ui.announce(change)
                st.rerun()
            except ValidationError as e:
                st.error(f"{e}. Type paces as minutes:seconds, for example {pace_text(p.paces[0], unit)}.")
        st.caption(f"Time = pace × distance; the times add up to the finish ({format_time(p.plan_finish)}). "
                   "The last interval is only 2.195 km, so it takes less time at the same pace.", help=ui.tip("intervals"))


def _strategy_changed(key: str) -> None:
    choice = st.session_state[key]
    if choice != "custom":
        ui.announce(ui.plan().choose_strategy(choice))


# ---- 5. check and save ---------------------------------------------------------------------------
def _apply_plan_details(fields: dict, text: str) -> None:
    """Voice or a refined version fills the save form (runs before its widgets are drawn)."""
    if fields.get("target_race"):
        st.session_state["target_race"] = fields["target_race"]
    if fields.get("race_date"):
        st.session_state["target_date"] = dt.date.fromisoformat(fields["race_date"])
    if fields.get("notes"):
        st.session_state["target_notes"] = fields["notes"]


def _apply_notes(fields: dict, text: str) -> None:
    st.session_state["target_notes"] = text


def save_section(p: PlanSession) -> None:
    c = p.check
    # Widget values live in session state under these keys, so opening a saved plan can fill them in.
    for k, default in (("target_race", ""), ("target_date", None), ("target_notes", "")):
        st.session_state.setdefault(k, default)
    with st.container(border=True):
        st.subheader("Check and save", help=ui.tip("verdict"))
        diff = c["difference_s"]
        st.metric("Your plan", format_time(c["finish_s"]),
                  None if round(diff) == 0 else f"{ui.delta(diff)} vs model", delta_color="inverse")
        box = {"realistic": st.success, "ambitious": st.warning, "cautious": st.info}[c["verdict"]]
        box(f"**{ui.VERDICT[c['verdict']][0]} target.** {c['note']}")
        for flag in c["flags"][:3]:
            st.warning(flag["message"])
        if len(c["flags"]) > 3:
            st.caption(f"…and {len(c['flags']) - 3} more intervals like these.")

        voice_panel("plan_details", "plan_voice", _apply_plan_details,
                    label="Say the target race, date and notes")
        a, b = st.columns([2, 1])
        race = a.text_input("Target race", key="target_race", placeholder="e.g. Toronto Waterfront Marathon 2026")
        picked = b.date_input("Race date", value=None, key="target_date")
        notes = st.text_area("Notes for this plan", key="target_notes",
                             placeholder="What do you know that the race data doesn't? Training, injuries, the runner's goal.")
        refine_box("plan_details", notes, "plan_notes", _apply_notes, label="Suggest clearer notes")
        date = picked.isoformat() if isinstance(picked, dt.date) else None
        s1, s2, s3 = st.columns(3)
        if s1.button("Save as new plan" if p.saved_plan_id else "Save plan", key="plan_save", type="primary"):
            if not race.strip():
                st.error("Enter the target race before saving.")
            else:
                p.save(race, date, notes)
                st.success("Plan saved. After the race, record the result under Saved plans. "
                           "Open Race-day watch to see it as a watch screen.")
        s2.download_button("Download plan (text)", p.plan_text(race, date, notes),
                           file_name=f"{p.name}-plan.txt".replace(" ", "_"), mime="text/plain", key="plan_txt")
        s3.download_button("Download splits (CSV)", racewatch.splits_csv(p.paces),
                           file_name=f"{p.name}-splits.csv".replace(" ", "_"), mime="text/csv", key="plan_csv")
        ui.link("watch", "See this plan as a race-day watch screen")
        ui.link("ask", "Ask PaceSense about this plan")

        st.markdown("**Your changes**", help=ui.tip("changes"))
        if p.changes:
            for n, change in enumerate(reversed(p.changes), start=1):
                st.markdown(f"{n}. {change.text()}")
        else:
            st.caption("No changes yet. The plan is the model's plan with equal race weights.")

