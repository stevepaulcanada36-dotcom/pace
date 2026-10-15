"""Race-day watch: a coach's plan as the runner would see it on a watch during the race.

Pick a moment (how far into the race, how far ahead or behind) or play a simulated race. The screen shows
the target pace, ahead or behind, the next split and a cue; below are the race so far and the pace band.
"""
from __future__ import annotations

import streamlit as st

from ... import racewatch as rw
from ...domain import MARATHON_KM, format_time
from .. import charts, ui

SPEEDS = {"60× (a marathon in about 3 minutes)": 60, "120×": 120, "300×": 300}
TICK_S = 0.5           # how often the simulated race redraws, in real seconds


def render() -> None:
    ui.page_header("Race-day watch", "A saved plan as the runner would see it during the race.")
    ui.why(ui.HELP["watch"].text)
    choice = choose_plan()
    if choice is None:
        return
    title, runner, race, race_date, paces = choice
    u = ui.unit()

    mode = st.radio("Show", ["A moment in the race", "Simulate a race"], horizontal=True, key="watch_mode")
    if mode == "A moment in the race":
        moment(paces, runner, u)
    else:
        simulation(paces, runner, u)

    st.divider()
    st.subheader("Pace band")
    every = st.radio("Show a split every", [1, 5], format_func=lambda n: f"{n} km", horizontal=True, key="band_every")
    splits = rw.km_splits(paces)
    rows = [s for s in splits if s.marker == "Finish" or int(s.distance_km) % every == 0]
    st.dataframe([{"Marker": s.marker, "Target elapsed": format_time(s.elapsed_s),
                   f"Pace (min/{u})": _pace(s.pace_s, u)} for s in rows], hide_index=True)
    stem = f"{runner}-{race or 'plan'}".replace(" ", "_")
    a, b, c = st.columns(3)
    a.download_button("Pace band (text)", rw.pace_band_text(paces, f"{runner}: {race or 'next marathon'}", u, every),
                      file_name=f"{stem}-paceband.txt", mime="text/plain", key="dl_band")
    b.download_button("Kilometre splits (CSV)", rw.splits_csv(paces), file_name=f"{stem}-splits.csv",
                      mime="text/csv", key="dl_csv")
    c.download_button("Plan for a watch app (JSON)", rw.plan_json(paces, runner, race, race_date),
                      file_name=f"{stem}.json", mime="application/json", key="dl_json",
                      help="A generic JSON description of the plan. It is not a specific watch brand's file format.")


def choose_plan():
    coach = ui.services().coach
    options: dict[str, tuple] = {}
    current = ui.plan()
    if current is not None:
        options["current"] = ("Current plan (Plan a race)", current.name, "", None, list(current.paces))
    for p in coach.list_plans():
        label = f"{p['runner_name']} · {p['target_race']}{' · ' + p['race_date'] if p['race_date'] else ''} · {p['finish']}"
        options[p["id"]] = (label, p["runner_name"], p["target_race"], p["race_date"], p["paces"])
    if not options:
        st.info("There is no plan yet. Plan a race and save it, then come back here.")
        ui.link("plan", "Plan a race")
        return None
    key = st.selectbox("Plan", list(options), format_func=lambda k: options[k][0], key="watch_plan")
    return options[key]


def moment(paces, runner: str, unit: str) -> None:
    left, right = st.columns([1, 1.25], gap="large")
    with right:
        d = st.slider("Distance run (km)", 0.0, MARATHON_KM, value=10.0, step=0.1, key="watch_distance")
        offset = st.slider("Seconds behind (+) or ahead (−) of plan", -300, 300, value=0, step=5, key="watch_offset")
        elapsed = max(0.0, rw.target_elapsed(paces, d) + offset)
        status = rw.status(paces, d, elapsed, unit)
        readout(status, unit)
    with left:
        ui.show(charts.watch_face(status, unit, runner))


def simulation(paces, runner: str, unit: str) -> None:
    a, b = st.columns(2)
    profile = a.selectbox("Simulated runner", list(rw.PROFILES), format_func=rw.PROFILES.get, key="sim_profile")
    speed = SPEEDS[b.selectbox("Speed", list(SPEEDS), key="sim_speed")]
    sim = st.session_state.setdefault("sim", {"clock": 0.0, "running": False, "profile": profile})
    if sim["profile"] != profile:                       # a new runner starts a new race
        sim.update(clock=0.0, running=False, profile=profile)
    track = rw.simulate(paces, profile)
    finish = track[-1][1]

    c1, c2, c3 = st.columns(3)
    if c1.button("Pause" if sim["running"] else ("Restart" if sim["clock"] >= finish else "Start"),
                 key="sim_toggle", type="primary"):
        if sim["clock"] >= finish:
            sim["clock"] = 0.0
        sim["running"] = not sim["running"]
        st.rerun()
    if c2.button("Reset", key="sim_reset"):
        sim.update(clock=0.0, running=False)
        st.rerun()
    if c3.button("Jump to 30 km", key="sim_jump"):
        sim["clock"] = min((t for dist, t in track if dist >= 30), default=finish)
        st.rerun()

    @st.fragment(run_every=TICK_S if sim["running"] else None)
    def live() -> None:
        if sim["running"]:
            sim["clock"] = min(finish, sim["clock"] + TICK_S * speed)
            if sim["clock"] >= finish:
                sim["running"] = False
                st.rerun()                 # full rerun: stops the timer and relabels the Start button
        d, t = rw.position_at(track, sim["clock"])
        status = rw.status(paces, d, t, unit)
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            ui.show(charts.watch_face(status, unit, runner))
        with right:
            readout(status, unit)
        ui.show(charts.ahead_behind_chart(track, paces, upto_km=d))
        if sim["clock"] >= finish:
            st.caption("Race finished. Press Restart to run it again.")

    live()


def readout(status: rw.WatchStatus, unit: str) -> None:
    box = {"start": st.info, "on": st.info, "ahead": st.success, "behind": st.warning, "slipping": st.warning,
           "done": st.success if status.delta_s <= 0 else st.warning}[status.level]
    box(status.cue)
    a, b = st.columns(2)
    a.metric("Target pace", f"{_pace(status.target_pace_s, unit)}/{unit}")
    if status.level not in ("start",):
        a.metric("Ahead / behind", "on plan" if abs(status.delta_s) <= rw.ON_PLAN_S else
                 f"{_clock(status.delta_s)} {'ahead' if status.delta_s < 0 else 'behind'}")
    if status.level != "done":
        b.metric(f"{status.next_marker} target", format_time(status.next_marker_target_s))
        b.metric("Projected finish", format_time(status.projected_finish_s),
                 help="Elapsed time so far plus the rest of the plan as written.")
        if status.required_pace_s == status.required_pace_s:          # not NaN
            st.caption(f"To finish on the plan's {format_time(status.plan_finish_s)}, the runner needs "
                       f"{_pace(status.required_pace_s, unit)}/{unit} from here.")


def _pace(s_per_km: float, unit: str) -> str:
    r = int(round(s_per_km * (1.609344 if unit == "mi" else 1.0)))
    return f"{r // 60}:{r % 60:02d}"


def _clock(s: float) -> str:
    r = int(round(abs(s)))
    return f"{r // 60}:{r % 60:02d}" if r < 3600 else f"{r // 3600}:{r % 3600 // 60:02d}:{r % 60:02d}"
