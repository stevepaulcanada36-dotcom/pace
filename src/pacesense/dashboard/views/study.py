"""Research study: the procedure of Muijlwijk et al. (IUI '24, Section 4.1), unchanged.

Consent, then two of the coach's own runners (or none), then four runners: for each, the coach's own
estimate (Coach_pre), the model's prediction (Model_initial, adjustable in the interactive condition:
Model_adapted), and the coach's final advice (Coach_post). Then the questionnaire. Every answer is stored by
`StudyService`, the same code the analysis reads.
"""
from __future__ import annotations

import uuid

import streamlit as st

from ...coach import runner_payload_races
from ...domain import Runner, ValidationError, parse_time
from ...study import RUNNER_MOTIVATIONS
from .. import charts, ui
from .runners import BLANK_RACES, time_field

FAMILIARITY = ("Not at all", "Slightly", "Moderately", "Well", "Very well")


def render() -> None:
    st.title("Marathon prediction study")
    state = st.session_state.setdefault("study", {"sid": None, "picked": [], "hand": []})
    study = ui.services().study
    if state["sid"] is None:
        consent(state)
        return
    summary = study.session_summary(state["sid"])
    if summary["completed"]:
        thanks(summary)
        return
    if not summary["runners"]:
        choose_runners(state)
        return
    nxt = next((r for r in summary["runners"] if r["status"] != "done"), None)
    if nxt is None:
        questionnaire(state["sid"])
        return
    view = study.trial_view(state["sid"], nxt["seq"])
    st.progress(0.1 + 0.2 * (nxt["seq"] - 1) + (0.1 if view["status"] != "planned" else 0.0),
                text=f"Runner {nxt['seq']} of 4")
    (runner_intro if view["status"] == "planned" else model_page)(state["sid"], view)


def consent(state: dict) -> None:
    st.subheader("Informed consent")
    st.markdown("This study is about how running coaches work with a model that predicts marathon finish times. "
                "You will give advice on a target finish time and pacing strategy for four runners, look at the "
                "model's prediction for each, and answer a short questionnaire. This takes about 30 minutes.")
    st.markdown("Participation is voluntary and you can stop at any time. Your answers are stored for research and "
                "reported anonymously.")
    agree = st.checkbox("I have read the information above and agree to take part.", key="study_consent")
    if st.button("Start", key="study_start", type="primary", disabled=not agree):
        s = ui.services()
        state["sid"] = s.study.create_session(s.assignment)["session_id"]
        st.rerun()


def choose_runners(state: dict) -> None:
    study = ui.services().study
    st.subheader("Introduction")
    st.markdown("For each runner you will first see their previous marathons: the finish times and the pace for every "
                "5 km. You give a challenging yet realistic target finish time and a pacing strategy. Next, you see "
                "the advice of a model, and you may update your advice. You will do this for four runners: two of "
                "your own runners and two runners you do not know.")
    st.subheader("Select your runners")
    q = st.text_input("Search for two runners you coach, by name or runner number", key="study_query")
    for r in study.search_runners(q) if q.strip() else []:
        a, b = st.columns([4, 1])
        a.markdown(f"**{r['name'] or r['runner_id']}** · {'Female' if r['gender'] == 'F' else 'Male'}"
                   f"{', ' + str(r['age']) if r['age'] else ''} · " + ", ".join(f"{x['event']} {x['year']} ({x['finish']})" for x in r["races"]))
        if b.button("Select", key=f"study_pick_{r['runner_id']}",
                    disabled=r["runner_id"] in state["picked"] or len(state["picked"]) + len(state["hand"]) >= 2):
            state["picked"].append(r["runner_id"])
            st.rerun()
    chosen = [study.runner(rid).display_name for rid in state["picked"]] + [h.name for h in state["hand"]]
    st.markdown("**Your runners:** " + (", ".join(chosen) if chosen else "none selected yet"))
    if chosen and st.button("Clear my selection", key="study_clear"):
        state["picked"], state["hand"] = [], []
        st.rerun()
    with st.expander("Add a runner by hand (if your runner is missing, or races are missing)"):
        with st.form("study_hand"):
            a, b, c = st.columns([2, 1, 1])
            name = a.text_input("Name")
            gender = b.selectbox("Gender", ["F", "M"], format_func={"F": "Female", "M": "Male"}.get)
            age = c.number_input("Age", min_value=10, max_value=100, value=None, step=1)
            races = st.data_editor(BLANK_RACES, num_rows="dynamic", hide_index=True, key="study_hand_races")
            if st.form_submit_button("Add this runner"):
                try:
                    state["hand"].append(hand_runner(name, gender, int(age) if age is not None else None, races))
                    st.rerun()
                except ValidationError as e:
                    st.error(str(e))
    a, b = st.columns(2)
    if a.button("Next", key="study_next", type="primary", disabled=len(chosen) != 2):
        _plan(state, unfamiliar_only=False)
    if b.button("Continue without my own runners", key="study_skip"):
        _plan(state, unfamiliar_only=True)


def _plan(state: dict, unfamiliar_only: bool) -> None:
    try:
        ui.services().study.plan_runners(state["sid"], [] if unfamiliar_only else state["picked"] + state["hand"],
                                         unfamiliar_only=unfamiliar_only)
        st.rerun()
    except ValidationError as e:
        st.error(str(e))


def hand_runner(name: str, gender: str, age: int | None, races) -> Runner:
    if not name.strip():
        raise ValidationError("enter the runner's name")
    items = []
    for _, row in races.iterrows():
        event, finish = str(row.get("Marathon") or "").strip(), str(row.get("Finish (h:mm:ss)") or "").strip()
        if not event and not finish:
            continue
        sp = str(row.get("Splits at 5…40 km (optional)") or "").strip()
        splits = [x.strip() for x in sp.replace(";", ",").split(",")] if sp else None
        if splits is not None and len(splits) != 8:
            raise ValidationError("enter 8 split times (5 to 40 km), or leave splits empty")
        year = row.get("Year")
        items.append({"event": event or "Marathon", "year": int(year) if year == year and year not in (None, "") else 0,
                      "finish": finish, "splits": splits})
    if not items:
        raise ValidationError("add at least one previous marathon")
    return Runner(f"H-{uuid.uuid4().hex[:12]}", gender, runner_payload_races(items, age, time_field), name=name.strip()[:200], age=age)


def _race_grid(view: dict, sliders: bool = False, sid: str | None = None) -> None:
    cols = st.columns(3)
    weights = view.get("model", {}).get("weights") or []
    for i, race in enumerate(view["races"]):
        with cols[i % 3]:
            st.markdown(f"**{race['event']} {race['year'] or ''}**  \nFinish time: {race['finish']}")
            if race["pace_per_km_s"]:
                ui.show(charts.race_bars(race["pace_per_km_s"]))
            else:
                st.caption("No 5 km splits for this race.")
            if sliders:
                key = f"study_w_{sid}_{view['seq']}_{i}"
                st.slider("How representative is this race?", 0.0, 1.0, value=float(weights[i]), step=0.1, key=key,
                          help="Left: not representative. Right: very representative.",
                          on_change=_weights_changed, args=(sid, view["seq"], len(view["races"])))


def _weights_changed(sid: str, seq: int, n: int) -> None:
    view = ui.services().study.trial_view(sid, seq)
    weights = [st.session_state.get(f"study_w_{sid}_{seq}_{i}", view["model"]["weights"][i]) for i in range(n)]
    if sum(weights) == 0:
        st.session_state["study_error"] = "Give at least one race a weight above zero."
        return
    ui.services().study.set_weights(sid, seq, weights)


def runner_intro(sid: str, view: dict) -> None:
    who = view["name"]
    st.subheader(f"Runner {view['seq']} of 4: {who}")
    st.markdown(f"{who} is a {'female' if view['gender'] == 'F' else 'male'} runner"
                f"{' of ' + str(view['age']) + ' years old' if view['age'] else ''}. Below are the previous marathon "
                f"races of {who}, with the pace per 5 km.")
    _race_grid(view)
    with st.form(f"study_pre_{view['seq']}"):
        familiarity, motivations = None, []
        if view["familiar"]:
            st.markdown("**About this runner**")
            familiarity = st.selectbox("How well do you know this runner?", range(1, 6), index=None,
                                       format_func=lambda v: FAMILIARITY[v - 1], placeholder="Select")
            motivations = st.multiselect("What motivates this runner to run marathons?", RUNNER_MOTIVATIONS)
        st.markdown("**Finish time**")
        finish = st.text_input("What do you find a challenging yet realistic finish time for this runner's next "
                               "marathon? (h:mm:ss)")
        st.markdown("**Pacing strategy**")
        pace = st.slider("How should the pace be distributed over the race? (−1 = end fast, 0 = even, 1 = start fast)",
                         -1.0, 1.0, 0.0, 0.1)
        if st.form_submit_button("Next", type="primary"):
            seconds = parse_time(finish)
            if seconds is None or not 7200 <= seconds <= 25200:
                st.error("Please enter a finish time as h:mm:ss, between 2:00:00 and 7:00:00.")
            elif view["familiar"] and familiarity is None:
                st.error("Please indicate how well you know this runner.")
            else:
                ui.services().study.submit_pre(sid, view["seq"], seconds, pace, familiarity, motivations)
                st.rerun()


def model_page(sid: str, view: dict) -> None:
    inter = view["condition"] == "interactive"
    who, n = view["name"], len(view["races"])
    if "study_error" in st.session_state:
        st.error(st.session_state.pop("study_error"))
    st.subheader("Model prediction")
    st.markdown(f"Based on the {n} marathon race{'s' if n > 1 else ''} of {who}, the model predicts the following:  \n"
                f"A challenging yet realistic target finish time is: **{_clock(view['model_initial_s'])}**")
    with st.expander("Click here to see how the model generated this advice"):
        st.markdown("The model uses case-based reasoning: it solves a new problem by reusing solutions to similar "
                    "problems from the past. For each previous race of the runner, it looks for runners who ran a very "
                    "similar race (similar finish time and pace over the 5 km sections) and who later ran a personal "
                    "best. How much those runners improved, and how they paced their personal best, is used to "
                    "predict a challenging yet realistic finish time and a pacing strategy.")
        st.markdown("*Example:* a runner finished in 3:50 after slowing strongly in the last 10 km. The model finds "
                    "runners who also ran about 3:50 with a similar slow-down and later ran their best in, say, 3:38 "
                    "with a more even pace. Their improvement forms the basis for the advice.")
        if n > 1:
            st.markdown("The predictions for each previous race are combined into one prediction" +
                        (", using the weights you set with the sliders." if inter else ", where every race counts equally."))
    if inter:
        st.markdown(f"Below, the marathon races of {who} are shown, which the model used. Not every race may be "
                    "equally representative of what this runner can do. Please use the sliders below the graphs to "
                    "indicate how representative you consider each race. Moving a slider to the left gives that race "
                    "less weight; to the right, more. The prediction below updates when you move a slider.")
    else:
        st.markdown(f"Below, the marathon races of {who} are shown, which the model used to determine a target finish time.")
    _race_grid(view, sliders=inter, sid=sid)
    with st.container(border=True):
        st.markdown(f"**Output model**  \nThe predicted finish time is: **{view['model']['finish']}**  \n"
                    "Below, the model recommends a pacing strategy that should make it possible to achieve this finish time.")
        ui.show(charts.race_bars(view["model"]["pace_per_km_s"], color="#3FB3C5"))
    with st.form(f"study_post_{view['seq']}"):
        a, b = st.columns(2)
        finish = a.text_input("New finish time: now you have assessed the model, what do you find a challenging yet "
                              "realistic finish time for this runner? (h:mm:ss)")
        pace = b.slider("New pacing strategy (−1 = end fast, 0 = even, 1 = start fast)", -1.0, 1.0, 0.0, 0.1)
        why = st.text_area("Motivate your answer: please shortly explain your finish time and pacing strategy.")
        label = {1: "Go to the next runner", 2: "Go to the next runner", 3: "Go to the last runner", 4: "Go to the questionnaire"}
        if st.form_submit_button(label[view["seq"]], type="primary"):
            seconds = parse_time(finish)
            if seconds is None or not 7200 <= seconds <= 25200:
                st.error("Please enter your finish time as h:mm:ss, between 2:00:00 and 7:00:00.")
            else:
                ui.services().study.submit_post(sid, view["seq"], seconds, pace, why)
                st.rerun()


def questionnaire(sid: str) -> None:
    q = ui.services().study.questionnaire_for(sid)
    st.subheader("Questionnaire")
    st.markdown("Thank you for assessing the four runners. Finally, we would like to ask you some questions.")
    with st.form("study_questionnaire"):
        answers = {}
        for scale in q["scales"]:
            st.markdown(f"**{scale['title']}**")
            st.caption(f"{scale.get('instruction', '')} 1 = {scale['anchors'][0]}, {scale['points']} = {scale['anchors'][1]}.".strip())
            for item in scale["items"]:
                answers[item["id"]] = st.radio(item["text"], list(range(1, scale["points"] + 1)), index=None,
                                               horizontal=True, key=f"study_q_{item['id']}")
        st.markdown("**About you**")
        cols = st.columns(2)
        for k, d in enumerate(q["demographics"]):
            with cols[k % 2]:
                if d["type"] == "number":
                    answers[d["id"]] = st.number_input(d["text"], min_value=d["min"], max_value=d["max"], value=None, step=1)
                else:
                    answers[d["id"]] = st.selectbox(d["text"], d["options"], index=None, placeholder="Select")
        answers["comment"] = st.text_area("Comments (optional)")
        if st.form_submit_button("Submit", type="primary"):
            missing = [i["text"] for s in q["scales"] for i in s["items"] if answers.get(i["id"]) is None]
            if missing:
                st.error(f"Please answer all statements ({len(missing)} left).")
                return
            clean = {k: v for k, v in answers.items() if v not in (None, "")}
            try:
                ui.services().study.submit_questionnaire(sid, clean)
                st.rerun()
            except ValidationError as e:
                st.error(str(e))


def thanks(summary: dict) -> None:
    st.subheader("Thank you")
    st.markdown("Your answers have been saved. Thank you for taking part in this study.")
    st.dataframe([{"Runner": r["seq"], "Yours": "yes" if r["familiar"] else "", "Your first": r["coach_pre"],
                   "Model": r["model_initial"], "Model, adjusted": r["model_adapted"] or "", "Your final": r["coach_post"],
                   "Best recent race (hidden)": r["best_recent_race"] or ""} for r in summary["runners"]], hide_index=True)
    if st.button("Start a new session", key="study_new"):
        st.session_state["study"] = {"sid": None, "picked": [], "hand": []}
        st.rerun()


def _clock(s: float) -> str:
    r = int(round(s))
    return f"{r // 3600}:{r % 3600 // 60:02d}:{r % 60:02d}"
