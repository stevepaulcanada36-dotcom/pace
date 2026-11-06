"""PaceSense dashboard. Run with:  pacesense dashboard   (or: streamlit run src/pacesense/dashboard/app.py)

This file only sets up the page and the navigation; each page lives in dashboard/views/.
"""
from __future__ import annotations

import streamlit as st

from pacesense.dashboard import ui
from pacesense.dashboard.views import (assistant, howitworks, lab, plan, record, results, runners, saved, settings,
                                       study, watch)


def main() -> None:
    st.set_page_config(page_title="PaceSense", page_icon="🏃", layout="wide")
    pages = {
        "plan": st.Page(plan.render, title="Plan a race", icon="📋", url_path="plan", default=True),
        "watch": st.Page(watch.render, title="Race-day watch", icon="⌚", url_path="watch"),
        "ask": st.Page(assistant.render, title="Ask PaceSense", icon="💬", url_path="ask"),
        "runners": st.Page(runners.render, title="Runners", icon="👟", url_path="runners"),
        "saved": st.Page(saved.render, title="Saved plans", icon="🗂️", url_path="saved"),
        "record": st.Page(record.render, title="Track record", icon="🎯", url_path="record"),
        "study": st.Page(study.render, title="Research study", icon="🧪", url_path="study"),
        "results": st.Page(results.render, title="Study results", icon="📊", url_path="results"),
        "lab": st.Page(lab.render, title="Model lab", icon="🧮", url_path="model-lab"),
        "help": st.Page(howitworks.render, title="How it works", icon="❓", url_path="how-it-works"),
        "settings": st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
    }
    ui.PAGES.update(pages)
    ui.keep_widget_values()
    nav = st.navigation({
        "Coach": [pages["plan"], pages["watch"], pages["ask"], pages["runners"], pages["saved"], pages["record"]],
        "Research": [pages["study"], pages["results"], pages["lab"]],
        "Help": [pages["help"], pages["settings"]],
    })
    with st.sidebar:
        st.caption(f"Race data: {ui.services().data_source}")
    nav.run()


main()
