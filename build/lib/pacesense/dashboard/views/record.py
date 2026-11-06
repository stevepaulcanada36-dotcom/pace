"""Track record: how saved plans and the model compared with actual results."""
from __future__ import annotations

import streamlit as st

from .. import ui


def render() -> None:
    ui.page_header("Track record", "How your saved plans and the model compared with actual results.")
    t = ui.services().coach.track_record()
    if not t["n"]:
        st.info("No results recorded yet. Record race results under Saved plans to build your track record.")
        ui.link("saved", "Saved plans")
        return
    a, b, c = st.columns(3)
    a.metric("Races with a result", t["n"])
    b.metric("Average error of your plans", f"{t['plan_error_pct']:.1f}%")
    c.metric("Average error of the model (equal weights)", f"{t['model_default_error_pct']:.1f}%")
    d, e, _ = st.columns(3)
    d.metric("Races where your plan was closer", f"{t['plan_better_than_model']} of {t['n']}")
    if t["in_range_pct"] is not None:
        e.metric("Results inside the likely range", f"{t['in_range_pct']:.0f}%", help="About 80% is expected.")
    st.caption("Results are kept with each plan, so the model's defaults can later be checked against real outcomes.")
