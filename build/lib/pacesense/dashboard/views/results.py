"""Study results: the paper's Section 5 analyses of everyone who took part, plus CSV exports.

Locked behind PACESENSE_ADMIN_TOKEN, because it shows every participant's answers.
"""
from __future__ import annotations

import hmac

import streamlit as st

from ...analysis import analyze_repository
from .. import ui


def render() -> None:
    ui.page_header("Study results", "The paper's analyses (Section 5) of every completed study session.")
    s = ui.services()
    if not s.admin_token:
        st.info("This page is turned off. Start the dashboard with PACESENSE_ADMIN_TOKEN set to use it.")
        return
    if not st.session_state.get("admin_ok"):
        token = st.text_input("Admin token", type="password", key="admin_token")
        if st.button("Unlock", key="admin_unlock", type="primary"):
            if hmac.compare_digest(token.encode(), s.admin_token.encode()):
                st.session_state["admin_ok"] = True
                st.rerun()
            st.error("That token is not right.")
        return
    trials = s.repo.trials_frame(completed_only=False)
    sessions = s.repo.sessions_frame()
    a, b = st.columns(2)
    a.download_button("Trials (CSV)", trials.to_csv(index=False), file_name="pacesense_trials.csv", mime="text/csv",
                      key="dl_trials")
    b.download_button("Sessions (CSV)", sessions.to_csv(index=False), file_name="pacesense_sessions.csv",
                      mime="text/csv", key="dl_sessions")
    try:
        report = analyze_repository(s.repo).to_markdown()
    except ValueError as e:
        st.info(f"No analysis yet: {e}. Results appear once coaches have completed the study.")
        return
    st.download_button("Report (Markdown)", report, file_name="pacesense_report.md", mime="text/markdown", key="dl_report")
    st.markdown(report)
