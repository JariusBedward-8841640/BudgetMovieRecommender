"""Executive overview page."""

from __future__ import annotations

import streamlit as st

from dashboard.context import comparison_load
from dashboard.data_loader import compute_kpis
from dashboard.ui import fmt_num, fmt_pct, metric_card


def render() -> None:
    data = comparison_load()
    for warning in data.warnings:
        st.warning(warning)

    st.header("Executive Overview")
    st.subheader("Budgeted Interactive Movie Recommender")
    st.write(
        "A recommendation engine that decides when to ask a short clarifying question and when to "
        "recommend immediately, maximizing relevance while minimizing user friction."
    )

    kpis = compute_kpis(data.df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Best Method", str(kpis["best_method"]))
    with c2:
        metric_card("Best Avg Reward", fmt_num(float(kpis["best_avg_reward"])))
    with c3:
        metric_card("Best Acceptance Rate", fmt_pct(float(kpis["best_acceptance_rate"])))
    with c4:
        metric_card("Avg Session Length", fmt_num(float(kpis["avg_session_length"])))
    with c5:
        metric_card("Avg Questions Asked", fmt_num(float(kpis["avg_questions_asked"])))

    st.info(
        "RL objective: maximize cumulative reward by balancing information gain from questions "
        "against friction and abandonment risk."
    )
    st.markdown("### Why This Matters")
    st.write(
        "Traditional recommenders often guess too early. This product adds adaptive interaction: "
        "ask only when needed, then recommend with improved confidence."
    )
    if data.source:
        st.caption(f"Current metrics source: `{data.source}`")
