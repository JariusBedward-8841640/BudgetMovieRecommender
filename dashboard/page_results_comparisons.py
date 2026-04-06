"""Results and comparison analytics page."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import bar_by_algorithm, grouped_by_budget, grouped_by_profile
from dashboard.context import comparison_load


def render() -> None:
    data = comparison_load()
    for warning in data.warnings:
        st.warning(warning)

    st.header("Results and Comparisons")
    if data.df.empty:
        st.info("No comparison rows found. Run experiments and comparison scripts first.")
        return

    df = data.df.copy()
    algorithms = sorted(df["algorithm"].dropna().unique().tolist())
    selected = st.multiselect("Algorithms", options=algorithms, default=algorithms)
    df = df[df["algorithm"].isin(selected)]
    if df.empty:
        st.info("No rows match current filters.")
        return

    specs = [
        ("avg_cumulative_reward", "Reward by Method"),
        ("avg_questions_asked", "Questions Asked by Method"),
        ("acceptance_rate", "Acceptance Rate by Method"),
        ("abandonment_rate", "Abandonment Rate by Method"),
    ]
    for metric, title in specs:
        fig = bar_by_algorithm(df, metric, title)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    fig_budget = grouped_by_budget(df, "avg_cumulative_reward", "Reward Across Question Budgets")
    if fig_budget is not None:
        st.plotly_chart(fig_budget, use_container_width=True)

    fig_profile = grouped_by_profile(df, "avg_cumulative_reward", "Reward Across User Profiles")
    if fig_profile is not None:
        st.plotly_chart(fig_profile, use_container_width=True)

    st.subheader("Metric Summary Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
