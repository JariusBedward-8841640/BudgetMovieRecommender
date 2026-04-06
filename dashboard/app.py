"""Executive-style Streamlit dashboard for BudgetMovieRecommender."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repo root is importable when launched via `streamlit run dashboard/app.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.context import ensure_context
from dashboard.page_executive_overview import render as render_executive_page
from dashboard.page_investor_technical import render as render_investor_technical_page
from dashboard.page_product_demo import render as render_product_demo_page
from dashboard.page_results_comparisons import render as render_results_page


st.set_page_config(
    page_title="Budgeted Interactive Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


def _metric_card(title: str, value: str) -> None:
    st.metric(label=title, value=value)


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{100.0 * value:.1f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def render_executive_overview(comparison_df: pd.DataFrame, source: str | None) -> None:
    st.header("Executive Overview")
    st.subheader("Budgeted Interactive Movie Recommender")
    st.write(
        "An adaptive recommender that decides when to ask a clarifying question and when to "
        "recommend immediately, maximizing recommendation quality under a strict question budget."
    )
    st.caption(
        "Problem: users dislike long questionnaires, but blind recommendations reduce relevance. "
        "Our policy optimizes cumulative reward by balancing information gain vs user friction."
    )

    kpis = compute_kpis(comparison_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric_card("Best Method", str(kpis["best_method"]))
    with c2:
        _metric_card("Best Avg Reward", _fmt_num(float(kpis["best_avg_reward"])))
    with c3:
        _metric_card("Best Acceptance Rate", _fmt_pct(float(kpis["best_acceptance_rate"])))
    with c4:
        _metric_card("Avg Session Length", _fmt_num(float(kpis["avg_session_length"])))
    with c5:
        _metric_card("Avg Questions Asked", _fmt_num(float(kpis["avg_questions_asked"])))

    st.info(
        "Optimization target: maximize long-term cumulative reward while minimizing unnecessary "
        "questions and abandonment risk."
    )
    if source:
        st.caption(f"Using comparison data source: `{source}`")


def render_product_demo(model_paths: dict[str, str]) -> None:
    st.header("Product Demo View")
    st.write("Interactive session replay that simulates a live product conversation.")

    profiles = available_profiles()
    policy_map = build_policy_map(model_paths)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        profile = st.selectbox("User Profile", options=profiles, index=0)
    with col2:
        budget = st.slider("Question Budget", min_value=0, max_value=5, value=2, step=1)
    with col3:
        policy_key = st.selectbox(
            "Algorithm / Policy",
            options=list(policy_map.keys()),
            format_func=lambda key: policy_map[key],
            index=0,
        )
    with col4:
        seed = st.number_input("Seed", min_value=0, value=7, step=1)

    max_steps = st.slider("Session Max Steps", min_value=4, max_value=12, value=8, step=1)
    if st.button("Start Simulated Session", type="primary"):
        replay = replay_session(
            policy_key=policy_key,
            profile=profile,
            question_budget=int(budget),
            max_steps=int(max_steps),
            seed=int(seed),
            model_paths=model_paths,
        )
        if replay.warning:
            st.warning(replay.warning)
            return

        st.subheader("Session Timeline")
        st.dataframe(pd.DataFrame(replay.steps), use_container_width=True, hide_index=True)

        st.subheader("Session Summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.metric("Accepted", replay.summary["accepted_recommendations"])
        with s2:
            st.metric("Skipped", replay.summary["skipped_recommendations"])
        with s3:
            st.metric("Abandoned", "Yes" if replay.summary["abandoned"] else "No")
        with s4:
            st.metric("Total Reward", replay.summary["total_reward"])
        with s5:
            st.metric("Questions Used", replay.summary["questions_used"])


def render_why_better_section() -> None:
    st.header("Why This Is Better Than a Standard Recommender")
    left, right = st.columns(2)
    with left:
        st.subheader("Standard Recommender")
        st.write("- Recommends immediately with limited context")
        st.write("- Higher risk of irrelevant recommendations")
        st.write("- No dynamic trade-off between friction and information")
    with right:
        st.subheader("Budgeted Interactive Recommender")
        st.write("- Adapts: asks only when expected value is high")
        st.write("- Improves personalization under strict question limits")
        st.write("- Optimizes both recommendation quality and user experience")
    st.markdown(
        """
        **Decision flow:**  
        uncertainty high -> ask brief question -> update belief -> recommend  
        uncertainty low -> recommend immediately
        """
    )


def render_results_section(comparison_df: pd.DataFrame) -> None:
    st.header("Results and Comparisons")
    if comparison_df.empty:
        st.warning("No comparison results available. Run experiments to populate this section.")
        return

    available_algorithms = sorted(comparison_df["algorithm"].dropna().unique().tolist())
    selected_algorithms = st.multiselect(
        "Filter algorithms",
        options=available_algorithms,
        default=available_algorithms,
    )
    view_df = comparison_df[comparison_df["algorithm"].isin(selected_algorithms)].copy()
    if view_df.empty:
        st.info("No rows after filters.")
        return

    chart_specs = [
        ("avg_cumulative_reward", "Average Cumulative Reward by Method"),
        ("avg_questions_asked", "Average Questions Asked by Method"),
        ("acceptance_rate", "Acceptance Rate by Method"),
        ("abandonment_rate", "Abandonment Rate by Method"),
    ]
    for metric, title in chart_specs:
        fig = bar_by_algorithm(view_df, metric, title)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    budget_fig = grouped_by_budget(view_df, "avg_cumulative_reward", "Reward Across Question Budgets")
    if budget_fig is not None:
        st.plotly_chart(budget_fig, use_container_width=True)

    profile_fig = grouped_by_profile(view_df, "avg_cumulative_reward", "Reward Across User Profiles")
    if profile_fig is not None:
        st.plotly_chart(profile_fig, use_container_width=True)

    st.subheader("Metrics Table")
    st.dataframe(view_df, use_container_width=True, hide_index=True)


def render_investor_product_framing() -> None:
    st.header("Product Framing")
    st.write(
        "The core product thesis is controlled interactivity: ask fewer but smarter questions to "
        "deliver stronger personalization with less user friction."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Fewer unnecessary questions**")
        st.caption("Budget-aware policy avoids long, frustrating flows.")
    with c2:
        st.markdown("**Better personalization**")
        st.caption("Questions refine belief, improving downstream recommendation decisions.")
    with c3:
        st.markdown("**Adaptive decision-making**")
        st.caption("Policy adapts behavior across multiple user profiles and budgets.")


def render_technical_credibility() -> None:
    st.header("Technical Credibility")
    st.write("- Custom Gymnasium environment with budgeted ask-vs-recommend actions")
    st.write("- Simulated users with hidden preferences and profile-specific behavior")
    st.write("- Tabular Q-Learning and DQN (Stable Baselines3) with baseline comparisons")
    st.write("- Reproducible experiment runners and aggregated result outputs")
    st.write("- Automated tests and implementation-aligned documentation")


def main() -> None:
    ensure_context()
    pages = [
        st.Page(
            render_executive_page,
            title="Executive Overview",
            icon=":material/insights:",
            url_path="executive-overview",
            default=True,
        ),
        st.Page(
            render_product_demo_page,
            title="Product Demo",
            icon=":material/play_circle:",
            url_path="product-demo",
        ),
        st.Page(
            render_results_page,
            title="Results and Comparisons",
            icon=":material/bar_chart:",
            url_path="results-comparisons",
        ),
        st.Page(
            render_investor_technical_page,
            title="Technical Credibility",
            icon=":material/verified:",
            url_path="investor-technical-credibility",
        ),
    ]
    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
