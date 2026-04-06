"""Product demo page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.context import model_paths
from dashboard.session_demo import available_profiles, build_policy_map, replay_session


def _default_policy_index(keys: list[str]) -> int:
    if "dqn" in keys:
        return keys.index("dqn")
    if "q_learning" in keys:
        return keys.index("q_learning")
    return 0


def render() -> None:
    st.header("Product Demo")
    st.write("Run a simulated recommendation session with interactive policy selection.")

    paths = model_paths()
    profiles = available_profiles()
    policy_map = build_policy_map(paths)
    policy_keys = list(policy_map.keys())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        profile = st.selectbox("User Profile", options=profiles, index=0)
    with c2:
        budget = st.slider("Question Budget", min_value=0, max_value=5, value=2, step=1)
    with c3:
        policy_key = st.selectbox(
            "Policy",
            options=policy_keys,
            format_func=lambda key: policy_map[key],
            index=_default_policy_index(policy_keys),
        )
    with c4:
        seed = st.number_input("Seed", min_value=0, value=7, step=1)

    max_steps = st.slider("Session Max Steps", min_value=4, max_value=12, value=8, step=1)
    advanced = st.toggle("Show advanced details", value=False)

    if st.button("Start Simulated Session", type="primary"):
        replay = replay_session(
            policy_key=policy_key,
            profile=profile,
            question_budget=int(budget),
            max_steps=int(max_steps),
            seed=int(seed),
            model_paths=paths,
        )
        if replay.warning:
            st.warning(replay.warning)
            return

        st.subheader("Session Timeline")
        frame = pd.DataFrame(replay.steps)
        if not advanced and not frame.empty:
            keep_cols = [
                "step",
                "action",
                "outcome",
                "reward",
                "engagement",
                "remaining_question_budget",
            ]
            frame = frame[[col for col in keep_cols if col in frame.columns]]
        st.dataframe(frame, use_container_width=True, hide_index=True)

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
