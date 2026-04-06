"""Executive and technical credibility page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("Technical Credibility")

    st.subheader("Executive Framing")
    st.write(
        "This product operationalizes an adaptive recommendation conversation: ask only when "
        "needed, then recommend with higher confidence."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Lower interaction friction**")
        st.caption("Fewer unnecessary questions improve user experience.")
    with c2:
        st.markdown("**Higher personalization quality**")
        st.caption("Clarifying questions are used strategically, not by default.")
    with c3:
        st.markdown("**Scalable decision policy**")
        st.caption("Approach generalizes across user types and budget constraints.")

    st.subheader("Technical Methodology")
    st.write("- Custom Gymnasium environment for budgeted ask-vs-recommend control")
    st.write("- Simulated users with hidden preferences and behavior profiles")
    st.write("- Baseline policies + tabular Q-Learning + DQN (Stable Baselines3)")
    st.write("- Experiment runners with comparison outputs across methods/budgets/profiles")
    st.write("- Automated tests and documentation for reproducibility")
