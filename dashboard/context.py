"""Shared dashboard context loading helpers."""

from __future__ import annotations

import streamlit as st

from dashboard.data_loader import discover_model_artifacts, load_comparison_results


def ensure_context() -> None:
    """Load common data once per session."""
    if "comparison_load" not in st.session_state:
        st.session_state["comparison_load"] = load_comparison_results()
    if "model_paths" not in st.session_state:
        st.session_state["model_paths"] = discover_model_artifacts()


def comparison_load():
    ensure_context()
    return st.session_state["comparison_load"]


def model_paths() -> dict[str, str]:
    ensure_context()
    return st.session_state["model_paths"]
