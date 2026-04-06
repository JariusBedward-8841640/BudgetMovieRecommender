"""UI formatting helpers for dashboard pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{100.0 * value:.1f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def metric_card(title: str, value: str) -> None:
    st.metric(label=title, value=value)
