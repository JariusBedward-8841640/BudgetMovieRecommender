"""Plotly chart helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def bar_by_algorithm(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df.columns:
        return None
    show = (
        df.groupby("algorithm", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )
    return px.bar(
        show,
        x="algorithm",
        y=metric,
        title=title,
        template="plotly_dark",
        color="algorithm",
    )


def grouped_by_budget(df: pd.DataFrame, metric: str, title: str):
    if df.empty or "question_budget" not in df.columns or metric not in df.columns:
        return None
    budget_df = df.dropna(subset=["question_budget"]).copy()
    if budget_df.empty:
        return None
    show = (
        budget_df.groupby(["question_budget", "algorithm"], as_index=False)[metric]
        .mean()
        .sort_values(["question_budget", metric], ascending=[True, False])
    )
    return px.bar(
        show,
        x="question_budget",
        y=metric,
        color="algorithm",
        barmode="group",
        title=title,
        template="plotly_dark",
    )


def grouped_by_profile(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df.columns:
        return None
    prof = df.copy()
    if "profile" not in prof.columns:
        return None
    prof["profile_label"] = prof["profile"].fillna("mixed")
    show = (
        prof.groupby(["profile_label", "algorithm"], as_index=False)[metric]
        .mean()
        .sort_values(["profile_label", metric], ascending=[True, False])
    )
    return px.bar(
        show,
        x="profile_label",
        y=metric,
        color="algorithm",
        barmode="group",
        title=title,
        template="plotly_dark",
    )
