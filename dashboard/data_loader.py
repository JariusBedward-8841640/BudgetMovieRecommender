"""Load and normalize dashboard data from results artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


COMPARISON_COLUMNS = [
    "algorithm",
    "kind",
    "source",
    "question_budget",
    "profile",
    "profile_mix",
    "avg_cumulative_reward",
    "acceptance_rate",
    "skip_rate",
    "abandonment_rate",
    "avg_session_length",
    "avg_questions_asked",
    "question_efficiency",
]


@dataclass
class LoadResult:
    df: pd.DataFrame
    source: str | None
    warnings: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _empty_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPARISON_COLUMNS)


def load_comparison_results() -> LoadResult:
    """Load best available comparison table from known result locations."""
    root = _repo_root()
    warnings: list[str] = []
    candidates = [
        root / "results" / "manual_sweep" / "comparison_summary.csv",
        root / "results" / "manual_check" / "comparison_summary.csv",
        root / "results" / "manual_sweep" / "comparison_summary.json",
        root / "results" / "manual_check" / "comparison_summary.json",
        root / "results" / "comparison_summary.csv",
        root / "results" / "comparison_summary.json",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                rows = payload.get("rows", [])
                df = pd.DataFrame(rows)

            if df.empty:
                warnings.append(f"Comparison file was empty: {path}")
                continue

            missing_cols = [col for col in COMPARISON_COLUMNS if col not in df.columns]
            for col in missing_cols:
                df[col] = None
            df = df[COMPARISON_COLUMNS]
            df = _coerce_numeric(
                df,
                [
                    "question_budget",
                    "avg_cumulative_reward",
                    "acceptance_rate",
                    "skip_rate",
                    "abandonment_rate",
                    "avg_session_length",
                    "avg_questions_asked",
                    "question_efficiency",
                ],
            )
            return LoadResult(df=df, source=str(path), warnings=warnings)
        except Exception as exc:  # pragma: no cover - safety fallback
            warnings.append(f"Failed reading {path}: {exc}")

    warnings.append(
        "No comparison results found. Run experiments first (see README dashboard section)."
    )
    return LoadResult(df=_empty_comparison_df(), source=None, warnings=warnings)


def load_baseline_summary() -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Load baseline summary json if available."""
    root = _repo_root()
    warnings: list[str] = []
    candidates = [
        root / "results" / "manual_check" / "baselines" / "summary.json",
        root / "results" / "baselines" / "summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload, str(path), warnings
        except Exception as exc:  # pragma: no cover - safety fallback
            warnings.append(f"Failed reading {path}: {exc}")

    warnings.append("No baseline summary file found.")
    return None, None, warnings


def compute_kpis(comparison_df: pd.DataFrame) -> dict[str, float | str]:
    """Compute top-level KPI card values from comparison rows."""
    if comparison_df.empty:
        return {
            "best_method": "N/A",
            "best_avg_reward": float("nan"),
            "best_acceptance_rate": float("nan"),
            "avg_session_length": float("nan"),
            "avg_questions_asked": float("nan"),
        }

    sorted_df = comparison_df.sort_values("avg_cumulative_reward", ascending=False)
    top = sorted_df.iloc[0]
    return {
        "best_method": str(top["algorithm"]),
        "best_avg_reward": float(top["avg_cumulative_reward"]),
        "best_acceptance_rate": float(comparison_df["acceptance_rate"].max()),
        "avg_session_length": float(comparison_df["avg_session_length"].mean()),
        "avg_questions_asked": float(comparison_df["avg_questions_asked"].mean()),
    }


def discover_model_artifacts() -> dict[str, str]:
    """Find the newest model artifacts for optional replay policies."""
    root = _repo_root()
    paths: dict[str, str] = {}

    q_candidates = list(root.glob("results/**/q_learning*.pkl"))
    if q_candidates:
        newest_q = max(q_candidates, key=lambda p: p.stat().st_mtime)
        paths["q_learning"] = str(newest_q)

    dqn_candidates = list(root.glob("results/**/dqn_model*.zip"))
    if dqn_candidates:
        newest_dqn = max(dqn_candidates, key=lambda p: p.stat().st_mtime)
        paths["dqn"] = str(newest_dqn)

    return paths
