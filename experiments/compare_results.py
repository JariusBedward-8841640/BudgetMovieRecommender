"""Aggregate and compare experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS_ORDER = [
    "avg_cumulative_reward",
    "acceptance_rate",
    "skip_rate",
    "abandonment_rate",
    "avg_session_length",
    "avg_questions_asked",
    "question_efficiency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved experiment results.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-json", type=str, default="results/comparison_summary.json")
    parser.add_argument("--output-csv", type=str, default="results/comparison_summary.csv")
    return parser.parse_args()


def _load_jsons(results_root: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in results_root.rglob("*.json"):
        if path.name in ("comparison_summary.json",):
            continue
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "algorithm" in data and "metrics" in data:
            data["_source"] = str(path)
            payloads.append(data)
    return payloads


def _flatten_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "algorithm": payload.get("algorithm", "unknown"),
        "kind": payload.get("kind", "unknown"),
        "source": payload.get("_source", ""),
    }
    metrics = payload.get("metrics", {})
    for metric in METRICS_ORDER:
        row[metric] = metrics.get(metric, None)
    config = payload.get("config", {})
    row["question_budget"] = config.get("question_budget")
    row["profile"] = config.get("profile")
    row["profile_mix"] = config.get("profile_mix")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "kind",
        "source",
        "question_budget",
        "profile",
        "profile_mix",
        *METRICS_ORDER,
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_readable(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No result files found to compare.")
        return
    rows_sorted = sorted(rows, key=lambda r: (r["avg_cumulative_reward"] or -999), reverse=True)
    print("Comparison (sorted by avg_cumulative_reward):")
    for row in rows_sorted:
        print(
            f"- {row['algorithm']} [{row['kind']}] | "
            f"avg_reward={row['avg_cumulative_reward']:.4f} | "
            f"accept_rate={row['acceptance_rate']:.4f} | "
            f"abandon_rate={row['abandonment_rate']:.4f} | "
            f"q_budget={row['question_budget']} | profile={row['profile']} | mix={row['profile_mix']}"
        )


def run_comparison(results_root: Path, output_json: Path, output_csv: Path) -> list[dict[str, Any]]:
    """Load experiment JSON files, save aggregate outputs, and print summary."""
    payloads = _load_jsons(results_root)
    rows = [_flatten_row(p) for p in payloads]

    _write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)

    _print_readable(rows)
    print(f"Saved comparison csv to: {output_csv}")
    print(f"Saved comparison json to: {output_json}")
    return rows


def main() -> None:
    args = parse_args()
    run_comparison(
        results_root=Path(args.results_root),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
