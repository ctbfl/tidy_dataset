#!/usr/bin/env python3
"""Summarize VLM eval order results by model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset")
INPUT_PATH = PROJECT_ROOT / "open_source_eval" / "vlm_eval_results.pkl"
OUTPUT_PATH = PROJECT_ROOT / "open_source_eval" / "vlm_eval_model_summary.csv"
JSON_OUTPUT_PATH = PROJECT_ROOT / "open_source_eval" / "vlm_eval_model_summary.json"

SUMMARY_COLUMNS = ("pure win", "pure lose", "tie", "inconsistent", "weak win", "weak lose")


def final_label(order1: str, order2: str) -> str:
    if order1 == "tie" and order2 == "tie":
        return "tie"
    if order1 == "ours" and order2 == "ours":
        return "pure win"
    if order1 == "sg_bot" and order2 == "sg_bot":
        return "pure lose"
    if {order1, order2} == {"ours", "tie"}:
        return "weak win"
    if {order1, order2} == {"sg_bot", "tie"}:
        return "weak lose"
    if {order1, order2} == {"ours", "sg_bot"}:
        return "inconsistent"
    raise ValueError(f"unexpected winners: {order1!r}, {order2!r}")


def build_summary() -> pd.DataFrame:
    df = pd.read_pickle(INPUT_PATH)
    required_columns = {"scene_id", "model", "perspective", "order_idx", "winner"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    pairs = df.pivot(
        index=["model", "scene_id", "perspective"],
        columns="order_idx",
        values="winner",
    )
    if set(pairs.columns) != {0, 1}:
        raise ValueError(f"unexpected order_idx columns: {sorted(pairs.columns)}")

    pairs["final"] = [final_label(order1, order2) for order1, order2 in zip(pairs[0], pairs[1], strict=True)]
    summary = (
        pairs.reset_index()
        .groupby(["model", "perspective", "final"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SUMMARY_COLUMNS, fill_value=0)
        .reset_index()
    )
    return summary.sort_values(["model", "perspective"]).reset_index(drop=True)


def main() -> None:
    summary = build_summary()
    summary.to_csv(OUTPUT_PATH, index=False)
    summary.to_json(JSON_OUTPUT_PATH, orient="records", indent=2)
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
