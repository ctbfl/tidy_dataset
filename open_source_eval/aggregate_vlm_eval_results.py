#!/usr/bin/env python3
"""Aggregate raw VLM comparison order results into one pandas pickle."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset")
DATA_ROOT = PROJECT_ROOT / "data" / "sgbot" / "exp0"
OUTPUT_PATH = PROJECT_ROOT / "open_source_eval" / "vlm_eval_results.pkl"

DEBUG_DIR = "debug_comparison"
PERSPECTIVES = ("tidy", "user_intent")
JSON_ORDERS = ("order1", "order2")
SCENE_RE = re.compile(r"^([A-Za-z0-9]{6})[A-Za-z0-9]*_(\d{4})_(\d+)_(scene-[^_]+)_type-(.+)$")

MODEL_FILES = {
    "gpt_5.5": "sg_bot_vs_ours_with_obstacle.json",
    "opus_4.8": "anthropic_claude_opus_4_8_sg_bot_vs_ours_with_obstacle.json",
    "sonnet_5": "anthropic_claude_sonnet_5_sg_bot_vs_ours_with_obstacle.json",
    "gemini_3.1_pro": "gemini_3_1_pro_preview_sg_bot_vs_ours_with_obstacle.json",
    "gemma_4_12B_it": "gemma_4_12B_it_sg_bot_vs_ours_with_obstacle.json",
    "gemma_4_26B_A4B": "gemma_4_26B_A4B_it_sg_bot_vs_ours_with_obstacle.json",
    "glm_4.6v_flash": "glm_4_6v_flash_sg_bot_vs_ours_with_obstacle.json",
    "kimi_k2.6": "kimi_k2_6_sg_bot_vs_ours_with_obstacle.json",
    "qwen3.6_27B": "qwen3_6_27B_sg_bot_vs_ours_with_obstacle.json",
    "qwen3_vl_8B": "qwen3_vl_8B_instruct_sg_bot_vs_ours_with_obstacle.json",
}


def parse_scene_name(scene_name: str) -> dict[str, Any]:
    match = SCENE_RE.match(scene_name)
    if match is None:
        raise ValueError(f"unexpected scene name: {scene_name}")

    scene_prefix, case_id, object_count, scene_spec, type_spec = match.groups()
    return {
        "scene_id": f"{scene_prefix}_{case_id}_{object_count}",
        "scene_name": scene_name,
        "scene_prefix": scene_prefix,
        "case_id": case_id,
        "object_count": int(object_count),
        "scene_spec": scene_spec,
        "type_spec": type_spec,
    }


def parse_layout(layout: str) -> dict[str, str]:
    label_by_letter: dict[str, str] = {}
    for part in layout.split(","):
        letter, label = part.split("=", 1)
        label_by_letter[letter.strip()] = label.strip()

    if set(label_by_letter) != {"A", "B"}:
        raise ValueError(f"unexpected layout: {layout}")
    if set(label_by_letter.values()) != {"ours", "sg_bot"}:
        raise ValueError(f"unexpected layout labels: {layout}")
    return label_by_letter


def order_from_layout(label_by_letter: dict[str, str]) -> tuple[int, str]:
    if label_by_letter["A"] == "ours":
        return 0, "our_first"
    return 1, "sgbot_first"


def winner_from_raw(raw_winner: str, label_by_letter: dict[str, str]) -> str:
    if raw_winner == "tie":
        return "tie"
    if raw_winner in label_by_letter:
        return label_by_letter[raw_winner]
    raise ValueError(f"unexpected winner: {raw_winner}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return data


def rows_for_result(scene_dir: Path, model: str, json_path: Path) -> list[dict[str, Any]]:
    data = load_json(json_path)
    scene_info = parse_scene_name(scene_dir.name)

    rows: list[dict[str, Any]] = []
    for json_order in JSON_ORDERS:
        order_data = data[json_order]
        label_by_letter = parse_layout(order_data["layout"])
        order_idx, order = order_from_layout(label_by_letter)

        for perspective in PERSPECTIVES:
            parsed = order_data["parsed"][perspective]
            rows.append(
                {
                    **scene_info,
                    "model": model,
                    "order_idx": order_idx,
                    "order": order,
                    "perspective": perspective,
                    "winner": winner_from_raw(parsed["winner"], label_by_letter),
                    "reason": parsed["reason"],
                    "json_order": json_order,
                    "raw_winner": parsed["winner"],
                    "layout": order_data["layout"],
                    "json_path": str(json_path),
                }
            )

    return rows


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scene_dirs = sorted(path for path in DATA_ROOT.iterdir() if (path / DEBUG_DIR).is_dir())
    if not scene_dirs:
        raise RuntimeError(f"no scene debug directories found under {DATA_ROOT}")

    for scene_dir in scene_dirs:
        debug_dir = scene_dir / DEBUG_DIR
        for model, file_name in MODEL_FILES.items():
            json_path = debug_dir / file_name
            if not json_path.is_file():
                raise FileNotFoundError(json_path)
            rows.extend(rows_for_result(scene_dir, model, json_path))

    return rows


def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(collect_rows())
    expected_rows_per_scene = len(MODEL_FILES) * len(PERSPECTIVES) * len(JSON_ORDERS)
    scene_count = df["scene_id"].nunique()
    if len(df) != scene_count * expected_rows_per_scene:
        raise RuntimeError(f"unexpected row count: got {len(df)}, scenes={scene_count}")
    return df.sort_values(["scene_id", "model", "order_idx", "perspective"]).reset_index(drop=True)


def main() -> None:
    df = build_dataframe()
    df.to_pickle(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")
    print(f"rows={len(df)} scenes={df['scene_id'].nunique()} models={df['model'].nunique()}")


if __name__ == "__main__":
    main()
