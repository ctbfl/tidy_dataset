#!/usr/bin/env python3
"""Sample tidy scenes from one constraint-annotation template.

Default target:
  data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2/<NNN>/tidy.json

Each sampled scene is rendered as reference_goal.png with the existing v2
goal renderer. Messy-scene capture files are intentionally not written.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HANDCRAFT_DIR = REPO / "handcraft"
SIMULATIONS_DIR = REPO / "simulations"
DATA_ROOT = REPO / "data" / "organize_it_dataset_v2"
SCENARIO = "dining_table"
VARIATION = "after_meal_cleanup_v2"
DEFAULT_TEMPLATE = "single_person"
ARRANGEMENT = "tidy"
GOAL_IMAGE = "reference_goal.png"
RENDER_SCRIPT = REPO / "tools" / "deprecated" / "export_v2.py"

if str(HANDCRAFT_DIR) not in sys.path:
    sys.path.insert(0, str(HANDCRAFT_DIR))
if str(SIMULATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATIONS_DIR))

from constrain_annotation_server import studio  # noqa: E402
from robotwin_utils import curated_textures  # noqa: E402


def next_scene_ids(root: Path, sample_count: int, start: int | None) -> list[str]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    root.mkdir(parents=True, exist_ok=True)
    if start is not None:
        if start <= 0:
            raise ValueError("--start must be positive")
        return [f"{start + i:03d}" for i in range(sample_count)]
    existing = [
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{3}", path.name)
    ]
    start = (max(existing) + 1) if existing else 1
    return [f"{start + i:03d}" for i in range(sample_count)]


def object_slot(ref: dict) -> str:
    return f"{ref['category']}-{int(ref['set']) + 1}-{int(ref['slot']) + 1}"


def table_texture_ids() -> list[str]:
    ids = [
        texture["id"] for texture in curated_textures("table")
        if texture["id"].startswith(("Marble", "Wood"))
    ]
    if not ids:
        raise ValueError("no Marble/Wood table textures found")
    return sorted(ids)


def sampled_scene(scene_id: str, template_name: str) -> dict:
    records = studio._object_records()
    missing = [record["key"] for record in records if record["key"] not in studio.scene_ids]
    if missing:
        raise ValueError(f"template left objects unplaced: {', '.join(missing)}")

    manifest = []
    items = []
    for record in records:
        ref = record["ref"]
        slot = object_slot(ref)
        sid = studio.scene_ids[record["key"]]
        obj = studio.editor.objects[sid]
        manifest.append({"slot": slot, "role": ref["category"], "asset_id": record["asset_id"]})
        items.append({
            "slot": slot,
            "asset_id": record["asset_id"],
            "transform": obj.get_pose().to_transformation_matrix().tolist(),
        })

    bg = studio.editor.background_state()
    return {
        "version": 2,
        "scenario": SCENARIO,
        "scene_id": scene_id,
        "arrangement": ARRANGEMENT,
        "template": VARIATION,
        "constraint_template": template_name,
        "table": bg["table"],
        "table_texture": bg["table_texture"],
        "wall_texture": bg["wall_texture"],
        "manifest": manifest,
        "items": items,
    }


def render_goal(folder: Path) -> None:
    cmd = [sys.executable, str(RENDER_SCRIPT), "--render-one", str(folder)]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_count", type=int)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--start", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = DATA_ROOT / SCENARIO / VARIATION
    scene_ids = next_scene_ids(out_root, args.sample_count, args.start)
    studio.load_variation(SCENARIO, VARIATION, clear=True)
    studio.load_template(args.template)
    textures = table_texture_ids()

    for scene_id in scene_ids:
        folder = out_root / scene_id
        if folder.exists():
            raise FileExistsError(f"scene folder already exists: {folder}")
        studio.editor.set_background(table_texture_id=random.choice(textures), wall_texture_id=None)
        studio.randomize_sets()
        tidy = sampled_scene(scene_id, args.template)
        folder.mkdir()
        (folder / f"{ARRANGEMENT}.json").write_text(json.dumps(tidy, indent=2, ensure_ascii=False))
        render_goal(folder)
        print(f"[write] {folder.relative_to(REPO)}  {len(tidy['items'])} objects  +{GOAL_IMAGE}")


if __name__ == "__main__":
    main()
