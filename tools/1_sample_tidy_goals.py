#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HANDCRAFT_DIR = REPO / "handcraft"
SIMULATIONS_DIR = REPO / "simulations"
ARRANGEMENT = "tidy"
GOAL_IMAGE = "reference_goal.png"

if str(HANDCRAFT_DIR) not in sys.path:
    sys.path.insert(0, str(HANDCRAFT_DIR))
if str(SIMULATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATIONS_DIR))

studio = None
LIBRARY = None
write_asset_json_backup = None
curated_textures = None
render_reference_goal = None
settle_scene = None
load_reference_module = None
add_calibrated_camera = None
DEFAULT_CALIBRATION = None


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


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def resolve_constraint_path(path_text: str) -> tuple[Path, Path, str, str, str]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix != ".json":
        raise ValueError(f"constraint path must be a .json file: {path}")
    if path.parent.name != "constraints" or path.parent.parent.name != "template":
        raise ValueError(f"constraint path must end with <scenario>/<variation>/template/constraints/<name>.json: {path}")
    variation_dir = path.parent.parent.parent
    scenario_dir = variation_dir.parent
    dataset_root = scenario_dir.parent
    if not scenario_dir.is_dir() or not variation_dir.is_dir():
        raise ValueError(f"invalid scenario/variation directory for constraint path: {path}")
    return path, dataset_root, scenario_dir.name, variation_dir.name, path.stem


def validate_dataset_format(constraint_path: Path, dataset_root: Path, scenario: str, variation: str) -> None:
    global_available_path = dataset_root / "available_assets.json"
    variation_available_path = dataset_root / scenario / variation / "template" / "available_assets.json"
    if not global_available_path.is_file():
        raise FileNotFoundError(global_available_path)
    if not variation_available_path.is_file():
        raise FileNotFoundError(variation_available_path)

    global_available = load_json(global_available_path).get("available_assets")
    if not isinstance(global_available, dict):
        raise ValueError(f"{global_available_path} missing available_assets object")

    variation_available = load_json(variation_available_path)
    group_ids = variation_available.get("assets_group")
    if not isinstance(group_ids, list):
        raise ValueError(f"{variation_available_path} missing assets_group list")
    missing_groups = [str(group_id) for group_id in group_ids if str(group_id) not in global_available]
    if missing_groups:
        raise ValueError(f"{variation_available_path} references unknown assets_group: {missing_groups[0]}")

    constraint = load_json(constraint_path)
    enabled_groups = {str(group_id) for group_id in group_ids}
    missing_categories = [
        str(object_set.get("category"))
        for object_set in constraint.get("object_sets", [])
        if str(object_set.get("category")) not in enabled_groups
    ]
    if missing_categories:
        raise ValueError(f"{constraint_path.name}: category is not enabled for this variation: {missing_categories[0]}")


def import_runtime_modules(dataset_root: Path) -> None:
    global studio, LIBRARY, write_asset_json_backup, curated_textures, render_reference_goal, settle_scene
    global load_reference_module, add_calibrated_camera, DEFAULT_CALIBRATION
    os.environ["TIDY_DATASET_DIR"] = str(dataset_root)

    from constrain_annotation_server import studio as constraint_studio  # noqa: E402
    from objects import write_asset_json_backup as write_backup  # noqa: E402
    from robotwin_utils import curated_textures as texture_catalog  # noqa: E402
    from scene import LIBRARY as asset_library  # noqa: E402
    from scene_runtime import (  # noqa: E402
        DEFAULT_CALIBRATION as default_calibration,
        add_calibrated_camera as add_camera,
        load_reference_module as load_ref_module,
        render_reference_goal as render_goal,
        settle_scene as settle,
    )

    studio = constraint_studio
    LIBRARY = asset_library
    write_asset_json_backup = write_backup
    curated_textures = texture_catalog
    render_reference_goal = render_goal
    settle_scene = settle
    load_reference_module = load_ref_module
    add_calibrated_camera = add_camera
    DEFAULT_CALIBRATION = default_calibration


def table_texture_ids() -> list[str]:
    ids = [
        texture["id"] for texture in curated_textures("table")
        if texture["id"].startswith(("Marble", "Wood"))
    ]
    if not ids:
        raise ValueError("no Marble/Wood table textures found")
    return sorted(ids)


def tidy_scene_dict(scene_id: str, scenario: str, variation: str, template_name: str) -> dict:
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
        "scenario": scenario,
        "scene_id": scene_id,
        "arrangement": ARRANGEMENT,
        "template": variation,
        "constraint_template": template_name,
        "user_prompt": studio.user_prompt,
        "table": bg["table"],
        "table_texture": bg["table_texture"],
        "wall_texture": bg["wall_texture"],
        "manifest": manifest,
        "items": items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("constraint_path")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="force re-run scene ids in this batch even if folders already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constraint_path, dataset_root, scenario, variation, template_name = resolve_constraint_path(args.constraint_path)
    validate_dataset_format(constraint_path, dataset_root, scenario, variation)
    out_root = dataset_root / scenario / variation
    scene_ids = next_scene_ids(out_root, args.count, args.start)
    existing_folders = [out_root / scene_id for scene_id in scene_ids if (out_root / scene_id).exists()]
    if existing_folders and not args.overwrite:
        raise FileExistsError(f"scene folder already exists: {existing_folders[0]}")

    import_runtime_modules(dataset_root)
    studio.load_variation(scenario, variation, clear=True)
    studio.load_template(template_name)
    textures = table_texture_ids()
    mod = load_reference_module()
    calibration = mod.preview._load_robotwin_camera_calibration(DEFAULT_CALIBRATION)
    goal_camera, _ = add_calibrated_camera(mod, studio.editor.scene_wrap, calibration["camera"])

    for scene_id in scene_ids:
        folder = out_root / scene_id
        studio.editor.set_background(table_texture_id=random.choice(textures), wall_texture_id=None)
        studio.randomize_sets()
        settle_scene(studio.editor.scene_wrap)
        tidy = tidy_scene_dict(scene_id, scenario, variation, template_name)
        folder.mkdir(exist_ok=args.overwrite)
        tidy_path = folder / f"{ARRANGEMENT}.json"
        tidy_path.write_text(json.dumps(tidy, indent=2, ensure_ascii=False))
        if tidy.get("user_prompt"):
            (folder / "user_prompt.txt").write_text(tidy["user_prompt"].strip() + "\n")
        else:
            print(f"[warn] {display_path(folder)}: template has no user_prompt")
        write_asset_json_backup(tidy_path, tidy, LIBRARY)
        render_reference_goal(studio.editor.scene_wrap, folder / GOAL_IMAGE, mod, goal_camera)
        tag = "overwrite" if args.overwrite and folder in existing_folders else "write"
        print(f"[{tag}] {display_path(folder)}  {len(tidy['items'])} objects  +{GOAL_IMAGE}")


if __name__ == "__main__":
    main()
