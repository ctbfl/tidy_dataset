#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from sapien import render

REPO = Path(__file__).resolve().parents[1]
SIM_DIR = REPO / "simulations"
TOOLS_DIR = REPO / "tools"
DATA_ROOT = REPO / "data" / "organize_it_dataset_v2"
DEFAULT_PARENT = DATA_ROOT / "dining_table" / "after_meal_cleanup_v2"

if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import export_to_organize_it as EXP  # noqa: E402
import gen_messy as M  # noqa: E402
from objects import asset_json_backup_dir, write_asset_json_backup  # noqa: E402
from scene import LIBRARY, create_scene  # noqa: E402
from scene_runtime import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_CATALOG,
    add_calibrated_camera,
    capture_current,
    load_reference_module,
    scene_dict_from_loaded_objects,
    settle_scene,
)


def scene_dirs(parent: Path, scene_range: list[int] | None) -> list[Path]:
    if scene_range is None:
        return [
            path for path in sorted(parent.iterdir())
            if path.is_dir() and path.name.isdigit() and (path / "tidy.json").is_file()
        ]
    start, end = scene_range
    if start > end:
        raise ValueError("--range START END requires START <= END")
    return [parent / f"{i:03d}" for i in range(start, end + 1)]


def convert_messy_to_pipeline_inputs(messy_path: Path, scene_type: str | None) -> tuple[dict, object]:
    catalog_path = DEFAULT_CATALOG.expanduser().resolve()
    calibration_path = DEFAULT_CALIBRATION.expanduser().resolve()
    registry = EXP.AssetRegistry.load(catalog_path)
    calibration = json.loads(calibration_path.read_text())
    scene = EXP.convert_scene(
        messy_path,
        registry,
        calibration,
        catalog_path,
        calibration_path,
        scene_type,
    )
    folder = messy_path.parent
    scene_json = folder / "scene.json"
    scene_json.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    tabletop_area = EXP.tabletop_area_payload(scene["generation"]["workspace_bounds_ur_base"], scene_json)
    (folder / "tabletop_area.json").write_text(json.dumps(tabletop_area, indent=2), encoding="utf-8")

    mod = load_reference_module()
    backup_dir = asset_json_backup_dir(messy_path)
    asset_registry = mod.AssetRegistry.load(
        catalog_path,
        asset_json_overwrite_dir=backup_dir if backup_dir.is_dir() else None,
    )
    return scene, asset_registry


def ensure_user_prompt_file(scene_dir: Path, tidy: dict) -> None:
    path = scene_dir / "user_prompt.txt"
    if path.is_file():
        return
    prompt = str(tidy.get("user_prompt", "")).strip()
    if prompt:
        path.write_text(prompt + "\n")
        return
    print(f"[warn] {scene_dir.name}: no user_prompt.txt and tidy.json has no user_prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=Path, nargs="?", default=DEFAULT_PARENT)
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent = args.parent_dir.expanduser().resolve()
    if not parent.is_dir():
        raise SystemExit(f"not a directory: {parent}")
    dirs = scene_dirs(parent, args.range)
    if not dirs:
        raise SystemExit(f"no scene folders with tidy.json under {parent}")

    gen_args = argparse.Namespace(
        height_thresh=0.30,
        area_thresh=None,
        min_visible=0.80,
        corner_frac=0.25,
        gap=0.5,
        max_attempts=300,
    )
    ts = create_scene(headless=True, table_length=1.2, table_width=0.7, table_height=0.74)
    mod = load_reference_module()
    calibration = mod.preview._load_robotwin_camera_calibration(DEFAULT_CALIBRATION)
    camera_info = calibration["camera"]
    cam, _ = add_calibrated_camera(mod, ts, camera_info)
    M.CAM_W = int(camera_info["width"])
    M.CAM_H = int(camera_info["height"])
    render.set_ray_tracing_samples_per_pixel(1)
    render.set_ray_tracing_path_depth(1)
    cmask = M.corner_mask(gen_args.corner_frac)

    written = failed = skipped = 0
    for idx, scene_dir in enumerate(dirs):
        tidy_path = scene_dir / "tidy.json"
        messy_path = scene_dir / "messy.json"
        if not tidy_path.is_file():
            print(f"[skip]  {scene_dir.name}: no tidy.json")
            skipped += 1
            continue
        if messy_path.exists() and not args.overwrite:
            print(f"[skip]  {scene_dir.name}: messy.json exists")
            skipped += 1
            continue

        LIBRARY.load_asset_json_backup(asset_json_backup_dir(tidy_path))
        tidy = json.loads(tidy_path.read_text())
        ensure_user_prompt_file(scene_dir, tidy)
        rng = random.Random(idx)
        items, attempts, status = M.generate(ts, cam, tidy, gen_args, rng, cmask, [])
        if items is None:
            print(f"[FAIL]  {scene_dir.name}: {status} ({attempts} attempts)")
            failed += 1
            continue

        messy = M.messy_dict(tidy, items)
        settle_scene(ts)
        messy = scene_dict_from_loaded_objects(messy, ts)
        messy_path.write_text(json.dumps(messy, indent=2, ensure_ascii=False))
        write_asset_json_backup(messy_path, messy, LIBRARY)

        scene_data, asset_registry = convert_messy_to_pipeline_inputs(messy_path, None)
        capture_current(ts, scene_dir, scene_data, asset_registry)
        print(f"[write] {scene_dir.name}/messy.json + current.*  {len(messy['items'])} items  attempt {attempts}")
        written += 1

    print(f"done: {written} written, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
