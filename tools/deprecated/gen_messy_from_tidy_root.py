#!/usr/bin/env python3
"""Generate messy.json for scene folders under one parent directory.

Usage:
    python tools/deprecated/gen_messy_from_tidy_root.py data/organize_it_dataset_v2/dining_table/after_meal_cleanup
    python tools/deprecated/gen_messy_from_tidy_root.py data/organize_it_dataset_v2/dining_table/after_meal_cleanup --range 1 10
"""
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
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gen_messy as M  # noqa: E402
import render_organize_it_scene_v2 as RND  # noqa: E402
from objects import asset_json_backup_dir, write_asset_json_backup  # noqa: E402
from scene import LIBRARY, create_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=Path)
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"))
    return parser.parse_args()


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
    mod = RND.load_reference_module()
    calibration = mod.preview._load_robotwin_camera_calibration(RND.DEFAULT_CALIBRATION)
    camera_info = calibration["camera"]
    cam, _ = RND.add_calibrated_camera(mod, ts, camera_info, 0.02, 10.0, "wrist_camera")
    M.CAM_W = int(camera_info["width"])
    M.CAM_H = int(camera_info["height"])
    render.set_ray_tracing_samples_per_pixel(1)
    render.set_ray_tracing_path_depth(1)
    cmask = M.corner_mask(gen_args.corner_frac)

    written = failed = skipped = 0
    for idx, scene_dir in enumerate(dirs):
        tidy_path = scene_dir / "tidy.json"
        out_path = scene_dir / "messy.json"
        if not tidy_path.is_file():
            print(f"[skip]  {scene_dir.name}: no tidy.json")
            skipped += 1
            continue
        if out_path.exists():
            print(f"[skip]  {scene_dir.name}: messy.json exists")
            skipped += 1
            continue
        LIBRARY.load_asset_json_backup(asset_json_backup_dir(tidy_path))
        tidy = json.loads(tidy_path.read_text())
        rng = random.Random(idx)
        items, attempts, status = M.generate(ts, cam, tidy, gen_args, rng, cmask, [])
        if items is None:
            print(f"[FAIL]  {scene_dir.name}: {status} ({attempts} attempts)")
            failed += 1
            continue
        messy = M.messy_dict(tidy, items)
        out_path.write_text(json.dumps(messy, indent=2, ensure_ascii=False))
        write_asset_json_backup(out_path, messy, LIBRARY)
        print(f"[write] {scene_dir.name}/messy.json  {len(items)} items  attempt {attempts}")
        written += 1

    print(f"done: {written} written, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
