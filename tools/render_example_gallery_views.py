#!/usr/bin/env python3
"""Render organize_it_dataset_v2 examples for visual review.

This is presentation-only. It does not touch the pipeline capture contract
(`current.png`, depth, intrinsics, extrinsics, segmentation). It renders:

  messy.json -> current_view.png
  tidy.json  -> reference_goal_view.png

using the same camera pose and fovy as the handcraft editor, at 1280x720.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "organize_it_dataset_v2"
HANDCRAFT_DIR = REPO_ROOT / "handcraft"
WIDTH = 1280
HEIGHT = 720


def scene_dirs(root: Path) -> list[Path]:
    return [
        path.parent for path in sorted(root.rglob("messy.json"))
        if (path.parent / "tidy.json").is_file()
    ]


def has_pair(path: Path) -> bool:
    return (path / "messy.json").is_file() and (path / "tidy.json").is_file()


def render(scene_json: Path, out_png: Path) -> None:
    if str(HANDCRAFT_DIR) not in sys.path:
        sys.path.insert(0, str(HANDCRAFT_DIR))
    from editor import SceneEditor

    editor = SceneEditor(camera_width=WIDTH, camera_height=HEIGHT)
    editor.load_scene_dict(json.loads(scene_json.read_text()))
    editor.settle_all()
    Image.fromarray(editor.render()).save(out_png)


def main() -> None:
    root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DATASET_ROOT
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if has_pair(root):
        render(root / "messy.json", root / "current_view.png")
        render(root / "tidy.json", root / "reference_goal_view.png")
        print(f"[gallery-render] {root}")
        return

    folders = scene_dirs(root)
    if not folders:
        raise SystemExit(f"no folders with messy.json and tidy.json under {root}")

    for folder in folders:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(folder)], check=True)
    print(f"[gallery-render] done: {len(folders)} folder(s)")


if __name__ == "__main__":
    main()
