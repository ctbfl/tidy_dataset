#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


TIDY_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset")
DATA_ROOT = TIDY_ROOT / "data" / "sgbot" / "exp0"
SGBOT_DIR = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/SG-Bot")
LIVE_RESULTS = SGBOT_DIR / "sg_bot_oracle_results"
WRAPPER = TIDY_ROOT / "sgbot_comparison" / "run_sgbot_oracle_with_tiny_observation.py"
RERENDER = TIDY_ROOT / "sgbot_comparison" / "rerender_sgbot_result.py"


def is_done(scene_dir: Path) -> bool:
    result = scene_dir / "sgbot_output" / "result.png"
    pkl = scene_dir / "sgbot_output" / "oracle_tiny_observation" / "result.pkl"
    if not result.is_file() or not pkl.is_file():
        return False
    image = Image.open(result)
    return image.size == (1280, 960) and image.mode == "RGB"


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=TIDY_ROOT, env=env, check=True)


def run_one(scene_dir: Path) -> None:
    scene = scene_dir.name
    out_dir = scene_dir / "sgbot_output"
    live_dir = LIVE_RESULTS / scene
    original_live = Path(tempfile.mkdtemp(prefix="sgbot_live_backup_")) / scene
    demo_file = Path(tempfile.mkdtemp(prefix="sgbot_demo_")) / "demo.txt"
    demo_file.write_text(scene + "\n")

    shutil.copytree(live_dir, original_live)
    original_tidy_render = out_dir / "result_tidy_render.png"
    saved_tidy_render = None
    if original_tidy_render.is_file():
        saved_tidy_render = Path(tempfile.mkdtemp(prefix="sgbot_tidy_render_")) / "result_tidy_render.png"
        shutil.copy2(original_tidy_render, saved_tidy_render)

    try:
        env = dict(os.environ)
        env["SGBOT_DEMO_FILE"] = str(demo_file)
        run_cmd([
            "xvfb-run", "-a", "conda", "run", "--no-capture-output", "-n", "sgbot",
            "python", str(WRAPPER),
        ], env=env)

        rerun_dir = out_dir / "oracle_tiny_observation"
        if rerun_dir.exists():
            shutil.rmtree(rerun_dir)
        shutil.copytree(live_dir, rerun_dir)

        run_cmd([
            "xvfb-run", "-a", "conda", "run", "--no-capture-output", "-n", "sgbot",
            "python", str(RERENDER), str(scene_dir),
        ])

        source = out_dir / "result_tidy_render.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        old_result = out_dir / "result.png"
        backup = out_dir / "result_opengl_original.png"
        if old_result.is_file() and not backup.exists():
            shutil.copy2(old_result, backup)
        shutil.copy2(source, old_result)
    finally:
        if live_dir.exists():
            shutil.rmtree(live_dir)
        shutil.copytree(original_live, live_dir)
        if saved_tidy_render is not None:
            shutil.copy2(saved_tidy_render, original_tidy_render)


def main() -> None:
    scenes = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    for index, scene_dir in enumerate(scenes, 1):
        if is_done(scene_dir):
            print(f"[{index}/{len(scenes)}] skip {scene_dir.name}", flush=True)
            continue
        print(f"[{index}/{len(scenes)}] run {scene_dir.name}", flush=True)
        run_one(scene_dir)
        print(f"[{index}/{len(scenes)}] done {scene_dir.name}", flush=True)


if __name__ == "__main__":
    main()
