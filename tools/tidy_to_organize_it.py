#!/usr/bin/env python3
"""End-to-end: one TidyDataset v0 scene JSON -> one organize_it-ready folder.

This is the single command you run. It chains the two stages internally:

  1. export_to_organize_it.py   -> scene.json + tabletop_area.json   (no GPU)
  2. render_organize_it_scene.py -> current.png / current_depth.pkl /
       current_intrinsics.yaml / current_extrinsics.yaml / GT segmentation (GPU)

so the output directory ends up holding the full capture set the pipeline consumes.

Usage
-----
    python tools/tidy_to_organize_it.py <v0_scene.json> <out_dir>

    # e.g. plate_fork_messy -> .../tidy_data_rolling/data/plate_fork_1
    python tools/tidy_to_organize_it.py \
        data/tidy_scene_v0/plate_fork_messy.json \
        /home/hjs/.../tidy_data_rolling/data/plate_fork_1

<out_dir> IS the target subdataset folder (everything is written directly inside it).
Optional flags (--scene-type / --settle-steps / --camera-calibration) are forwarded
to the underlying stages; you normally need none of them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
EXPORT_SCRIPT = TOOLS_DIR / "export_to_organize_it.py"
RENDER_SCRIPT = TOOLS_DIR / "render_organize_it_scene.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("v0_scene", type=Path, help="TidyDataset v0 scene JSON (path or stem).")
    p.add_argument("out_dir", type=Path, help="Target folder for the organize_it dataset (created if absent).")
    p.add_argument("--scene-type", default=None, help="Force organize_it scene_type (else inferred).")
    p.add_argument("--settle-steps", type=int, default=0,
                   help="Physics steps before capture (0 keeps the handcrafted layout).")
    p.add_argument("--camera-calibration", type=Path, default=None,
                   help="Override camera calibration json (default: pipeline's step15 calibration).")
    return p.parse_args()


def run(stage: str, cmd: list[str]) -> None:
    print(f"\n=== [{stage}] {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[{stage}] failed (exit {result.returncode})")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()

    # Stage 1 writes <out-root>/<name>/{scene.json,tabletop_area.json}; point those at out_dir.
    export_cmd = [
        sys.executable, str(EXPORT_SCRIPT), str(args.v0_scene),
        "--out-root", str(out_dir.parent), "--name", out_dir.name,
    ]
    if args.scene_type:
        export_cmd += ["--scene-type", args.scene_type]
    if args.camera_calibration:
        export_cmd += ["--camera-calibration", str(args.camera_calibration)]
    run("export", export_cmd)

    # Stage 2 reads out_dir/scene.json (and finds the source v0 from it) and renders into out_dir.
    render_cmd = [
        sys.executable, str(RENDER_SCRIPT), str(out_dir),
        "--settle-steps", str(args.settle_steps),
    ]
    if args.camera_calibration:
        render_cmd += ["--camera-calibration", str(args.camera_calibration)]
    run("render", render_cmd)

    print(f"\n[done] organize_it dataset -> {out_dir}")
    print("[done] files:", ", ".join(sorted(p.name for p in out_dir.iterdir())))


if __name__ == "__main__":
    main()
