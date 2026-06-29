#!/usr/bin/env python3
"""export_v1 — batch-export a whole scenario directory for the organize_it pipeline.

Point it at a scenario folder (e.g. ``data/scenarios/office_desk``). For every scene
folder inside it (``001/``, ``002/`` … each holding ``messy.json`` + ``tidy.json``)
this writes, *in place* in that same folder:

  From ``messy.json`` — the pipeline's FULL input capture (the starting observation):
    - ``scene.json`` + ``tabletop_area.json``   (export_to_organize_it.convert_scene; no GPU)
    - ``current.png`` / ``current_depth.pkl`` / ``current_intrinsics.yaml`` /
      ``current_extrinsics.yaml`` + GT segmentation (``current_pybullet_segmentation.npy``,
      ``extract_meta.json``)  — rendered from the calibrated camera by
      render_organize_it_scene.py, i.e. the exact capture contract the pipeline consumes.

  From ``tidy.json`` — the goal state:
    - ``reference_goal.png``  — rendered from the SAME calibrated camera (so the goal lines
      up with ``current.png``). Render only: no depth / intrinsics / extrinsics / seg.

So ``messy`` becomes the pipeline's starting scene + observation, ``tidy`` its visual target.
Each scene renders in its own subprocess (SAPIEN never frees a scene's GPU resources, so a
couple dozen in one process exhausts the device) — a clean process == clean GPU state.

Usage
-----
    python tools/deprecated/export_v1.py data/scenarios/office_desk
    python tools/deprecated/export_v1.py data/scenarios/office_desk --scene-type Desk
    python tools/deprecated/export_v1.py data/scenarios/office_desk/001        # a single scene folder
    python tools/deprecated/export_v1.py data/scenarios/office_desk --no-render # scene.json only (no GPU)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
ACTIVE_TOOLS_DIR = TOOLS_DIR.parent
for path in (TOOLS_DIR, ACTIVE_TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_to_organize_it as EXP          # noqa: E402  (messy.json -> scene.json; no GPU at import)
import render_organize_it_scene as RND        # noqa: E402  (tidy.json -> rgb; SAPIEN loaded lazily)

RENDER_SCRIPT = TOOLS_DIR / "render_organize_it_scene.py"  # messy.json -> full capture contract
INPUT_ARRANGEMENT = "messy"   # -> scene.json + tabletop_area.json + current.* capture
GOAL_ARRANGEMENT = "tidy"     # -> reference_goal.png
GOAL_IMAGE_NAME = "reference_goal.png"


def scene_folders(root: Path) -> list[Path]:
    """The scene folders to process: ``root`` itself if it directly holds the
    arrangements, else each immediate subfolder that does."""
    has = lambda d: (d / f"{INPUT_ARRANGEMENT}.json").is_file() or (d / f"{GOAL_ARRANGEMENT}.json").is_file()
    if has(root):
        return [root]
    return [d for d in sorted(root.iterdir()) if d.is_dir() and has(d)]


# --------------------------------------------------------------------------- #
# stage 1 — convert messy.json -> scene.json + tabletop_area.json (no GPU)
# --------------------------------------------------------------------------- #
def convert_messy(folder: Path, ctx: dict, scene_type: str | None) -> int:
    """Returns the object count written, or -1 if there is no messy.json."""
    src = folder / f"{INPUT_ARRANGEMENT}.json"
    if not src.is_file():
        return -1
    scene = EXP.convert_scene(
        src, ctx["registry"], ctx["calibration"],
        ctx["catalog_path"], ctx["calibration_path"], scene_type,
    )
    out_json = folder / "scene.json"
    out_json.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    ta = EXP.tabletop_area_payload(scene["generation"]["workspace_bounds_ur_base"], out_json)
    (folder / "tabletop_area.json").write_text(json.dumps(ta, indent=2), encoding="utf-8")
    return len(scene["objects"])


# --------------------------------------------------------------------------- #
# stage 2 — render messy -> full capture contract (current.png / depth / intr / extr / seg)
# --------------------------------------------------------------------------- #
# Delegated to render_organize_it_scene.py, run as a subprocess (it reads scene.json,
# rebuilds the scene from messy.json, and writes the current.* capture files in place).
def capture_messy_subprocess(folder: Path, args: argparse.Namespace) -> tuple[bool, str]:
    """Render the messy scene's full input-capture contract into the folder. (ok, detail)."""
    cmd = [
        sys.executable, str(RENDER_SCRIPT.resolve()), str(folder),
        "--v0", str(folder / f"{INPUT_ARRANGEMENT}.json"),
        "--asset-catalog", str(args.asset_catalog),
        "--camera-calibration", str(args.camera_calibration),
        "--camera-name", args.camera_name,
        "--near-m", str(args.near_m), "--far-m", str(args.far_m),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, "current.png + depth/intrinsics/extrinsics/seg"
    tail = (proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    return False, tail


# --------------------------------------------------------------------------- #
# stage 3 — render tidy.json -> reference_goal.png (same calibrated camera, GPU)
# --------------------------------------------------------------------------- #
# Each render builds a fresh SAPIEN scene + camera; the renderer never frees the
# previous one, so a couple dozen in a single process exhausts the GPU. We therefore
# render every scene in its OWN subprocess (this script re-invoked with --render-one):
# a clean process == clean GPU state, at the cost of one interpreter start per scene.
def render_goal(folder: Path, near: float, far: float, camera_name: str, calibration_path: Path) -> int:
    """Render tidy.json from the calibrated camera into reference_goal.png, in-process.
    Returns the object count, or -1 if there is no tidy.json."""
    src = folder / f"{GOAL_ARRANGEMENT}.json"
    if not src.is_file():
        return -1
    mod = RND.load_reference_module()
    calibration = mod.preview._load_robotwin_camera_calibration(calibration_path.resolve())
    tidy = json.loads(src.read_text())
    ts = RND.build_scene_with_tidy_loader(tidy, tidy.get("table_texture"))
    camera, _ = RND.add_calibrated_camera(mod, ts, calibration["camera"], near, far, camera_name)
    ts.scene.update_render()
    capture = mod.capture_camera(camera)
    Image.fromarray(capture["rgb"]).save(folder / GOAL_IMAGE_NAME)
    return len(ts.objects)


def render_goal_subprocess(folder: Path, args: argparse.Namespace) -> tuple[bool, str]:
    """Render one folder's reference_goal.png in a clean subprocess. Returns (ok, detail)."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--render-one", str(folder),
        "--camera-calibration", str(args.camera_calibration),
        "--camera-name", args.camera_name,
        "--near-m", str(args.near_m), "--far-m", str(args.far_m),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, (proc.stdout.strip().splitlines() or [""])[-1]
    tail = (proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    return False, tail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario_dir", type=Path, nargs="?", default=None,
                   help="Scenario folder (e.g. data/scenarios/office_desk) or a single scene folder.")
    p.add_argument("--scene-type", choices=EXP.CATEGORIES, default=None,
                   help="Force organize_it scene_type for every scene (else per-file map / majority vote).")
    p.add_argument("--asset-catalog", type=Path, default=EXP.DEFAULT_CATALOG)
    p.add_argument("--camera-calibration", type=Path, default=EXP.DEFAULT_CALIBRATION)
    p.add_argument("--camera-name", default="wrist_camera")
    p.add_argument("--near-m", type=float, default=0.02)
    p.add_argument("--far-m", type=float, default=10.0)
    p.add_argument("--no-render", action="store_true",
                   help="Convert only (scene.json + tabletop_area.json); skip both the messy capture and the goal render.")
    p.add_argument("--render-one", type=Path, default=None,
                   help="Internal: render this one scene folder's reference_goal.png and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Internal single-folder render mode (one clean process per scene).
    if args.render_one is not None:
        n = render_goal(args.render_one.expanduser().resolve(),
                        args.near_m, args.far_m, args.camera_name, args.camera_calibration)
        print(f"rendered {GOAL_IMAGE_NAME} objects={n}")
        return

    if args.scenario_dir is None:
        raise SystemExit("scenario_dir is required (e.g. data/scenarios/office_desk)")
    root = args.scenario_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    folders = scene_folders(root)
    if not folders:
        raise SystemExit(f"no scene folders with {INPUT_ARRANGEMENT}.json / {GOAL_ARRANGEMENT}.json under {root}")

    catalog_path = args.asset_catalog.expanduser().resolve()
    calibration_path = args.camera_calibration.expanduser().resolve()
    ctx = {
        "registry": EXP.AssetRegistry.load(catalog_path),
        "calibration": json.loads(calibration_path.read_text()),
        "catalog_path": catalog_path,
        "calibration_path": calibration_path,
    }

    print(f"[export_v1] {root}  ({len(folders)} scene folder(s))")
    n_scene = n_capture = n_goal = n_err = 0
    for folder in folders:
        rel = "/".join(folder.parts[-2:])  # e.g. "office_desk/001" — readable, always defined

        converted = False
        try:  # stage 1: convert messy.json -> scene.json + tabletop_area.json (in-process, no GPU)
            n_obj = convert_messy(folder, ctx, args.scene_type)
            if n_obj >= 0:
                converted = True
                n_scene += 1
                print(f"  [convert] {rel}/scene.json  objects={n_obj}")
            else:
                print(f"  [skip]    {rel}: no {INPUT_ARRANGEMENT}.json")
        except Exception as exc:
            n_err += 1
            print(f"  [ERROR]   {rel} convert: {type(exc).__name__}: {exc}")

        if args.no_render:
            continue

        if converted:  # stage 2: render messy -> full capture contract (subprocess, GPU)
            ok, detail = capture_messy_subprocess(folder, args)
            if ok:
                n_capture += 1
                print(f"  [capture] {rel}/  {detail}")
            else:
                n_err += 1
                print(f"  [ERROR]   {rel} capture: {detail}")

        if (folder / f"{GOAL_ARRANGEMENT}.json").is_file():  # stage 3: render tidy -> reference_goal.png
            ok, detail = render_goal_subprocess(folder, args)
            if ok:
                n_goal += 1
                print(f"  [goal]    {rel}/{GOAL_IMAGE_NAME}  ({detail})")
            else:
                n_err += 1
                print(f"  [ERROR]   {rel} goal: {detail}")
        else:
            print(f"  [skip]    {rel}: no {GOAL_ARRANGEMENT}.json")

    print(f"[export_v1] done — {n_scene} scene.json, {n_capture} capture set(s), {n_goal} {GOAL_IMAGE_NAME}"
          + (f", {n_err} error(s)" if n_err else ""))


if __name__ == "__main__":
    main()
