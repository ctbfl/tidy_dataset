#!/usr/bin/env python3
"""Render a tidy scene and write the organize_it pipeline's *input capture contract*.

Two reference contracts, used deliberately:

1. Camera capture files (``current.png`` / ``current_depth.pkl`` /
   ``current_intrinsics.yaml`` / ``current_extrinsics.yaml``) follow the **real
   pipeline contract** as used by every real-robot capture under
   ``organize_it_v2/test_data/*``. ``current_extrinsics.yaml`` has exactly
   ``camera_pose_world {p,q}`` / ``T_world_from_cam`` / ``T_cam_from_world``, and
   ``T_world_from_cam`` is the camera pose in the **table-centered frame** (z-up, table
   top at z=0, table centered at XY origin; camera axes x-forward / y-left / z-up). That
   frame matches the scene.json object poses, ``generation.workspace_bounds_ur_base`` and
   ``tabletop_area.json``, so step6's workspace projection (corners at z=0) and step2's
   table detection line up. We do NOT emit the RoboTwin collect script's
   ``sapien_scene_debug`` aliases.

2. The synthetic GT segmentation (``current_pybullet_segmentation.npy`` +
   ``extract_meta.json``) reuses the collect script's own
   ``save_streamline_gt_seg_outputs`` verbatim, so the ``--use-gt-seg`` contract
   matches byte-for-byte.

Scene construction uses our loader (``simulations/scene.py``) so the textured
table, curated PBR materials and handcrafted stable-frame poses render exactly as
authored. Object world poses == handcraft poses; the calibration world coincides
with ours (RoboTwin convention, table top z=0.74), so absolute origin is moot
(pipeline step2 re-detects table z from the point cloud). ``--settle-steps 0``
(default) renders the curated layout untouched.

Usage
-----
    python tools/render_organize_it_scene.py <scene_dir>            # dir holding scene.json
    python tools/render_organize_it_scene.py <scene_dir> --v0 path/to/v0.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "simulations"
ORGANIZE_IT_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
REF_DIR = ORGANIZE_IT_ROOT / "experiments" / "pybullet_ur5_test_simple"
REF_SCRIPT = REF_DIR / "collect_asset_library_camera_scene_sample.py"
DEFAULT_CATALOG = ORGANIZE_IT_ROOT / "data" / "asset_library" / "catalog.json"
DEFAULT_CALIBRATION = REF_DIR / "camera_adjust_step15_calibration.json"


def load_reference_module():
    """Import the collect script (only for capture_camera + the GT-seg writer)."""
    if str(REF_DIR) not in sys.path:  # importlib won't add the script's own dir (needed for `preview`)
        sys.path.insert(0, str(REF_DIR))
    spec = importlib.util.spec_from_file_location("oi_capture", REF_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_wxyz_from_matrix(rotation: np.ndarray) -> list[float]:
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z]); q /= np.linalg.norm(q)
    return [float(v) for v in (q if q[0] >= 0 else -q)]


def build_scene_with_tidy_loader(v0: dict, table_texture_id: str | None):
    """Build ground + textured table + objects via our own SAPIEN loader."""
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))
    import scene as tidy_scene  # simulations/scene.py

    table = v0["table"]
    ts = tidy_scene.create_scene(
        headless=True, use_hdri=False, random_background=False,
        table_length=table["length"], table_width=table["width"],
        table_height=table["height"], table_thickness=table["thickness"],
        table_texture_id=table_texture_id, wall_texture_id=None,
    )
    items = [{"asset_id": it["asset_id"], "transform": np.asarray(it["transform"], dtype=np.float64)}
             for it in v0.get("items", [])]
    tidy_scene.load_items(ts, items)
    return ts


def add_calibrated_camera(mod, ts, camera_info: dict, near: float, far: float, name: str):
    """Camera at the calibrated world pose with the calibrated intrinsics.

    Returns (camera, T_world_from_cam) where T_world_from_cam is the SAPIEN-world
    camera pose (x-forward / y-left / z-up) -- the value the contract's
    ``T_world_from_cam`` field carries.
    """
    T_world_from_cam = np.asarray(
        mod.preview.T_world_from_sapien_camera_from_calibration(camera_info), dtype=np.float64
    ).reshape(4, 4)
    width, height = int(camera_info["width"]), int(camera_info["height"])
    fovy = 2.0 * math.atan(0.5 * height / float(camera_info["fy"]))
    camera = ts.scene.add_camera(name=name, width=width, height=height, fovy=float(fovy), near=near, far=far)
    mod.preview._configure_robotwin_camera_from_calibration(camera, camera_info)
    camera.entity.set_pose(mod.preview._sapien_pose_from_matrix(T_world_from_cam))
    return camera, T_world_from_cam


# --------------------------------------------------------------------------- #
# output writers — test_data capture contract
# --------------------------------------------------------------------------- #
def write_intrinsics(scene_dir: Path, prefix: str, camera_info: dict) -> None:
    height, fy = int(camera_info["height"]), float(camera_info["fy"])
    data = {
        "width": int(camera_info["width"]),
        "height": height,
        "fov_vertical_deg": math.degrees(2.0 * math.atan(0.5 * height / fy)),
        "fx": float(camera_info["fx"]),
        "fy": fy,
        "cx": float(camera_info["cx"]),
        "cy": float(camera_info["cy"]),
    }
    with (scene_dir / f"{prefix}_intrinsics.yaml").open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_extrinsics(scene_dir: Path, prefix: str, T_world_from_cam: np.ndarray) -> None:
    T = np.asarray(T_world_from_cam, dtype=np.float64).reshape(4, 4)
    data = {
        "camera_pose_world": {
            "p": [float(v) for v in T[:3, 3]],
            "q": quat_wxyz_from_matrix(T[:3, :3]),
        },
        "T_world_from_cam": T.astype(float).tolist(),
        "T_cam_from_world": np.linalg.inv(T).astype(float).tolist(),
    }
    with (scene_dir / f"{prefix}_extrinsics.yaml").open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_depth(scene_dir: Path, prefix: str, depth_m: np.ndarray) -> None:
    depth_mm = (np.asarray(depth_m, dtype=np.float32) * 1000.0).astype(np.float32)
    if depth_mm.ndim == 2:  # contract: HxWx1
        depth_mm = depth_mm[..., None]
    with (scene_dir / f"{prefix}_depth.pkl").open("wb") as f:
        pickle.dump(depth_mm, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_dir", type=Path, help="Directory holding scene.json (outputs written here).")
    p.add_argument("--v0", type=Path, default=None, help="Source v0 scene json (default: generation.source_v0_scene).")
    p.add_argument("--prefix", default="current")
    p.add_argument("--settle-steps", type=int, default=0,
                   help="Physics steps before capture. 0 (default) preserves the handcrafted arrangement.")
    p.add_argument("--asset-catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--camera-calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--camera-name", default="wrist_camera")
    p.add_argument("--near-m", type=float, default=0.02)
    p.add_argument("--far-m", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    scene_json = scene_dir / "scene.json"
    if not scene_json.is_file():
        raise SystemExit(f"No scene.json in {scene_dir}")
    scene_data = json.loads(scene_json.read_text())

    v0_path = (args.v0 or Path(scene_data.get("generation", {}).get("source_v0_scene", ""))).expanduser()
    if not v0_path.is_file():
        raise SystemExit(f"Source v0 scene not found: {v0_path!r} (pass --v0)")
    v0 = json.loads(v0_path.read_text())

    mod = load_reference_module()
    calibration = mod.preview._load_robotwin_camera_calibration(args.camera_calibration.expanduser().resolve())
    camera_info = calibration["camera"]
    asset_registry = mod.AssetRegistry.load(args.asset_catalog.expanduser().resolve())

    ts = build_scene_with_tidy_loader(v0, v0.get("table_texture"))
    camera, T_world_from_sapien_cam = add_calibrated_camera(mod, ts, camera_info, args.near_m, args.far_m, args.camera_name)
    # Contract frame is table-centered (table top z=0): shift the SAPIEN-world camera pose
    # down by the table-top height so it matches the scene.json object poses and
    # generation.workspace_bounds_ur_base / tabletop_area.json. Pure z-translation.
    table_top_z = float(v0["table"]["height"])
    T_world_from_cam = T_world_from_sapien_cam.copy()
    T_world_from_cam[2, 3] -= table_top_z

    for _ in range(max(0, int(args.settle_steps))):
        ts.scene.step()
    ts.scene.update_render()
    capture = mod.capture_camera(camera)  # {"rgb","segmentation","depth_m"}

    loaded_objects = [
        {"name": scene_obj.get("name"), "entity": sobj.entity,
         "per_scene_id": int(getattr(sobj.entity, "per_scene_id", -1)), "scene_object": scene_obj}
        for sobj, scene_obj in zip(ts.objects.values(), scene_data["objects"])
    ]

    # camera capture (test_data contract) + GT segmentation (collect-script contract)
    Image.fromarray(capture["rgb"]).save(scene_dir / f"{args.prefix}.png")
    write_depth(scene_dir, args.prefix, capture["depth_m"])
    write_intrinsics(scene_dir, args.prefix, camera_info)
    write_extrinsics(scene_dir, args.prefix, T_world_from_cam)
    mod.save_streamline_gt_seg_outputs(
        scene_dir=scene_dir, prefix=args.prefix, segmentation=capture["segmentation"],
        loaded_objects=loaded_objects, asset_registry=asset_registry,
    )

    seg_ids = np.asarray(capture["segmentation"])[..., 1]
    print(f"[render] {scene_dir}  table_texture={v0.get('table_texture')}  settle_steps={args.settle_steps}")
    print(f"[render] T_world_from_cam p={[round(float(v),4) for v in T_world_from_cam[:3,3]]}")
    for o in loaded_objects:
        print(f"   obj {o['per_scene_id']:>3}  {o['scene_object']['asset_id']:24s} "
              f"visible_px={int(np.count_nonzero(seg_ids == o['per_scene_id']))}")
    print("[render] files:", ", ".join(sorted(p.name for p in scene_dir.iterdir())))


if __name__ == "__main__":
    main()
