#!/usr/bin/env python3
"""Convert handcraft v0 scene JSONs into the organize_it pipeline format.

The pipeline (see organize_it_v2/experiments/pybullet_ur5_test_simple/
collect_asset_library_camera_scene_sample.py) consumes a ``scene.json`` of
``schema_version: asset_library_camera_sample_v1``: a ``generation`` block, a
``table`` block, and an ``objects`` list whose loader reads ``asset_id``,
``name``, ``pos`` and ``quat`` (wxyz). We emit a full, directly-loadable scene
that preserves the handcrafted arrangement.

Frames
------
Both tools use the RoboTwin SAPIEN world convention (table top at world z=0.74),
so object world poses transfer unchanged. The only nuance is the *stable frame*:
our v0 ``transform`` is the object pose in its stable frame, while organize_it
sets the *raw mesh* entity pose directly. So the raw-mesh world pose is::

    T_world_from_raw = T_world_from_stable @ stable_rotation

and ``pos``/``quat`` are its translation/rotation. UR-base metadata is filled
from the calibration's ``T_world_from_ur_base`` (UR base = world at (0,-0.5,0.74)
rotated 180 deg about z), so downstream UR-frame code/cameras line up.

Usage
-----
    python tools/export_to_organize_it.py                 # convert all v0 scenes
    python tools/export_to_organize_it.py 0001 makeup_table
    python tools/export_to_organize_it.py --scene-type Kitchen 0002
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
V0_DIR = REPO_ROOT / "data" / "tidy_scene_v0"
ORGANIZE_IT_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
DEFAULT_CATALOG = ORGANIZE_IT_ROOT / "data" / "asset_library" / "catalog.json"
DEFAULT_CALIBRATION = (
    ORGANIZE_IT_ROOT
    / "experiments" / "pybullet_ur5_test_simple" / "camera_adjust_step15_calibration.json"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "organize_it_v1"

# organize_it scene_type vocabulary.
CATEGORIES = ("Kitchen", "Tools", "Desk")
# Per-file scene_type (decided by content; override with --scene-type). Files not
# listed fall back to a majority vote over each scene's asset category tags.
SCENE_TYPE_BY_STEM: dict[str, str] = {
    "0001": "Kitchen",
    "0002": "Desk",
    "0003": "Kitchen",
    "gaming_table": "Desk",
    "makeup_table": "Desk",
    "plate_and_fork": "Kitchen",
    "plate_fork_messy": "Kitchen",
}

SIM_TIMESTEP = 0.004
SCHEMA_VERSION = "asset_library_camera_sample_v1"

# Make the organize_it AssetRegistry importable (lightweight: no SAPIEN import).
SRC_DIR = ORGANIZE_IT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from organize_it.assets.registry import AssetHandle, AssetRegistry  # noqa: E402


# --------------------------------------------------------------------------- #
# pose / quaternion helpers
# --------------------------------------------------------------------------- #
def quat_wxyz_from_matrix(rotation: np.ndarray) -> list[float]:
    """3x3 rotation -> unit quaternion [w, x, y, z]."""
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:  # canonical: non-negative scalar part
        q = -q
    return [float(v) for v in q]


def matrix_to_pose7_wxyz(T: np.ndarray) -> list[float]:
    mat = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return [float(v) for v in mat[:3, 3].tolist()] + quat_wxyz_from_matrix(mat[:3, :3])


def stable_rotation_4x4(stable_rotation: Any) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(stable_rotation, dtype=np.float64).reshape(3, 3)
    return T


def _safe_entity_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "asset"


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def table_from_world(table_top_z: float) -> np.ndarray:
    """Table-centered frame: the handcraft/SAPIEN world shifted so the table top is z=0.

    Pure z-translation (table is already centered at the world XY origin), z stays up
    and the camera-axis convention is unchanged -- which is exactly what the pipeline's
    workspace projection (corners at z=0) and table detection assume. No UR calibration
    needed for the frame; only the table-top height matters.
    """
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = -float(table_top_z)
    return T


def tabletop_bounds(table: dict[str, Any]) -> list[float]:
    """Table footprint in the table-centered frame: [x_min,x_max,y_min,y_max,z_min,z_max]."""
    half_l, half_w = float(table["length"]) / 2.0, float(table["width"]) / 2.0
    return [-half_l, half_l, -half_w, half_w, 0.0, 0.5]


def tabletop_area_payload(bounds: list[float], scene_json_path: Path) -> dict[str, Any]:
    """The tabletop_area.json the pipeline's layout step reads directly (same schema it
    would otherwise derive from scene.json). min/max are [x, y, 0.0] in the table frame."""
    x_min, x_max = sorted((float(bounds[0]), float(bounds[1])))
    y_min, y_max = sorted((float(bounds[2]), float(bounds[3])))
    return {
        "frame": "table_centered_z0",
        "min": [x_min, y_min, 0.0],
        "max": [x_max, y_max, 0.0],
        "size": [x_max - x_min, y_max - y_min],
        "source": str(scene_json_path),
    }


def object_record(
    index: int,
    item: dict[str, Any],
    handle: AssetHandle,
    T_table_from_world: np.ndarray,
) -> dict[str, Any]:
    record = handle.record
    terms = record.source_specific_terms
    model_name = str(terms.get("model_name") or record.label or record.asset_id)
    model_id = str(terms.get("model_id") or terms.get("source_asset_id") or record.asset_id)

    # v0 transform is the object pose in its stable frame; organize_it wants the
    # raw-mesh world pose, which is stable_world @ stable_rotation.
    T_world_from_stable = np.asarray(item["transform"], dtype=np.float64).reshape(4, 4)
    T_world_from_raw = T_world_from_stable @ stable_rotation_4x4(record.geometry.stable_rotation)
    T_table_from_raw = T_table_from_world @ T_world_from_raw

    # Primary pos/quat are in the table-centered frame (table top z=0): the pipeline's
    # workspace projection and table detection assume this frame. The SAPIEN-world pose
    # is kept for reference.
    return {
        "item_id": f"obj:{index}",
        "name": _safe_entity_name(f"{model_name}_{model_id}_{index - 1}"),
        "asset_id": record.asset_id,
        "label": record.label,
        "source": record.source,
        "model_name": model_name,
        "model_id": model_id,
        "model_type": record.model_type,
        "scale": [float(v) for v in record.geometry.scale],
        "stable_rotation": [[float(v) for v in row] for row in record.geometry.stable_rotation],
        "aabb_m": dict(record.geometry.aabb_m),
        "pos": [float(v) for v in T_table_from_raw[:3, 3].tolist()],
        "quat": quat_wxyz_from_matrix(T_table_from_raw[:3, :3]),
        "mass": float(record.physics.mass),
        "pose_source": "handcraft_v0_stable_transform",
        "pose7_table": matrix_to_pose7_wxyz(T_table_from_raw),
        "pose7_sim_world": matrix_to_pose7_wxyz(T_world_from_raw),
    }


def majority_scene_type(items: list[dict[str, Any]], registry: AssetRegistry) -> str:
    counts: dict[str, int] = {}
    for item in items:
        try:
            tags = registry.get(item["asset_id"]).record.semantics.tags
        except KeyError:
            continue
        for tag in tags:
            if tag in CATEGORIES:
                counts[tag] = counts.get(tag, 0) + 1
    return max(counts, key=counts.get) if counts else "Kitchen"


def convert_scene(
    v0_path: Path,
    registry: AssetRegistry,
    calibration: dict[str, Any],
    catalog_path: Path,
    calibration_path: Path,
    scene_type_override: str | None,
) -> dict[str, Any]:
    data = json.loads(v0_path.read_text())
    if data.get("version") != 1:
        raise ValueError(f"{v0_path}: unsupported v0 version {data.get('version')!r}")
    items = data.get("items", [])
    table = data["table"]
    table_texture = data.get("table_texture")  # curated PBR set id under assets/textures/table/, or None
    wall_texture = data.get("wall_texture")

    table_top_z = float(table["height"])  # table-top height in the handcraft/SAPIEN world
    T_table_from_world = table_from_world(table_top_z)

    objects = [
        object_record(i, item, registry.get(item["asset_id"]), T_table_from_world)
        for i, item in enumerate(items, start=1)
    ]

    bounds = tabletop_bounds(table)
    scene_type = (
        scene_type_override
        or SCENE_TYPE_BY_STEM.get(v0_path.stem)
        or majority_scene_type(items, registry)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": 0,
        "scene_type": scene_type,
        "sim_timestep": SIM_TIMESTEP,
        "generation": {
            "script": Path(__file__).name,
            "source_v0_scene": str(v0_path.resolve()),
            "asset_catalog": str(catalog_path.resolve()),
            "camera_calibration": str(calibration_path.resolve()),
            # Key name is the pipeline's (it reads generation.workspace_bounds_ur_base and
            # treats it as the table/world-frame bounds); values are the table-centered frame.
            "workspace_bounds_ur_base": bounds,
            "table_height_m": float(table["height"]),
            "table_thickness_m": float(table["thickness"]),
            "wall_texture": wall_texture,
            "frame": "table_centered_z0",
            "pose_convention": (
                "objects[].pos/quat are raw-mesh poses in the table-centered frame: handcraft/SAPIEN "
                "world shifted so the table top is z=0 (table centered at XY origin, z up). Consistent "
                "with generation.workspace_bounds_ur_base, tabletop_area.json and current_extrinsics "
                "T_world_from_cam. pose7_sim_world keeps the SAPIEN-world pose for reference."
            ),
        },
        "table": {
            "pose": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "length": float(table["length"]),
            "width": float(table["width"]),
            "thickness": float(table["thickness"]),
            "texture": table_texture,  # curated PBR set id; renderer applies it to the tabletop
        },
        "robot": {},
        "cameras": {},
        "objects": objects,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenes", nargs="*", help="v0 scene names/stems or paths (default: all in data/tidy_scene_v0).")
    p.add_argument("--v0-dir", type=Path, default=V0_DIR)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--asset-catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--camera-calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--scene-type", choices=CATEGORIES, default=None,
                   help="Force scene_type for every converted scene (else per-file map / majority vote).")
    p.add_argument("--name", default=None,
                   help="Output subdataset name (the dir under --out-root). Only valid with a single input scene.")
    return p.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if not args.scenes:
        return sorted(args.v0_dir.glob("*.json"))
    paths: list[Path] = []
    for name in args.scenes:
        cand = Path(name)
        if cand.is_file():
            paths.append(cand)
            continue
        stem = cand.stem if cand.suffix == ".json" else name
        match = args.v0_dir / f"{stem}.json"
        if not match.is_file():
            raise FileNotFoundError(f"No v0 scene for {name!r} (looked for {match})")
        paths.append(match)
    return paths


def main() -> None:
    args = parse_args()
    catalog_path = args.asset_catalog.expanduser().resolve()
    calibration_path = args.camera_calibration.expanduser().resolve()
    registry = AssetRegistry.load(catalog_path)
    calibration = json.loads(calibration_path.read_text())

    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(args)
    if args.name is not None and len(inputs) != 1:
        raise SystemExit(f"--name requires exactly one input scene, got {len(inputs)}")

    for v0_path in inputs:
        scene = convert_scene(v0_path, registry, calibration, catalog_path, calibration_path, args.scene_type)
        scene_dir = out_root / (args.name or v0_path.stem)
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_json = scene_dir / "scene.json"
        out_json.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        # Emit tabletop_area.json ourselves -- the data producer owns it; the pipeline
        # reads it directly (and only falls back to deriving it from scene.json if absent).
        ta_payload = tabletop_area_payload(scene["generation"]["workspace_bounds_ur_base"], out_json)
        (scene_dir / "tabletop_area.json").write_text(json.dumps(ta_payload, indent=2), encoding="utf-8")
        print(f"[export] {v0_path.name:24s} -> {out_json}  "
              f"(scene_type={scene['scene_type']}, objects={len(scene['objects'])}, "
              f"+tabletop_area.json {ta_payload['min'][:2]}..{ta_payload['max'][:2]})")


if __name__ == "__main__":
    main()
