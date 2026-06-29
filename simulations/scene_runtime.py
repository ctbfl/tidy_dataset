from __future__ import annotations

import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import sapien.core as sapien
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "simulations"
ORGANIZE_IT_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
REF_DIR = ORGANIZE_IT_ROOT / "experiments" / "pybullet_ur5_test_simple"
REF_SCRIPT = REF_DIR / "collect_asset_library_camera_scene_sample.py"
DEFAULT_CATALOG = ORGANIZE_IT_ROOT / "data" / "asset_library" / "catalog.json"
DEFAULT_CALIBRATION = REF_DIR / "camera_adjust_step15_calibration.json"
DEFAULT_SETTLE_STEPS = 100

if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from objects import asset_json_backup_dir  # noqa: E402
from scene import LIBRARY, create_scene, load_items  # noqa: E402


def load_reference_module():
    if str(REF_DIR) not in sys.path:
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
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return [float(v) for v in (q if q[0] >= 0 else -q)]


def load_scene_dict_to_sapien(scene_dict: dict, scene_json_path: Path | None = None):
    if scene_json_path is not None:
        LIBRARY.load_asset_json_backup(asset_json_backup_dir(scene_json_path))
    table = scene_dict["table"]
    ts = create_scene(
        headless=True,
        use_hdri=False,
        random_background=False,
        table_length=table["length"],
        table_width=table["width"],
        table_height=table["height"],
        table_thickness=table["thickness"],
        table_texture_id=scene_dict.get("table_texture"),
        wall_texture_id=scene_dict.get("wall_texture"),
    )
    load_items(ts, scene_dict.get("items", []))
    return ts


def _dynamic(entity):
    return entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)


def settle_scene(ts, steps: int = DEFAULT_SETTLE_STEPS) -> None:
    bodies = []
    for obj in ts.objects.values():
        body = _dynamic(obj.entity)
        if body is None:
            continue
        bodies.append(body)
        body.set_kinematic(False)
        body.set_locked_motion_axes([False] * 6)
        body.set_linear_velocity([0, 0, 0])
        body.set_angular_velocity([0, 0, 0])
    for _ in range(max(0, int(steps))):
        ts.scene.step()
    for body in bodies:
        body.set_linear_velocity([0, 0, 0])
        body.set_angular_velocity([0, 0, 0])
        body.set_kinematic(True)
    ts.scene.update_render()


def scene_dict_from_loaded_objects(original_dict: dict, ts) -> dict:
    out = dict(original_dict)
    items = []
    for item, obj in zip(original_dict.get("items", []), ts.objects.values()):
        updated = dict(item)
        updated["transform"] = obj.get_pose().to_transformation_matrix().tolist()
        items.append(updated)
    out["items"] = items
    return out


def add_calibrated_camera(mod, ts, camera_info: dict, near: float = 0.02, far: float = 10.0, name: str = "wrist_camera"):
    T_world_from_cam = np.asarray(
        mod.preview.T_world_from_sapien_camera_from_calibration(camera_info), dtype=np.float64
    ).reshape(4, 4)
    width, height = int(camera_info["width"]), int(camera_info["height"])
    fovy = 2.0 * math.atan(0.5 * height / float(camera_info["fy"]))
    camera = ts.scene.add_camera(name=name, width=width, height=height, fovy=float(fovy), near=near, far=far)
    mod.preview._configure_robotwin_camera_from_calibration(camera, camera_info)
    camera.entity.set_pose(mod.preview._sapien_pose_from_matrix(T_world_from_cam))
    return camera, T_world_from_cam


def render_reference_goal(ts, out_path: Path, calibration_path: Path = DEFAULT_CALIBRATION) -> None:
    mod = load_reference_module()
    calibration = mod.preview._load_robotwin_camera_calibration(Path(calibration_path).expanduser().resolve())
    camera, _ = add_calibrated_camera(mod, ts, calibration["camera"])
    ts.scene.update_render()
    Image.fromarray(mod.capture_camera(camera)["rgb"]).save(out_path)


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
    if depth_mm.ndim == 2:
        depth_mm = depth_mm[..., None]
    with (scene_dir / f"{prefix}_depth.pkl").open("wb") as f:
        pickle.dump(depth_mm, f, protocol=pickle.HIGHEST_PROTOCOL)


def capture_current(ts, scene_dir: Path, scene_data: dict, asset_registry, calibration_path: Path = DEFAULT_CALIBRATION, prefix: str = "current") -> None:
    mod = load_reference_module()
    calibration = mod.preview._load_robotwin_camera_calibration(Path(calibration_path).expanduser().resolve())
    camera_info = calibration["camera"]
    camera, T_world_from_sapien_cam = add_calibrated_camera(mod, ts, camera_info)
    table_top_z = float(scene_data["generation"]["table_height_m"])
    T_world_from_cam = T_world_from_sapien_cam.copy()
    T_world_from_cam[2, 3] -= table_top_z
    ts.scene.update_render()
    capture = mod.capture_camera(camera)
    loaded_objects = [
        {
            "name": scene_obj.get("name"),
            "entity": sobj.entity,
            "per_scene_id": int(getattr(sobj.entity, "per_scene_id", -1)),
            "scene_object": scene_obj,
        }
        for sobj, scene_obj in zip(ts.objects.values(), scene_data["objects"])
    ]
    Image.fromarray(capture["rgb"]).save(scene_dir / f"{prefix}.png")
    write_depth(scene_dir, prefix, capture["depth_m"])
    write_intrinsics(scene_dir, prefix, camera_info)
    write_extrinsics(scene_dir, prefix, T_world_from_cam)
    mod.save_streamline_gt_seg_outputs(
        scene_dir=scene_dir,
        prefix=prefix,
        segmentation=capture["segmentation"],
        loaded_objects=loaded_objects,
        asset_registry=asset_registry,
    )
