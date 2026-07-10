#!/usr/bin/env python3
"""Run our v2 transforms through SG-Bot's rollout executor for one scene.

Input is one tidy_dataset scene directory containing:
  - ta_real_scene.pkl
  - sgbot_input/random_scene.json

Output:
  - <scene_dir>/our_result.png
  - <scene_dir>/our_output/teleport.mp4
  - <scene_dir>/our_output/teleport.log

With --no-obstacle, the same files are written as:
  - <scene_dir>/our_result_no_obstacle.png
  - <scene_dir>/our_output_no_obstacle/

After rollout, VLM comparison runs by default and writes debug_comparison/*.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import pkgutil
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ORGANIZE_IT_V2_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
MAIN_SGBOT_COMPARE = ORGANIZE_IT_V2_ROOT / "experiments" / "main_sgbot_compare"
CUSTOM_SGBOT_DIR = MAIN_SGBOT_COMPARE / "custom_sgbot"
SGBOT_DIR = ORGANIZE_IT_V2_ROOT / "SG-Bot"

sys.path.insert(0, str(CUSTOM_SGBOT_DIR))
sys.path.insert(0, str(SGBOT_DIR))
sys.path.insert(0, str(ORGANIZE_IT_V2_ROOT / "src"))

import run_custom_sgbot as rcs  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
CUTLERY = {"fork", "knife", "tablespoon", "teaspoon"}
OCC_THRESHOLD_M = 0.01
EMPTY_GRID_STEP = 0.04
VIDEO_FPS = 30
RECORD_EVERY_STEPS = 16
VIDEO_ZNEAR = 0.01
VIDEO_ZFAR = 10.0
SETTLE_MAX_STEPS = 5000
SETTLE_MIN_STEPS = 20
SETTLE_STABLE_STEPS = 20
SETTLE_LIN_EPS = 0.01
SETTLE_ANG_EPS = 0.2
COLLISION_LIFT_STEP = 0.01
COLLISION_LIFT_MAX_STEPS = 40
COLLISION_PENETRATION_TOL = 1e-4


def install_moviepy_editor_shim() -> None:
    try:
        __import__("moviepy.editor")
        return
    except ModuleNotFoundError:
        pass

    import moviepy

    editor = types.ModuleType("moviepy.editor")
    editor.ImageSequenceClip = moviepy.ImageSequenceClip
    sys.modules["moviepy.editor"] = editor


def pose_from_pos_quat(pos: Any, quat_xyzw: Any) -> np.ndarray:
    import pybullet as pb

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.reshape(pb.getMatrixFromQuaternion(quat_xyzw), (3, 3))
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def transformed_pose(rel_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rel_pose[:3, :3] @ current_pose[:3, :3]
    out[:3, 3] = rel_pose[:3, :3] @ current_pose[:3, 3] + rel_pose[:3, 3]
    return out


def table_frame_transform_to_sim(rel_pose: np.ndarray, table_height: float) -> np.ndarray:
    table_from_sim = np.eye(4, dtype=np.float64)
    table_from_sim[2, 3] = -table_height
    sim_from_table = np.eye(4, dtype=np.float64)
    sim_from_table[2, 3] = table_height
    return sim_from_table @ rel_pose @ table_from_sim


def object_local_cloud(obj: dict, n_points: int = 1024) -> np.ndarray:
    import os
    import trimesh
    import xml.etree.ElementTree as ET

    asset = obj.get("instance", obj["name"])
    base = os.path.join(rcs.MODELS_ROOT, obj["class"], asset)

    scale = np.ones(3)
    offset = np.zeros(3)
    vis = ET.parse(base + ".urdf").getroot().find(".//visual")
    if vis is not None:
        mesh = vis.find("geometry/mesh")
        if mesh is not None and mesh.get("scale"):
            scale = np.array([float(x) for x in mesh.get("scale").split()])
        origin = vis.find("origin")
        if origin is not None and origin.get("xyz"):
            offset = np.array([float(x) for x in origin.get("xyz").split()])

    mesh = trimesh.load(base + ".obj", force="mesh")
    points = np.asarray(mesh.vertices, dtype=np.float64) * scale + offset
    if len(points) > n_points:
        points = points[np.linspace(0, len(points) - 1, n_points).astype(int)]
    return points


def world_xy_cloud(local_cloud: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return ((pose[:3, :3] @ local_cloud.T).T + pose[:3, 3])[:, :2]


def world_cloud(local_cloud: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ local_cloud.T).T + pose[:3, 3]


def chamfer_xy_min(a: np.ndarray, b: np.ndarray) -> float:
    d2 = np.sum(a ** 2, axis=1, keepdims=True) - 2 * a @ b.T + np.sum(b ** 2, axis=1)
    return float(np.sqrt(np.maximum(d2, 0.0).min()))


def find_empty_spot(
    local_clouds: dict[str, np.ndarray],
    current_pose: dict[str, np.ndarray],
    own_cls: str,
    own_pose: np.ndarray,
    table_aabb_xy: tuple[float, float, float, float],
) -> np.ndarray | None:
    xmin, ymin, xmax, ymax = table_aabb_xy
    margin = 0.06
    xmin += margin
    ymin += margin
    xmax -= margin
    ymax -= margin

    own_rotated = (own_pose[:3, :3] @ local_clouds[own_cls].T).T
    others = [cls for cls in current_pose if cls != own_cls]
    xs = np.arange(xmin, xmax + 1e-6, EMPTY_GRID_STEP)
    ys = np.arange(ymin, ymax + 1e-6, EMPTY_GRID_STEP)
    candidates = sorted(
        [(min(abs(x), abs(y)), x, y) for x in xs for y in ys],
        key=lambda item: -item[0],
    )
    for _, x, y in candidates:
        cand_xy = own_rotated[:, :2] + np.array([x, y])
        clear = True
        for other in others:
            other_xy = world_xy_cloud(local_clouds[other], current_pose[other])
            if chamfer_xy_min(cand_xy, other_xy) < OCC_THRESHOLD_M:
                clear = False
                break
        if clear:
            new_pose = own_pose.copy()
            new_pose[0, 3] = x
            new_pose[1, 3] = y
            return new_pose
    return None


def lift_rel_pose_to_table(
    rel_pose: np.ndarray,
    current_pose: np.ndarray,
    local_cloud: np.ndarray,
    table_height: float,
) -> tuple[np.ndarray, float]:
    target_pose = transformed_pose(rel_pose, current_pose)
    min_z = float(world_cloud(local_cloud, target_pose)[:, 2].min())
    lift = max(0.0, table_height - min_z)
    if lift == 0.0:
        return rel_pose, 0.0
    lifted = rel_pose.copy()
    lifted[2, 3] += lift
    return lifted, lift


def lift_pose_to_table(
    pose: np.ndarray,
    local_cloud: np.ndarray,
    table_height: float,
) -> tuple[np.ndarray, float]:
    min_z = float(world_cloud(local_cloud, pose)[:, 2].min())
    lift = max(0.0, table_height - min_z)
    if lift == 0.0:
        return pose, 0.0
    lifted = pose.copy()
    lifted[2, 3] += lift
    return lifted, lift


def table_pose_to_sim(pose: np.ndarray, table_height: float) -> np.ndarray:
    out = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
    out[2, 3] += float(table_height)
    return out


def scene_objects_by_class(scene: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for obj in scene["objects"]:
        cls = obj["class"]
        if cls == "support_table":
            continue
        if cls in out:
            raise ValueError(f"duplicate SG-Bot class in scene: {cls}")
        out[cls] = obj
    return out


def load_context(scene_dir: Path, out_dir_name: str) -> dict:
    scene_dir = scene_dir.expanduser().resolve()
    pkl_path = scene_dir / "ta_real_scene.pkl"
    random_scene_path = scene_dir / "sgbot_input" / "random_scene.json"
    intrinsics_path = scene_dir / "current_intrinsics.yaml"
    extrinsics_path = scene_dir / "current_extrinsics.yaml"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"missing {pkl_path}")
    if not random_scene_path.is_file():
        raise FileNotFoundError(f"missing {random_scene_path}")
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"missing {intrinsics_path}")
    if not extrinsics_path.is_file():
        raise FileNotFoundError(f"missing {extrinsics_path}")

    with pkl_path.open("rb") as f:
        ta_scene = pickle.load(f)
    scene = json.loads(random_scene_path.read_text())
    intrinsics = yaml.safe_load(intrinsics_path.read_text())
    extrinsics = yaml.safe_load(extrinsics_path.read_text())
    rcs._uniquify_scene_object_names(scene)

    out_dir = scene_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = out_dir / "steps"
    if steps_dir.exists():
        shutil.rmtree(steps_dir)
    steps_dir.mkdir()
    for video_path in out_dir.glob("step_*.mp4"):
        video_path.unlink()
    for frame_path in out_dir.glob("teleport_frame*.png"):
        frame_path.unlink()
    for stale_name in (
        "teleport.mp4",
        "teleport.log",
        "rollout.log",
        "final_poses.json",
        "step_poses.json",
    ):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    synthetic_view = out_dir / "synthetic_view-1.json"
    synthetic_view.write_text(json.dumps(scene, ensure_ascii=False, indent=2))

    sim_by_class = scene_objects_by_class(scene)
    if "table" in scene and "aabb" in scene["table"]:
        aabb = scene["table"]["aabb"]
        table_aabb_xy = (aabb[0][0], aabb[0][1], aabb[1][0], aabb[1][1])
    else:
        tabletop_area = json.loads((scene_dir / "tabletop_area.json").read_text())
        table_aabb_xy = (
            tabletop_area["min"][0],
            tabletop_area["min"][1],
            tabletop_area["max"][0],
            tabletop_area["max"][1],
        )
    ta_by_class = ta_scene.objects
    missing_ta = sorted(set(sim_by_class) - set(ta_by_class))
    extra_ta = sorted(set(ta_by_class) - set(sim_by_class))
    if missing_ta or extra_ta:
        raise ValueError(f"class mismatch: missing_ta={missing_ta} extra_ta={extra_ta}")

    for cls, obj in ta_by_class.items():
        transform = getattr(obj, "v2_real_pt_transform", None)
        if np.asarray(transform).shape != (4, 4):
            raise ValueError(f"{cls}: invalid v2_real_pt_transform")

    return {
        "scene_dir": scene_dir,
        "out_dir": out_dir,
        "scene": scene,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "ta_scene": ta_scene,
        "synthetic_view": synthetic_view,
        "steps_dir": steps_dir,
        "sim_by_class": sim_by_class,
        "table_aabb_xy": table_aabb_xy,
    }


def scene_has_obstacle(scene_dir: Path) -> bool:
    scene_path = scene_dir.expanduser().resolve() / "sgbot_input" / "random_scene.json"
    if not scene_path.is_file():
        raise FileNotFoundError(f"missing {scene_path}")
    scene = json.loads(scene_path.read_text())
    return any(obj["class"] == "obstacle" for obj in scene["objects"])


def copy_with_obstacle_outputs_to_no_obstacle(scene_dir: Path) -> bool:
    scene_dir = scene_dir.expanduser().resolve()
    source_result = scene_dir / "our_result.png"
    source_out = scene_dir / "our_output"
    source_video = source_out / "teleport.mp4"
    source_log = source_out / "teleport.log"
    source_steps = source_out / "steps"
    if not (
        source_result.is_file()
        and source_video.is_file()
        and source_log.is_file()
        and (source_steps / "final.json").is_file()
    ):
        return False

    target_result = scene_dir / "our_result_no_obstacle.png"
    target_out = scene_dir / "our_output_no_obstacle"
    target_out.mkdir(parents=True, exist_ok=True)
    for path in target_out.glob("step_*.mp4"):
        path.unlink()
    for path in target_out.glob("teleport_frame*.png"):
        path.unlink()
    for name in (
        "teleport.mp4",
        "teleport.log",
        "rollout.log",
        "final_poses.json",
        "step_poses.json",
        "synthetic_view-1.json",
    ):
        path = target_out / name
        if path.exists():
            path.unlink()
    target_steps = target_out / "steps"
    if target_steps.exists():
        shutil.rmtree(target_steps)

    shutil.copy2(source_result, target_result)
    shutil.copy2(source_video, target_out / "teleport.mp4")
    shutil.copytree(source_steps, target_steps)
    copied_log = (
        "[copy] scene has no obstacle; copied existing with-obstacle output\n"
        f"[copy-source] {source_out}\n"
        + source_log.read_text()
    )
    (target_out / "teleport.log").write_text(copied_log)
    return True


@contextlib.contextmanager
def force_pybullet_direct(pb: Any):
    original_connect = pb.connect

    def connect_direct(mode: int, *args: Any, **kwargs: Any) -> int:
        return original_connect(pb.DIRECT, *args, **kwargs)

    pb.connect = connect_direct
    try:
        yield
    finally:
        pb.connect = original_connect


def load_egl_renderer(pb: Any) -> int:
    egl = pkgutil.get_loader("eglRenderer")
    if egl is None:
        raise RuntimeError("missing PyBullet eglRenderer plugin")
    plugin_id = pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    if plugin_id < 0:
        raise RuntimeError("failed to load PyBullet eglRenderer plugin")
    return int(plugin_id)


def asset_base_path(sim_obj: dict) -> Path:
    asset = sim_obj.get("instance", sim_obj["name"])
    return Path(rcs.MODELS_ROOT) / sim_obj["class"] / asset


def read_asset_material(sim_obj: dict) -> dict[str, Any]:
    obj_path = asset_base_path(sim_obj).with_suffix(".obj")
    if not obj_path.is_file():
        raise FileNotFoundError(f"missing OBJ: {obj_path}")

    mtl_path = None
    for line in obj_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0] == "mtllib":
            mtl_path = obj_path.parent / parts[1]
            break
    if mtl_path is None or not mtl_path.is_file():
        raise FileNotFoundError(f"missing mtllib for {obj_path}")

    material: dict[str, Any] = {"texture_path": None, "rgba": None, "specular": None}
    for line in mtl_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "Kd" and len(parts) >= 4:
            material["rgba"] = [float(v) for v in parts[1:4]] + [1.0]
        elif parts[0] == "Ks" and len(parts) >= 4:
            material["specular"] = [float(v) for v in parts[1:4]]
        elif parts[0] == "map_Kd" and len(parts) >= 2:
            texture_path = mtl_path.parent / parts[-1]
            if not texture_path.is_file():
                raise FileNotFoundError(f"missing texture: {texture_path}")
            material["texture_path"] = texture_path
    if material["rgba"] is None:
        raise ValueError(f"missing Kd in {mtl_path}")
    return material


def apply_visual_textures(pb: Any, env: Any, sim_by_class: dict[str, dict]) -> None:
    for sim_obj in sim_by_class.values():
        bid = env.pybullet_id_dict[sim_obj["name"]]
        material = read_asset_material(sim_obj)
        texture_path = material["texture_path"]
        texture_id = pb.loadTexture(str(texture_path)) if texture_path is not None else -1
        for visual in pb.getVisualShapeData(bid):
            link_index = int(visual[1])
            visual_args = {"rgbaColor": material["rgba"]}
            if material["specular"] is not None:
                visual_args["specularColor"] = material["specular"]
            if texture_id >= 0:
                visual_args["textureUniqueId"] = texture_id
            pb.changeVisualShape(bid, link_index, **visual_args)


def dataset_camera(ctx: dict, table_height: float) -> dict[str, Any]:
    intr = ctx["intrinsics"]
    T_world_from_cam = np.asarray(ctx["extrinsics"]["T_world_from_cam"], dtype=np.float64)
    if T_world_from_cam.shape != (4, 4):
        raise ValueError("current_extrinsics.yaml: T_world_from_cam must be 4x4")

    T_sim_from_cam = T_world_from_cam.copy()
    T_sim_from_cam[2, 3] += float(table_height)
    eye = T_sim_from_cam[:3, 3]
    forward = T_sim_from_cam[:3, 0]
    return {
        "width": int(intr["width"]),
        "height": int(intr["height"]),
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "eye": eye,
        "at": eye + forward,
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    }


def render_dataset_camera(pb: Any, camera: dict[str, Any], renderer: int | None = None) -> np.ndarray:
    width = camera["width"]
    height = camera["height"]
    view = pb.computeViewMatrix(
        camera["eye"].tolist(),
        camera["at"].tolist(),
        camera["up"].tolist(),
    )
    fovh = 180.0 * np.arctan((height / 2.0) / camera["fy"]) * 2.0 / np.pi
    proj = pb.computeProjectionMatrixFOV(fovh, width / height, VIDEO_ZNEAR, VIDEO_ZFAR)
    _, _, rgba, _, _ = pb.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=proj,
        shadow=1,
        renderer=pb.ER_TINY_RENDERER if renderer is None else renderer,
    )
    return np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]


def snapshot_body_poses(pb: Any, env: Any) -> dict[str, tuple[list[float], list[float]]]:
    poses = {}
    for name, bid in env.pybullet_id_dict.items():
        pos, quat = pb.getBasePositionAndOrientation(bid)
        poses[name] = ([float(x) for x in pos], [float(x) for x in quat])
    return poses


def pose_matrix_json(pose: np.ndarray) -> dict[str, list[float]]:
    import pybullet_utils.transformations as trans

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = pose[:3, :3]
    quat = trans.quaternion_from_matrix(mat)
    return {
        "pos": [float(x) for x in pose[:3, 3]],
        "quat_xyzw": [float(x) for x in quat],
    }


def snapshot_step_state(
    pb: Any,
    env: Any,
    ctx: dict,
    *,
    phase: str,
    step_index: int,
    action: dict[str, Any] | None,
    completed: bool = False,
) -> dict[str, Any]:
    objects = {}
    for cls, sim_obj in sorted(ctx["sim_by_class"].items()):
        bid = env.pybullet_id_dict[sim_obj["name"]]
        pos, quat = pb.getBasePositionAndOrientation(bid)
        aabb_min, aabb_max = pb.getAABB(bid)
        objects[cls] = {
            "class": cls,
            "name": sim_obj["name"],
            "instance": sim_obj.get("instance", sim_obj["name"]),
            "pos": [float(x) for x in pos],
            "quat_xyzw": [float(x) for x in quat],
            "aabb_min": [float(x) for x in aabb_min],
            "aabb_max": [float(x) for x in aabb_max],
        }

    table_aabb_min, table_aabb_max = pb.getAABB(env.pybullet_id_dict["support_table"])
    state = {
        "schema_version": 1,
        "scene_id": ctx["scene_dir"].name,
        "phase": phase,
        "step_index": int(step_index),
        "action": action,
        "frame": {
            "name": "pybullet_sim_world",
            "table_height": float(env.table_height),
            "table_aabb_min": [float(x) for x in table_aabb_min],
            "table_aabb_max": [float(x) for x in table_aabb_max],
        },
        "objects": objects,
    }
    if completed:
        state["completed"] = True
    return state


def write_step_state(
    pb: Any,
    env: Any,
    ctx: dict,
    path: Path,
    *,
    phase: str,
    step_index: int,
    action: dict[str, Any] | None,
    completed: bool = False,
) -> dict[str, Any]:
    state = snapshot_step_state(
        pb,
        env,
        ctx,
        phase=phase,
        step_index=step_index,
        action=action,
        completed=completed,
    )
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return state


def render_clean_tiny_final_in_subprocess(
    ctx: dict,
    body_poses: dict[str, tuple[list[float], list[float]]],
    result_path: Path,
) -> None:
    pose_path = ctx["out_dir"] / "final_render_poses.json"
    pose_path.write_text(json.dumps(body_poses, indent=2))
    code = r"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

repo_root = Path(sys.argv[1])
synthetic_view = Path(sys.argv[2])
pose_path = Path(sys.argv[3])
scene_dir = Path(sys.argv[4])
result_path = Path(sys.argv[5])

sys.path.insert(0, str(repo_root))
import sgbot_comparison.run_sgbot_compare as runner  # noqa: E402

os.chdir(str(runner.SGBOT_DIR))
runner.install_moviepy_editor_shim()
from initial import RearrangeEnv  # noqa: E402
import pybullet as pb  # noqa: E402

with runner.force_pybullet_direct(pb):
    env = RearrangeEnv([(str(synthetic_view), str(synthetic_view))], runner.rcs.MODELS_ROOT, None, 5)
env.reset_rearrange(0)

poses = json.loads(pose_path.read_text())
for name, pose in poses.items():
    if name in ("robot", "ghost", "gripper", "support_table"):
        continue
    bid = env.pybullet_id_dict.get(name)
    if bid is None:
        continue
    pos, quat = pose
    pb.resetBasePositionAndOrientation(bid, pos, quat)
    pb.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

camera = runner.dataset_camera(
    {
        "intrinsics": yaml.safe_load((scene_dir / "current_intrinsics.yaml").read_text()),
        "extrinsics": yaml.safe_load((scene_dir / "current_extrinsics.yaml").read_text()),
    },
    float(env.table_height),
)
Image.fromarray(
    runner.render_dataset_camera(pb, camera, renderer=pb.ER_TINY_RENDERER)
).save(result_path)
pb.disconnect()
"""
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(REPO_ROOT),
                str(ctx["synthetic_view"]),
                str(pose_path),
                str(ctx["scene_dir"]),
                str(result_path),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
    finally:
        if pose_path.exists():
            pose_path.unlink()


def run_offline_verifier(ctx: dict, log: Any) -> None:
    verifier = Path(__file__).with_name("verify_teleport_steps.py")
    result = subprocess.run(
        [sys.executable, str(verifier), str(ctx["steps_dir"])],
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        log("[offline-verify] stdout:")
        for line in result.stdout.strip().splitlines():
            log(f"[offline-verify] {line}")
    if result.stderr.strip():
        log("[offline-verify] stderr:")
        for line in result.stderr.strip().splitlines():
            log(f"[offline-verify] {line}")
    if result.returncode != 0:
        raise RuntimeError(f"offline verifier failed with exit code {result.returncode}")


class VideoRecorder:
    def __init__(self, path: Path):
        import imageio.v2 as imageio

        self.path = path
        self.frame_count = 0
        self.writer = imageio.get_writer(
            str(path),
            fps=VIDEO_FPS,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )

    def capture(self, frame: np.ndarray) -> None:
        self.writer.append_data(frame)
        self.frame_count += 1

    def close(self) -> None:
        self.writer.close()


def capture_video_frame(pb: Any, recorder: VideoRecorder, camera: dict[str, Any]) -> None:
    recorder.capture(render_dataset_camera(pb, camera, renderer=pb.ER_BULLET_HARDWARE_OPENGL))


def settle_and_record(pb: Any, recorder: VideoRecorder, bid: int, camera: dict[str, Any]) -> int:
    stable = 0
    for step in range(SETTLE_MAX_STEPS):
        pb.stepSimulation()
        if (step + 1) % RECORD_EVERY_STEPS == 0:
            capture_video_frame(pb, recorder, camera)
        linear, angular = pb.getBaseVelocity(bid)
        lin_norm = float(np.linalg.norm(linear))
        ang_norm = float(np.linalg.norm(angular))
        if step + 1 >= SETTLE_MIN_STEPS and lin_norm < SETTLE_LIN_EPS and ang_norm < SETTLE_ANG_EPS:
            stable += 1
            if stable >= SETTLE_STABLE_STEPS:
                capture_video_frame(pb, recorder, camera)
                return step + 1
        else:
            stable = 0
    capture_video_frame(pb, recorder, camera)
    return SETTLE_MAX_STEPS


def pose_has_penetration(pb: Any, bid: int, pos: np.ndarray, ori: Any) -> bool:
    pb.resetBasePositionAndOrientation(bid, pos.tolist(), ori)
    pb.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    pb.performCollisionDetection()
    for contact in pb.getContactPoints(bodyA=bid):
        if float(contact[8]) < -COLLISION_PENETRATION_TOL:
            return True
    return False


def release_position_for_pose(pb: Any, bid: int, pose: np.ndarray, ori: Any) -> np.ndarray:
    base_pos = np.asarray(pose[:3, 3], dtype=np.float64)
    if pose_has_penetration(pb, bid, base_pos, ori):
        for step in range(1, COLLISION_LIFT_MAX_STEPS + 1):
            candidate = base_pos.copy()
            candidate[2] += step * COLLISION_LIFT_STEP
            if not pose_has_penetration(pb, bid, candidate, ori):
                return candidate
        raise RuntimeError(f"no non-colliding release pose found above {base_pos.tolist()}")

    release_pos = base_pos.copy()
    for step in range(1, COLLISION_LIFT_MAX_STEPS + 1):
        candidate = base_pos.copy()
        candidate[2] -= step * COLLISION_LIFT_STEP
        if pose_has_penetration(pb, bid, candidate, ori):
            return release_pos
        release_pos = candidate
    raise RuntimeError(f"no collision found below {base_pos.tolist()}")


def teleport_rel_pose_and_settle(
    pb: Any,
    env: Any,
    bid: int,
    rel_pose: np.ndarray | None,
    obj_class: str,
    recorder: VideoRecorder,
    camera: dict[str, Any],
) -> int:
    import pybullet_utils.transformations as trans
    import sgbot_pybullet as sg

    current_pos, current_ori = pb.getBasePositionAndOrientation(bid)
    if rel_pose is None:
        new_ori = current_ori
        new_pos = np.array([0.5, -0.4, 0.05], dtype=np.float64)
    else:
        current_rot = np.reshape(pb.getMatrixFromQuaternion(current_ori), (3, 3))
        new_rot = np.eye(4, dtype=np.float64)
        new_rot[:3, :3] = rel_pose[:3, :3] @ current_rot
        new_ori = trans.quaternion_from_matrix(new_rot)
        new_pos = rel_pose[:3, :3] @ np.asarray(current_pos, dtype=np.float64) + rel_pose[:3, 3]
        new_pos = new_pos + np.array([0.0, 0.0, 0.05], dtype=np.float64)

    sg.change_dynamics_obj(bid, obj_class)
    pb.resetBasePositionAndOrientation(bid, new_pos.tolist(), new_ori)
    pb.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    capture_video_frame(pb, recorder, camera)
    return settle_and_record(pb, recorder, bid, camera)


def teleport_pose_and_settle(
    pb: Any,
    env: Any,
    bid: int,
    pose: np.ndarray,
    obj_class: str,
    recorder: VideoRecorder,
    camera: dict[str, Any],
) -> int:
    import pybullet_utils.transformations as trans
    import sgbot_pybullet as sg

    rot_pose = np.eye(4, dtype=np.float64)
    rot_pose[:3, :3] = pose[:3, :3]
    sg.change_dynamics_obj(bid, obj_class)
    ori = trans.quaternion_from_matrix(rot_pose)
    release_pos = release_position_for_pose(pb, bid, pose, ori)
    pb.resetBasePositionAndOrientation(
        bid,
        release_pos.tolist(),
        ori,
    )
    pb.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    capture_video_frame(pb, recorder, camera)
    return settle_and_record(pb, recorder, bid, camera)


def dump_final_poses(pb: Any, env: Any, sim_by_class: dict[str, dict], out_dir: Path) -> None:
    sim_by_name = {obj["name"]: obj for obj in sim_by_class.values()}
    final_poses: dict[str, Any] = {
        "table": {"height": float(env.table_height)},
        "objects": {},
    }
    for name, bid in env.pybullet_id_dict.items():
        if name in ("robot", "ghost", "gripper"):
            continue
        aabb_min, aabb_max = pb.getAABB(bid)
        if name == "support_table":
            final_poses["table"]["aabb_min"] = [float(x) for x in aabb_min]
            final_poses["table"]["aabb_max"] = [float(x) for x in aabb_max]
            continue
        pos, quat = pb.getBasePositionAndOrientation(bid)
        info = sim_by_name[name]
        final_poses["objects"][name] = {
            "class": info["class"],
            "instance": info.get("instance", name),
            "pos": [float(x) for x in pos],
            "quat_xyzw": [float(x) for x in quat],
            "aabb_min": [float(x) for x in aabb_min],
            "aabb_max": [float(x) for x in aabb_max],
        }
    (out_dir / "final_poses.json").write_text(json.dumps(final_poses, indent=2))


def snapshot_object_positions(pb: Any, env: Any, sim_by_class: dict[str, dict]) -> dict[str, list[float]]:
    out = {}
    for cls, obj in sim_by_class.items():
        pos, _ = pb.getBasePositionAndOrientation(env.pybullet_id_dict[obj["name"]])
        out[cls] = [float(x) for x in pos]
    return out


def run_scene(scene_dir: Path, no_obstacle: bool, run_vlm: bool) -> None:
    scene_dir = scene_dir.expanduser().resolve()
    if no_obstacle and not scene_has_obstacle(scene_dir):
        if copy_with_obstacle_outputs_to_no_obstacle(scene_dir):
            print(
                "[copy] scene has no obstacle; reused existing with-obstacle outputs",
                flush=True,
            )
            if run_vlm:
                from run_vlm_compare import run as run_vlm_compare

                print("[vlm] comparing SG-Bot result against ours...", flush=True)
                run_vlm_compare(scene_dir, no_obstacle=True)
                print("[vlm] comparison saved", flush=True)
            return

    out_dir_name = "our_output_no_obstacle" if no_obstacle else "our_output"
    result_name = "our_result_no_obstacle.png" if no_obstacle else "our_result.png"
    ctx = load_context(scene_dir, out_dir_name)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    os.chdir(str(SGBOT_DIR))
    install_moviepy_editor_shim()
    from initial import RearrangeEnv
    import pybullet as pb
    from PIL import Image

    with force_pybullet_direct(pb):
        env = RearrangeEnv(
            [(str(ctx["synthetic_view"]), str(ctx["synthetic_view"]))],
            rcs.MODELS_ROOT,
            None,
            5,
        )
    egl_plugin_id = load_egl_renderer(pb)
    env.reset_rearrange(0)
    env.real_cam_to_world = np.eye(4, dtype=np.float64)
    sim_by_class = ctx["sim_by_class"]
    apply_visual_textures(pb, env, sim_by_class)
    video_camera = dataset_camera(ctx, float(env.table_height))
    recorder = VideoRecorder(ctx["out_dir"] / "teleport.mp4")
    log(f"[render] pybullet DIRECT + EGL plugin {egl_plugin_id}")
    log(
        f"[camera] current yaml view {video_camera['width']}x{video_camera['height']} "
        f"fx={video_camera['fx']:.3f} fy={video_camera['fy']:.3f}"
    )
    capture_video_frame(pb, recorder, video_camera)
    write_step_state(
        pb,
        env,
        ctx,
        ctx["steps_dir"] / "step_init.json",
        phase="init",
        step_index=0,
        action=None,
    )

    local_clouds = {
        cls: object_local_cloud(sim_obj)
        for cls, sim_obj in sim_by_class.items()
    }
    current_pose: dict[str, np.ndarray] = {}
    rel_pose_by_class: dict[str, np.ndarray] = {}
    target_pose: dict[str, np.ndarray] = {}
    for cls, sim_obj in sim_by_class.items():
        bid = env.pybullet_id_dict[sim_obj["name"]]
        pos, quat = pb.getBasePositionAndOrientation(bid)
        current_pose[cls] = pose_from_pos_quat(pos, quat)
        if no_obstacle and cls == "obstacle":
            rel_pose_by_class[cls] = None
            target_pose[cls] = current_pose[cls]
            log("[plan] obstacle: rel_pose=None (SG-Bot move-away)")
            continue
        rel_pose = table_frame_transform_to_sim(
            np.asarray(ctx["ta_scene"].objects[cls].v2_real_pt_transform, dtype=np.float64),
            float(env.table_height),
        )
        rel_pose, lift = lift_rel_pose_to_table(
            rel_pose,
            current_pose[cls],
            local_clouds[cls],
            float(env.table_height),
        )
        rel_pose_by_class[cls] = rel_pose
        target_pose[cls] = transformed_pose(rel_pose_by_class[cls], current_pose[cls])
        if lift:
            log(f"[lift] {cls}: target mesh below table, z += {lift:.3f}m")
        log(f"[plan] {cls}: target xy=({target_pose[cls][0, 3]:.3f}, {target_pose[cls][1, 3]:.3f})")

    from organize_it.modules.teleport_video import build_v0703_plan

    plan = build_v0703_plan(ctx["scene_dir"])
    buffer_count = sum(1 for step_info in plan if step_info.get("kind") == "buffer")
    log(f"[planner-v0703] using dependency plan: {len(plan)} steps ({buffer_count} buffer detours)")

    finished: set[str] = set()
    step = 0
    last_action: dict[str, Any] | None = None
    prev_plan_pose = {
        cls: np.asarray(ctx["ta_scene"].objects[cls].any6d_original_pose, dtype=np.float64).reshape(4, 4).copy()
        for cls in sim_by_class
    }
    for step_info in plan:
        cls = step_info.get("obj_id")
        if cls not in sim_by_class:
            raise KeyError(f"planner object not in SG-Bot scene: {cls}")
        kind = step_info.get("kind")
        sim_obj = sim_by_class[cls]
        bid = env.pybullet_id_dict[sim_obj["name"]]
        step += 1
        action: dict[str, Any]

        if kind == "buffer":
            if "target_world_pose" not in step_info:
                raise KeyError(f"planner buffer step missing target_world_pose: {step_info}")
            target_plan_pose = np.asarray(step_info["target_world_pose"], dtype=np.float64).reshape(4, 4)
            planner_delta = target_plan_pose @ np.linalg.inv(prev_plan_pose[cls])
            buffer_pose = transformed_pose(
                table_frame_transform_to_sim(planner_delta, float(env.table_height)),
                current_pose[cls],
            )
            buffer_pose, lift = lift_pose_to_table(
                buffer_pose,
                local_clouds[cls],
                float(env.table_height),
            )
            log(
                f"[step {step}] buffer {cls} -> "
                f"({buffer_pose[0, 3]:.3f}, {buffer_pose[1, 3]:.3f})"
            )
            if lift:
                log(f"[lift] buffer {cls}: target mesh below table, z += {lift:.3f}m")
            action = {
                "kind": "buffer",
                "object_class": cls,
                "object_name": sim_obj["name"],
                "target_pose_sim": pose_matrix_json(buffer_pose),
            }
            settled_steps = teleport_pose_and_settle(
                pb,
                env,
                bid,
                buffer_pose,
                cls,
                recorder,
                video_camera,
            )
            log(f"[settle] buffer {cls}: {settled_steps} sim steps")
            prev_plan_pose[cls] = target_plan_pose.copy()
        elif kind == "goal":
            if rel_pose_by_class[cls] is None:
                log(f"[step {step}] move {cls} with rel_pose=None (SG-Bot move-away)")
                action = {
                    "kind": "goal",
                    "object_class": cls,
                    "object_name": sim_obj["name"],
                    "target_pose_sim": None,
                    "move_away": True,
                }
                settled_steps = teleport_rel_pose_and_settle(
                    pb,
                    env,
                    bid,
                    None,
                    cls,
                    recorder,
                    video_camera,
                )
            else:
                log(
                    f"[step {step}] move {cls} -> absolute goal "
                    f"({target_pose[cls][0, 3]:.3f}, {target_pose[cls][1, 3]:.3f})"
                )
                action = {
                    "kind": "goal",
                    "object_class": cls,
                    "object_name": sim_obj["name"],
                    "target_pose_sim": pose_matrix_json(target_pose[cls]),
                    "move_away": False,
                }
                settled_steps = teleport_pose_and_settle(
                    pb,
                    env,
                    bid,
                    target_pose[cls],
                    cls,
                    recorder,
                    video_camera,
                )
            log(f"[settle] {cls}: {settled_steps} sim steps")
            finished.add(cls)
            if "target_world_pose" in step_info:
                prev_plan_pose[cls] = np.asarray(step_info["target_world_pose"], dtype=np.float64).reshape(4, 4).copy()
        else:
            raise ValueError(f"unknown planner step kind: {kind}")

        pos, quat = pb.getBasePositionAndOrientation(bid)
        current_pose[cls] = pose_from_pos_quat(pos, quat)
        action["settled_steps"] = int(settled_steps)
        action["final_pose_sim"] = {
            "pos": [float(x) for x in pos],
            "quat_xyzw": [float(x) for x in quat],
        }
        if kind == "goal" and rel_pose_by_class[cls] is not None:
            xy_err = float(np.linalg.norm(current_pose[cls][:2, 3] - target_pose[cls][:2, 3]))
            z_err = float(current_pose[cls][2, 3] - target_pose[cls][2, 3])
            action["verify_error"] = {
                "xy_m": xy_err,
                "z_m": z_err,
            }
            log(f"[verify] {cls}: final xy err={xy_err:.3f}m z err={z_err:.3f}m")
        write_step_state(
            pb,
            env,
            ctx,
            ctx["steps_dir"] / f"step_{step}.json",
            phase="step",
            step_index=step,
            action=action,
        )
        last_action = action

    movable = set(sim_by_class)
    log(f"[done] moved {len(finished)}/{len(movable)} objects")
    write_step_state(
        pb,
        env,
        ctx,
        ctx["steps_dir"] / "final.json",
        phase="final",
        step_index=step,
        action=last_action,
        completed=True,
    )
    final_body_poses = snapshot_body_poses(pb, env)
    recorder.close()
    log(f"[video] frames={recorder.frame_count} fps={VIDEO_FPS} path={ctx['out_dir'] / 'teleport.mp4'}")
    result_path = ctx["scene_dir"] / result_name
    pb.disconnect()
    render_clean_tiny_final_in_subprocess(ctx, final_body_poses, result_path)
    log(f"[render] final still clean subprocess TinyRenderer path={result_path}")
    ctx["synthetic_view"].unlink()
    try:
        run_offline_verifier(ctx, log)
    except Exception:
        (ctx["out_dir"] / "teleport.log").write_text("\n".join(log_lines) + "\n")
        raise
    (ctx["out_dir"] / "teleport.log").write_text("\n".join(log_lines) + "\n")
    if run_vlm:
        from run_vlm_compare import run as run_vlm_compare

        print("[vlm] comparing SG-Bot result against ours...", flush=True)
        run_vlm_compare(ctx["scene_dir"], no_obstacle=no_obstacle)
        print("[vlm] comparison saved", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument(
        "--no-obstacle",
        action="store_true",
        help="move obstacle with SG-Bot's rel_pose=None fallback instead of v2_real_pt_transform",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="only run rollout; do not call the VLM comparison",
    )
    args = parser.parse_args()
    run_scene(args.scene_dir, args.no_obstacle, not args.no_vlm)


if __name__ == "__main__":
    main()
