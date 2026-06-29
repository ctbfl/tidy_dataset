#!/usr/bin/env python3
"""VLM coordinate-frame experiment for tabletop object center estimation.

Compares two prompt coordinate definitions on the same rendered scenes:

A. center_pm1: table center is (0, 0), x/y ranges are [-1, 1].
B. corner_01: back-left table corner is (0, 0), front-right is (1, 1).

The scene uses the tidy_dataset SAPIEN simulator and the calibrated camera used by
``tools/render_organize_it_scene_v2.py``.  Objects are dropped, physics-settled,
then their tight visual AABB centers are recorded as ground truth.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sapien.core as sapien
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "simulations"
HANDCRAFT_DIR = REPO_ROOT / "handcraft"
ORGANIZE_IT_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
ORGANIZE_IT_SRC = ORGANIZE_IT_ROOT / "src"
RENDER_V2 = REPO_ROOT / "tools" / "render_organize_it_scene_v2.py"
DEFAULT_CALIBRATION = (
    ORGANIZE_IT_ROOT
    / "experiments"
    / "pybullet_ur5_test_simple"
    / "camera_adjust_step15_calibration.json"
)
DEFAULT_OUT = REPO_ROOT / "experiments" / "vlm_coordinate_frame"

for p in (str(SIM_DIR), str(HANDCRAFT_DIR), str(ORGANIZE_IT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import scene as tidy_scene  # noqa: E402
from objects import Asset, spawn  # noqa: E402
from editor import _world_aabb  # noqa: E402


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    prompt_label: str


ASSETS: list[AssetSpec] = [
    AssetSpec("gso:obj:Cole_Hardware_Mug_Classic_Blue", "blue mug"),
    AssetSpec("gso:obj:Now_Designs_Bowl_Akita_Black", "black bowl"),
    AssetSpec("gso:obj:Cole_Hardware_Hammer_Black", "black hammer"),
    AssetSpec("gso:obj:ACE_Coffee_Mug_Kristen_16_oz_cup", "white coffee mug"),
    AssetSpec("gso:obj:Cole_Hardware_Saucer_Glazed_6", "white saucer plate"),
    AssetSpec("lightwheel:obj:digital_scale:005", "digital scale"),
]

TABLE = {"length": 1.2, "width": 0.7, "height": 0.74, "thickness": 0.05}

# Positions are in the prompted center_pm1 convention: x right, y front.
SINGLE_POSITIONS = [
    (-0.70, -0.65), (0.00, -0.65), (0.70, -0.65),
    (-0.70, 0.00), (0.00, 0.00), (0.70, 0.00),
    (-0.70, 0.65), (0.00, 0.65), (0.70, 0.65),
    (-0.35, 0.35), (0.35, -0.35), (0.45, 0.45),
]

PAIR_POSITIONS = [
    ((-0.55, -0.45), (0.50, 0.40)),
    ((0.55, -0.45), (-0.50, 0.40)),
    ((-0.60, 0.45), (0.55, -0.35)),
    ((0.58, 0.45), (-0.55, -0.35)),
    ((-0.65, 0.00), (0.65, 0.00)),
    ((0.00, -0.65), (0.00, 0.65)),
    ((-0.35, -0.55), (-0.35, 0.45)),
    ((0.42, 0.48), (0.42, -0.48)),
    ((-0.20, 0.55), (0.55, 0.20)),
    ((0.20, -0.55), (-0.55, -0.20)),
]


def load_render_module():
    spec = importlib.util.spec_from_file_location("render_v2", RENDER_V2)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rotation_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return m


def prompt_xy_to_world_xy(x_pm1: float, y_front_pm1: float) -> tuple[float, float]:
    """Prompted coordinates -> SAPIEN/table world XY.

    The calibrated camera is on the negative table-y side; we define the front
    edge as the near/camera side, so prompted +y_front maps to negative world y.
    """
    return (float(x_pm1) * TABLE["length"] * 0.5, -float(y_front_pm1) * TABLE["width"] * 0.5)


def world_xy_to_center_pm1(x_world: float, y_world: float) -> dict[str, float]:
    return {
        "x": float(x_world) / (TABLE["length"] * 0.5),
        "y": -float(y_world) / (TABLE["width"] * 0.5),
    }


def center_pm1_to_corner_01(x: float, y: float) -> dict[str, float]:
    return {"x": (float(x) + 1.0) * 0.5, "y": (float(y) + 1.0) * 0.5}


def world_xy_to_corner_01(x_world: float, y_world: float) -> dict[str, float]:
    c = world_xy_to_center_pm1(x_world, y_world)
    return center_pm1_to_corner_01(c["x"], c["y"])


def create_base_scene():
    tidy_scene.LIBRARY.load_asset_json_backup(None)
    ts = tidy_scene.create_scene(
        headless=True,
        use_hdri=False,
        random_background=False,
        table_length=TABLE["length"],
        table_width=TABLE["width"],
        table_height=TABLE["height"],
        table_thickness=TABLE["thickness"],
        table_texture_id="Marble011",
        wall_texture_id=None,
        scene_fps=100.0,
    )
    return ts


def spawn_at_prompt_xy(ts, spec: AssetSpec, obj_name: str, xy_pm1: tuple[float, float], yaw: float):
    asset = Asset(tidy_scene.LIBRARY[spec.asset_id].handle)
    obj = spawn(ts.scene, asset, obj_name)
    ts.objects[obj_name] = obj

    xw, yw = prompt_xy_to_world_xy(*xy_pm1)
    pose_m = rotation_z(yaw)
    pose_m[:3, 3] = [xw, yw, TABLE["height"] + 0.20]
    obj.set_pose(sapien.Pose(pose_m))
    ts.scene.update_render()

    # Put the visual/collision bottom just above tabletop before physics settle.
    aabb = _world_aabb(obj.entity)
    pose = obj.get_pose()
    pose.p[2] += TABLE["height"] + 0.01 - float(aabb[0][2])
    obj.set_pose(pose)
    return obj


def settle_scene(ts, steps: int = 500) -> None:
    for _ in range(steps):
        ts.scene.step()
    ts.scene.update_render()


def render_scene(ts, out_png: Path, calibration_path: Path) -> np.ndarray:
    mod = load_render_module()
    ref_mod = mod.load_reference_module()
    calibration = ref_mod.preview._load_robotwin_camera_calibration(calibration_path.resolve())
    camera, _ = mod.add_calibrated_camera(
        ref_mod, ts, calibration["camera"], near=0.02, far=10.0, name="wrist_camera"
    )
    ts.scene.update_render()
    capture = ref_mod.capture_camera(camera)
    rgb = capture["rgb"]
    Image.fromarray(rgb).save(out_png)
    return rgb


def object_truth(obj) -> dict[str, Any]:
    aabb = _world_aabb(obj.entity)
    center = (aabb[0] + aabb[1]) * 0.5
    center_pm1 = world_xy_to_center_pm1(float(center[0]), float(center[1]))
    corner_01 = center_pm1_to_corner_01(center_pm1["x"], center_pm1["y"])
    return {
        "world_aabb_min": [float(v) for v in aabb[0]],
        "world_aabb_max": [float(v) for v in aabb[1]],
        "world_center_xyz": [float(v) for v in center],
        "center_pm1": center_pm1,
        "corner_01": corner_01,
    }


def generate_dataset(out_dir: Path, n_single: int, n_pair: int, seed: int, settle_steps: int, calibration_path: Path) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    scenes: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    single_positions = SINGLE_POSITIONS[:]
    pair_positions = PAIR_POSITIONS[:]
    rng.shuffle(single_positions)
    rng.shuffle(pair_positions)

    for i in range(n_single):
        scene_id = f"single_{i:03d}"
        scene_dir = out_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        ts = create_base_scene()
        spec = ASSETS[i % len(ASSETS)]
        xy = single_positions[i % len(single_positions)]
        obj = spawn_at_prompt_xy(ts, spec, "object_A", xy, yaw=rng.uniform(-math.pi, math.pi))
        settle_scene(ts, settle_steps)
        image_path = scene_dir / "current.png"
        render_scene(ts, image_path, calibration_path)
        record = {
            "scene_id": scene_id,
            "task": "single",
            "image": str(image_path),
            "objects": [{"id": "object_A", "prompt_label": "the only object", "asset_id": spec.asset_id, **object_truth(obj)}],
        }
        (scene_dir / "truth.json").write_text(json.dumps(record, indent=2))
        scenes.append(record)

    for i in range(n_pair):
        scene_id = f"pair_{i:03d}"
        scene_dir = out_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        ts = create_base_scene()
        spec_a = ASSETS[(2 * i) % len(ASSETS)]
        spec_b = ASSETS[(2 * i + 1) % len(ASSETS)]
        xy_a, xy_b = pair_positions[i % len(pair_positions)]
        obj_a = spawn_at_prompt_xy(ts, spec_a, "object_A", xy_a, yaw=rng.uniform(-math.pi, math.pi))
        obj_b = spawn_at_prompt_xy(ts, spec_b, "object_B", xy_b, yaw=rng.uniform(-math.pi, math.pi))
        settle_scene(ts, settle_steps)
        image_path = scene_dir / "current.png"
        render_scene(ts, image_path, calibration_path)
        a_truth = object_truth(obj_a)
        b_truth = object_truth(obj_b)
        rel_center = {
            "dx": b_truth["center_pm1"]["x"] - a_truth["center_pm1"]["x"],
            "dy": b_truth["center_pm1"]["y"] - a_truth["center_pm1"]["y"],
        }
        rel_corner = {
            "dx": b_truth["corner_01"]["x"] - a_truth["corner_01"]["x"],
            "dy": b_truth["corner_01"]["y"] - a_truth["corner_01"]["y"],
        }
        record = {
            "scene_id": scene_id,
            "task": "pair",
            "image": str(image_path),
            "objects": [
                {"id": "object_A", "prompt_label": spec_a.prompt_label, "asset_id": spec_a.asset_id, **a_truth},
                {"id": "object_B", "prompt_label": spec_b.prompt_label, "asset_id": spec_b.asset_id, **b_truth},
            ],
            "relative_gt": {"center_pm1": rel_center, "corner_01": rel_corner},
        }
        (scene_dir / "truth.json").write_text(json.dumps(record, indent=2))
        scenes.append(record)

    manifest = {"table": TABLE, "seed": seed, "settle_steps": settle_steps, "scenes": scenes}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return scenes


def load_manifest(out_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return manifest["scenes"]


def prompt_for(scene: dict[str, Any], frame: str) -> str:
    if frame == "center_pm1":
        coord = (
            "Use this coordinate system: the tabletop center is (0, 0). "
            "The x coordinate ranges from -1 at the left edge to +1 at the right edge. "
            "The y coordinate ranges from -1 at the far/back edge to +1 at the near/front edge (closest to the camera)."
        )
        scale_note = "Coordinates must be in [-1, 1]."
    elif frame == "corner_01":
        coord = (
            "Use this coordinate system: the far/back-left tabletop corner is (0, 0), "
            "and the near/front-right tabletop corner is (1, 1). "
            "Thus x ranges from 0 at the left edge to 1 at the right edge, and y ranges from 0 at the far/back edge to 1 at the near/front edge (closest to the camera)."
        )
        scale_note = "Coordinates must be in [0, 1]."
    else:
        raise ValueError(frame)

    common = (
        "You are estimating tabletop positions from an oblique camera image. "
        "Estimate the geometric center of each visible object projected onto the tabletop plane, not the contact point and not the top of the object. "
        f"{coord} {scale_note} Be as numerically precise as possible. "
        "Return ONLY valid JSON, with no markdown or explanation."
    )
    if scene["task"] == "single":
        return common + (
            "\nThere is one object. Output schema exactly: "
            '{"objects":{"object_A":{"x":number,"y":number}},"relative":{}}'
        )
    a = scene["objects"][0]["prompt_label"]
    b = scene["objects"][1]["prompt_label"]
    return common + (
        f"\nThere are two target objects: object_A is the {a}; object_B is the {b}. "
        "Also report object_B's relative displacement from object_A in the same coordinate units: dx = x_B - x_A, dy = y_B - y_A. "
        "Output schema exactly: "
        '{"objects":{"object_A":{"x":number,"y":number},"object_B":{"x":number,"y":number}},"relative":{"B_from_A":{"dx":number,"dy":number,"x_relation":"left|right|same","y_relation":"front|back|same"}}}'
    )


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def call_vlm(prompt: str, image: str) -> str:
    from organize_it.modules import vlm  # noqa: WPS433

    # vlm.py currently references base64 but does not import it; patch the module
    # namespace locally instead of changing the shared project module.
    vlm.base64 = base64
    return vlm.codex(prompt, images=[image], temperature=0, reasoning_effort="low")


def run_vlm(out_dir: Path, frames: list[str], sleep_s: float = 0.0) -> None:
    scenes = load_manifest(out_dir)
    responses_dir = out_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        for frame in frames:
            response_path = responses_dir / f"{scene['scene_id']}__{frame}.json"
            if response_path.is_file():
                continue
            prompt = prompt_for(scene, frame)
            print(f"[VLM] {scene['scene_id']} {frame}", flush=True)
            raw = None
            call_error = None
            for attempt in range(3):
                try:
                    raw = call_vlm(prompt, scene["image"])
                    break
                except Exception as exc:  # noqa: BLE001
                    call_error = repr(exc)
                    print(f"[VLM][retry {attempt + 1}/3] {scene['scene_id']} {frame}: {call_error}", flush=True)
                    if attempt < 2:
                        time.sleep(10 * (attempt + 1))
            parsed = None
            parse_error = None
            if raw is not None:
                try:
                    parsed = extract_json(raw)
                except Exception as exc:  # noqa: BLE001
                    parse_error = repr(exc)
            payload = {
                "scene_id": scene["scene_id"], "frame": frame, "prompt": prompt,
                "raw": raw, "parsed": parsed, "parse_error": parse_error, "call_error": call_error if raw is None else None,
            }
            response_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            if sleep_s > 0:
                time.sleep(sleep_s)


def as_float(v: Any) -> float | None:
    try:
        if isinstance(v, str):
            v = v.strip().strip("%")
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def clamp_or_none(v: float | None, lo: float, hi: float) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    return max(lo, min(hi, float(v)))


def sign_label(v: float, eps: float = 0.05, axis: str = "x") -> str:
    if abs(v) <= eps:
        return "same"
    if axis == "x":
        return "right" if v > 0 else "left"
    return "front" if v > 0 else "back"


def analyze(out_dir: Path) -> dict[str, Any]:
    scenes = load_manifest(out_dir)
    responses_dir = out_dir / "responses"
    rows = []
    rel_rows = []
    for scene in scenes:
        for frame in ("center_pm1", "corner_01"):
            response_path = responses_dir / f"{scene['scene_id']}__{frame}.json"
            if not response_path.is_file():
                continue
            resp = json.loads(response_path.read_text())
            parsed = resp.get("parsed") or {}
            pred_objects = parsed.get("objects") if isinstance(parsed, dict) else None
            if not isinstance(pred_objects, dict):
                for obj in scene["objects"]:
                    rows.append({"scene_id": scene["scene_id"], "task": scene["task"], "frame": frame, "object_id": obj["id"], "valid": False})
                continue
            for obj in scene["objects"]:
                pred = pred_objects.get(obj["id"], {})
                px = clamp_or_none(as_float(pred.get("x")), -1.0 if frame == "center_pm1" else 0.0, 1.0)
                py = clamp_or_none(as_float(pred.get("y")), -1.0 if frame == "center_pm1" else 0.0, 1.0)
                gt = obj[frame]
                valid = px is not None and py is not None
                if valid:
                    err_x = float(px) - float(gt["x"])
                    err_y = float(py) - float(gt["y"])
                    # Convert all errors to center_pm1-scale units for fair comparison.
                    scale = 1.0 if frame == "center_pm1" else 2.0
                    err_l2_pm1 = math.sqrt((err_x * scale) ** 2 + (err_y * scale) ** 2)
                    abs_x_pm1 = abs(err_x * scale)
                    abs_y_pm1 = abs(err_y * scale)
                else:
                    err_l2_pm1 = abs_x_pm1 = abs_y_pm1 = None
                rows.append({
                    "scene_id": scene["scene_id"], "task": scene["task"], "frame": frame, "object_id": obj["id"], "valid": valid,
                    "pred_x": px, "pred_y": py, "gt_x": gt["x"], "gt_y": gt["y"],
                    "err_l2_pm1": err_l2_pm1, "abs_x_pm1": abs_x_pm1, "abs_y_pm1": abs_y_pm1,
                })
            if scene["task"] == "pair":
                # Evaluate relative position from predicted coordinates; it is the most comparable across prompts.
                pa = pred_objects.get("object_A", {})
                pb = pred_objects.get("object_B", {})
                ax = clamp_or_none(as_float(pa.get("x")), -1.0 if frame == "center_pm1" else 0.0, 1.0)
                ay = clamp_or_none(as_float(pa.get("y")), -1.0 if frame == "center_pm1" else 0.0, 1.0)
                bx = clamp_or_none(as_float(pb.get("x")), -1.0 if frame == "center_pm1" else 0.0, 1.0)
                by = clamp_or_none(as_float(pb.get("y")), -1.0 if frame == "center_pm1" else 0.0, 1.0)
                valid = None not in (ax, ay, bx, by)
                gt_rel = scene["relative_gt"][frame]
                if valid:
                    pred_dx = bx - ax
                    pred_dy = by - ay
                    scale = 1.0 if frame == "center_pm1" else 2.0
                    rel_l2_pm1 = math.sqrt(((pred_dx - gt_rel["dx"]) * scale) ** 2 + ((pred_dy - gt_rel["dy"]) * scale) ** 2)
                    sign_x_ok = sign_label(pred_dx, axis="x") == sign_label(gt_rel["dx"], axis="x")
                    sign_y_ok = sign_label(pred_dy, axis="y") == sign_label(gt_rel["dy"], axis="y")
                else:
                    pred_dx = pred_dy = rel_l2_pm1 = None
                    sign_x_ok = sign_y_ok = False
                rel_rows.append({
                    "scene_id": scene["scene_id"], "frame": frame, "valid": bool(valid),
                    "pred_dx": pred_dx, "pred_dy": pred_dy, "gt_dx": gt_rel["dx"], "gt_dy": gt_rel["dy"],
                    "rel_l2_pm1": rel_l2_pm1, "sign_x_ok": sign_x_ok, "sign_y_ok": sign_y_ok,
                })

    def summarize(subrows: list[dict[str, Any]]) -> dict[str, Any]:
        vals = [r["err_l2_pm1"] for r in subrows if r.get("valid") and r.get("err_l2_pm1") is not None]
        xs = [r["abs_x_pm1"] for r in subrows if r.get("valid") and r.get("abs_x_pm1") is not None]
        ys = [r["abs_y_pm1"] for r in subrows if r.get("valid") and r.get("abs_y_pm1") is not None]
        return {
            "n": len(subrows),
            "valid_n": len(vals),
            "mean_l2_pm1": float(np.mean(vals)) if vals else None,
            "median_l2_pm1": float(np.median(vals)) if vals else None,
            "mean_abs_x_pm1": float(np.mean(xs)) if xs else None,
            "mean_abs_y_pm1": float(np.mean(ys)) if ys else None,
        }

    summary: dict[str, Any] = {"object_position": {}, "relative": {}}
    for task in ("single", "pair"):
        for frame in ("center_pm1", "corner_01"):
            summary["object_position"][f"{task}__{frame}"] = summarize([r for r in rows if r.get("task") == task and r.get("frame") == frame])
    for frame in ("center_pm1", "corner_01"):
        rs = [r for r in rel_rows if r.get("frame") == frame]
        vals = [r["rel_l2_pm1"] for r in rs if r.get("valid") and r.get("rel_l2_pm1") is not None]
        summary["relative"][frame] = {
            "n": len(rs),
            "valid_n": len(vals),
            "mean_rel_l2_pm1": float(np.mean(vals)) if vals else None,
            "median_rel_l2_pm1": float(np.median(vals)) if vals else None,
            "x_relation_acc": float(np.mean([r["sign_x_ok"] for r in rs])) if rs else None,
            "y_relation_acc": float(np.mean([r["sign_y_ok"] for r in rs])) if rs else None,
            "both_relation_acc": float(np.mean([r["sign_x_ok"] and r["sign_y_ok"] for r in rs])) if rs else None,
        }

    result = {"summary": summary, "position_rows": rows, "relative_rows": rel_rows}
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--camera-calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--n-single", type=int, default=9)
    ap.add_argument("--n-pair", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--settle-steps", type=int, default=500)
    ap.add_argument("--generate", action="store_true", help="Generate settled scenes and images.")
    ap.add_argument("--vlm", action="store_true", help="Call organize_it.modules.vlm.codex for both coordinate prompts.")
    ap.add_argument("--analyze", action="store_true", help="Analyze cached VLM responses.")
    ap.add_argument("--sleep-s", type=float, default=0.0)
    args = ap.parse_args()

    if not (args.generate or args.vlm or args.analyze):
        args.generate = args.vlm = args.analyze = True

    if args.generate:
        generate_dataset(args.out, args.n_single, args.n_pair, args.seed, args.settle_steps, args.camera_calibration)
    if args.vlm:
        run_vlm(args.out, ["center_pm1", "corner_01"], sleep_s=args.sleep_s)
    if args.analyze:
        result = analyze(args.out)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
