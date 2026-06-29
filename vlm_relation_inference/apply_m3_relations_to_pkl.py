from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_goal_segmentation import load_scene


ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
INPUT_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2")
OUTPUT_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2_3")
M3_REL = Path("debug_vlm_relation_extraction/exp1/m3")
RELATIONS_NAME = "relations.json"
VALIDATION_NAME = "validation_result.json"
DEBUG_NAME = "m3_metric_pose_debug.json"


FIELDS = ("x", "y", "rotation")
AXIS_INDEX = {"x": 0, "y": 1}


def load_organize_it():
    src = str(ORGANIZE_IT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from organize_it.modules.goal_confine import backproject_goal_world
    from organize_it.modules.utils import save_ta_real_scene

    return backproject_goal_world, save_ta_real_scene


@dataclass
class ObjectMetric:
    matched: bool
    raw_goal_mask_id: int | None
    center_xy: np.ndarray | None
    points: np.ndarray | None


@dataclass
class RelationDebug:
    index: int
    relation: dict[str, Any]
    metric: dict[str, Any]


class PoseState:
    def __init__(self, object_ids: list[str]):
        self.values: dict[str, dict[str, float | np.ndarray | None]] = {
            object_id: {"x": None, "y": None, "rotation": None}
            for object_id in object_ids
        }

    def set_scalar(self, object_id: str, field: str, value: float) -> None:
        if self.values[object_id][field] is not None:
            raise ValueError(f"{object_id}.{field} is already defined")
        self.values[object_id][field] = float(value)

    def set_rotation(self, object_id: str, rotation: np.ndarray) -> None:
        if self.values[object_id]["rotation"] is not None:
            raise ValueError(f"{object_id}.rotation is already defined")
        arr = np.asarray(rotation, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"{object_id}.rotation must be 3x3, got {arr.shape}")
        self.values[object_id]["rotation"] = arr

    def require_scalar(self, object_id: str, field: str) -> float:
        value = self.values[object_id][field]
        if value is None or isinstance(value, np.ndarray):
            raise ValueError(f"{object_id}.{field} is undefined")
        return float(value)

    def require_rotation(self, object_id: str) -> np.ndarray:
        value = self.values[object_id]["rotation"]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{object_id}.rotation is undefined")
        return value


def case_dirs(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.isdigit() and len(path.name) == 3
    ]


def table_axis(table: dict[str, Any], axis: str) -> tuple[float, float]:
    idx = AXIS_INDEX[axis]
    minimum = float(table["min"][idx])
    maximum = float(table["max"][idx])
    size = float(table["size"][idx])
    return (minimum + maximum) * 0.5, size


def norm_abs_to_metric(table: dict[str, Any], axis: str, value: float) -> float:
    center, size = table_axis(table, axis)
    return center + float(value) * size * 0.5


def norm_delta_to_metric(table: dict[str, Any], axis: str, value: float) -> float:
    _, size = table_axis(table, axis)
    return float(value) * size * 0.5


def load_relations(case_dir: Path) -> list[dict[str, Any]]:
    path = case_dir / M3_REL / RELATIONS_NAME
    data = json.loads(path.read_text())
    constraints = data.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError(f"{path} must contain constraints list")
    return constraints


def require_full_ok(case_dir: Path) -> None:
    path = case_dir / M3_REL / VALIDATION_NAME
    data = json.loads(path.read_text())
    objects = data.get("objects")
    if data.get("ok") is not True or not isinstance(objects, dict):
        raise ValueError(f"{path} is not ok")
    for object_id, fields in objects.items():
        for field in FIELDS:
            if fields.get(field) != "ok":
                raise ValueError(f"{path}: {object_id}.{field} is not ok")


def best_goal_matches(case_dir: Path, object_ids: set[str]) -> dict[int, str]:
    path = case_dir / "goal_matching" / "summary.json"
    data = json.loads(path.read_text())
    matches = data.get("matches")
    if not isinstance(matches, list):
        raise ValueError(f"{path} must contain matches list")
    best: dict[int, tuple[float, str]] = {}
    for match in matches:
        object_id = match.get("obj_id")
        raw_id = match.get("raw_goal_mask_id")
        score = match.get("score")
        if object_id not in object_ids:
            continue
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        prev = best.get(raw_id)
        if prev is None or float(score) > prev[0]:
            best[raw_id] = (float(score), str(object_id))
    return {raw_id: object_id for raw_id, (_, object_id) in best.items()}


def object_metrics(scene, case_dir: Path) -> dict[str, ObjectMetric]:
    backproject_goal_world, _ = load_organize_it()
    points_world, valid, _ = backproject_goal_world(scene)
    object_ids = set(str(object_id) for object_id in scene.objects)
    best_by_raw = best_goal_matches(case_dir, object_ids)
    out: dict[str, ObjectMetric] = {}
    for object_id, obj in scene.objects.items():
        object_id = str(object_id)
        raw_id = getattr(obj, "raw_goal_mask_id", None)
        mask = getattr(obj, "goal_mask", None)
        matched = isinstance(raw_id, int) and best_by_raw.get(raw_id) == object_id
        if not matched or mask is None:
            out[object_id] = ObjectMetric(False, raw_id if isinstance(raw_id, int) else None, None, None)
            continue
        use = np.asarray(mask).astype(bool) & valid
        if int(use.sum()) < 8:
            out[object_id] = ObjectMetric(False, int(raw_id), None, None)
            continue
        points = np.asarray(points_world[use], dtype=np.float64)
        center_xy = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) * 0.5
        out[object_id] = ObjectMetric(True, int(raw_id), center_xy, points)
    return out


def metric_center(metrics: dict[str, ObjectMetric], object_id: str, axis: str) -> float | None:
    item = metrics[object_id]
    if item.center_xy is None:
        return None
    return float(item.center_xy[AXIS_INDEX[axis]])


def rotation_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def local_long_axis(obj) -> str:
    vertices = np.asarray(obj.any6d_scaled_mesh["vertices"], dtype=np.float64)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return "x" if float(extent[0]) >= float(extent[1]) else "y"


def axis_rotation(scene, object_id: str, axis: str, matched: bool) -> np.ndarray:
    if axis in {"any", "custom"}:
        if matched:
            pose = np.asarray(scene.objects[object_id].pose_after_render_fix, dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"{object_id}.pose_after_render_fix must be 4x4")
            return pose[:3, :3]
        return np.eye(3, dtype=np.float64)
    long_axis = local_long_axis(scene.objects[object_id])
    if axis == "horizontal":
        return rotation_z(0.0 if long_axis == "x" else -math.pi / 2.0)
    if axis == "vertical":
        return rotation_z(math.pi / 2.0 if long_axis == "x" else 0.0)
    raise ValueError(f"invalid align_axis.axis: {axis}")


def footprint_axis_size(scene, state: PoseState, object_id: str, axis: str) -> float:
    vertices = np.asarray(scene.objects[object_id].any6d_scaled_mesh["vertices"], dtype=np.float64)
    rotation = state.require_rotation(object_id)
    rotated = (rotation @ vertices.T).T
    values = rotated[:, AXIS_INDEX[axis]]
    return float(values.max() - values.min())


def robust_center_spacing(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    distances = np.diff(ordered)
    distances = distances[distances > 1e-6]
    if len(distances) == 0:
        return None
    if len(distances) >= 3:
        median = float(np.median(distances))
        keep = distances <= max(median * 1.8, median + 0.05)
        if int(keep.sum()) >= 1:
            distances = distances[keep]
    return float(np.mean(distances))


def fallback_even_spacing(scene, state: PoseState, table: dict[str, Any], relation: dict[str, Any]) -> float:
    axis = relation["axis"]
    spacing = norm_delta_to_metric(table, axis, relation["spacing"])
    if relation["mode"] != "footprint":
        return spacing
    sizes = []
    for object_id in relation["objects"]:
        try:
            sizes.append(footprint_axis_size(scene, state, object_id, axis))
        except ValueError:
            vertices = np.asarray(scene.objects[object_id].any6d_scaled_mesh["vertices"], dtype=np.float64)
            values = vertices[:, AXIS_INDEX[axis]]
            sizes.append(float(values.max() - values.min()))
    return spacing + float(np.mean(sizes)) if sizes else spacing


def estimate_even_spacing(
    scene,
    state: PoseState,
    table: dict[str, Any],
    metrics: dict[str, ObjectMetric],
    relation: dict[str, Any],
) -> tuple[float, str]:
    axis = relation["axis"]
    values = [
        float(metrics[object_id].center_xy[AXIS_INDEX[axis]])
        for object_id in relation["objects"]
        if metrics[object_id].matched and metrics[object_id].center_xy is not None
    ]
    spacing = robust_center_spacing(values)
    if spacing is not None:
        return spacing, "goal_mask_center_spacing"
    return fallback_even_spacing(scene, state, table, relation), "vlm_fallback"


def apply_even_spacing(state: PoseState, relation: dict[str, Any], spacing: float) -> None:
    axis = relation["axis"]
    order = relation["order"]
    anchor = relation["anchor"]
    anchor_index = order.index(anchor)
    values: list[float | None] = [None] * len(order)
    values[anchor_index] = state.require_scalar(anchor, axis)
    for i in range(anchor_index + 1, len(order)):
        values[i] = float(values[i - 1]) + spacing
    for i in range(anchor_index - 1, -1, -1):
        values[i] = float(values[i + 1]) - spacing
    for object_id, value in zip(order, values):
        if object_id != anchor:
            state.set_scalar(object_id, axis, float(value))


def apply_relation(
    scene,
    state: PoseState,
    table: dict[str, Any],
    metrics: dict[str, ObjectMetric],
    relation: dict[str, Any],
) -> dict[str, Any]:
    kind = relation["type"]
    if kind == "table_x":
        object_id = relation["object"]
        value = metric_center(metrics, object_id, "x")
        source = "goal_mask_center" if value is not None else "vlm_fallback"
        if value is None:
            value = norm_abs_to_metric(table, "x", relation["x"])
        state.set_scalar(object_id, "x", value)
        return {"x": value, "source": source}
    if kind == "table_y":
        object_id = relation["object"]
        value = metric_center(metrics, object_id, "y")
        source = "goal_mask_center" if value is not None else "vlm_fallback"
        if value is None:
            value = norm_abs_to_metric(table, "y", relation["y"])
        state.set_scalar(object_id, "y", value)
        return {"y": value, "source": source}
    if kind == "table_xy":
        object_id = relation["object"]
        x = metric_center(metrics, object_id, "x")
        y = metric_center(metrics, object_id, "y")
        source = "goal_mask_center" if x is not None and y is not None else "vlm_fallback"
        if x is None:
            x = norm_abs_to_metric(table, "x", relation["x"])
        if y is None:
            y = norm_abs_to_metric(table, "y", relation["y"])
        state.set_scalar(object_id, "x", x)
        state.set_scalar(object_id, "y", y)
        return {"x": x, "y": y, "source": source}
    if kind == "align_axis":
        object_id = relation["object"]
        rotation = axis_rotation(scene, object_id, relation["axis"], metrics[object_id].matched)
        state.set_rotation(object_id, rotation)
        return {"axis": relation["axis"], "source": "pose_after_render_fix" if relation["axis"] in {"any", "custom"} and metrics[object_id].matched else "axis_rule"}
    if kind == "in_same_vertical_line":
        anchor = relation["anchor"]
        x = state.require_scalar(anchor, "x")
        for object_id in relation["objects"]:
            if object_id != anchor:
                state.set_scalar(object_id, "x", x)
        return {"x": x, "source": "anchor"}
    if kind == "in_same_horizontal_line":
        anchor = relation["anchor"]
        y = state.require_scalar(anchor, "y")
        for object_id in relation["objects"]:
            if object_id != anchor:
                state.set_scalar(object_id, "y", y)
        return {"y": y, "source": "anchor"}
    if kind == "x_offset_from":
        return apply_offset(state, table, metrics, relation, ("x",))
    if kind == "y_offset_from":
        return apply_offset(state, table, metrics, relation, ("y",))
    if kind == "xy_offset_from":
        return apply_offset(state, table, metrics, relation, ("x", "y"))
    if kind == "evenly_spaced_from_anchor":
        spacing, source = estimate_even_spacing(scene, state, table, metrics, relation)
        apply_even_spacing(state, relation, spacing)
        return {"axis": relation["axis"], "center_spacing": spacing, "source": source}
    if kind == "on_top_of":
        return {"source": "z_support"}
    if kind == "in_holder":
        holder = relation["holder"]
        object_id = relation["object"]
        state.set_scalar(object_id, "x", state.require_scalar(holder, "x"))
        state.set_scalar(object_id, "y", state.require_scalar(holder, "y"))
        try:
            rotation = state.require_rotation(holder)
        except ValueError:
            rotation = np.eye(3, dtype=np.float64)
        state.set_rotation(object_id, rotation)
        return {"holder": holder, "source": "holder_pose"}
    raise ValueError(f"unknown relation type: {kind}")


def apply_offset(
    state: PoseState,
    table: dict[str, Any],
    metrics: dict[str, ObjectMetric],
    relation: dict[str, Any],
    axes: tuple[str, ...],
) -> dict[str, Any]:
    object_id = relation["object"]
    anchor = relation["anchor"]
    both_matched = metrics[object_id].matched and metrics[anchor].matched
    out = {"source": "goal_mask_delta" if both_matched else "vlm_fallback"}
    for axis in axes:
        key = "d" + axis
        if both_matched:
            delta = float(metrics[object_id].center_xy[AXIS_INDEX[axis]] - metrics[anchor].center_xy[AXIS_INDEX[axis]])
        else:
            delta = norm_delta_to_metric(table, axis, relation[key])
        value = state.require_scalar(anchor, axis) + delta
        state.set_scalar(object_id, axis, value)
        out[key] = delta
        out[axis] = value
    return out


def support_map(relations: list[dict[str, Any]]) -> dict[str, str]:
    supports: dict[str, str] = {}
    for relation in relations:
        if relation["type"] != "on_top_of":
            continue
        child = relation["object"]
        parent = relation["anchor"]
        if child in supports and supports[child] != parent:
            raise ValueError(f"{child} has multiple on_top_of supports")
        supports[child] = parent
    return supports


def mesh_world_vertices(obj, pose: np.ndarray) -> np.ndarray:
    vertices = np.asarray(obj.any6d_scaled_mesh["vertices"], dtype=np.float64)
    return (pose[:3, :3] @ vertices.T).T + pose[:3, 3]


def rotated_bbox_center_xy(obj, rotation: np.ndarray) -> np.ndarray:
    vertices = np.asarray(obj.any6d_scaled_mesh["vertices"], dtype=np.float64)
    rotated = (rotation @ vertices.T).T
    return (rotated[:, :2].min(axis=0) + rotated[:, :2].max(axis=0)) * 0.5


def set_bottom_z(obj, pose: np.ndarray, target_bottom: float) -> None:
    points = mesh_world_vertices(obj, pose)
    pose[2, 3] += float(target_bottom) - float(points[:, 2].min())


def top_surface_z(obj, pose: np.ndarray) -> float:
    z = mesh_world_vertices(obj, pose)[:, 2]
    q90, q95 = np.quantile(z, [0.90, 0.95])
    use = z[(z >= q90) & (z <= q95)]
    if len(use) == 0:
        return float(q95)
    return float(np.mean(use))


def z_order(object_ids: list[str], supports: dict[str, str]) -> list[str]:
    children_by_parent: dict[str, list[str]] = {}
    for child, parent in supports.items():
        children_by_parent.setdefault(parent, []).append(child)
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(object_id: str) -> None:
        if object_id in visited:
            return
        if object_id in visiting:
            raise ValueError(f"cycle in on_top_of relations at {object_id}")
        visiting.add(object_id)
        parent = supports.get(object_id)
        if parent is not None:
            visit(parent)
        visiting.remove(object_id)
        visited.add(object_id)
        order.append(object_id)

    for object_id in object_ids:
        visit(object_id)
    return order


def build_poses(scene, state: PoseState, relations: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for object_id in sorted(scene.objects):
        pose = np.eye(4, dtype=np.float64)
        rotation = state.require_rotation(object_id)
        pose[:3, :3] = rotation
        bbox_center_xy = rotated_bbox_center_xy(scene.objects[object_id], rotation)
        pose[0, 3] = state.require_scalar(object_id, "x") - float(bbox_center_xy[0])
        pose[1, 3] = state.require_scalar(object_id, "y") - float(bbox_center_xy[1])
        poses[object_id] = pose

    supports = support_map(relations)
    for object_id in z_order(sorted(scene.objects), supports):
        parent = supports.get(object_id)
        target_bottom = 0.0 if parent is None else top_surface_z(scene.objects[parent], poses[parent])
        set_bottom_z(scene.objects[object_id], poses[object_id], target_bottom)
    return poses


def process_case(input_case: Path, output_case: Path) -> dict[str, Any]:
    require_full_ok(input_case)
    scene = load_scene(output_case / "ta_real_scene.pkl")
    table = json.loads((output_case / "tabletop_area.json").read_text())
    relations = load_relations(input_case)
    metrics = object_metrics(scene, input_case)
    state = PoseState(sorted(scene.objects))

    relation_debug = []
    for index, relation in enumerate(relations):
        metric = apply_relation(scene, state, table, metrics, relation)
        relation_debug.append(RelationDebug(index, relation, metric))

    poses = build_poses(scene, state, relations)
    for object_id, pose in poses.items():
        scene.objects[object_id].pose_after_layout = pose.astype(float).tolist()
        scene.objects[object_id].pose_after_physics = None
        scene.objects[object_id].final_target_pose = None

    _, save_ta_real_scene = load_organize_it()
    save_ta_real_scene(scene, str(output_case / "ta_real_scene.pkl"))

    debug = {
        "case": input_case.name,
        "method": "m3_metric_pose_v1",
        "object_goal_matches": {
            object_id: {
                "matched": item.matched,
                "raw_goal_mask_id": item.raw_goal_mask_id if item.matched else -1,
                "center_xy": None if item.center_xy is None else [float(v) for v in item.center_xy],
                "points": 0 if item.points is None else int(len(item.points)),
            }
            for object_id, item in sorted(metrics.items())
        },
        "relations": [
            {
                "index": item.index,
                "relation": item.relation,
                "metric": item.metric,
            }
            for item in relation_debug
        ],
        "poses": {
            object_id: poses[object_id].astype(float).tolist()
            for object_id in sorted(poses)
        },
    }
    debug_path = output_case / M3_REL / DEBUG_NAME
    debug_path.write_text(json.dumps(debug, indent=2, ensure_ascii=False))
    return {
        "case": input_case.name,
        "objects": len(scene.objects),
        "matched": sum(1 for item in metrics.values() if item.matched),
        "missing": sum(1 for item in metrics.values() if not item.matched),
        "debug": str(debug_path),
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(INPUT_ROOT)

    shutil.copytree(INPUT_ROOT, OUTPUT_ROOT)
    results = []
    for input_case in case_dirs(INPUT_ROOT):
        output_case = OUTPUT_ROOT / input_case.name
        result = process_case(input_case, output_case)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "total": len(results),
        "results": results,
    }
    summary_path = OUTPUT_ROOT / "m3_metric_pose_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(summary_path)


if __name__ == "__main__":
    main()
