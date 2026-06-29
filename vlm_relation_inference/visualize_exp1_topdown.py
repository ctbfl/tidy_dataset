from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from render_goal_segmentation import load_scene


DATA_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2")
CASES = ("001", "011", "021", "031")
METHODS = ("m1", "m2")
EXP_REL = Path("debug_vlm_relation_extraction/exp1")

CANVAS_WORKSPACE_WIDTH = 1100
CANVAS_MARGIN = 90
OUT_IMAGE = "topdown_layout.png"
OUT_POSES = "topdown_layout_poses.json"
OUT_SUMMARY = DATA_ROOT / "exp1_topdown_summary.json"


def is_full_ok(validation: dict[str, Any]) -> bool:
    if validation.get("ok") is not True:
        return False
    objects = validation.get("objects")
    if not isinstance(objects, dict):
        return False
    return all(fields.get("x") == "ok" and fields.get("y") == "ok" and fields.get("rotation") == "ok" for fields in objects.values())


def load_table(data_dir: Path) -> dict[str, Any]:
    data = json.loads((data_dir / "tabletop_area.json").read_text())
    size = data.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"invalid tabletop_area.json: {data_dir}")
    return data


def object_asset_ids(data_dir: Path) -> dict[str, str]:
    scene = load_scene(data_dir / "ta_real_scene.pkl")
    out = {}
    for object_id, obj in scene.objects.items():
        asset_id = None
        for entry in getattr(obj, "meta_data", []):
            payload = entry.get("payload", {})
            metadata = payload.get("metadata", {})
            if "asset_id" in metadata:
                asset_id = metadata["asset_id"]
                break
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError(f"missing asset_id for {object_id}")
        out[str(object_id)] = asset_id
    return out


def load_object_dims(data_dir: Path, table_size: tuple[float, float]) -> dict[str, dict[str, Any]]:
    asset_ids = object_asset_ids(data_dir)
    out = {}
    for object_id, asset_id in asset_ids.items():
        path = data_dir / "asset_json_backup" / f"{asset_id}.json"
        data = json.loads(path.read_text())
        size = data["geometry"]["aabb_m"]["size"]
        sx, sy = float(size[0]), float(size[1])
        out[object_id] = {
            "asset_id": asset_id,
            "size_m": [sx, sy],
            "size_norm": [sx / table_size[0] * 2.0, sy / table_size[1] * 2.0],
        }
    return out


def axis_rotation(axis: Any, size_norm: list[float]) -> float:
    if isinstance(axis, (int, float)) and not isinstance(axis, bool):
        return float(axis)
    if axis == "any":
        return 0.0
    if axis == "horizontal":
        return 0.0 if size_norm[0] >= size_norm[1] else 90.0
    if axis == "vertical":
        return 0.0 if size_norm[1] >= size_norm[0] else 90.0
    if axis == "custom":
        raise ValueError("custom axis has no numeric rotation")
    raise ValueError(f"invalid axis: {axis}")


def footprint_axis_size(object_id: str, axis: str, poses: dict[str, dict[str, float | None]], dims: dict[str, dict[str, Any]]) -> float:
    w, h = dims[object_id]["size_norm"]
    angle = math.radians(float(poses[object_id]["rotation"] or 0.0))
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    x_size = w * c + h * s
    y_size = w * s + h * c
    return x_size if axis == "x" else y_size


def set_field(poses: dict[str, dict[str, float | None]], object_id: str, field: str, value: float) -> None:
    if poses[object_id][field] is not None:
        raise ValueError(f"over-defined during visualization: {object_id}.{field}")
    poses[object_id][field] = float(value)


def require_field(poses: dict[str, dict[str, float | None]], object_id: str, field: str) -> float:
    value = poses[object_id][field]
    if value is None:
        raise ValueError(f"undefined during visualization: {object_id}.{field}")
    return float(value)


def apply_even_spacing(relation: dict[str, Any], poses: dict[str, dict[str, float | None]], dims: dict[str, dict[str, Any]]) -> None:
    axis = relation["axis"]
    field = axis
    order = relation["order"]
    anchor = relation["anchor"]
    spacing = float(relation["spacing"])
    mode = relation["mode"]
    anchor_index = order.index(anchor)
    values = [None] * len(order)
    values[anchor_index] = require_field(poses, anchor, field)

    for i in range(anchor_index + 1, len(order)):
        prev_id, object_id = order[i - 1], order[i]
        delta = spacing
        if mode == "footprint":
            delta += 0.5 * footprint_axis_size(prev_id, axis, poses, dims)
            delta += 0.5 * footprint_axis_size(object_id, axis, poses, dims)
        values[i] = float(values[i - 1]) + delta

    for i in range(anchor_index - 1, -1, -1):
        next_id, object_id = order[i + 1], order[i]
        delta = spacing
        if mode == "footprint":
            delta += 0.5 * footprint_axis_size(next_id, axis, poses, dims)
            delta += 0.5 * footprint_axis_size(object_id, axis, poses, dims)
        values[i] = float(values[i + 1]) - delta

    for object_id, value in zip(order, values):
        if object_id != anchor:
            set_field(poses, object_id, field, float(value))


def compute_poses(relations: list[dict[str, Any]], object_ids: list[str], dims: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    poses: dict[str, dict[str, float | None]] = {object_id: {"x": None, "y": None, "rotation": None} for object_id in object_ids}
    for relation in relations:
        if relation["type"] == "align_axis":
            object_id = relation["object"]
            set_field(poses, object_id, "rotation", axis_rotation(relation["axis"], dims[object_id]["size_norm"]))

    for relation in relations:
        kind = relation["type"]
        if kind == "table_x":
            set_field(poses, relation["object"], "x", relation["x"])
        elif kind == "table_y":
            set_field(poses, relation["object"], "y", relation["y"])
        elif kind == "table_xy":
            set_field(poses, relation["object"], "x", relation["x"])
            set_field(poses, relation["object"], "y", relation["y"])
        elif kind == "align_axis":
            continue
        elif kind == "in_same_vertical_line":
            x = require_field(poses, relation["anchor"], "x")
            for object_id in relation["objects"]:
                if object_id != relation["anchor"]:
                    set_field(poses, object_id, "x", x)
        elif kind == "in_same_horizontal_line":
            y = require_field(poses, relation["anchor"], "y")
            for object_id in relation["objects"]:
                if object_id != relation["anchor"]:
                    set_field(poses, object_id, "y", y)
        elif kind == "evenly_spaced_from_anchor":
            apply_even_spacing(relation, poses, dims)
        elif kind == "x_offset_from":
            x = require_field(poses, relation["anchor"], "x")
            set_field(poses, relation["object"], "x", x + float(relation["dx"]))
        elif kind == "y_offset_from":
            y = require_field(poses, relation["anchor"], "y")
            set_field(poses, relation["object"], "y", y + float(relation["dy"]))
        elif kind == "xy_offset_from":
            x = require_field(poses, relation["anchor"], "x")
            y = require_field(poses, relation["anchor"], "y")
            set_field(poses, relation["object"], "x", x + float(relation["dx"]))
            set_field(poses, relation["object"], "y", y + float(relation["dy"]))
        elif kind == "on_top_of":
            continue
        elif kind == "in_holder":
            holder = relation["holder"]
            set_field(poses, relation["object"], "x", require_field(poses, holder, "x"))
            set_field(poses, relation["object"], "y", require_field(poses, holder, "y"))
            set_field(poses, relation["object"], "rotation", require_field(poses, holder, "rotation") if poses[holder]["rotation"] is not None else 0.0)
        else:
            raise ValueError(f"unknown relation type: {kind}")

    complete = {}
    for object_id, pose in poses.items():
        complete[object_id] = {
            "x": require_field(poses, object_id, "x"),
            "y": require_field(poses, object_id, "y"),
            "rotation": require_field(poses, object_id, "rotation"),
        }
    return complete


def render_topdown(out_dir: Path, table_size: tuple[float, float], poses: dict[str, dict[str, float]], dims: dict[str, dict[str, Any]]) -> None:
    workspace_w = CANVAS_WORKSPACE_WIDTH
    workspace_h = round(workspace_w * table_size[1] / table_size[0])
    width = workspace_w + 2 * CANVAS_MARGIN
    height = workspace_h + 2 * CANVAS_MARGIN
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    x0, y0 = CANVAS_MARGIN, CANVAS_MARGIN
    x1, y1 = x0 + workspace_w, y0 + workspace_h
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 2)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return x0 + (x + 1.0) * 0.5 * workspace_w, y0 + (1.0 - (y + 1.0) * 0.5) * workspace_h

    groups: dict[tuple[float, float, float, float, float], list[str]] = {}
    for object_id, pose in poses.items():
        key = (
            round(pose["x"], 5),
            round(pose["y"], 5),
            round(pose["rotation"], 5),
            round(dims[object_id]["size_norm"][0], 5),
            round(dims[object_id]["size_norm"][1], 5),
        )
        groups.setdefault(key, []).append(object_id)

    for object_ids in sorted(groups.values(), key=lambda values: values[0]):
        object_id = sorted(object_ids)[0]
        pose = poses[object_id]
        cx, cy = to_px(pose["x"], pose["y"])
        w = dims[object_id]["size_norm"][0] * 0.5 * workspace_w
        h = dims[object_id]["size_norm"][1] * 0.5 * workspace_h
        theta = math.radians(pose["rotation"])
        corners = []
        for lx, ly in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            rx = math.cos(theta) * lx - math.sin(theta) * ly
            ry = math.sin(theta) * lx + math.cos(theta) * ly
            corners.append((round(cx + rx), round(cy - ry)))
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(image, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)

        labels = sorted(object_ids)
        scale = 0.45
        thickness = 1
        line_h = 16
        start_y = cy - (len(labels) - 1) * line_h / 2
        for index, text in enumerate(labels):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            text_y = round(start_y + index * line_h + th / 2)
            cv2.putText(image, text, (round(cx - tw / 2), text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / OUT_IMAGE), image)


def render_if_full_ok(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    validation = json.loads((out_dir / "validation_result.json").read_text())
    full_ok = is_full_ok(validation)
    result = {"full_ok": full_ok, "rendered": False}
    if not full_ok:
        return result

    relations = json.loads((out_dir / "relations.json").read_text())["constraints"]
    table = load_table(data_dir)
    table_size = (float(table["size"][0]), float(table["size"][1]))
    dims = load_object_dims(data_dir, table_size)
    poses = compute_poses(relations, sorted(dims), dims)

    pose_dump = {
        "tabletop_area": table,
        "objects": {
            object_id: {**poses[object_id], **dims[object_id]}
            for object_id in sorted(poses)
        },
    }
    (out_dir / OUT_POSES).write_text(json.dumps(pose_dump, indent=2, ensure_ascii=False))
    render_topdown(out_dir, table_size, poses, dims)
    result["rendered"] = True
    result["image"] = str(out_dir / OUT_IMAGE)
    result["poses"] = str(out_dir / OUT_POSES)
    return result


def process_one(case_id: str, method: str) -> dict[str, Any]:
    data_dir = DATA_ROOT / case_id
    out_dir = data_dir / EXP_REL / method
    result = render_if_full_ok(data_dir, out_dir)
    result["case"] = case_id
    result["method"] = method
    return result


def main() -> None:
    results = [process_one(case_id, method) for case_id in CASES for method in METHODS]
    full_ok_count = sum(1 for result in results if result["full_ok"])
    summary = {
        "total": len(results),
        "full_ok": full_ok_count,
        "by_method": {
            method: {
                "total": sum(1 for result in results if result["method"] == method),
                "full_ok": sum(1 for result in results if result["method"] == method and result["full_ok"]),
            }
            for method in METHODS
        },
        "results": results,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
