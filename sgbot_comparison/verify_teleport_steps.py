#!/usr/bin/env python3
"""Verify that each recorded teleport step has a clear vertical lift path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path("/home/hjs/Datasets/sgbot/sgbot_dataset/models")
LIFT_STEP_M = 0.01
LIFT_MARGIN_M = 0.10
PENETRATION_TOL = 0.002


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def step_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step_") or stem == "step_init":
        raise ValueError(f"invalid step file name: {path}")
    return int(stem.split("_", 1)[1])


def ordered_step_files(steps_dir: Path) -> list[Path]:
    files = sorted(
        (p for p in steps_dir.glob("step_*.json") if p.name != "step_init.json"),
        key=step_index,
    )
    if not files:
        raise FileNotFoundError(f"no step_*.json files in {steps_dir}")
    expected = list(range(1, len(files) + 1))
    actual = [step_index(p) for p in files]
    if actual != expected:
        raise ValueError(f"non-contiguous step files: expected={expected} actual={actual}")
    return files


def asset_urdf(obj: dict[str, Any]) -> Path:
    path = MODELS_ROOT / obj["class"] / f"{obj['instance']}.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"missing URDF: {path}")
    return path


def load_state(pb: Any, state: dict[str, Any]) -> dict[str, int]:
    pb.resetSimulation()
    body_by_class: dict[str, int] = {}
    for cls, obj in state["objects"].items():
        body_id = pb.loadURDF(
            str(asset_urdf(obj)),
            obj["pos"],
            obj["quat_xyzw"],
            useFixedBase=True,
        )
        body_by_class[cls] = int(body_id)
    pb.performCollisionDetection()
    return body_by_class


def max_scene_z(state: dict[str, Any]) -> float:
    return max(float(obj["aabb_max"][2]) for obj in state["objects"].values())


def colliding_classes(pb: Any, moving_body: int, body_by_class: dict[str, int], moving_cls: str) -> list[str]:
    collisions = []
    for other_cls, other_body in body_by_class.items():
        if other_cls == moving_cls:
            continue
        points = pb.getClosestPoints(moving_body, other_body, distance=0.0)
        if any(float(point[8]) < -PENETRATION_TOL for point in points):
            collisions.append(other_cls)
    return collisions


def verify_clear_lift(pb: Any, pre_state: dict[str, Any], moving_cls: str) -> dict[str, Any]:
    if moving_cls not in pre_state["objects"]:
        raise KeyError(f"{moving_cls}: missing from pre-state objects")
    body_by_class = load_state(pb, pre_state)
    moving_body = body_by_class[moving_cls]
    start_pos = list(pre_state["objects"][moving_cls]["pos"])
    quat = pre_state["objects"][moving_cls]["quat_xyzw"]
    target_z = max_scene_z(pre_state) + LIFT_MARGIN_M
    lift = max(LIFT_STEP_M, target_z - float(start_pos[2]))
    samples = max(1, int(math.ceil(lift / LIFT_STEP_M)))

    for sample in range(1, samples + 1):
        pos = start_pos.copy()
        pos[2] = float(start_pos[2]) + sample * LIFT_STEP_M
        pb.resetBasePositionAndOrientation(moving_body, pos, quat)
        pb.performCollisionDetection()
        blockers = colliding_classes(pb, moving_body, body_by_class, moving_cls)
        if blockers:
            return {
                "ok": False,
                "blocked_at_lift_m": sample * LIFT_STEP_M,
                "moving_z": pos[2],
                "blockers": blockers,
            }
    return {
        "ok": True,
        "checked_lift_m": samples * LIFT_STEP_M,
        "target_z": float(start_pos[2]) + samples * LIFT_STEP_M,
    }


def load_current_support_graph(scene_dir: Path) -> dict[str, list[str]]:
    path = scene_dir / "current_support_graph.json"
    if not path.is_file():
        return {}
    graph: dict[str, list[str]] = {}
    for top, bottom in load_json(path):
        graph.setdefault(top, []).append(bottom)
    return graph


def current_support_override(
    moving_cls: str,
    lift_check: dict[str, Any],
    moved_before: set[str],
    current_support: dict[str, list[str]],
) -> dict[str, Any] | None:
    if lift_check["ok"]:
        return None
    blockers = set(lift_check.get("blockers", []))
    bottoms = set(current_support.get(moving_cls, []))
    if not blockers or not bottoms:
        return None
    if not blockers.issubset(bottoms):
        return None
    moved_bottoms = sorted(bottoms & moved_before)
    if moved_bottoms:
        return None
    return {
        "reason": "current_support_top_lift_override",
        "top_object": moving_cls,
        "support_bottoms": sorted(bottoms),
        "blocked_by": sorted(blockers),
        "note": "lift collision accepted because moving object is a top object in current_support_graph and all supported bottom objects have not moved yet",
    }


def verify_steps_dir(steps_dir: Path) -> dict[str, Any]:
    import pybullet as pb

    steps_dir = steps_dir.expanduser().resolve()
    init_path = steps_dir / "step_init.json"
    final_path = steps_dir / "final.json"
    if not init_path.is_file():
        raise FileNotFoundError(f"missing {init_path}")
    if not final_path.is_file():
        raise FileNotFoundError(f"missing {final_path}")

    step_files = ordered_step_files(steps_dir)
    states = {"step_init": load_json(init_path)}
    for path in step_files:
        states[path.stem] = load_json(path)
    scene_dir = steps_dir.parent.parent
    current_support = load_current_support_graph(scene_dir)

    client_id = pb.connect(pb.DIRECT)
    if client_id < 0:
        raise RuntimeError("failed to connect pybullet DIRECT")
    try:
        checks = []
        pre_state = states["step_init"]
        moved_before: set[str] = set()
        for path in step_files:
            state = states[path.stem]
            action = state.get("action")
            if not action:
                raise ValueError(f"{path}: missing action")
            moving_cls = action["object_class"]
            lift_check = verify_clear_lift(pb, pre_state, moving_cls)
            support_override = current_support_override(
                moving_cls,
                lift_check,
                moved_before,
                current_support,
            )
            check = {
                "step_index": state["step_index"],
                "step_file": path.name,
                "kind": action["kind"],
                "object_class": moving_cls,
                "object_name": action["object_name"],
                "ok": lift_check["ok"] or support_override is not None,
                "lift_check": lift_check,
            }
            if support_override is not None:
                check["support_override"] = support_override
            checks.append(check)
            moved_before.add(moving_cls)
            pre_state = state
    finally:
        pb.disconnect()

    final_state = load_json(final_path)
    last_state = states[step_files[-1].stem]
    final_matches_last = final_state["objects"] == last_state["objects"]
    return {
        "schema_version": 1,
        "steps_dir": str(steps_dir),
        "scene_id": states["step_init"]["scene_id"],
        "ok": all(check["ok"] for check in checks) and final_matches_last,
        "final_matches_last_step": final_matches_last,
        "checks": checks,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def report_support_overrides(report: dict[str, Any]) -> list[str]:
    lines = []
    for check in report["checks"]:
        override = check.get("support_override")
        if override is None:
            continue
        lines.append(
            "step={step} object={obj} reason={reason} blocked_by={blocked_by} support_bottoms={bottoms}".format(
                step=check["step_index"],
                obj=check["object_class"],
                reason=override["reason"],
                blocked_by=",".join(override["blocked_by"]),
                bottoms=",".join(override["support_bottoms"]),
            )
        )
    return lines


def write_verified_marker(out_dir: Path, report: dict[str, Any]) -> None:
    video_path = out_dir / "teleport.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    lines = [
        "ok: true",
        f"time: {datetime.now(timezone.utc).isoformat()}",
        f"video_hash: {sha256_file(video_path)}",
    ]
    support_overrides = report_support_overrides(report)
    if support_overrides:
        lines.append("reverify: current_support_lift_override")
        lines.extend(f"reverify_detail: {line}" for line in support_overrides)
    lines.append("")
    text = "\n".join(lines)
    (out_dir / "RESULT_VERIFIED").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps_dir", type=Path)
    args = parser.parse_args()

    report = verify_steps_dir(args.steps_dir)
    marker_path = args.steps_dir.expanduser().resolve().parent / "RESULT_VERIFIED"
    if report["ok"]:
        write_verified_marker(marker_path.parent, report)
    elif marker_path.exists():
        marker_path.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
