from __future__ import annotations

import base64
import io
import json
import random
import sys
import threading
from pathlib import Path

import numpy as np
import sapien.core as sapien
import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

HERE = Path(__file__).resolve().parent
SIMULATIONS_DIR = HERE.parent / "simulations"
if str(SIMULATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATIONS_DIR))

from editor import SceneEditor, _world_aabb  # noqa: E402
from objects import Asset, spawn  # noqa: E402
from preview import PreviewRenderer  # noqa: E402
from scene import LIBRARY  # noqa: E402

DATASET_DIR = HERE.parent / "data" / "organize_it_dataset_v2"
GPU = threading.Lock()
TRANS_STEP = 0.01
TRANS_FINE = 0.002
YAW_STEP = np.radians(15)
YAW_FINE = np.radians(5)
LOCAL_IDS = "abcdefghijklmnopqrstuvwxyz"

previews = PreviewRenderer()


class IncompleteRelation(ValueError):
    pass


def _variation_dir(scenario: str, variation: str) -> Path:
    path = (DATASET_DIR / Path(scenario).stem / Path(variation).stem).resolve()
    if path.parent.parent != DATASET_DIR.resolve():
        raise ValueError(f"invalid scenario/variation: {scenario}/{variation}")
    return path


def _available_assets_path(scenario: str, variation: str) -> Path:
    return _variation_dir(scenario, variation) / "template" / "available_assets.json"


def _constraints_dir(scenario: str, variation: str) -> Path:
    return _variation_dir(scenario, variation) / "template" / "constraints"


def _constraint_path(scenario: str, variation: str, name: str) -> Path:
    path = (_constraints_dir(scenario, variation) / f"{Path(name).stem}.json").resolve()
    if path.parent != _constraints_dir(scenario, variation).resolve():
        raise ValueError(f"invalid constraint template name: {name}")
    return path


def _read_available_assets(scenario: str, variation: str) -> dict:
    path = _available_assets_path(scenario, variation)
    if not path.is_file():
        return {"version": 1, "available_assets": {}}
    return json.loads(path.read_text())


def _list_scenarios() -> list[dict]:
    out = []
    if not DATASET_DIR.is_dir():
        return out
    for scenario_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        variations = [
            p.name for p in sorted(scenario_dir.iterdir())
            if p.is_dir() and (p / "template").is_dir()
        ]
        out.append({"scenario": scenario_dir.name, "variations": variations})
    return out


def _mid(value):
    if isinstance(value, list):
        return float(sum(value) / len(value))
    return float(value)


def _ref_key(ref: dict) -> str:
    return f"{ref['category']}:{int(ref['set'])}:{int(ref['slot'])}"


def _clean_ref(ref: dict) -> dict:
    return {"category": str(ref["category"]), "set": int(ref["set"]), "slot": int(ref["slot"])}


class ConstraintStudio:
    def __init__(self):
        self.editor = SceneEditor(camera_width=1024, camera_height=768)
        self.scenario = "dining_table"
        self.variation = "after_meal_cleanup"
        self.template_name = "draft"
        self.available = {}
        self.object_sets: list[dict] = []
        self.constraints: list[dict] = []
        self.scene_ids: dict[str, str] = {}
        self.number_by_key: dict[str, int] = {}
        self.key_by_scene_id: dict[str, str] = {}
        self.placed_keys: set[str] = set()
        self.selected_keys: set[str] = set()
        self.fields: dict[str, dict] = {}
        self.relation_errors: list[str | None] = []
        self.relation_incomplete: list[str | None] = []
        self._num = 0
        self.load_variation(self.scenario, self.variation, clear=True)

    def load_variation(self, scenario: str, variation: str, clear: bool = True) -> None:
        self.scenario = Path(scenario).stem
        self.variation = Path(variation).stem
        self.available = _read_available_assets(self.scenario, self.variation).get("available_assets", {})
        if clear:
            self.object_sets = []
            self.constraints = []
            self.placed_keys.clear()
            self.template_name = "draft"
        self._rebuild_scene()

    def list_constraint_templates(self) -> list[str]:
        path = _constraints_dir(self.scenario, self.variation)
        if not path.is_dir():
            return []
        return sorted(p.stem for p in path.glob("*.json"))

    def annotation(self) -> dict:
        return {
            "version": 1,
            "scenario": self.scenario,
            "variation": self.variation,
            "object_sets": [dict(s) for s in self.object_sets],
            "constraints": [dict(c) for c in self.constraints],
        }

    def load_template(self, name: str) -> None:
        data = json.loads(_constraint_path(self.scenario, self.variation, name).read_text())
        self.template_name = Path(name).stem
        self.object_sets = [dict(s) for s in data.get("object_sets", [])]
        self.constraints = [self._normalize_relation(c) for c in data.get("constraints", [])]
        self.placed_keys = set(self._mentioned_keys_in_order())
        self._rebuild_scene()

    def save_template(self, name: str) -> str:
        self.template_name = Path(name).stem
        path = _constraint_path(self.scenario, self.variation, self.template_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.annotation(), indent=2, ensure_ascii=False))
        return str(path)

    def new_template(self, name: str) -> str:
        self.template_name = Path(name).stem
        if not self.template_name:
            raise ValueError("template name is empty")
        path = _constraint_path(self.scenario, self.variation, self.template_name)
        if path.exists():
            raise ValueError(f"template already exists: {self.template_name}")
        self.object_sets = []
        self.constraints = []
        self.placed_keys.clear()
        self._rebuild_scene()
        return self.save_template(self.template_name)

    def randomize_sets(self) -> None:
        by_category: dict[str, list[int]] = {}
        for index, object_set in enumerate(self.object_sets):
            by_category.setdefault(object_set["category"], []).append(index)

        for category, indices in by_category.items():
            entries = self.available[category]["entries"]
            if len(entries) < len(indices):
                raise ValueError(f"{category} has {len(indices)} sets but only {len(entries)} available entries")
            choices = random.sample(range(len(entries)), len(indices))
            for set_index, entry_index in zip(indices, choices):
                self.object_sets[set_index]["entry_index"] = entry_index

        self.placed_keys = set(self._mentioned_keys_in_order())
        self._rebuild_scene(use_jitter=True)

    def add_set(self, category: str) -> None:
        entries = self.available[category]["entries"]
        if not entries:
            raise ValueError(f"{category} has no available asset sets")
        self.object_sets.append({"category": category, "entry_index": random.randrange(len(entries))})
        self._refresh_object_index()
        for record in self._object_records():
            self.fields.setdefault(record["key"], {"x": None, "y": None, "rotation": None})

    def delete_object(self, key: str) -> None:
        ref = self._ref_from_key(key)
        category_sets = [i for i, s in enumerate(self.object_sets) if s["category"] == ref["category"]]
        if ref["set"] >= len(category_sets):
            return
        self.object_sets.pop(category_sets[ref["set"]])
        self.constraints = [c for c in self.constraints if key not in self._relation_keys(c)]
        self.placed_keys.discard(key)
        self._rebuild_scene()

    def _ref_from_key(self, key: str) -> dict:
        category, set_index, slot = key.split(":")
        return {"category": category, "set": int(set_index), "slot": int(slot)}

    def _object_records(self) -> list[dict]:
        counters: dict[str, int] = {}
        records = []
        for object_set in self.object_sets:
            category = object_set["category"]
            set_index = counters.get(category, 0)
            counters[category] = set_index + 1
            entries = self.available[category]["entries"]
            entry_index = int(object_set.get("entry_index", 0)) % len(entries)
            for slot, asset_id in enumerate(entries[entry_index]):
                ref = {"category": category, "set": set_index, "slot": slot}
                key = _ref_key(ref)
                records.append({"key": key, "ref": ref, "asset_id": asset_id})
        return records

    def _refresh_object_index(self) -> None:
        self.number_by_key.clear()
        records = self._object_records()
        record_keys = {record["key"] for record in records}
        self.placed_keys &= record_keys
        self.selected_keys &= record_keys
        spawn_order = {key: i for i, key in enumerate(self._mentioned_keys_in_order())}
        ordered_records = [
            record for _, record in sorted(
                enumerate(records),
                key=lambda item: (item[1]["key"] not in spawn_order,
                                  spawn_order.get(item[1]["key"], len(records)),
                                  item[0]),
            )
        ]
        for i, record in enumerate(ordered_records):
            self.number_by_key[record["key"]] = len(self.number_by_key) + 1

    def _rebuild_scene(self, use_jitter: bool = False) -> None:
        self.editor.clear()
        self.scene_ids.clear()
        self.key_by_scene_id.clear()
        self.selected_keys.clear()
        self._refresh_object_index()
        self.apply_constraints(use_jitter=use_jitter)

    def _ensure_spawned(self, key: str) -> str:
        sid = self.scene_ids.get(key)
        if sid is not None:
            return sid
        records = {record["key"]: record for record in self._object_records()}
        if key not in records:
            raise ValueError(f"unknown object key: {key}")
        self._num += 1
        index = self.number_by_key.get(key, 1) - 1
        pose = sapien.Pose([10.0 + index, 10.0, self.editor.scene_wrap.table["height"] + 0.08], [1, 0, 0, 0])
        sid = self._spawn(records[key]["asset_id"], key, index, pose)
        self.scene_ids[key] = sid
        self.key_by_scene_id[sid] = key
        self.placed_keys.add(key)
        return sid

    def _spawn(self, asset_id: str, key: str, index: int, pose: sapien.Pose | None = None) -> str:
        asset = Asset(LIBRARY[asset_id].handle)
        sid = f"{key}::{asset_id}#{self._num}"
        obj = spawn(self.editor.scene, asset, sid)
        self.editor.objects[sid] = obj
        self.editor.slot_of[sid] = None
        self.editor._freeze(obj)
        x = -0.45 + (index % 5) * 0.22
        y = -0.22 + (index // 5) * 0.18
        if pose is None:
            obj.set_pose(sapien.Pose([x, y, self.editor.scene_wrap.table["height"] + 0.08], [1, 0, 0, 0]))
            self._move_center_to(obj, x, y)
        else:
            obj.set_pose(pose)
        self.editor._settle(obj)
        return sid

    def place_object(self, key: str, x: int, y: int) -> None:
        records = {record["key"]: record for record in self._object_records()}
        if key not in records:
            raise ValueError(f"unknown object key: {key}")
        point = self.editor._world_point(x, y)
        if point is None:
            raise ValueError("drop object on the table")
        if key not in self.scene_ids:
            self.placed_keys.add(key)
            self._num += 1
            sid = self._spawn(records[key]["asset_id"], key, self.number_by_key.get(key, 1) - 1)
            self.scene_ids[key] = sid
            self.key_by_scene_id[sid] = key
        obj = self.editor.objects[self.scene_ids[key]]
        pose = obj.get_pose()
        obj.set_pose(sapien.Pose([point[0], point[1], pose.p[2]], pose.q))
        self._move_center_to(obj, point[0], point[1])
        self.editor._settle(obj)
        self.selected_keys = {key}
        self._sync_selection()
        self.apply_constraints({key})

    def _move_center_to(self, obj, x: float | None = None, y: float | None = None) -> None:
        aabb = _world_aabb(obj.entity)
        center = (aabb[0] + aabb[1]) * 0.5
        pose = obj.get_pose()
        nx = pose.p[0] if x is None else pose.p[0] + float(x) - float(center[0])
        ny = pose.p[1] if y is None else pose.p[1] + float(y) - float(center[1])
        obj.set_pose(sapien.Pose([nx, ny, pose.p[2]], pose.q))

    def _set_rotation(self, obj, axis: str) -> None:
        yaw = float(axis) if not isinstance(axis, str) else self._yaw_for_axis(obj, axis)
        pose = obj.get_pose()
        obj.set_pose(sapien.Pose(pose.p, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))

    def _yaw_for_axis(self, obj, axis: str) -> float:
        if axis == "any":
            return 0.0
        aabb = obj.asset.handle.record.geometry.aabb_m
        mn = np.asarray(aabb["min"], dtype=float)
        mx = np.asarray(aabb["max"], dtype=float)
        dims = mx - mn
        long_is_x = dims[0] >= dims[1]
        if axis == "horizontal":
            return 0.0 if long_is_x else -np.pi / 2
        if axis == "vertical":
            return np.pi / 2 if long_is_x else 0.0
        raise ValueError(f"unknown axis: {axis}")

    def select_scene_id(self, scene_id: str | None) -> None:
        key = self.key_by_scene_id.get(scene_id or "")
        self.selected_keys = {key} if key else set()
        self._sync_selection()

    def select_at(self, x: int, y: int) -> None:
        self.editor.select_at(x, y)
        self.select_scene_id(self.editor.selected)

    def select_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.editor.select_rect(x0, y0, x1, y1)
        self.selected_keys = {self.key_by_scene_id[sid] for sid in self.editor.selected_ids if sid in self.key_by_scene_id}
        self._sync_selection()

    def _sync_selection(self) -> None:
        self.editor.selected_ids = {self.scene_ids[k] for k in self.selected_keys if k in self.scene_ids}
        self.editor.selected = next(iter(self.editor.selected_ids), None)

    def add_relation(self, relation_type: str) -> None:
        refs = [self._ref_from_key(k) for k in self._stable_keys(self.selected_keys)]
        if not refs:
            raise ValueError("select at least one object")
        relation = self._default_relation(relation_type, refs)
        self._init_relation_params_from_current(relation, None)
        self.constraints.append(relation)
        self.apply_constraints(set(self._relation_writes(relation)))

    def update_relation(self, index: int, relation: dict) -> None:
        old = self.constraints[index]
        relation = self._normalize_relation(relation)
        self._init_relation_params_from_current(relation, old)
        self.constraints[index] = relation
        self.apply_constraints(set(self._relation_writes(self.constraints[index])))

    def pick_relation_ref(self, index: int, field: str, scene_id: str | None) -> None:
        key = self.key_by_scene_id.get(scene_id or "")
        if not key:
            raise ValueError("click an object")
        relation = dict(self.constraints[index])
        if key not in self._relation_keys(relation):
            raise ValueError("picked object must belong to this relation")
        relation[field] = self._ref_from_key(key)
        self.update_relation(index, relation)

    def delete_relation(self, index: int) -> None:
        self.constraints.pop(index)
        self.apply_constraints()

    def _default_relation(self, relation_type: str, refs: list[dict]) -> dict:
        first = refs[0]
        if relation_type == "table_x":
            if len(refs) != 1:
                raise ValueError("table_x needs exactly one selected object")
            return {"type": relation_type, "objects": refs, "target": first, "x": self._current_norm(first, "x")}
        if relation_type == "table_y":
            if len(refs) != 1:
                raise ValueError("table_y needs exactly one selected object")
            return {"type": relation_type, "objects": refs, "target": first, "y": self._current_norm(first, "y")}
        if relation_type == "table_xy":
            if len(refs) != 1:
                raise ValueError("table_xy needs exactly one selected object")
            return {"type": relation_type, "objects": refs, "target": first,
                    "x": self._current_norm(first, "x"), "y": self._current_norm(first, "y")}
        if relation_type == "align_axis":
            if len(refs) != 1:
                raise ValueError("align_axis needs exactly one selected object")
            return {"type": relation_type, "objects": refs, "target": first, "axis": "horizontal", "jitter_deg": 0}
        if relation_type in ("in_same_vertical_line", "in_same_horizontal_line", "evenly_spaced_from_anchor"):
            if len(refs) < 2:
                raise ValueError(f"{relation_type} needs at least two selected objects")
            relation = {"type": relation_type, "objects": refs}
            if relation_type == "evenly_spaced_from_anchor":
                relation["mode"] = "footprint"
            return relation
        if relation_type in ("x_offset_from", "y_offset_from", "xy_offset_from"):
            if len(refs) != 2:
                raise ValueError(f"{relation_type} needs exactly two selected objects")
            return {"type": relation_type, "objects": refs}
        if relation_type == "pen_in_holder":
            if len(refs) != 2:
                raise ValueError("pen_in_holder needs exactly two selected objects")
            return {"type": relation_type, "objects": refs}
        raise ValueError(f"unknown relation type: {relation_type}")

    def _even_spacing_complete(self, relation: dict) -> bool:
        return (
            relation.get("anchor") is not None
            and relation.get("axis") in ("x", "y")
            and relation.get("mode") in ("obj_center", "footprint")
            and bool(relation.get("order"))
        )

    def _even_spacing_from_current(self, relation: dict) -> float:
        before, after = self._even_order_sides(relation)
        targets = after or before
        if not targets:
            raise ValueError("evenly_spaced_from_anchor needs targets")
        axis = relation["axis"]
        if relation["mode"] == "obj_center":
            return round(abs(self._current_norm(targets[0], axis) - self._current_norm(relation["anchor"], axis)), 3)
        anchor_min, anchor_max = self._bbox_norm(relation["anchor"], axis)
        target_min, target_max = self._bbox_norm(targets[0], axis)
        anchor_center = (anchor_min + anchor_max) * 0.5
        target_center = (target_min + target_max) * 0.5
        if target_center >= anchor_center:
            return round(abs(target_min - anchor_max), 3)
        return round(abs(target_max - anchor_min), 3)

    def _axis_from_current_bbox(self, refs: list[dict]) -> str:
        xs = [self._current_norm(ref, "x") for ref in refs]
        ys = [self._current_norm(ref, "y") for ref in refs]
        return "x" if max(xs) - min(xs) >= max(ys) - min(ys) else "y"

    def _ordered_refs_by_axis(self, refs: list[dict], axis: str) -> list[dict]:
        return [
            ref for _, _, ref in sorted(
                (self._current_norm(ref, axis), index, ref)
                for index, ref in enumerate(refs)
            )
        ]

    def _default_even_anchor(self, ordered_refs: list[dict], axis: str) -> dict:
        for ref in ordered_refs:
            fields = self.fields.get(_ref_key(ref))
            if fields and fields.get(axis) is not None:
                return ref
        return ordered_refs[0]

    def _even_order_sides(self, relation: dict) -> tuple[list[dict], list[dict]]:
        order = self._even_order_refs(relation)
        anchor_key = _ref_key(relation["anchor"])
        keys = [_ref_key(ref) for ref in order]
        if anchor_key not in keys:
            raise ValueError("evenly_spaced_from_anchor order must include anchor")
        anchor_index = keys.index(anchor_key)
        return list(reversed(order[:anchor_index])), order[anchor_index + 1:]

    def _even_local_refs(self, relation: dict) -> list[dict]:
        refs = [_clean_ref(ref) for ref in relation.get("objects", [])]
        if len(refs) > len(LOCAL_IDS):
            raise ValueError("evenly_spaced_from_anchor supports at most 26 objects")
        return refs

    def _even_order_from_refs(self, relation: dict, ordered_refs: list[dict]) -> str:
        keys = [_ref_key(ref) for ref in self._even_local_refs(relation)]
        out = []
        for ref in ordered_refs:
            key = _ref_key(ref)
            if key not in keys:
                raise ValueError("evenly_spaced_from_anchor order references an object outside this relation")
            out.append(LOCAL_IDS[keys.index(key)])
        if len(set(out)) != len(out):
            raise ValueError("evenly_spaced_from_anchor order has duplicate objects")
        if len(out) != len(keys):
            raise ValueError("evenly_spaced_from_anchor order must include every object")
        return "".join(out)

    def _normalize_even_order(self, relation: dict, value) -> str:
        if value is None or value == "":
            return ""
        refs = self._even_local_refs(relation)
        if isinstance(value, list):
            if not refs:
                relation["objects"] = [_clean_ref(ref) for ref in value]
            return self._even_order_from_refs(relation, [_clean_ref(ref) for ref in value])
        if isinstance(value, str):
            order = "".join(ch for ch in value.lower() if not ch.isspace() and ch != ",")
            allowed = set(LOCAL_IDS[:len(refs)])
            unknown = [ch for ch in order if ch not in allowed]
            if unknown:
                raise ValueError(f"unknown local id in order: {unknown[0]}")
            if len(set(order)) != len(order):
                raise ValueError("evenly_spaced_from_anchor order has duplicate local ids")
            if len(order) != len(refs):
                raise ValueError("evenly_spaced_from_anchor order must include every local id")
            return order
        raise ValueError("evenly_spaced_from_anchor order must be a local-id string")

    def _even_order_refs(self, relation: dict) -> list[dict]:
        refs = self._even_local_refs(relation)
        order = str(relation.get("order") or "")
        if not order:
            return []
        return [refs[LOCAL_IDS.index(ch)] for ch in order]

    def _stable_keys(self, keys) -> list[str]:
        return sorted(keys, key=lambda k: self.number_by_key.get(k, 10_000))

    def _relation_refs(self, relation: dict) -> list[dict]:
        if relation.get("objects"):
            refs = relation["objects"]
        else:
            refs = []
            for name in ("target", "anchor", "holder"):
                if relation.get(name):
                    refs.append(relation[name])
            refs.extend(relation.get("targets", []))
            if isinstance(relation.get("order"), list):
                refs.extend(relation["order"])
        out = []
        seen = set()
        for ref in refs:
            key = _ref_key(ref)
            if key in seen:
                continue
            seen.add(key)
            out.append(_clean_ref(ref))
        return out

    def _targets_from_relation(self, relation: dict, anchor_field: str) -> list[dict]:
        anchor = relation.get(anchor_field)
        if not anchor:
            return []
        anchor_key = _ref_key(anchor)
        return [ref for ref in self._relation_refs(relation) if _ref_key(ref) != anchor_key]

    def _single_target_from(self, relation: dict, anchor_field: str) -> dict:
        targets = self._targets_from_relation(relation, anchor_field)
        if len(targets) != 1:
            raise ValueError(f"{relation['type']} needs exactly one target beside {anchor_field}")
        return targets[0]

    def _target_ref(self, relation: dict) -> dict:
        target = relation.get("target")
        if target:
            return target
        refs = self._relation_refs(relation)
        if len(refs) != 1:
            raise ValueError(f"{relation['type']} needs exactly one target")
        return refs[0]

    def _init_relation_params_from_current(self, relation: dict, old: dict | None = None) -> None:
        kind = relation["type"]
        if kind == "evenly_spaced_from_anchor":
            refs = self._relation_refs(relation)
            if not refs:
                return
            if len(refs) > len(LOCAL_IDS):
                raise ValueError("evenly_spaced_from_anchor supports at most 26 objects")
            relation["objects"] = [_clean_ref(ref) for ref in refs]
            if relation.get("axis") not in ("x", "y"):
                relation["axis"] = self._axis_from_current_bbox(refs)
            if not relation.get("order"):
                ordered_refs = self._ordered_refs_by_axis(refs, relation["axis"])
                relation["order"] = self._even_order_from_refs(relation, ordered_refs)
            if not relation.get("anchor"):
                relation["anchor"] = self._default_even_anchor(self._even_order_refs(relation), relation["axis"])
            changed = (
                old is None
                or old.get("axis") != relation.get("axis")
                or old.get("mode") != relation.get("mode")
                or old.get("anchor") != relation.get("anchor")
                or old.get("order") != relation.get("order")
            )
            if self._even_spacing_complete(relation) and (changed or relation.get("spacing") in (None, "")):
                relation["spacing"] = self._even_spacing_from_current(relation)
        elif kind == "x_offset_from" and relation.get("anchor") and relation.get("dx") in (None, ""):
            target = self._single_target_from(relation, "anchor")
            relation["dx"] = round(self._current_norm(target, "x") - self._current_norm(relation["anchor"], "x"), 3)
        elif kind == "y_offset_from" and relation.get("anchor") and relation.get("dy") in (None, ""):
            target = self._single_target_from(relation, "anchor")
            relation["dy"] = round(self._current_norm(target, "y") - self._current_norm(relation["anchor"], "y"), 3)
        elif kind == "xy_offset_from" and relation.get("anchor"):
            target = self._single_target_from(relation, "anchor")
            if relation.get("dx") in (None, ""):
                relation["dx"] = round(self._current_norm(target, "x") - self._current_norm(relation["anchor"], "x"), 3)
            if relation.get("dy") in (None, ""):
                relation["dy"] = round(self._current_norm(target, "y") - self._current_norm(relation["anchor"], "y"), 3)

    def _incomplete_reason(self, relation: dict) -> str | None:
        kind = relation["type"]
        if kind in ("table_x", "table_y", "table_xy", "align_axis"):
            try:
                self._target_ref(relation)
            except ValueError as exc:
                return str(exc)
        if kind in ("table_x", "table_xy") and relation.get("x") is None:
            return "set x"
        if kind in ("table_y", "table_xy") and relation.get("y") is None:
            return "set y"
        if kind == "align_axis" and not relation.get("axis"):
            return "choose axis"
        if kind in ("in_same_vertical_line", "in_same_horizontal_line", "evenly_spaced_from_anchor", "x_offset_from", "y_offset_from", "xy_offset_from"):
            if not relation.get("anchor"):
                return "choose anchor"
            if not self._targets_from_relation(relation, "anchor"):
                return "choose target objects"
        if kind == "evenly_spaced_from_anchor":
            if relation.get("axis") not in ("x", "y"):
                return "choose axis"
            if relation.get("mode") not in ("obj_center", "footprint"):
                return "choose mode"
            if relation.get("spacing") is None:
                return "set spacing"
        if kind in ("x_offset_from", "xy_offset_from") and relation.get("dx") is None:
            return "set dx"
        if kind in ("y_offset_from", "xy_offset_from") and relation.get("dy") is None:
            return "set dy"
        if kind == "pen_in_holder":
            if not relation.get("holder"):
                return "choose holder"
            if not self._targets_from_relation(relation, "holder"):
                return "choose object to place"
        return None

    def _bbox_norm(self, ref: dict, axis: str) -> tuple[float, float]:
        key = _ref_key(ref)
        self._ensure_spawned(key)
        aabb = _world_aabb(self.editor.objects[self.scene_ids[key]].entity)
        table = self.editor.scene_wrap.table
        idx = 0 if axis == "x" else 1
        scale = (table["length"] if axis == "x" else table["width"]) * 0.5
        return float(aabb[0][idx]) / scale, float(aabb[1][idx]) / scale

    def _footprint_center_width(self, ref: dict, axis: str) -> tuple[float, float]:
        mn, mx = self._bbox_norm(ref, axis)
        key = _ref_key(ref)
        center = self.fields.get(key, {}).get(axis)
        if center is None:
            center = (mn + mx) * 0.5
        return float(center), mx - mn

    def _footprint_next_center(self, previous: dict, target: dict, axis: str, spacing: float) -> float:
        prev_center, prev_width = self._footprint_center_width(previous, axis)
        _, width = self._footprint_center_width(target, axis)
        if spacing >= 0:
            return prev_center + prev_width * 0.5 + spacing + width * 0.5
        return prev_center - prev_width * 0.5 + spacing - width * 0.5

    def _current_norm(self, ref: dict, axis: str) -> float:
        key = _ref_key(ref)
        self._ensure_spawned(key)
        obj = self.editor.objects[self.scene_ids[key]]
        aabb = _world_aabb(obj.entity)
        center = (aabb[0] + aabb[1]) * 0.5
        table = self.editor.scene_wrap.table
        if axis == "x":
            return round(float(center[0]) / (table["length"] * 0.5), 3)
        return round(float(center[1]) / (table["width"] * 0.5), 3)

    def _current_yaw_deg(self, ref: dict) -> float:
        key = _ref_key(ref)
        self._ensure_spawned(key)
        obj = self.editor.objects[self.scene_ids[key]]
        q = obj.get_pose().q
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]))
        return round(float(np.degrees(yaw)), 3)

    def _normalize_relation(self, relation: dict) -> dict:
        r = dict(relation)
        if "objects" in r:
            r["objects"] = [_clean_ref(ref) for ref in r["objects"]]
        for name in ("target", "anchor", "holder"):
            if name in r:
                r[name] = _clean_ref(r[name]) if r[name] else None
        if "targets" in r:
            r["targets"] = [_clean_ref(ref) for ref in r["targets"]]
        if r.get("type") == "evenly_spaced_from_anchor":
            r["order"] = self._normalize_even_order(r, r.get("order"))
            if not r["order"] and r.get("objects"):
                r["order"] = LOCAL_IDS[:len(r["objects"])]
        if "objects" not in r:
            r["objects"] = self._relation_refs(r)
        return r

    def apply_constraints(self, settle_keys: set[str] | None = None, use_jitter: bool = False) -> None:
        self.fields = {record["key"]: {"x": None, "y": None, "rotation": None} for record in self._object_records()}
        self.relation_errors = [None] * len(self.constraints)
        self.relation_incomplete = [None] * len(self.constraints)
        settle_keys = settle_keys or set()
        for i, relation in enumerate(self.constraints):
            try:
                writes = self._relation_writes(relation)
                self._apply_relation(relation, use_jitter)
                self._apply_preview(set(writes), settle_keys)
            except IncompleteRelation as exc:
                self.relation_incomplete[i] = str(exc)
            except ValueError as exc:
                self.relation_errors[i] = str(exc)

    def _require(self, key: str, field: str) -> float | str:
        value = self.fields[key][field]
        if value is None:
            raise ValueError(f"{self._label(key)} requires {field} to be defined")
        return value

    def _write(self, key: str, field: str, value) -> None:
        if self.fields[key][field] is not None:
            raise ValueError(f"{self._label(key)} over-defines {field}")
        self.fields[key][field] = value

    def _jitter(self, relation: dict, name: str, use_jitter: bool) -> float:
        if not use_jitter:
            return 0.0
        amount = abs(float(relation.get(name, 0) or 0))
        return random.uniform(-amount, amount) if amount else 0.0

    def _apply_relation(self, relation: dict, use_jitter: bool = False) -> None:
        kind = relation["type"]
        reason = self._incomplete_reason(relation)
        if reason:
            raise IncompleteRelation(reason)
        if kind == "table_x":
            self._write(_ref_key(self._target_ref(relation)), "x", _mid(relation["x"]) + self._jitter(relation, "x_jitter", use_jitter))
        elif kind == "table_y":
            self._write(_ref_key(self._target_ref(relation)), "y", _mid(relation["y"]) + self._jitter(relation, "y_jitter", use_jitter))
        elif kind == "table_xy":
            key = _ref_key(self._target_ref(relation))
            self._write(key, "x", _mid(relation["x"]) + self._jitter(relation, "x_jitter", use_jitter))
            self._write(key, "y", _mid(relation["y"]) + self._jitter(relation, "y_jitter", use_jitter))
        elif kind == "align_axis":
            axis = relation.get("axis", "horizontal")
            if axis == "custom":
                self._write(_ref_key(self._target_ref(relation)), "rotation", np.radians(float(relation.get("yaw_deg", 0))))
            else:
                self._write(_ref_key(self._target_ref(relation)), "rotation", axis)
        elif kind == "in_same_vertical_line":
            anchor = _ref_key(relation["anchor"])
            x = self._require(anchor, "x")
            for target in self._targets_from_relation(relation, "anchor"):
                self._write(_ref_key(target), "x", x)
        elif kind == "in_same_horizontal_line":
            anchor = _ref_key(relation["anchor"])
            y = self._require(anchor, "y")
            for target in self._targets_from_relation(relation, "anchor"):
                self._write(_ref_key(target), "y", y)
        elif kind == "evenly_spaced_from_anchor":
            axis = relation.get("axis", "x")
            if axis not in ("x", "y"):
                raise ValueError("evenly_spaced_from_anchor axis must be x or y")
            anchor = _ref_key(relation["anchor"])
            base = float(self._require(anchor, axis))
            spacing = float(relation.get("spacing", 0)) + self._jitter(relation, "spacing_jitter", use_jitter)
            if spacing < 0:
                raise ValueError("evenly_spaced_from_anchor spacing + jitter must be non-negative")
            before_targets, after_targets = self._even_order_sides(relation)
            if relation.get("mode") == "obj_center":
                for idx, target in enumerate(before_targets, start=1):
                    self._write(_ref_key(target), axis, base - idx * spacing)
                for idx, target in enumerate(after_targets, start=1):
                    self._write(_ref_key(target), axis, base + idx * spacing)
            else:
                previous = relation["anchor"]
                for target in before_targets:
                    value = self._footprint_next_center(previous, target, axis, -spacing)
                    target_key = _ref_key(target)
                    self._write(target_key, axis, value)
                    self._apply_preview({target_key}, set())
                    previous = target
                previous = relation["anchor"]
                for target in after_targets:
                    value = self._footprint_next_center(previous, target, axis, spacing)
                    target_key = _ref_key(target)
                    self._write(target_key, axis, value)
                    self._apply_preview({target_key}, set())
                    previous = target
        elif kind == "x_offset_from":
            anchor = _ref_key(relation["anchor"])
            target = self._single_target_from(relation, "anchor")
            self._write(_ref_key(target), "x", float(self._require(anchor, "x")) + float(relation.get("dx", 0)) + self._jitter(relation, "dx_jitter", use_jitter))
        elif kind == "y_offset_from":
            anchor = _ref_key(relation["anchor"])
            target = self._single_target_from(relation, "anchor")
            self._write(_ref_key(target), "y", float(self._require(anchor, "y")) + float(relation.get("dy", 0)) + self._jitter(relation, "dy_jitter", use_jitter))
        elif kind == "xy_offset_from":
            anchor = _ref_key(relation["anchor"])
            target = _ref_key(self._single_target_from(relation, "anchor"))
            self._write(target, "x", float(self._require(anchor, "x")) + float(relation.get("dx", 0)) + self._jitter(relation, "dx_jitter", use_jitter))
            self._write(target, "y", float(self._require(anchor, "y")) + float(relation.get("dy", 0)) + self._jitter(relation, "dy_jitter", use_jitter))
        elif kind == "pen_in_holder":
            holder = _ref_key(relation["holder"])
            target = _ref_key(self._single_target_from(relation, "holder"))
            self._write(target, "x", self._require(holder, "x"))
            self._write(target, "y", self._require(holder, "y"))
            self._write(target, "rotation", "vertical")
        else:
            raise ValueError(f"unknown relation type: {kind}")

    def _apply_preview(self, keys: set[str], settle_keys: set[str]) -> None:
        table = self.editor.scene_wrap.table
        for key in keys:
            sid = self._ensure_spawned(key)
            obj = self.editor.objects[sid]
            fields = self.fields[key]
            changed = False
            if fields["rotation"] is not None:
                self._set_rotation(obj, fields["rotation"])
                changed = True
            x = None if fields["x"] is None else float(fields["x"]) * table["length"] * 0.5
            y = None if fields["y"] is None else float(fields["y"]) * table["width"] * 0.5
            if x is not None or y is not None:
                self._move_center_to(obj, x, y)
                changed = True
            if changed or key in settle_keys:
                self.editor._settle(obj)
        self.editor.scene.update_render()

    def key(self, name: str, fine: bool = False, relation_index: int | None = None) -> None:
        if name not in "wasdqe":
            return
        selected = [k for k in sorted(self.selected_keys, key=lambda k: self.number_by_key[k]) if k in self.scene_ids]
        if not selected:
            return
        field = "rotation" if name in "qe" else ("x" if name in "ad" else "y")
        allowed = self._allowed_keys_for_key_edit(field, relation_index)
        editable = [key for key in selected if key in allowed]
        if not editable:
            return

        if name in "wasd":
            step = TRANS_FINE if fine else TRANS_STEP
            dx, dy = {"w": (0, step), "s": (0, -step), "a": (-step, 0), "d": (step, 0)}[name]
            for key in editable:
                obj = self.editor.objects[self.scene_ids[key]]
                pose = obj.get_pose()
                obj.set_pose(sapien.Pose([pose.p[0] + dx, pose.p[1] + dy, pose.p[2]], pose.q))
        else:
            angle = (YAW_FINE if fine else YAW_STEP) * (1 if name == "q" else -1)
            spin = sapien.Pose(q=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
            for key in editable:
                obj = self.editor.objects[self.scene_ids[key]]
                pose = obj.get_pose()
                obj.set_pose(sapien.Pose(pose.p, (spin * sapien.Pose(q=pose.q)).q))

        if relation_index is not None and 0 <= relation_index < len(self.constraints):
            self._update_relation_from_preview(relation_index, editable, field)
            self.apply_constraints(set(editable))
        else:
            self.apply_constraints(set(editable))

    def _allowed_keys_for_key_edit(self, field: str, relation_index: int | None) -> set[str]:
        if relation_index is not None and 0 <= relation_index < len(self.constraints):
            writes = self._relation_writes(self.constraints[relation_index])
            return {key for key, fields in writes.items() if field in fields}
        return {key for key, fields in self.fields.items() if fields[field] is None}

    def _relation_writes(self, relation: dict) -> dict[str, set[str]]:
        if self._incomplete_reason(relation):
            return {}
        kind = relation["type"]
        out: dict[str, set[str]] = {}

        def add(ref, *fields):
            out.setdefault(_ref_key(ref), set()).update(fields)

        if kind == "table_x":
            add(self._target_ref(relation), "x")
        elif kind == "table_y":
            add(self._target_ref(relation), "y")
        elif kind == "table_xy":
            add(self._target_ref(relation), "x", "y")
        elif kind == "align_axis":
            add(self._target_ref(relation), "rotation")
        elif kind == "in_same_vertical_line":
            for ref in self._targets_from_relation(relation, "anchor"):
                add(ref, "x")
        elif kind == "in_same_horizontal_line":
            for ref in self._targets_from_relation(relation, "anchor"):
                add(ref, "y")
        elif kind == "evenly_spaced_from_anchor":
            axis = relation.get("axis")
            for ref in self._targets_from_relation(relation, "anchor"):
                add(ref, axis)
        elif kind == "x_offset_from":
            add(self._single_target_from(relation, "anchor"), "x")
        elif kind == "y_offset_from":
            add(self._single_target_from(relation, "anchor"), "y")
        elif kind == "xy_offset_from":
            add(self._single_target_from(relation, "anchor"), "x", "y")
        elif kind == "pen_in_holder":
            add(self._single_target_from(relation, "holder"), "x", "y", "rotation")
        return out

    def _update_relation_from_preview(self, index: int, moved_keys: list[str], field: str) -> None:
        relation = dict(self.constraints[index])
        kind = relation["type"]
        moved = set(moved_keys)
        target = self._target_ref(relation) if kind in ("table_x", "table_y", "table_xy", "align_axis") else None
        if kind == "table_x" and _ref_key(target) in moved and field == "x":
            relation["x"] = self._current_norm(target, "x")
        elif kind == "table_y" and _ref_key(target) in moved and field == "y":
            relation["y"] = self._current_norm(target, "y")
        elif kind == "table_xy" and _ref_key(target) in moved:
            if field == "x":
                relation["x"] = self._current_norm(target, "x")
            elif field == "y":
                relation["y"] = self._current_norm(target, "y")
        elif kind == "align_axis" and _ref_key(target) in moved and field == "rotation":
            relation["axis"] = "custom"
            relation["yaw_deg"] = self._current_yaw_deg(target)
        elif kind == "evenly_spaced_from_anchor":
            axis = relation.get("axis", "x")
            if field == axis:
                if self._even_spacing_complete(relation):
                    relation["spacing"] = self._even_spacing_from_current(relation)
        elif kind == "x_offset_from" and field == "x":
            target = self._single_target_from(relation, "anchor")
            anchor = self.fields[_ref_key(relation["anchor"])]["x"]
            if _ref_key(target) in moved and anchor is not None:
                relation["dx"] = round(self._current_norm(target, "x") - float(anchor), 3)
        elif kind == "y_offset_from" and field == "y":
            target = self._single_target_from(relation, "anchor")
            anchor = self.fields[_ref_key(relation["anchor"])]["y"]
            if _ref_key(target) in moved and anchor is not None:
                relation["dy"] = round(self._current_norm(target, "y") - float(anchor), 3)
        elif kind == "xy_offset_from":
            target = self._single_target_from(relation, "anchor")
            anchor_key = _ref_key(relation["anchor"])
            if _ref_key(target) in moved and field == "x" and self.fields[anchor_key]["x"] is not None:
                relation["dx"] = round(self._current_norm(target, "x") - float(self.fields[anchor_key]["x"]), 3)
            elif _ref_key(target) in moved and field == "y" and self.fields[anchor_key]["y"] is not None:
                relation["dy"] = round(self._current_norm(target, "y") - float(self.fields[anchor_key]["y"]), 3)
        self.constraints[index] = relation

    def _relation_keys(self, relation: dict) -> set[str]:
        return {_ref_key(ref) for ref in self._relation_refs(relation)}

    def _mentioned_keys_in_order(self) -> list[str]:
        keys = []
        seen = set()
        for relation in self.constraints:
            for ref in self._relation_refs(relation):
                key = _ref_key(ref)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def _label(self, key: str) -> str:
        return str(self.number_by_key.get(key, key))

    def state(self) -> dict:
        records = []
        for record in self._object_records():
            key = record["key"]
            fields = self.fields.get(key, {"x": None, "y": None, "rotation": None})
            records.append({
                **record,
                "num_id": self.number_by_key[key],
                "scene_id": self.scene_ids.get(key),
                "placed": key in self.scene_ids,
                "selected": key in self.selected_keys,
                "defined": {name: fields[name] is not None for name in ("x", "y", "rotation")},
            })
        return {
            "scenario": self.scenario,
            "variation": self.variation,
            "template_name": self.template_name,
            "available_categories": list(self.available),
            "objects": records,
            "constraints": self.constraints,
            "relation_errors": self.relation_errors,
            "relation_incomplete": self.relation_incomplete,
            "selected_keys": list(self.selected_keys),
            "templates": self.list_constraint_templates(),
        }


studio = ConstraintStudio()
app = FastAPI(title="constraint annotation")


def _frame() -> dict:
    with GPU:
        rgb = studio.editor.render()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return {"type": "frame", "image": base64.b64encode(buf.getvalue()).decode(), "state": studio.state()}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "constrain_annotation.html").read_text()


@app.get("/meta")
def meta():
    return {"scenarios": _list_scenarios()}


@app.get("/preview")
def preview(asset_id: str):
    with GPU:
        body = previews.image_bytes(asset_id)
    return Response(body, media_type="image/png")


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    await socket.send_json(_frame())
    try:
        while True:
            msg = await socket.receive_json()
            try:
                kind = msg["type"]
                with GPU:
                    if kind == "load_variation":
                        studio.load_variation(msg["scenario"], msg["variation"], clear=True)
                    elif kind == "load_template":
                        studio.load_template(msg["name"])
                    elif kind == "save_template":
                        path = studio.save_template(msg["name"])
                        await socket.send_json({"type": "saved", "path": path})
                    elif kind == "new_template":
                        path = studio.new_template(msg["name"])
                        await socket.send_json({"type": "new_template", "path": path})
                    elif kind == "randomize":
                        studio.randomize_sets()
                        await socket.send_json({"type": "randomized"})
                    elif kind == "add_set":
                        studio.add_set(msg["category"])
                    elif kind == "delete_object":
                        studio.delete_object(msg["key"])
                    elif kind == "place_object":
                        studio.place_object(msg["key"], int(msg["x"]), int(msg["y"]))
                    elif kind == "select":
                        studio.select_scene_id(msg["scene_id"])
                    elif kind == "select_at":
                        studio.select_at(msg["x"], msg["y"])
                    elif kind == "select_rect":
                        studio.select_rect(msg["x0"], msg["y0"], msg["x1"], msg["y1"])
                    elif kind == "add_relation":
                        studio.add_relation(msg["relation"])
                    elif kind == "pick_relation_ref":
                        scene_id = studio.editor.scene_id_at(int(msg["x"]), int(msg["y"]))
                        studio.pick_relation_ref(int(msg["index"]), msg["field"], scene_id)
                    elif kind == "update_relation":
                        studio.update_relation(int(msg["index"]), msg["relation"])
                    elif kind == "delete_relation":
                        studio.delete_relation(int(msg["index"]))
                    elif kind == "reapply":
                        studio.apply_constraints()
                    elif kind == "key":
                        relation_index = msg.get("relation_index")
                        studio.key(msg["name"], bool(msg.get("fine", False)),
                                   None if relation_index is None else int(relation_index))
                    else:
                        raise ValueError(f"unknown message type: {kind}")
            except Exception as exc:
                await socket.send_json({"type": "error", "message": str(exc)})
            await socket.send_json(_frame())
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    print("[constraint annotation] http://127.0.0.1:8104")
    uvicorn.run(app, host="127.0.0.1", port=8104, log_level="warning")
