from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import threading
from pathlib import Path

import numpy as np
import sapien.core as sapien
import uvicorn
from fastapi import Body, FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

HERE = Path(__file__).resolve().parent
SIMULATIONS_DIR = HERE.parent / "simulations"
if str(SIMULATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATIONS_DIR))

from editor import OUTLINE_COLOR, RELATION_OUTLINE_COLOR, SceneEditor, _world_aabb, _xy_overlaps  # noqa: E402
from objects import Asset, spawn  # noqa: E402
from preview import PreviewRenderer  # noqa: E402
from scene import LIBRARY  # noqa: E402

DATASET_DIR = Path(os.environ.get("TIDY_DATASET_DIR", HERE.parent / "data" / "organize_it_dataset_v2"))
GLOBAL_AVAILABLE_ASSETS_PATH = DATASET_DIR / "available_assets.json"
GPU = threading.Lock()
TRANS_STEP = 0.01
TRANS_FINE = 0.002
YAW_STEP = np.radians(15)
YAW_FINE = np.radians(5)
LOCAL_IDS = "abcdefghijklmnopqrstuvwxyz"
ALIGN_AXES = ("0", "90", "180", "270", "any", "custom")
DELETE_RELATION = "delete_relation"
KEEP_RELATION = "keep_relation"

previews = PreviewRenderer()
ENABLED_ASSET_IDS = {asset.id for asset in LIBRARY if LIBRARY.is_enabled(asset.id)}
SOURCES = sorted({asset.source for asset in LIBRARY if asset.id in ENABLED_ASSET_IDS})
TAGS = sorted({tag for asset in LIBRARY if asset.id in ENABLED_ASSET_IDS for tag in asset.tags})


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


def _variation_info_path(scenario: str, variation: str) -> Path:
    return _variation_dir(scenario, variation) / "template" / "info.json"


def _constraint_path(scenario: str, variation: str, name: str) -> Path:
    path = (_constraints_dir(scenario, variation) / f"{Path(name).stem}.json").resolve()
    if path.parent != _constraints_dir(scenario, variation).resolve():
        raise ValueError(f"invalid constraint template name: {name}")
    return path


def _read_available_assets(scenario: str, variation: str) -> dict:
    group_ids = _read_variation_asset_group_ids(scenario, variation)
    if not group_ids:
        return {"version": 1, "available_assets": {}}
    global_assets = _read_global_available_assets()["available_assets"]
    missing = [group_id for group_id in group_ids if group_id not in global_assets]
    if missing:
        raise ValueError(f"unknown assets_group: {missing[0]}")
    return {"version": 1, "available_assets": {group_id: global_assets[group_id] for group_id in group_ids}}


def _read_global_available_assets() -> dict:
    return _normalize_available_assets(json.loads(GLOBAL_AVAILABLE_ASSETS_PATH.read_text()))


def _read_variation_asset_group_ids(scenario: str, variation: str) -> list[str]:
    path = _available_assets_path(scenario, variation)
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    if "assets_group" in data:
        return [str(group_id) for group_id in data["assets_group"]]
    if "available_assets" in data:
        return list(_normalize_available_assets(data)["available_assets"])
    raise ValueError(f"{path} missing assets_group")


def _write_variation_asset_group_ids(scenario: str, variation: str, group_ids: list[str]) -> str:
    path = _available_assets_path(scenario, variation)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    path.write_text(json.dumps({"version": 1, "assets_group": group_ids}, indent=2, ensure_ascii=False))
    return str(path)


def _normalize_available_assets(payload: dict) -> dict:
    raw_categories = payload.get("available_assets", {})
    if not isinstance(raw_categories, dict):
        raise ValueError("available_assets must be an object")

    categories = {}
    for category_id, category in raw_categories.items():
        category_id = str(category_id).strip()
        if not category_id:
            raise ValueError("category_id cannot be empty")
        slot_count = int(category.get("slot_count", 1))
        if slot_count < 1:
            raise ValueError(f"{category_id}: slot_count must be >= 1")

        entries = []
        seen_entries = set()
        for entry in category.get("entries", []):
            if not isinstance(entry, list) or len(entry) != slot_count:
                raise ValueError(f"{category_id}: every entry must have {slot_count} slots")
            clean = [str(asset_id).strip() for asset_id in entry]
            if any(not asset_id for asset_id in clean):
                continue
            missing = [asset_id for asset_id in clean if asset_id not in LIBRARY.assets]
            if missing:
                raise ValueError(f"{category_id}: unknown asset_id {missing[0]}")
            disabled = [asset_id for asset_id in clean if asset_id not in ENABLED_ASSET_IDS]
            if disabled:
                raise ValueError(f"{category_id}: disabled asset_id {disabled[0]}")
            if len(set(clean)) != len(clean):
                raise ValueError(f"{category_id}: duplicate asset_id inside one set")
            key = tuple(clean)
            if key in seen_entries:
                raise ValueError(f"{category_id}: duplicate asset set")
            seen_entries.add(key)
            entries.append(clean)

        categories[category_id] = {"slot_count": slot_count, "entries": entries}
    return {"version": 1, "available_assets": categories}


def _read_variation_info(scenario: str, variation: str) -> dict:
    path = _variation_info_path(scenario, variation)
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.setdefault("scene_description", "")
    data.setdefault("user_note", "")
    return data


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


def _clean_set_ids(values) -> list[int]:
    sets = [int(v) for v in values]
    if len(set(sets)) != len(sets):
        raise ValueError("same_entry sets must not contain duplicates")
    return sets


def _empty_fields() -> dict:
    return {"x": None, "y": None, "rotation": None}


class ConstraintStudio:
    def __init__(self):
        self.editor = SceneEditor(camera_width=1024, camera_height=768)
        self.scenario = "dining_table"
        self.variation = "after_meal_cleanup"
        self.template_name = "draft"
        self.user_prompt = ""
        self.scene_description = ""
        self.user_note = ""
        self.variation_info = {}
        self.available = {}
        self.object_sets: list[dict] = []
        self.sample_entry_index: dict[str, list[int]] = {}
        self.selection_constraints: list[dict] = []
        self.constraints: list[dict] = []
        self.scene_ids: dict[str, str] = {}
        self.number_by_key: dict[str, int] = {}
        self.key_by_scene_id: dict[str, str] = {}
        self.placed_keys: set[str] = set()
        self.selected_keys: set[str] = set()
        self.fields: dict[str, dict] = {}
        self.selection_errors: list[str | None] = []
        self.relation_errors: list[str | None] = []
        self.relation_incomplete: list[str | None] = []
        self._active_support_snapshot: dict | None = None
        self._pending_support_snapshot: dict | None = None
        self._num = 0
        self.load_variation(self.scenario, self.variation, clear=True)

    def load_variation(self, scenario: str, variation: str, clear: bool = True) -> None:
        self.scenario = Path(scenario).stem
        self.variation = Path(variation).stem
        self.variation_info = _read_variation_info(self.scenario, self.variation)
        self.scene_description = str(self.variation_info["scene_description"])
        self.user_note = str(self.variation_info["user_note"])
        self.available = _read_available_assets(self.scenario, self.variation).get("available_assets", {})
        if clear:
            self.object_sets = []
            self.sample_entry_index = {}
            self.selection_constraints = []
            self.constraints = []
            self.placed_keys.clear()
            self.template_name = "draft"
            self.user_prompt = self.scene_description
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
            "user_prompt": self.user_prompt,
            "object_sets": [{"category": s["category"]} for s in self.object_sets],
            "sample_entry_index": {category: list(indices) for category, indices in self.sample_entry_index.items()},
            "selection_constraints": [dict(c) for c in self.selection_constraints],
            "constraints": [dict(c) for c in self.constraints],
        }

    def load_template(self, name: str) -> None:
        data = json.loads(_constraint_path(self.scenario, self.variation, name).read_text())
        object_sets = [{"category": str(s["category"])} for s in data.get("object_sets", [])]
        for object_set in object_sets:
            category = object_set["category"]
            if category not in self.available:
                raise ValueError(f"{Path(name).stem}: category is not enabled for this variation: {category}")
        self.template_name = Path(name).stem
        self.user_prompt = str(data.get("user_prompt", self.scene_description))
        self.object_sets = object_sets
        self.selection_constraints = [
            self._normalize_selection_constraint(c)
            for c in data.get("selection_constraints", [])
        ]
        self._load_sample_entry_index(data.get("sample_entry_index"))
        self.constraints = [self._normalize_relation(c) for c in data.get("constraints", [])]
        self.placed_keys = set(self._mentioned_keys_in_order())
        self._rebuild_scene()

    def save_template(self, name: str) -> str:
        self.template_name = Path(name).stem
        path = _constraint_path(self.scenario, self.variation, self.template_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.annotation(), indent=2, ensure_ascii=False))
        return str(path)

    def save_user_note(self, value: str) -> str:
        self.user_note = str(value)
        self.variation_info["user_note"] = self.user_note
        path = _variation_info_path(self.scenario, self.variation)
        path.write_text(json.dumps(self.variation_info, indent=2, ensure_ascii=False))
        return str(path)

    def save_user_prompt(self, value: str) -> str:
        self.user_prompt = str(value)
        return self.save_template(self.template_name)

    def new_template(self, name: str) -> str:
        self.template_name = Path(name).stem
        if not self.template_name:
            raise ValueError("template name is empty")
        path = _constraint_path(self.scenario, self.variation, self.template_name)
        if path.exists():
            raise ValueError(f"template already exists: {self.template_name}")
        self.object_sets = []
        self.sample_entry_index = {}
        self.selection_constraints = []
        self.constraints = []
        self.user_prompt = self.scene_description
        self.placed_keys.clear()
        self._rebuild_scene()
        return self.save_template(self.template_name)

    def _category_set_count(self, category: str) -> int:
        return sum(1 for object_set in self.object_sets if object_set["category"] == category)

    def _load_sample_entry_index(self, raw) -> None:
        if raw is None:
            self.sample_entry_index = self._new_sample_entry_index()
            self._sample_entries()
            return
        if not isinstance(raw, dict):
            raise ValueError("sample_entry_index must be an object")
        sample: dict[str, list[int]] = {}
        for category in self.available:
            count = self._category_set_count(category)
            values = raw.get(category, [])
            if count == 0:
                continue
            if len(values) != count:
                raise ValueError(f"sample_entry_index.{category} must have {count} entries")
            entries = self.available[category]["entries"]
            sample[category] = []
            for value in values:
                entry_index = int(value)
                if entry_index < 0 or entry_index >= len(entries):
                    raise ValueError(f"{category}: invalid sample entry index {entry_index}")
                sample[category].append(entry_index)
        self.sample_entry_index = sample
        self._apply_same_entry_samples()

    def _new_sample_entry_index(self) -> dict[str, list[int]]:
        out = {}
        for category in self.available:
            count = self._category_set_count(category)
            if count:
                out[category] = [0] * count
        return out

    def _entry_index(self, category: str, set_index: int) -> int:
        return self.sample_entry_index[category][set_index]

    def _set_entry_index(self, category: str, set_index: int, entry_index: int) -> None:
        self.sample_entry_index.setdefault(category, [])
        while len(self.sample_entry_index[category]) <= set_index:
            self.sample_entry_index[category].append(0)
        self.sample_entry_index[category][set_index] = int(entry_index)

    def _same_entry_units(self) -> list[tuple[str, list[int]]]:
        units = []
        claimed: set[tuple[str, int]] = set()
        for relation in self.selection_constraints:
            if relation["type"] != "same_entry":
                continue
            self._validate_selection_constraint(relation)
            category = relation["category"]
            sets = list(relation["sets"])
            for set_index in sets:
                key = (category, set_index)
                if key in claimed:
                    raise ValueError(f"{category}[{set_index}] belongs to multiple same_entry constraints")
                claimed.add(key)
            units.append((category, sets))
        for category in self.available:
            for set_index in range(self._category_set_count(category)):
                if (category, set_index) not in claimed:
                    units.append((category, [set_index]))
        return units

    def _apply_same_entry_samples(self) -> None:
        claimed: set[tuple[str, int]] = set()
        for category, sets in self._same_entry_units():
            if len(sets) == 1:
                continue
            entry_index = self.sample_entry_index[category][sets[0]]
            for set_index in sets:
                key = (category, set_index)
                if key in claimed:
                    raise ValueError(f"{category}[{set_index}] belongs to multiple same_entry constraints")
                claimed.add(key)
                self.sample_entry_index[category][set_index] = entry_index

    def _validate_assigned_selection_constraints(self, sample: dict[str, list[int]], assigned: set[tuple[str, int]]) -> None:
        for relation in self.selection_constraints:
            if relation["type"] != "bbox_larger_than":
                continue
            refs = (relation["larger"], relation["smaller"])
            if all((ref["category"], ref["set"]) in assigned for ref in refs):
                self._validate_selection_constraint(relation, sample)

    def _sample_entries(self) -> None:
        sample = self._new_sample_entry_index()
        units = self._same_entry_units()
        random.shuffle(units)
        assigned: set[tuple[str, int]] = set()

        def search(index: int) -> bool:
            if index == len(units):
                for relation in self.selection_constraints:
                    self._validate_selection_constraint(relation, sample)
                return True
            category, sets = units[index]
            choices = list(range(len(self.available[category]["entries"])))
            random.shuffle(choices)
            keys = {(category, set_index) for set_index in sets}
            for entry_index in choices:
                for set_index in sets:
                    sample[category][set_index] = entry_index
                assigned.update(keys)
                try:
                    self._validate_assigned_selection_constraints(sample, assigned)
                except ValueError:
                    assigned.difference_update(keys)
                    continue
                if search(index + 1):
                    return True
                assigned.difference_update(keys)
            return False

        if not search(0):
            raise ValueError("no valid entry sample satisfies selection_constraints")
        self.sample_entry_index = sample

    def randomize_sets(self) -> None:
        self._sample_entries()
        self.placed_keys = set(self._mentioned_keys_in_order())
        self._rebuild_scene(use_jitter=True)

    def add_set(self, category: str) -> None:
        entries = self.available[category]["entries"]
        if not entries:
            raise ValueError(f"{category} has no available asset sets")
        set_index = self._category_set_count(category)
        self.object_sets.append({"category": category})
        self._set_entry_index(category, set_index, random.randrange(len(entries)))
        self._refresh_object_index()
        for record in self._object_records():
            self.fields.setdefault(record["key"], _empty_fields())

    def clone_set(self, category: str, set_index: int) -> None:
        category = str(category)
        set_index = int(set_index)
        if category not in self.available:
            raise ValueError(f"unknown category: {category}")
        if set_index < 0 or set_index >= self._category_set_count(category):
            raise ValueError(f"{category}: unknown set {set_index}")
        new_set = self._category_set_count(category)
        self.object_sets.append({"category": category})
        self._set_entry_index(category, new_set, self._entry_index(category, set_index))
        relation = self._same_entry_relation_for(category, set_index)
        if relation is None:
            self.selection_constraints.append({"type": "same_entry", "category": category, "sets": [set_index, new_set]})
            self.selection_errors.append(None)
        else:
            relation["sets"].append(new_set)
        self._refresh_object_index()
        for record in self._object_records():
            self.fields.setdefault(record["key"], _empty_fields())
        self._sync_selection()
        self.apply_constraints()

    def delete_object(self, key: str) -> None:
        ref = self._ref_from_key(key)
        category_sets = [i for i, s in enumerate(self.object_sets) if s["category"] == ref["category"]]
        if ref["set"] >= len(category_sets):
            return
        records = self._object_records()
        record_keys = {record["key"] for record in records}
        if key not in record_keys:
            return
        removed_keys = {
            record["key"] for record in records
            if record["ref"]["category"] == ref["category"] and record["ref"]["set"] == ref["set"]
        }
        key_map = {
            record["key"]: self._key_after_deleted_set(record["ref"], ref["category"], ref["set"])
            for record in records
        }

        self.object_sets.pop(category_sets[ref["set"]])
        self.sample_entry_index[ref["category"]].pop(ref["set"])
        if not self.sample_entry_index[ref["category"]]:
            self.sample_entry_index.pop(ref["category"])
        for deleted_key in removed_keys:
            sid = self.scene_ids.get(deleted_key)
            if sid is not None:
                self.editor.delete(sid)
        self.selection_constraints = self._selection_constraints_after_deleted_set(ref["category"], ref["set"])
        self.constraints = [
            self._remap_relation(c, ref["category"], ref["set"])
            for c in self.constraints
            if not (self._relation_keys(c) & removed_keys)
        ]
        self.scene_ids = {new_key: sid for old_key, sid in self.scene_ids.items()
                          if (new_key := key_map.get(old_key)) is not None}
        self.key_by_scene_id = {sid: key for key, sid in self.scene_ids.items()}
        self.placed_keys = {new_key for old_key in self.placed_keys
                            if (new_key := key_map.get(old_key)) is not None}
        self.selected_keys = {new_key for old_key in self.selected_keys
                              if (new_key := key_map.get(old_key)) is not None}
        self.fields = {new_key: value for old_key, value in self.fields.items()
                       if (new_key := key_map.get(old_key)) is not None}
        self._refresh_object_index()
        for record in self._object_records():
            self.fields.setdefault(record["key"], _empty_fields())
        self._sync_selection()
        self.apply_constraints()

    def _ref_from_key(self, key: str) -> dict:
        category, set_index, slot = key.split(":")
        return {"category": category, "set": int(set_index), "slot": int(slot)}

    def _key_after_deleted_set(self, ref: dict, category: str, set_index: int) -> str | None:
        if ref["category"] != category:
            return _ref_key(ref)
        if ref["set"] == set_index:
            return None
        next_ref = dict(ref)
        if ref["set"] > set_index:
            next_ref["set"] = ref["set"] - 1
        return _ref_key(next_ref)

    def _ref_after_deleted_set(self, ref: dict | None, category: str, set_index: int) -> dict | None:
        if ref is None:
            return None
        if ref["category"] != category or ref["set"] < set_index:
            return dict(ref)
        if ref["set"] == set_index:
            raise ValueError("deleted ref should have removed its relation")
        next_ref = dict(ref)
        next_ref["set"] = ref["set"] - 1
        return next_ref

    def _remap_relation(self, relation: dict, category: str, set_index: int) -> dict:
        out = dict(relation)
        for name in ("target", "anchor", "holder"):
            if name in out:
                out[name] = self._ref_after_deleted_set(out[name], category, set_index)
        if "objects" in out:
            out["objects"] = [self._ref_after_deleted_set(ref, category, set_index) for ref in out["objects"]]
        if "targets" in out:
            out["targets"] = [self._ref_after_deleted_set(ref, category, set_index) for ref in out["targets"]]
        return out

    def _normalize_selection_constraint(self, relation: dict) -> dict:
        kind = relation.get("type")
        if kind == "same_entry":
            category = str(relation["category"])
            return {"type": kind, "category": category, "sets": _clean_set_ids(relation["sets"])}
        if kind == "bbox_larger_than":
            larger = _clean_ref(relation["larger"])
            smaller = _clean_ref(relation["smaller"])
            objects = [_clean_ref(ref) for ref in relation.get("objects", [larger, smaller])]
            if len(objects) != 2:
                raise ValueError("bbox_larger_than needs exactly two objects")
            if len({_ref_key(ref) for ref in objects}) != 2:
                raise ValueError("bbox_larger_than objects must be different")
            return {"type": kind, "objects": objects, "larger": larger, "smaller": smaller}
        raise ValueError(f"unknown selection constraint type: {kind}")

    def _same_entry_relation_for(self, category: str, set_index: int) -> dict | None:
        out = None
        for relation in self.selection_constraints:
            if relation["type"] != "same_entry":
                continue
            if relation["category"] == category and set_index in relation["sets"]:
                if out is not None:
                    raise ValueError(f"{category}[{set_index}] belongs to multiple same_entry constraints")
                out = relation
        return out

    def _selection_relation_involves_set(self, relation: dict, category: str, set_index: int) -> bool:
        kind = relation["type"]
        if kind == "same_entry":
            return relation["category"] == category and set_index in relation["sets"]
        if kind == "bbox_larger_than":
            return any(
                ref["category"] == category and ref["set"] == set_index
                for ref in relation["objects"]
            )
        raise ValueError(f"unknown selection constraint type: {kind}")

    def _selection_delete_hook(self, relation: dict, category: str, set_index: int) -> str:
        kind = relation["type"]
        if kind == "same_entry":
            if relation["category"] != category or set_index not in relation["sets"]:
                raise ValueError("same_entry delete hook received an unrelated set")
            if len(relation["sets"]) <= 2:
                return DELETE_RELATION
            relation["sets"] = [s for s in relation["sets"] if s != set_index]
            return KEEP_RELATION
        if kind == "bbox_larger_than":
            if not self._selection_relation_involves_set(relation, category, set_index):
                raise ValueError("bbox_larger_than delete hook received an unrelated set")
            return DELETE_RELATION
        raise ValueError(f"unknown selection constraint type: {kind}")

    def _remap_selection_constraint(self, relation: dict, category: str, set_index: int) -> dict:
        out = dict(relation)
        kind = out["type"]
        if kind == "same_entry":
            if out["category"] == category:
                out["sets"] = [s - 1 if s > set_index else s for s in out["sets"]]
            return out
        if kind == "bbox_larger_than":
            for name in ("larger", "smaller"):
                ref = dict(out[name])
                if ref["category"] == category and ref["set"] > set_index:
                    ref["set"] -= 1
                out[name] = ref
            out["objects"] = [
                {**ref, "set": ref["set"] - 1 if ref["category"] == category and ref["set"] > set_index else ref["set"]}
                for ref in out["objects"]
            ]
            return out
        raise ValueError(f"unknown selection constraint type: {kind}")

    def _selection_constraints_after_deleted_set(self, category: str, set_index: int) -> list[dict]:
        kept = []
        for relation in self.selection_constraints:
            relation = dict(relation)
            if self._selection_relation_involves_set(relation, category, set_index):
                action = self._selection_delete_hook(relation, category, set_index)
                if action == DELETE_RELATION:
                    continue
                if action != KEEP_RELATION:
                    raise ValueError(f"unknown selection delete action: {action}")
            kept.append(self._remap_selection_constraint(relation, category, set_index))
        return kept

    def _entry_asset_id(self, ref: dict, sample: dict[str, list[int]] | None = None) -> str:
        sample = self.sample_entry_index if sample is None else sample
        category = ref["category"]
        set_index = int(ref["set"])
        slot = int(ref["slot"])
        if category not in self.available:
            raise ValueError(f"unknown category: {category}")
        if set_index < 0 or set_index >= self._category_set_count(category):
            raise ValueError(f"{category}: unknown set {set_index}")
        entries = self.available[category]["entries"]
        entry_index = sample[category][set_index]
        entry = entries[entry_index]
        if slot < 0 or slot >= len(entry):
            raise ValueError(f"{category}[{set_index}]: unknown slot {slot}")
        return entry[slot]

    def _asset_bbox_xy_area(self, asset_id: str) -> float:
        geometry = LIBRARY[asset_id].handle.record.geometry
        aabb = geometry.aabb_m
        if hasattr(aabb, "size"):
            size = np.asarray(aabb.size, dtype=float)
        else:
            size = np.asarray(aabb["size"], dtype=float)
        return float(size[0] * size[1])

    def _validate_selection_constraint(self, relation: dict, sample: dict[str, list[int]] | None = None) -> None:
        kind = relation["type"]
        if kind == "same_entry":
            category = relation["category"]
            if category not in self.available:
                raise ValueError(f"unknown category: {category}")
            if len(relation["sets"]) < 2:
                raise ValueError("same_entry needs at least two sets")
            if len(set(relation["sets"])) != len(relation["sets"]):
                raise ValueError("same_entry sets must not contain duplicates")
            count = self._category_set_count(category)
            for set_index in relation["sets"]:
                if set_index < 0 or set_index >= count:
                    raise ValueError(f"{category}: unknown set {set_index}")
            return
        if kind == "bbox_larger_than":
            object_keys = {_ref_key(ref) for ref in relation["objects"]}
            if _ref_key(relation["larger"]) not in object_keys or _ref_key(relation["smaller"]) not in object_keys:
                raise ValueError("bbox_larger_than roles must refer to relation objects")
            if _ref_key(relation["larger"]) == _ref_key(relation["smaller"]):
                raise ValueError("bbox_larger_than larger and smaller must be different objects")
            larger = self._asset_bbox_xy_area(self._entry_asset_id(relation["larger"], sample))
            smaller = self._asset_bbox_xy_area(self._entry_asset_id(relation["smaller"], sample))
            if larger < smaller:
                raise ValueError("larger bbox xy area must be greater than or equal to smaller bbox xy area")
            return
        raise ValueError(f"unknown selection constraint type: {kind}")

    def _update_selection_errors(self) -> None:
        self.selection_errors = []
        seen_same_entry: set[tuple[str, int]] = set()
        for relation in self.selection_constraints:
            try:
                if relation["type"] == "same_entry":
                    for set_index in relation["sets"]:
                        key = (relation["category"], set_index)
                        if key in seen_same_entry:
                            raise ValueError(f"{relation['category']}[{set_index}] belongs to multiple same_entry constraints")
                        seen_same_entry.add(key)
                self._validate_selection_constraint(relation)
                self.selection_errors.append(None)
            except ValueError as exc:
                self.selection_errors.append(str(exc))

    def _object_records(self) -> list[dict]:
        counters: dict[str, int] = {}
        records = []
        for object_set in self.object_sets:
            category = object_set["category"]
            set_index = counters.get(category, 0)
            counters[category] = set_index + 1
            entries = self.available[category]["entries"]
            entry_index = self._entry_index(category, set_index)
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
        if key not in self._relation_key_set():
            self.editor.scene.update_render()
            return
        self.apply_constraints({key})

    def _move_center_to(self, obj, x: float | None = None, y: float | None = None) -> None:
        aabb = _world_aabb(obj.entity)
        center = (aabb[0] + aabb[1]) * 0.5
        pose = obj.get_pose()
        nx = pose.p[0] if x is None else pose.p[0] + float(x) - float(center[0])
        ny = pose.p[1] if y is None else pose.p[1] + float(y) - float(center[1])
        obj.set_pose(sapien.Pose([nx, ny, pose.p[2]], pose.q))

    def _set_rotation(self, obj, axis) -> None:
        yaw = self._yaw_for_axis(obj, axis)
        pose = obj.get_pose()
        obj.set_pose(sapien.Pose(pose.p, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))

    def _yaw_for_axis(self, obj, axis) -> float:
        if not isinstance(axis, str):
            return float(axis)
        if axis in ("0", "90", "180", "270"):
            return np.radians(float(axis))
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

    def _nearest_right_angle(self, ref: dict) -> str:
        yaw = self._current_yaw_deg(ref)
        return str(int(round(yaw / 90.0) * 90) % 360)

    def select_scene_id(self, scene_id: str | None) -> None:
        self.clear_highlight()
        key = self.key_by_scene_id.get(scene_id or "")
        self.selected_keys = {key} if key else set()
        self._sync_selection()

    def select_at(self, x: int, y: int) -> None:
        self.editor.select_at(x, y)
        self.select_scene_id(self.editor.selected)

    def select_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.clear_highlight()
        self.editor.select_rect(x0, y0, x1, y1)
        self.selected_keys = {self.key_by_scene_id[sid] for sid in self.editor.selected_ids if sid in self.key_by_scene_id}
        self._sync_selection()

    def _sync_selection(self) -> None:
        self.editor.selected_ids = {self.scene_ids[k] for k in self.selected_keys if k in self.scene_ids}
        self.editor.selected = next(iter(self.editor.selected_ids), None)

    def clear_highlight(self) -> None:
        self.editor.clear_extra_outlines()

    def highlight_relation(self, index: int, mode: str = "layout") -> None:
        self.clear_highlight()
        self.selected_keys.clear()
        self._sync_selection()
        blue_keys: set[str] = set()
        yellow_keys: set[str] = set()
        if mode == "layout":
            relation = self.constraints[index]
            keys = self._relation_keys(relation)
            anchor = relation.get("anchor")
            anchor_key = _ref_key(anchor) if anchor else None
            yellow_keys = {anchor_key} if anchor_key else set()
            blue_keys = keys - yellow_keys
        elif mode == "selection":
            relation = self.selection_constraints[index]
            if relation["type"] == "same_entry":
                refs = [
                    record["ref"] for record in self._object_records()
                    if record["ref"]["category"] == relation["category"]
                    and record["ref"]["set"] in relation["sets"]
                ]
                blue_keys = {_ref_key(ref) for ref in refs}
            else:
                blue_keys = self._relation_keys(relation)
        else:
            raise ValueError(f"unknown relation mode: {mode}")
        colors = {
            self.scene_ids[key]: RELATION_OUTLINE_COLOR
            for key in blue_keys
            if key in self.scene_ids
        }
        colors.update({
            self.scene_ids[key]: OUTLINE_COLOR
            for key in yellow_keys
            if key in self.scene_ids
        })
        self.editor.set_extra_outlines(colors)

    def add_relation(self, relation_type: str) -> None:
        refs = [self._ref_from_key(k) for k in self._stable_keys(self.selected_keys)]
        if not refs:
            raise ValueError("select at least one object")
        relation = self._default_relation(relation_type, refs)
        self._init_relation_params_from_current(relation, None)
        self.constraints.append(relation)
        if reason := self._incomplete_reason(relation):
            self._resize_relation_status()
            self.relation_incomplete[-1] = reason
            return
        self.apply_constraints(set(self._relation_writes(relation)))

    def add_selection_constraint(self, relation_type: str) -> None:
        refs = [self._ref_from_key(k) for k in self._stable_keys(self.selected_keys)]
        if relation_type == "bbox_larger_than":
            if len(refs) != 2:
                raise ValueError("bbox_larger_than needs exactly two selected objects")
            first_area = self._asset_bbox_xy_area(self._entry_asset_id(refs[0]))
            second_area = self._asset_bbox_xy_area(self._entry_asset_id(refs[1]))
            larger, smaller = (refs[0], refs[1]) if first_area >= second_area else (refs[1], refs[0])
            relation = {"type": relation_type, "objects": refs, "larger": larger, "smaller": smaller}
        else:
            raise ValueError(f"unknown selection constraint type: {relation_type}")
        relation = self._normalize_selection_constraint(relation)
        self.selection_constraints.append(relation)
        self.apply_constraints()

    def update_relation(self, index: int, relation: dict) -> None:
        old = self.constraints[index]
        relation = self._normalize_relation(relation)
        self._init_relation_params_from_current(relation, old)
        self.constraints[index] = relation
        self.apply_constraints(set(self._relation_writes(self.constraints[index])))

    def _picked_ref_from_relation(self, relation: dict, scene_id: str | None) -> dict:
        key = self.key_by_scene_id.get(scene_id or "")
        if not key:
            raise ValueError("click an object")
        if key not in self._relation_keys(relation):
            raise ValueError("picked object must belong to this relation")
        return self._ref_from_key(key)

    def pick_relation_ref(self, index: int, field: str, scene_id: str | None, mode: str = "layout") -> None:
        if mode == "layout":
            relation = dict(self.constraints[index])
            relation[field] = self._picked_ref_from_relation(relation, scene_id)
            self.update_relation(index, relation)
            return
        if mode == "selection":
            relation = dict(self.selection_constraints[index])
            if relation["type"] != "bbox_larger_than" or field not in ("larger", "smaller"):
                raise ValueError(f"{relation['type']} has no pickable {field}")
            relation[field] = self._picked_ref_from_relation(relation, scene_id)
            self.update_selection_relation(index, relation)
            return
        raise ValueError(f"unknown relation mode: {mode}")

    def delete_relation(self, index: int) -> None:
        self.constraints.pop(index)
        self.apply_constraints()

    def delete_selection_relation(self, index: int) -> None:
        self.selection_constraints.pop(index)
        self.apply_constraints()

    def update_selection_relation(self, index: int, relation: dict) -> None:
        self.selection_constraints[index] = self._normalize_selection_constraint(relation)
        self.apply_constraints()

    def _resize_relation_status(self) -> None:
        self.relation_errors = (self.relation_errors + [None] * len(self.constraints))[:len(self.constraints)]
        self.relation_incomplete = (self.relation_incomplete + [None] * len(self.constraints))[:len(self.constraints)]

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
            return {"type": relation_type, "objects": refs, "target": first,
                    "axis": self._nearest_right_angle(first), "jitter_deg": 0}
        if relation_type in ("in_same_vertical_line", "in_same_horizontal_line", "evenly_spaced_from_anchor"):
            if len(refs) < 2:
                raise ValueError(f"{relation_type} needs at least two selected objects")
            relation = {"type": relation_type, "objects": refs}
            if relation_type == "evenly_spaced_from_anchor":
                relation["mode"] = "footprint"
            return relation
        if relation_type in ("x_offset_from", "y_offset_from", "xy_offset_from", "on_top_of"):
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
        if kind in ("in_same_vertical_line", "in_same_horizontal_line", "evenly_spaced_from_anchor", "x_offset_from", "y_offset_from", "xy_offset_from", "on_top_of"):
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
            if relation.get("anchor"):
                anchor_key = _ref_key(relation["anchor"])
                axis = relation["axis"]
                if self.fields.get(anchor_key, {}).get(axis) is None:
                    return f"define anchor {axis}"
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
        if r.get("type") == "align_axis":
            axis = r.get("axis", "0")
            if axis == "horizontal":
                r["axis"] = "0"
            elif axis == "vertical":
                r["axis"] = "90"
            elif axis not in ALIGN_AXES:
                try:
                    value = int(round(float(axis))) % 360
                except (TypeError, ValueError):
                    raise ValueError(f"unknown align axis: {axis}")
                if str(value) in ALIGN_AXES:
                    r["axis"] = str(value)
                else:
                    r["axis"] = "custom"
                    r["yaw_deg"] = value
        if "objects" not in r:
            r["objects"] = self._relation_refs(r)
        return r

    def apply_constraints(self, settle_keys: set[str] | None = None, use_jitter: bool = False) -> None:
        self._update_selection_errors()
        support_snapshot = self._pending_support_snapshot or self._make_support_snapshot()
        self._pending_support_snapshot = None
        self._active_support_snapshot = support_snapshot
        self.fields = {record["key"]: _empty_fields() for record in self._object_records()}
        self.relation_errors = [None] * len(self.constraints)
        self.relation_incomplete = [None] * len(self.constraints)
        settle_keys = settle_keys or set()
        try:
            deferred = self._defer_unconstrained_objects(settle_keys)
            parkable_keys = set(self._parkable_relation_keys_in_order())
            self._park_constrained_objects(parkable_keys)
            self._preapply_alignments(parkable_keys)
            for i, relation in enumerate(self.constraints):
                try:
                    writes = self._relation_writes(relation)
                    self._apply_relation(relation, use_jitter)
                    if relation["type"] == "pen_in_holder":
                        self._apply_pen_in_holder_preview(relation)
                    else:
                        self._apply_preview(set(writes), settle_keys)
                except IncompleteRelation as exc:
                    self.relation_incomplete[i] = str(exc)
                except ValueError as exc:
                    self.relation_errors[i] = str(exc)
            self._restore_unconstrained_objects(deferred, settle_keys)
        finally:
            self._active_support_snapshot = None

    def _defer_unconstrained_objects(self, settle_keys: set[str]) -> list[tuple[str, sapien.Pose]]:
        relation_keys = set(self._mentioned_keys_in_order())
        out = []
        for key, sid in self.scene_ids.items():
            if key in relation_keys:
                continue
            obj = self.editor.objects[sid]
            out.append((key, obj.get_pose()))
            self._park_object(obj, len(out))
        if out:
            self._sync_physics_frames()
        out.sort(key=lambda item: item[0] in settle_keys)
        return out

    def _park_constrained_objects(self, parkable_keys: set[str]) -> None:
        moved = False
        for index, key in enumerate(self._active_relation_keys_in_order(), start=1):
            if key not in parkable_keys:
                continue
            sid = self.scene_ids.get(key)
            if sid is None:
                continue
            self._park_object(self.editor.objects[sid], index)
            moved = True
        if moved:
            self._sync_physics_frames()

    def _park_object(self, obj, index: int) -> None:
        obj.set_pose(sapien.Pose(
            [20.0 + index, 20.0, self.editor.scene_wrap.table["height"] + 1.0],
            obj.get_pose().q,
        ))

    def _preapply_alignments(self, parkable_keys: set[str]) -> None:
        for relation in self.constraints:
            if relation.get("type") != "align_axis" or self._incomplete_reason(relation):
                continue
            key = _ref_key(self._target_ref(relation))
            sid = self._ensure_spawned(key)
            self._set_rotation(self.editor.objects[sid], self._align_axis_value(relation))
            if key in parkable_keys:
                self._park_object(self.editor.objects[sid], self.number_by_key.get(key, 1))

    def _restore_unconstrained_objects(self, deferred: list[tuple[str, sapien.Pose]], settle_keys: set[str]) -> None:
        for key, pose in deferred:
            sid = self.scene_ids.get(key)
            if sid is None:
                continue
            obj = self.editor.objects[sid]
            obj.set_pose(pose)
            self.editor._settle(obj)

    def _sync_physics_frames(self) -> None:
        for _ in range(3):
            self.editor.scene.step()
        self.editor.scene.update_render()

    def _make_support_snapshot(self) -> dict:
        aabbs = {
            key: _world_aabb(self.editor.objects[sid].entity)
            for key, sid in self.scene_ids.items()
            if sid in self.editor.objects
        }
        overlaps = set()
        keys = sorted(aabbs, key=lambda key: self.number_by_key.get(key, 10_000))
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if _xy_overlaps(aabbs[a], aabbs[b]):
                    overlaps.add(frozenset((a, b)))
        return {
            "bottom": {key: float(aabb[0][2]) for key, aabb in aabbs.items()},
            "overlaps": overlaps,
        }

    def _strict_support_edges(self) -> dict[frozenset[str], tuple[str, str]]:
        edges = {}
        for relation in self.constraints:
            kind = relation.get("type")
            try:
                if kind == "on_top_of" and not self._incomplete_reason(relation):
                    lower = _ref_key(relation["anchor"])
                    upper = _ref_key(self._single_target_from(relation, "anchor"))
                elif kind == "pen_in_holder" and not self._incomplete_reason(relation):
                    lower = _ref_key(relation["holder"])
                    upper = _ref_key(self._single_target_from(relation, "holder"))
                else:
                    continue
            except (KeyError, ValueError):
                continue
            pair = frozenset((lower, upper))
            previous = edges.get(pair)
            if previous is not None and previous != (lower, upper):
                raise ValueError(f"conflicting support relation between {self._label(lower)} and {self._label(upper)}")
            edges[pair] = (lower, upper)
        return edges

    def _support_order_graph(self, moving_keys: set[str]) -> dict[str, set[str]]:
        keys = {
            key for key, sid in self.scene_ids.items()
            if sid in self.editor.objects
        }
        graph = {key: set() for key in keys}
        aabbs = {key: _world_aabb(self.editor.objects[self.scene_ids[key]].entity) for key in keys}
        snapshot = self._active_support_snapshot or self._make_support_snapshot()
        old_bottom = snapshot["bottom"]
        old_overlaps = snapshot["overlaps"]
        strict_edges = self._strict_support_edges()
        ordered = sorted(keys, key=lambda key: self.number_by_key.get(key, 10_000))
        eps = 1e-3
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if not _xy_overlaps(aabbs[a], aabbs[b]):
                    continue
                pair = frozenset((a, b))
                if pair in strict_edges:
                    lower, upper = strict_edges[pair]
                elif pair in old_overlaps and abs(old_bottom.get(a, aabbs[a][0][2]) - old_bottom.get(b, aabbs[b][0][2])) > eps:
                    lower, upper = (a, b) if old_bottom.get(a, 0.0) < old_bottom.get(b, 0.0) else (b, a)
                elif (a in moving_keys) != (b in moving_keys):
                    lower, upper = (b, a) if a in moving_keys else (a, b)
                elif abs(old_bottom.get(a, aabbs[a][0][2]) - old_bottom.get(b, aabbs[b][0][2])) > eps:
                    lower, upper = (a, b) if old_bottom.get(a, 0.0) < old_bottom.get(b, 0.0) else (b, a)
                else:
                    lower, upper = (a, b)
                graph[lower].add(upper)
        return graph

    def _topological_support_order(self, graph: dict[str, set[str]]) -> list[str]:
        indegree = {key: 0 for key in graph}
        for targets in graph.values():
            for target in targets:
                indegree[target] += 1
        ready = sorted(
            (key for key, degree in indegree.items() if degree == 0),
            key=lambda key: self.number_by_key.get(key, 10_000),
        )
        out = []
        while ready:
            key = ready.pop(0)
            out.append(key)
            for target in sorted(graph[key], key=lambda item: self.number_by_key.get(item, 10_000)):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=lambda item: self.number_by_key.get(item, 10_000))
        if len(out) != len(graph):
            cycle = [self._label(key) for key, degree in indegree.items() if degree > 0]
            raise ValueError(f"support order cycle: {', '.join(cycle)}")
        return out

    def _settle_keys_by_support_order(self, keys: set[str], moving_keys: set[str]) -> None:
        keys = {key for key in keys if key in self.scene_ids}
        if not keys:
            return
        graph = self._support_order_graph(moving_keys)
        reverse = {key: set() for key in graph}
        for lower, uppers in graph.items():
            for upper in uppers:
                reverse[upper].add(lower)
        for key in self._topological_support_order(graph):
            if key not in keys or key not in self.scene_ids:
                continue
            keep = reverse.get(key, set()) | {key}
            parked = []
            for other_key, sid in self.scene_ids.items():
                if other_key in keep or sid not in self.editor.objects:
                    continue
                obj = self.editor.objects[sid]
                parked.append((obj, obj.get_pose()))
                self._park_object(obj, len(parked))
            try:
                self.editor._settle(self.editor.objects[self.scene_ids[key]])
            finally:
                for obj, pose in parked:
                    obj.set_pose(pose)
                if parked:
                    self._sync_physics_frames()

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

    def _align_axis_value(self, relation: dict, use_jitter: bool = False):
        axis = str(relation.get("axis", "0"))
        if axis == "custom":
            yaw_deg = float(relation.get("yaw_deg", 0)) + self._jitter(relation, "jitter_deg", use_jitter)
            return np.radians(yaw_deg)
        if axis in ("0", "90", "180", "270"):
            yaw_deg = float(axis) + self._jitter(relation, "jitter_deg", use_jitter)
            return np.radians(yaw_deg)
        return axis

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
            self._write(_ref_key(self._target_ref(relation)), "rotation", self._align_axis_value(relation, use_jitter))
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
        elif kind == "on_top_of":
            self._single_target_from(relation, "anchor")
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
        changed_keys = set()
        for key in keys:
            sid = self._ensure_spawned(key)
            obj = self.editor.objects[sid]
            fields = self.fields[key]
            before_aabb = _world_aabb(obj.entity)
            before_center = (before_aabb[0] + before_aabb[1]) * 0.5
            before_q = np.asarray(obj.get_pose().q)
            changed = False
            if fields["rotation"] is not None:
                self._set_rotation(obj, fields["rotation"])
            x = None if fields["x"] is None else float(fields["x"]) * table["length"] * 0.5
            y = None if fields["y"] is None else float(fields["y"]) * table["width"] * 0.5
            if x is not None or y is not None:
                self._move_center_to(obj, x, y)
            after_aabb = _world_aabb(obj.entity)
            after_center = (after_aabb[0] + after_aabb[1]) * 0.5
            after_q = np.asarray(obj.get_pose().q)
            changed = (
                np.linalg.norm(after_center[:2] - before_center[:2]) > 1e-5
                or np.linalg.norm(after_q - before_q) > 1e-5
            )
            if changed:
                changed_keys.add(key)
        self._settle_keys_by_support_order(changed_keys | (set(keys) & settle_keys), changed_keys | settle_keys)
        self.editor.scene.update_render()

    def _apply_pen_in_holder_preview(self, relation: dict) -> None:
        holder_key = _ref_key(relation["holder"])
        target_key = _ref_key(self._single_target_from(relation, "holder"))
        holder_sid = self._ensure_spawned(holder_key)
        target_sid = self._ensure_spawned(target_key)
        self.editor.pen_in_holder(target_sid, holder_sid, select=False)
        self.editor.scene.update_render()

    def key(self, name: str, fine: bool = False, relation_index: int | None = None) -> None:
        if name not in "wasdqe":
            return
        selected = [k for k in sorted(self.selected_keys, key=lambda k: self.number_by_key[k]) if k in self.scene_ids]
        if not selected:
            return
        self._pending_support_snapshot = self._make_support_snapshot()
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

        if relation_index is None and not (set(editable) & self._relation_key_set()):
            for key in editable:
                self.editor._settle(self.editor.objects[self.scene_ids[key]])
            self.editor.scene.update_render()
            return

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
            for ref in self._relation_refs_for_load_order(relation):
                key = _ref_key(ref)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def _active_relation_keys_in_order(self) -> list[str]:
        keys = []
        seen = set()
        for relation in self.constraints:
            try:
                if not self._relation_writes(relation):
                    continue
            except ValueError:
                pass
            for ref in self._relation_refs_for_load_order(relation):
                key = _ref_key(ref)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def _parkable_relation_keys_in_order(self) -> list[str]:
        writes_by_key: dict[str, set[str]] = {}
        for relation in self.constraints:
            try:
                writes = self._relation_writes(relation)
            except ValueError:
                continue
            for key, fields in writes.items():
                writes_by_key.setdefault(key, set()).update(fields)
        return [
            key for key in self._active_relation_keys_in_order()
            if {"x", "y"} <= writes_by_key.get(key, set())
        ]

    def _relation_refs_for_load_order(self, relation: dict) -> list[dict]:
        if relation.get("type") == "on_top_of" and relation.get("anchor"):
            return [relation["anchor"], *self._targets_from_relation(relation, "anchor")]
        if relation.get("type") == "pen_in_holder" and relation.get("holder"):
            return [relation["holder"], *self._targets_from_relation(relation, "holder")]
        return self._relation_refs(relation)

    def _relation_key_set(self) -> set[str]:
        return set(self._active_relation_keys_in_order())

    def _label(self, key: str) -> str:
        return str(self.number_by_key.get(key, key))

    def state(self) -> dict:
        records = []
        for record in self._object_records():
            key = record["key"]
            fields = self.fields.get(key, _empty_fields())
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
            "user_note": self.user_note,
            "user_prompt": self.user_prompt,
            "scene_description": self.scene_description,
            "available_categories": list(self.available),
            "objects": records,
            "sample_entry_index": self.sample_entry_index,
            "selection_constraints": self.selection_constraints,
            "selection_errors": self.selection_errors,
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
    return {"sources": SOURCES, "tags": TAGS, "scenarios": _list_scenarios()}


@app.get("/assets")
def assets(search: str = "", tag: str = "", source: str = ""):
    search = search.lower()
    out = []
    for asset in LIBRARY:
        if asset.id not in ENABLED_ASSET_IDS:
            continue
        search_blob = " ".join([asset.id, asset.label, *asset.tags]).lower()
        if search and search not in search_blob:
            continue
        if tag and tag not in asset.tags:
            continue
        if source and asset.source != source:
            continue
        out.append({"id": asset.id, "label": asset.label, "source": asset.source, "tags": list(asset.tags)})
    return out


@app.get("/preview")
def preview(asset_id: str):
    with GPU:
        body = previews.image_bytes(asset_id)
    return Response(body, media_type="image/png")


@app.get("/available_assets")
def load_available_assets(scenario: str, variation: str):
    data = _read_global_available_assets()
    data["assets_group"] = _read_variation_asset_group_ids(scenario, variation)
    return data


@app.post("/available_assets")
def save_available_assets(scenario: str, variation: str, payload: dict = Body(...)):
    data = _normalize_available_assets(payload)
    group_ids = [str(group_id) for group_id in payload.get("assets_group", [])]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("assets_group must not contain duplicates")
    missing = [group_id for group_id in group_ids if group_id not in data["available_assets"]]
    if missing:
        raise ValueError(f"unknown assets_group: {missing[0]}")
    GLOBAL_AVAILABLE_ASSETS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    variation_path = _write_variation_asset_group_ids(scenario, variation, group_ids)
    return {
        "saved": str(GLOBAL_AVAILABLE_ASSETS_PATH),
        "variation_saved": variation_path,
        "category_count": len(data["available_assets"]),
        "enabled_count": len(group_ids),
    }


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
                    elif kind == "save_user_note":
                        path = studio.save_user_note(msg["value"])
                        await socket.send_json({"type": "saved", "path": path})
                    elif kind == "save_user_prompt":
                        path = studio.save_user_prompt(msg["value"])
                        await socket.send_json({"type": "saved", "path": path})
                    elif kind == "new_template":
                        path = studio.new_template(msg["name"])
                        await socket.send_json({"type": "new_template", "path": path})
                    elif kind == "randomize":
                        studio.randomize_sets()
                        await socket.send_json({"type": "randomized"})
                    elif kind == "add_set":
                        studio.add_set(msg["category"])
                    elif kind == "clone_set":
                        studio.clone_set(msg["category"], int(msg["set"]))
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
                    elif kind == "add_selection_constraint":
                        studio.add_selection_constraint(msg["relation"])
                    elif kind == "pick_relation_ref":
                        scene_id = studio.editor.scene_id_at(int(msg["x"]), int(msg["y"]))
                        studio.pick_relation_ref(int(msg["index"]), msg["field"], scene_id, msg.get("mode", "layout"))
                    elif kind == "update_relation":
                        studio.update_relation(int(msg["index"]), msg["relation"])
                    elif kind == "update_selection_relation":
                        studio.update_selection_relation(int(msg["index"]), msg["relation"])
                    elif kind == "delete_relation":
                        studio.delete_relation(int(msg["index"]))
                    elif kind == "delete_selection_relation":
                        studio.delete_selection_relation(int(msg["index"]))
                    elif kind == "highlight_relation":
                        studio.highlight_relation(int(msg["index"]), msg.get("mode", "layout"))
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
