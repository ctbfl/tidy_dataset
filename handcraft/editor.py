# Interactive scene editor: owns one TidyScene, places/moves/selects/deletes
# objects, and renders the RGB view with a yellow outline on the selected object.

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import sapien.core as sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulations"))

from scene import LIBRARY, add_camera, create_scene  # noqa: E402
from objects import spawn  # noqa: E402

TRANS_STEP = 0.02            # m per WASD press
YAW_STEP = np.radians(15)    # rad per Q/E press
OUTLINE_COLOR = (255, 220, 0)


def _world_aabb_min_z(entity) -> float:
    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    return float(np.asarray(body.compute_global_aabb_tight())[0][2])


class SceneEditor:
    def __init__(self, use_hdri=False):
        self.scene_wrap = create_scene(headless=True, use_hdri=use_hdri)
        self.scene = self.scene_wrap.scene
        self.camera = add_camera(self.scene_wrap)
        self.selected: str | None = None
        self._seg = self._pos = self._model = None  # filled by render()
        self._counter = 0  # monotonic; gives every spawn a unique scene id
        # scenario annotation state (set by load_scene_dict)
        self.scenario: str | None = None
        self.scene_id: str | None = None
        self.template: str | None = None
        self.manifest: list[dict] = []          # [{slot, role, asset_id}, ...]
        self.slot_of: dict[str, str | None] = {}  # scene_id -> manifest slot (None = extra)

    @property
    def objects(self) -> dict:
        return self.scene_wrap.objects

    # -- rendering -------------------------------------------------------- #
    def render(self) -> np.ndarray:
        self.scene.update_render()
        self.camera.take_picture()
        rgb = (self.camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        self._seg = self.camera.get_picture("Segmentation")[..., 1]
        self._pos = self.camera.get_picture("Position")
        self._model = np.asarray(self.camera.get_model_matrix())
        if self.selected:
            psid = self.objects[self.selected].entity.per_scene_id
            mask = (self._seg == psid).astype(np.uint8)
            if mask.any():
                edge = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8))
                rgb[edge > 0] = OUTLINE_COLOR
        return rgb

    def _world_point(self, x: int, y: int):
        view = self._pos[y, x]
        if not view[:3].any():  # background
            return None
        return (self._model @ np.array([view[0], view[1], view[2], 1.0]))[:3]

    # -- state for the client -------------------------------------------- #
    def state(self) -> dict:
        placed = {s for s in self.slot_of.values() if s}
        return {"objects": list(self.objects), "selected": self.selected,
                "background": self.background_state(),
                "scenario": self.scenario, "scene_id": self.scene_id,
                "slots": {oid: self.slot_of.get(oid) for oid in self.objects},
                "manifest": [{**m, "placed": m["slot"] in placed} for m in self.manifest]}

    def background_state(self) -> dict:
        ts = self.scene_wrap
        return {"table": ts.table,
                "table_texture": getattr(ts, "table_texture_id", None),
                "wall_texture": getattr(ts, "wall_texture_id", None)}

    def scene_dict(self) -> dict:
        ts = self.scene_wrap
        return {
            "version": 2,
            "scenario": self.scenario,
            "scene_id": self.scene_id,
            "template": self.template,
            "table": ts.table,
            "table_texture": getattr(ts, "table_texture_id", None),
            "wall_texture": getattr(ts, "wall_texture_id", None),
            "manifest": self.manifest,
            "items": [{"slot": self.slot_of.get(o.id), "asset_id": o.asset.id,
                       "transform": o.get_pose().to_transformation_matrix().tolist()}
                      for o in self.objects.values()],
        }

    def clear(self) -> None:
        """Remove placed objects but keep the manifest, so the same scene can be
        re-annotated from scratch."""
        for o in list(self.objects.values()):
            self.scene.remove_entity(o.entity)
        self.objects.clear()
        self.slot_of.clear()
        self.selected = None

    # -- background (table dims + textures) ------------------------------- #
    def rebuild_background(self, table=None, table_texture_id=None, wall_texture_id=None,
                           random_background=False) -> None:
        """Rebuild table + wall in place (the session keeps one scene/renderer).
        random_background=True ignores the manual texture ids."""
        ts = self.scene_wrap
        dims = table or ts.table
        ts.random_background = random_background
        ts.robotwin_create_table_and_wall(
            table_length=dims["length"], table_width=dims["width"],
            table_height=dims["height"], table_thickness=dims["thickness"],
            table_texture_id=table_texture_id, wall_texture_id=wall_texture_id,
        )
        ts.table = dict(dims)
        self.scene.update_render()

    def randomize_background(self) -> None:
        self.rebuild_background(random_background=True)

    def set_background(self, table_texture_id=None, wall_texture_id=None) -> None:
        self.rebuild_background(table_texture_id=table_texture_id,
                                wall_texture_id=wall_texture_id, random_background=False)

    def load_scene_dict(self, data: dict) -> None:
        """Replace the scene with a saved dict. v1 = table + textures + items;
        v2 adds scenario/scene_id/template + a manifest, and items carry a slot."""
        version = data.get("version")
        if version not in (1, 2):
            raise ValueError(f"unsupported scene version: {version!r} (expected 1 or 2)")
        self.clear()
        self.scenario = data.get("scenario")
        self.scene_id = data.get("scene_id")
        self.template = data.get("template")
        self.manifest = list(data.get("manifest", []))
        self.rebuild_background(
            table=data.get("table"),
            table_texture_id=data.get("table_texture"),
            wall_texture_id=data.get("wall_texture"),
            random_background=False,
        )
        for item in data.get("items", []):
            self._counter += 1
            obj = spawn(self.scene, LIBRARY[item["asset_id"]], f"{item['asset_id']}#{self._counter}")
            self.objects[obj.id] = obj
            self.slot_of[obj.id] = item.get("slot")
            obj.set_pose(sapien.Pose(np.asarray(item["transform"], dtype=float)))
        self.scene.update_render()

    # -- editing ---------------------------------------------------------- #
    def place(self, asset_id: str, x: int, y: int, slot: str | None = None) -> str | None:
        point = self._world_point(x, y)
        if point is None:
            return None
        self._counter += 1
        obj = spawn(self.scene, LIBRARY[asset_id], f"{asset_id}#{self._counter}")
        self.objects[obj.id] = obj
        self.slot_of[obj.id] = slot
        obj.set_pose(sapien.Pose([point[0], point[1], point[2]], [1, 0, 0, 0]))
        self.scene.update_render()
        pose = obj.get_pose()  # rest the bottom on the clicked surface
        obj.set_pose(sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + point[2] - _world_aabb_min_z(obj.entity)], pose.q))
        self.selected = obj.id
        return obj.id

    def place_slot(self, slot: str, x: int, y: int) -> str | None:
        """Place the asset a manifest slot points at. No-op if the slot is
        already satisfied or unknown."""
        if slot in self.slot_of.values():
            return None
        entry = next((m for m in self.manifest if m["slot"] == slot), None)
        if entry is None:
            return None
        return self.place(entry["asset_id"], x, y, slot=slot)

    def select(self, scene_id: str) -> None:
        self.selected = scene_id if scene_id in self.objects else None

    def select_at(self, x: int, y: int) -> None:
        psid = int(self._seg[y, x])
        self.selected = next((sid for sid, o in self.objects.items() if o.entity.per_scene_id == psid), None)

    def delete(self, scene_id: str) -> None:
        self.scene.remove_entity(self.objects.pop(scene_id).entity)
        self.slot_of.pop(scene_id, None)
        if self.selected == scene_id:
            self.selected = None

    def key(self, name: str) -> None:
        if not self.selected:
            return
        obj = self.objects[self.selected]
        pose = obj.get_pose()
        if name in "wasd":
            dx, dy = {"w": (0, TRANS_STEP), "s": (0, -TRANS_STEP), "a": (-TRANS_STEP, 0), "d": (TRANS_STEP, 0)}[name]
            obj.set_pose(sapien.Pose([pose.p[0] + dx, pose.p[1] + dy, pose.p[2]], pose.q))
        elif name in "qe":
            angle = YAW_STEP if name == "q" else -YAW_STEP
            spin = sapien.Pose(q=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
            obj.set_pose(sapien.Pose(pose.p, (spin * sapien.Pose(q=pose.q)).q))
