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
from objects import Asset, spawn  # noqa: E402

TRANS_STEP = 0.01            # m per WASD press (coarse)
TRANS_FINE = 0.002           # m per Shift+WASD press (fine — for centering a bowl on a plate)
YAW_STEP = np.radians(15)    # rad per Q/E press (coarse)
YAW_FINE = np.radians(5)     # rad per Shift+Q/E press (fine)
SETTLE_STEPS = 400           # max physics sub-steps when dropping an object to rest
SETTLE_REST_V = 1e-3         # m/s below which the falling object counts as settled
SETTLE_DROP_CLEARANCE = 0.02
OUTLINE_COLOR = (255, 220, 0)
RELATION_OUTLINE_COLOR = (80, 170, 255)


def _world_aabb(entity) -> np.ndarray:
    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    return np.asarray(body.compute_global_aabb_tight())  # [[min x,y,z], [max x,y,z]]


def _world_aabb_min_z(entity) -> float:
    return float(_world_aabb(entity)[0][2])


def _xy_overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a[0][0] <= b[1][0] and a[1][0] >= b[0][0]
                and a[0][1] <= b[1][1] and a[1][1] >= b[0][1])


def _aabb_corners(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (mn[0], mx[0])
                     for y in (mn[1], mx[1])
                     for z in (mn[2], mx[2])], dtype=float)


def _dynamic(entity):
    """The rigid-body component we toggle for the vertical settle (None for articulations)."""
    return entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)


def _uses_nonconvex_collision(obj) -> bool:
    shape = obj.asset.collision_shape
    return shape == "nonconvex" or (not shape and "holder" in {tag.lower() for tag in obj.asset.tags})


class SceneEditor:
    def __init__(self, use_hdri=False, camera_width=1024, camera_height=768):
        self.scene_wrap = create_scene(headless=True, use_hdri=use_hdri)
        self.scene = self.scene_wrap.scene
        self.camera = add_camera(self.scene_wrap, width=camera_width, height=camera_height)
        self.selected: str | None = None
        self.selected_ids: set[str] = set()
        self.extra_outline_colors: dict[str, tuple[int, int, int]] = {}
        self._seg = self._pos = self._model = None  # filled by render()
        self._counter = 0  # monotonic; gives every spawn a unique scene id
        # scenario annotation state (set by load_scene_dict)
        self.scenario: str | None = None
        self.scene_id: str | None = None
        self.arrangement: str | None = None  # "tidy" / "messy" / ... (== the file stem)
        self.template: str | None = None
        self.manifest: list[dict] = []          # [{slot, role, asset_id}, ...]
        self.slot_of: dict[str, str | None] = {}  # scene_id -> manifest slot (None = extra)
        self.pending_asset_scales: dict[str, tuple[float, float, float]] = {}
        self._special_relation: dict | None = None

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
        outline_colors = dict(self.extra_outline_colors)
        for sid in self.selected_ids:
            outline_colors[sid] = OUTLINE_COLOR
        for sid, color in outline_colors.items():
            if sid not in self.objects:
                continue
            psid = self.objects[sid].entity.per_scene_id
            mask = (self._seg == psid).astype(np.uint8)
            if mask.any():
                edge = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8))
                rgb[edge > 0] = color
        return rgb

    def set_extra_outlines(self, colors: dict[str, tuple[int, int, int]]) -> None:
        self.extra_outline_colors = dict(colors)

    def clear_extra_outlines(self) -> None:
        self.extra_outline_colors.clear()

    def _world_point(self, x: int, y: int):
        view = self._pos[y, x]
        if not view[:3].any():  # background
            return None
        return (self._model @ np.array([view[0], view[1], view[2], 1.0]))[:3]

    # -- state for the client -------------------------------------------- #
    def state(self) -> dict:
        placed = {s for s in self.slot_of.values() if s}
        return {"objects": list(self.objects), "selected": self.selected,
                "selected_ids": list(self.selected_ids),
                "background": self.background_state(),
                "size_target": self.size_target(),
                "scenario": self.scenario, "scene_id": self.scene_id,
                "arrangement": self.arrangement,
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
            "arrangement": self.arrangement,
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
        self.selected_ids.clear()
        self.pending_asset_scales.clear()
        self._special_relation = None

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
        self.arrangement = data.get("arrangement")
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
            asset_id = item["asset_id"]
            asset = Asset(LIBRARY[asset_id].handle, self.pending_asset_scales.get(asset_id))
            obj = spawn(self.scene, asset, f"{asset_id}#{self._counter}")
            self.objects[obj.id] = obj
            self.slot_of[obj.id] = item.get("slot")
            self._freeze(obj)  # loaded poses are already final — keep them as frozen colliders
            obj.set_pose(sapien.Pose(np.asarray(item["transform"], dtype=float)))
        self.scene.update_render()

    def settle_all(self) -> None:
        """Re-rest loaded objects after asset geometry changes.

        This is order-dependent and only uses geometric XY overlap to infer support.
        It does not know constraint semantics such as explicit on_top_of edges, so do
        not use it after constraint solving/sampling.
        """
        for obj in list(self.objects.values()):
            self._settle(obj)

    def size_target(self) -> dict | None:
        if not self.selected or self.selected not in self.objects:
            return None
        obj = self.objects[self.selected]
        base = np.asarray(LIBRARY[obj.asset.id].scale, dtype=float)
        scale = np.asarray(obj.asset.scale, dtype=float)
        factor = float(np.mean(scale / base))
        return {"scene_id": self.selected, "asset_id": obj.asset.id,
                "scale": scale.tolist(), "factor": factor}

    def set_selected_asset_scale_factor(self, factor: float) -> None:
        if not self.selected or self.selected not in self.objects:
            return
        if factor <= 0:
            raise ValueError(f"invalid scale factor: {factor}")
        asset_id = self.objects[self.selected].asset.id
        scale = tuple((np.asarray(LIBRARY[asset_id].scale, dtype=float) * factor).tolist())
        self.pending_asset_scales[asset_id] = scale
        for sid, obj in list(self.objects.items()):
            if obj.asset.id == asset_id:
                self._respawn_with_scale(sid, scale)
        self.scene.update_render()

    def _respawn_with_scale(self, scene_id: str, scale: tuple[float, float, float]) -> None:
        old = self.objects[scene_id]
        pose = old.get_pose()
        slot = self.slot_of.get(scene_id)
        self.scene.remove_entity(old.entity)
        obj = spawn(self.scene, Asset(LIBRARY[old.asset.id].handle, scale), scene_id)
        self.objects[scene_id] = obj
        self.slot_of[scene_id] = slot
        self._freeze(obj)
        obj.set_pose(pose)
        self._settle(obj)

    # -- physics: height auto-rest --------------------------------------- #
    def _freeze(self, obj) -> None:
        """Make an object an immovable (kinematic) collider. Placed objects are posed
        by hand, so by default nothing falls; only the object being settled is woken."""
        body = _dynamic(obj.entity)
        if body is not None:
            body.set_kinematic(True)

    def _settle(self, obj) -> None:
        """Drop the object straight down — x/y and orientation locked — until it rests
        on whatever its footprint overlaps: the table, or another object. This is what
        makes a bowl nestle into a plate and a fruit into a bowl, with the height (z)
        set automatically while the user keeps full control of x/y and yaw. Every other
        object stays frozen, so settling one never disturbs the rest of the scene."""
        body = _dynamic(obj.entity)
        if body is None:  # articulation/URDF link: leave its pose untouched
            return
        floor = self.scene_wrap.table["height"]            # table top: a hard lower bound
        support_top = self._support_top(obj)
        if _uses_nonconvex_collision(obj):
            self._rest_on_height(obj, support_top)
            return
        bottom = _world_aabb_min_z(obj.entity)
        if bottom < support_top + SETTLE_DROP_CLEARANCE:
            pose = obj.entity.get_pose()
            obj.entity.set_pose(sapien.Pose(
                [pose.p[0], pose.p[1], pose.p[2] + support_top + SETTLE_DROP_CLEARANCE - bottom],
                pose.q,
            ))
        offset = _world_aabb_min_z(obj.entity) - obj.entity.get_pose().p[2]  # bottom vs origin (yaw is locked)
        body.set_kinematic(False)
        body.set_locked_motion_axes([True, True, False, True, True, True])  # free Z only
        body.set_linear_velocity([0, 0, 0])
        body.set_angular_velocity([0, 0, 0])
        for i in range(SETTLE_STEPS):
            self.scene.step()
            if offset + obj.entity.get_pose().p[2] < floor - 1e-3:  # nothing under it -> stop before it drifts off
                break
            if i > 15 and float(np.linalg.norm(body.get_linear_velocity())) < SETTLE_REST_V:
                break
        body.set_locked_motion_axes([False] * 6)
        body.set_kinematic(True)
        bottom = _world_aabb_min_z(obj.entity)             # clamp: never let it sink below the table
        if bottom < floor:
            p = obj.entity.get_pose()
            obj.entity.set_pose(sapien.Pose([p.p[0], p.p[1], p.p[2] + floor - bottom], p.q))
        self.scene.update_render()

    def _rest_on_height(self, obj, z: float) -> None:
        bottom = _world_aabb_min_z(obj.entity)
        p = obj.entity.get_pose()
        obj.entity.set_pose(sapien.Pose([p.p[0], p.p[1], p.p[2] + z - bottom], p.q))
        self.scene.update_render()

    def _support_top(self, obj) -> float:
        obj_aabb = _world_aabb(obj.entity)
        top = float(self.scene_wrap.table["height"])
        for other in self.objects.values():
            if other is obj:
                continue
            other_aabb = _world_aabb(other.entity)
            if _xy_overlaps(obj_aabb, other_aabb):
                top = max(top, float(other_aabb[1][2]))
        return top

    def _free_settle(self, obj) -> None:
        body = _dynamic(obj.entity)
        if body is None:
            raise ValueError(f"{obj.id} is not a dynamic rigid object")
        body.set_kinematic(False)
        body.set_locked_motion_axes([False] * 6)
        body.set_linear_velocity([0, 0, 0])
        body.set_angular_velocity([0, 0, 0])
        for i in range(SETTLE_STEPS * 3):
            self.scene.step()
            lv = float(np.linalg.norm(body.get_linear_velocity()))
            av = float(np.linalg.norm(body.get_angular_velocity()))
            if i > 30 and lv < SETTLE_REST_V and av < SETTLE_REST_V:
                break
        body.set_linear_velocity([0, 0, 0])
        body.set_angular_velocity([0, 0, 0])
        body.set_kinematic(True)
        self.scene.update_render()

    def _stable_aabb(self, obj) -> tuple[np.ndarray, np.ndarray]:
        aabb = dict(obj.asset.handle.record.geometry.aabb_m)
        mn = np.asarray(aabb["min"], dtype=float)
        mx = np.asarray(aabb["max"], dtype=float)
        base_scale = np.asarray(obj.asset.handle.record.geometry.scale, dtype=float)
        scale = np.asarray(obj.asset.scale, dtype=float)
        factor = float(np.mean(scale / base_scale))
        return mn * factor, mx * factor

    def pen_in_holder(self, mover_id: str, holder_id: str) -> None:
        if mover_id == holder_id:
            raise ValueError("moving object and holder must be different")
        if mover_id not in self.objects:
            raise ValueError(f"unknown moving object: {mover_id}")
        if holder_id not in self.objects:
            raise ValueError(f"unknown holder object: {holder_id}")

        mover = self.objects[mover_id]
        holder = self.objects[holder_id]
        mn, mx = self._stable_aabb(mover)
        dims = np.sort(mx - mn)
        if dims[-2] <= 0 or dims[-1] / dims[-2] <= 2:
            raise ValueError(f"{mover_id} is not long and thin enough for pen_in_holder")

        holder_aabb = _world_aabb(holder.entity)
        target = np.array([
            (holder_aabb[0][0] + holder_aabb[1][0]) * 0.5,
            (holder_aabb[0][1] + holder_aabb[1][1]) * 0.5,
            holder_aabb[1][2],
        ], dtype=float)

        q = [float(np.sqrt(0.5)), float(np.sqrt(0.5)), 0.0, 0.0]  # stable +Y -> world +Z
        rot = sapien.Pose(q=q).to_transformation_matrix()[:3, :3]
        corners = (rot @ _aabb_corners(mn, mx).T).T
        bottom = corners[np.isclose(corners[:, 2], corners[:, 2].min())].mean(axis=0)
        mover.set_pose(sapien.Pose(target - bottom, q))
        self._free_settle(mover)
        self.selected = mover_id
        self.selected_ids = {mover_id}

    # -- editing ---------------------------------------------------------- #
    def place(self, asset_id: str, x: int, y: int, slot: str | None = None) -> str | None:
        point = self._world_point(x, y)
        if point is None:
            return None
        self._counter += 1
        asset = Asset(LIBRARY[asset_id].handle, self.pending_asset_scales.get(asset_id))
        obj = spawn(self.scene, asset, f"{asset_id}#{self._counter}")
        self.objects[obj.id] = obj
        self.slot_of[obj.id] = slot
        self._freeze(obj)
        obj.set_pose(sapien.Pose([point[0], point[1], point[2]], [1, 0, 0, 0]))
        self.scene.update_render()
        pose = obj.get_pose()  # rest the bottom on the clicked surface, then let it settle/nest
        obj.set_pose(sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + point[2] - _world_aabb_min_z(obj.entity)], pose.q))
        self._settle(obj)
        self.selected = obj.id
        self.selected_ids = {obj.id}
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
        self.selected_ids = {self.selected} if self.selected else set()

    def select_at(self, x: int, y: int) -> None:
        self.selected = self.scene_id_at(x, y)
        self.selected_ids = {self.selected} if self.selected else set()

    def scene_id_at(self, x: int, y: int) -> str | None:
        psid = int(self._seg[y, x])
        return next((sid for sid, o in self.objects.items() if o.entity.per_scene_id == psid), None)

    def start_special_relation(self, relation: str) -> None:
        if relation != "pen_in_holder":
            raise ValueError(f"unknown special relation: {relation}")
        self._special_relation = {"relation": relation, "ids": []}
        self.selected = None
        self.selected_ids.clear()

    def cancel_special_relation(self) -> None:
        self._special_relation = None

    def special_relation_active(self, relation: str) -> bool:
        return bool(self._special_relation and self._special_relation["relation"] == relation)

    def special_pick(self, relation: str, x: int, y: int) -> dict:
        if not self.special_relation_active(relation):
            self.start_special_relation(relation)
        sid = self.scene_id_at(x, y)
        if sid is None:
            raise ValueError("select an object")
        ids = self._special_relation["ids"]
        if ids and sid == ids[0]:
            raise ValueError("select a different holder object")
        ids.append(sid)
        self.selected = sid
        self.selected_ids = {sid}
        if len(ids) == 1:
            return {"done": False, "message": "Pen in Holder: now select the holder/container."}
        self._special_relation = None
        self.pen_in_holder(ids[0], ids[1])
        return {"done": True, "message": "Pen in Holder complete."}

    def select_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        x0 = min(max(0, x0), self._seg.shape[1] - 1)
        x1 = min(max(0, x1), self._seg.shape[1] - 1)
        y0 = min(max(0, y0), self._seg.shape[0] - 1)
        y1 = min(max(0, y1), self._seg.shape[0] - 1)
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        seg = self._seg[ymin:ymax + 1, xmin:xmax + 1]
        ids = set(int(i) for i in np.unique(seg) if int(i))
        selected = {sid for sid, o in self.objects.items() if o.entity.per_scene_id in ids}
        self.selected_ids = selected
        self.selected = next((sid for sid in self.objects if sid in selected), None)

    def delete(self, scene_id: str) -> None:
        self.scene.remove_entity(self.objects.pop(scene_id).entity)
        self.slot_of.pop(scene_id, None)
        if self.selected == scene_id:
            self.selected = None
        self.selected_ids.discard(scene_id)

    def key(self, name: str, fine: bool = False) -> None:
        selected = [self.objects[sid] for sid in self.selected_ids if sid in self.objects]
        if not selected:
            return
        if name in "wasd":
            step = TRANS_FINE if fine else TRANS_STEP
            dx, dy = {"w": (0, step), "s": (0, -step), "a": (-step, 0), "d": (step, 0)}[name]
            for obj in selected:
                pose = obj.get_pose()
                obj.set_pose(sapien.Pose([pose.p[0] + dx, pose.p[1] + dy, pose.p[2]], pose.q))
            for obj in selected:
                self._settle(obj)  # re-rest at the new x/y (height follows the support beneath it)
        elif name in "qe":
            angle = (YAW_FINE if fine else YAW_STEP) * (1 if name == "q" else -1)
            spin = sapien.Pose(q=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
            for obj in selected:
                pose = obj.get_pose()
                obj.set_pose(sapien.Pose(pose.p, (spin * sapien.Pose(q=pose.q)).q))
            for obj in selected:
                self._settle(obj)  # footprint changed — re-rest at the new yaw
