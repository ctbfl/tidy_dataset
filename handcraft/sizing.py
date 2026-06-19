# Asset sizing studio: one SAPIEN scene with a metric grid floor, three calibrated
# reference objects, and one target asset that the user scales UNIFORMLY back to a
# sensible real-world size. Calibration aids: visual anchors (the references), a 5/10cm
# grid floor, a live W*D*H readout in cm, and a "set this axis to N cm" control.
#
# Saving writes geometry.scale *= f and geometry.aabb_m *= f straight back into the
# shared asset.json (organize_it_v2/data/asset_library). Uniform scaling makes both
# exact, so no mesh / pybullet recompute is needed.

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import sapien.core as sapien
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulations"))

from scene import LIBRARY, look_at  # noqa: E402
from objects import Asset, spawn  # noqa: E402

ASSET_ROOT = LIBRARY.root  # .../organize_it_v2/data/asset_library

# The asset library is the single source of truth; we don't snapshot it. We only keep a
# plain append-only log of every scale edit, so there's a record of what changed.
LOG_PATH = Path(__file__).resolve().parent / "scale_edits.jsonl"

GRID_HALF = 0.60          # metric floor is 2*GRID_HALF m on a side
CELL_MINOR = 0.05         # 5 cm minor grid
CELL_MAJOR = 0.10         # 10 cm major grid (heavier line)
REF_Y = 0.0               # all objects sit on this row
REF_X0 = -0.55            # left edge of the first (smallest) reference
REF_GAP = 0.09            # gap between neighbours, m

DEFAULT_REFS = [
    {"asset_id": "gso:obj:ACE_Coffee_Mug_Kristen_16_oz_cup", "real_cm": 13.5},  # small mug
    {"asset_id": "robotwin:urdf:bottle:017", "real_cm": 16.3},                  # medium bottle
    {"asset_id": "objaverse:glb:table_lamp:1", "real_cm": 40.0},               # large lamp
]


def _grid_texture(px: int = 1400) -> np.ndarray:
    """RGBA grid: light board, 5cm minor + 10cm major lines, blue centre axes."""
    span = 2 * GRID_HALF
    im = Image.new("RGBA", (px, px), (236, 237, 240, 255))
    d = ImageDraw.Draw(im)
    n = int(round(span / CELL_MINOR))
    for k in range(n + 1):
        m = k * CELL_MINOR
        p = m / span * px
        major = abs((m % CELL_MAJOR)) < 1e-6 or abs((m % CELL_MAJOR) - CELL_MAJOR) < 1e-6
        col = (120, 122, 130, 255) if major else (203, 205, 210, 255)
        w = 3 if major else 1
        d.line([(p, 0), (p, px)], fill=col, width=w)
        d.line([(0, p), (px, p)], fill=col, width=w)
    c = GRID_HALF / span * px
    d.line([(c, 0), (c, px)], fill=(86, 138, 200, 255), width=3)
    d.line([(0, c), (px, c)], fill=(86, 138, 200, 255), width=3)
    return np.asarray(im)


class SizingStudio:
    def __init__(self) -> None:
        self.index = {e["asset_id"]: (ASSET_ROOT / e["asset_json"]).resolve()
                      for e in json.loads((ASSET_ROOT / "assets.json").read_text())["assets"]}
        self.refs: list[dict] = []      # [{asset_id, real_cm, entity, size_m}, ...]
        self._ref_right = REF_X0        # world-x of the right edge of the reference row
        self.target: dict | None = None  # {asset_id, asset, s0, size0, factor, entity}
        self._build_scene()
        self.set_refs(DEFAULT_REFS)

    # -- scene ------------------------------------------------------------- #
    def _build_scene(self) -> None:
        scene = sapien.Scene()
        scene.set_ambient_light([0.45, 0.45, 0.45])
        scene.add_directional_light([-1, -1, -1], [1.0, 1.0, 1.0], shadow=False)
        scene.add_directional_light([1, 0.5, -0.8], [0.5, 0.5, 0.5], shadow=False)
        scene.add_directional_light([0, 1, -0.6], [0.4, 0.4, 0.4], shadow=False)
        self.camera = scene.add_camera("sizing", 1100, 760, fovy=0.78, near=0.02, far=100)
        mat = sapien.render.RenderMaterial()
        mat.set_base_color_texture(sapien.render.RenderTexture2D(_grid_texture(), "R8G8B8A8Unorm", srgb=True))
        mat.base_color = [1, 1, 1, 1]
        mat.metallic = 0.0
        mat.roughness = 1.0
        b = scene.create_actor_builder()
        b.set_physx_body_type("static")
        b.add_box_visual(pose=sapien.Pose([0, 0, -0.004]), half_size=[GRID_HALF, GRID_HALF, 0.004], material=mat)
        self.floor = b.build(name="grid_floor")
        self.scene = scene

    def _aabb(self, entity) -> tuple[np.ndarray, np.ndarray]:
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        lo, hi = np.asarray(body.compute_global_aabb_tight())
        return lo, hi

    def _place(self, entity, left_x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Translate `entity` (already in its stable, upright pose at the origin) so its
        footprint left edge is at world x=left_x, centred on y, bottom resting on z=0.
        Returns its world aabb afterwards."""
        self.scene.update_render()
        lo, hi = self._aabb(entity)
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        hx = (hi[0] - lo[0]) / 2
        p = entity.get_pose()
        entity.set_pose(sapien.Pose([p.p[0] + (left_x + hx - cx), p.p[1] + (y - cy), p.p[2] - lo[2]], p.q))
        self.scene.update_render()
        return self._aabb(entity)

    # -- references -------------------------------------------------------- #
    def set_refs(self, refs: list[dict]) -> None:
        for r in self.refs:
            self.scene.remove_entity(r["entity"])
        self.refs = []
        x = REF_X0
        for spec in refs[:3]:
            try:
                asset = LIBRARY[spec["asset_id"]]
            except KeyError:
                continue
            obj = spawn(self.scene, asset, f"ref::{spec['asset_id']}")
            obj.set_pose(sapien.Pose())
            lo, hi = self._place(obj.entity, x, REF_Y)
            size = (hi - lo)
            self.refs.append({"asset_id": spec["asset_id"], "real_cm": float(spec.get("real_cm") or 0),
                              "entity": obj.entity, "size_m": size.tolist()})
            x = x + size[0] + REF_GAP
        self._ref_right = x
        if self.target:
            self._relayout_target()

    # -- target ------------------------------------------------------------ #
    def load_target(self, asset_id: str) -> None:
        asset = LIBRARY[asset_id]
        rec = json.loads(self.index[asset_id].read_text())
        g = rec.get("geometry", {})
        s0 = tuple(float(c) for c in g.get("scale", [1.0, 1.0, 1.0]))
        size0 = (g.get("aabb_m") or {}).get("size")
        if self.target:
            self.scene.remove_entity(self.target["entity"])
            self.target = None
        self.target = {"asset_id": asset_id, "asset": asset, "s0": s0,
                       "size0": list(map(float, size0)) if size0 else None,
                       "factor": 1.0, "entity": None}
        self._spawn_target()

    def _spawn_target(self) -> None:
        t = self.target
        if t["entity"] is not None:
            self.scene.remove_entity(t["entity"])
        f = t["factor"]
        scaled = dataclasses.replace(t["asset"], scale=tuple(c * f for c in t["s0"]))
        obj = spawn(self.scene, scaled, f"target::{t['asset_id']}")
        obj.set_pose(sapien.Pose())
        t["entity"] = obj.entity
        lo, hi = self._place(obj.entity, self._ref_right + REF_GAP, REF_Y)
        if t["size0"] is None:                       # asset.json had no aabb_m -> measure once
            t["size0"] = [float(v / f) for v in (hi - lo)]

    def _relayout_target(self) -> None:
        if self.target and self.target["entity"] is not None:
            self._place(self.target["entity"], self._ref_right + REF_GAP, REF_Y)

    def set_factor(self, f: float) -> None:
        if not self.target:
            return
        self.target["factor"] = float(min(max(f, 1e-3), 1e3))
        self._spawn_target()

    def set_absolute(self, axis, cm: float) -> None:
        """Scale so the chosen axis measures `cm` centimetres. axis: 0/1/2 or 'longest'."""
        if not self.target or not self.target["size0"]:
            return
        size0 = self.target["size0"]
        a = int(np.argmax(size0)) if axis == "longest" else int(axis)
        if size0[a] <= 0:
            return
        self.set_factor((cm / 100.0) / size0[a])

    def reset(self) -> None:
        self.set_factor(1.0)

    # -- save -------------------------------------------------------------- #
    def save(self) -> dict:
        t = self.target
        if not t:
            raise ValueError("no target loaded")
        f = t["factor"]
        path = self.index[t["asset_id"]]
        rec = json.loads(path.read_text())
        g = rec["geometry"]
        old_scale = [float(c) for c in g["scale"]]
        g["scale"] = [c * f for c in old_scale]
        ab = g.get("aabb_m")
        if ab:
            for k in ("min", "max", "aabb_center", "size"):
                if k in ab and ab[k] is not None:
                    ab[k] = [v * f for v in ab[k]]
            if ab.get("bottom_z") is not None:
                ab["bottom_z"] = ab["bottom_z"] * f
        path.write_text(json.dumps(rec, indent=2))
        dims_cm = [round(v * 100, 2) for v in (ab["size"] if ab and ab.get("size") else [])]
        self._log(t["asset_id"], old_scale, g["scale"], f, dims_cm)
        # fold the factor into the in-memory baseline so further edits compose cleanly
        new_s0 = tuple(c * f for c in t["s0"])
        t["s0"] = new_s0
        t["size0"] = [v * f for v in t["size0"]] if t["size0"] else None
        t["factor"] = 1.0
        t["asset"] = dataclasses.replace(t["asset"], scale=new_s0)
        LIBRARY.assets[t["asset_id"]] = t["asset"]
        self._spawn_target()
        return {"asset_id": t["asset_id"], "scale": list(new_s0),
                "dims_cm": [round(v * 100, 2) for v in (t["size0"] or [])], "path": str(path)}

    @staticmethod
    def _log(asset_id, old_scale, new_scale, factor, dims_cm) -> None:
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "asset_id": asset_id,
                 "old_scale": [round(c, 8) for c in old_scale],
                 "new_scale": [round(c, 8) for c in new_scale],
                 "factor": round(factor, 6), "dims_cm": dims_cm}
        with LOG_PATH.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- rendering / state ------------------------------------------------- #
    def render(self) -> np.ndarray:
        self.scene.update_render()
        self._frame_camera()
        self.scene.update_render()
        self.camera.take_picture()
        rgb = (self.camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        seg = self.camera.get_picture("Segmentation")[..., 1]
        rgb[seg == 0] = 250  # clean background; the floor keeps its own segmentation id
        return rgb

    def _frame_camera(self) -> None:
        pts = []
        for r in self.refs:
            lo, hi = self._aabb(r["entity"]); pts += [lo, hi]
        if self.target and self.target["entity"] is not None:
            lo, hi = self._aabb(self.target["entity"]); pts += [lo, hi]
        if pts:
            arr = np.array(pts); lo = arr.min(0); hi = arr.max(0)
            center = (lo + hi) / 2; radius = float(np.linalg.norm(hi - lo) / 2 + 1e-3)
        else:
            center = np.array([0.0, 0.0, 0.1]); radius = 0.4
        direction = np.array([0.05, -1.0, 0.5]); direction /= np.linalg.norm(direction)
        self.camera.set_local_pose(look_at(center + direction * radius * 2.6, center))

    def current_state(self) -> dict:
        t = self.target
        target = None
        if t:
            f = t["factor"]
            dims = [round(v * f * 100, 2) for v in t["size0"]] if t["size0"] else None
            target = {"asset_id": t["asset_id"], "source": t["asset"].source,
                      "tags": list(t["asset"].tags), "factor": round(f, 6),
                      "base_scale": [round(c, 8) for c in t["s0"]],
                      "scale": [round(c * f, 8) for c in t["s0"]],
                      "base_dims_cm": [round(v * 100, 2) for v in t["size0"]] if t["size0"] else None,
                      "dims_cm": dims}
        return {"target": target,
                "refs": [{"asset_id": r["asset_id"], "real_cm": r["real_cm"],
                          "dims_cm": [round(v * 100, 2) for v in r["size_m"]]} for r in self.refs]}
