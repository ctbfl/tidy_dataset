# Renders a thumbnail per asset (visual mesh, stable-upright, white background)
# in its own SAPIEN scene, cached on disk so each asset is rendered only once.

from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien.core as sapien
from PIL import Image

from editor import _world_aabb_min_z  # noqa: F401  (keeps the AABB helper in one place)
from scene import LIBRARY, look_at
from objects import spawn

CACHE_DIR = Path("/tmp/tidy_previews")


class PreviewRenderer:
    def __init__(self, size=256):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.scene = sapien.Scene()
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light([-1, -1, -1], [1.0, 1.0, 1.0])
        self.scene.add_directional_light([1, 1, -0.5], [0.4, 0.4, 0.4])
        self.camera = self.scene.add_camera("preview", size, size, fovy=0.7, near=0.01, far=50)

    def path(self, asset_id: str) -> Path:
        return CACHE_DIR / (asset_id.replace(":", "_").replace("/", "_") + ".png")

    def render(self, asset_id: str) -> Path:
        out = self.path(asset_id)
        if out.exists():
            return out
        obj = spawn(self.scene, LIBRARY[asset_id], asset_id)
        obj.set_pose(sapien.Pose())
        self.scene.update_render()
        body = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        lo, hi = np.asarray(body.compute_global_aabb_tight())
        center = (lo + hi) / 2
        radius = np.linalg.norm(hi - lo) / 2 + 1e-3
        self.camera.set_local_pose(look_at(center + np.array([1.0, 1.0, 0.8]) * radius * 2.2, center))
        self.scene.update_render()
        self.camera.take_picture()
        rgb = (self.camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        rgb[self.camera.get_picture("Segmentation")[..., 1] == 0] = 255  # white background
        self.scene.remove_entity(obj.entity)
        Image.fromarray(rgb).save(out)
        return out
