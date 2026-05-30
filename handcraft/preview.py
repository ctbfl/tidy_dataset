# Shared asset-preview cache, read-first with a SAPIEN fallback.
#
# The asset browser (organize_it_v2/.../asset_brower_app) is the authoritative
# renderer: it writes previews (by asset_id, plus a .key fingerprint) into
# <asset_library>/.preview_cache/ and re-renders when an asset's config changes.
#
# Handcraft reads that cache first. On a miss it renders the asset itself with
# SAPIEN (which is stable here, unlike the browser's Open3D) and writes the PNG
# into the same cache so the thumbnail shows up immediately without opening the
# browser. It does not write a .key, so the browser still re-renders its own
# authoritative version later.

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import sapien.core as sapien
from PIL import Image

from scene import LIBRARY, look_at
from objects import spawn

CACHE_DIR = LIBRARY.root / ".preview_cache"


def _placeholder_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (235, 236, 238)).save(buf, format="PNG")
    return buf.getvalue()


class PreviewRenderer:
    def __init__(self, size=256):
        self._size = size
        self._placeholder = _placeholder_png()
        self._lock = threading.Lock()
        self._scene = None
        self._camera = None

    def path(self, asset_id: str) -> Path:
        return CACHE_DIR / (asset_id.replace(":", "_").replace("/", "_") + ".png")

    def image_bytes(self, asset_id: str) -> bytes:
        out = self.path(asset_id)
        if out.is_file():
            return out.read_bytes()
        try:
            return self._render(asset_id, out)
        except Exception:
            return self._placeholder

    def _ensure_scene(self):
        if self._scene is None:
            scene = sapien.Scene()
            scene.set_ambient_light([0.4, 0.4, 0.4])
            scene.add_directional_light([-1, -1, -1], [1.0, 1.0, 1.0])
            scene.add_directional_light([1, 1, -0.5], [0.4, 0.4, 0.4])
            self._camera = scene.add_camera("preview", self._size, self._size, fovy=0.7, near=0.01, far=50)
            self._scene = scene
        return self._scene, self._camera

    def _render(self, asset_id: str, out: Path) -> bytes:
        with self._lock:
            if out.is_file():  # filled while we waited for the lock
                return out.read_bytes()
            scene, camera = self._ensure_scene()
            obj = spawn(scene, LIBRARY[asset_id], asset_id)
            try:
                obj.set_pose(sapien.Pose())
                scene.update_render()
                body = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
                lo, hi = np.asarray(body.compute_global_aabb_tight())
                center = (lo + hi) / 2
                radius = np.linalg.norm(hi - lo) / 2 + 1e-3
                camera.set_local_pose(look_at(center + np.array([1.0, 1.0, 0.8]) * radius * 2.2, center))
                scene.update_render()
                camera.take_picture()
                rgb = (camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
                rgb[camera.get_picture("Segmentation")[..., 1] == 0] = 255  # white background
            finally:
                scene.remove_entity(obj.entity)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(out)
            return out.read_bytes()
