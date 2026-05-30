# Read-only access to the shared asset-preview cache.
#
# The asset browser (organize_it_v2/.../asset_brower_app) is the sole renderer and
# writer of previews: it renders each asset once (by asset_id) into
# <asset_library>/.preview_cache/ and re-renders when an asset's config changes.
# Handcraft only reads those PNGs; if one hasn't been rendered yet, it shows a
# neutral placeholder (open the asset browser to generate it).

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from scene import LIBRARY

CACHE_DIR = LIBRARY.root / ".preview_cache"


def _placeholder_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (235, 236, 238)).save(buf, format="PNG")
    return buf.getvalue()


class PreviewRenderer:
    def __init__(self):
        self._placeholder = _placeholder_png()

    def path(self, asset_id: str) -> Path:
        return CACHE_DIR / (asset_id.replace(":", "_").replace("/", "_") + ".png")

    def image_bytes(self, asset_id: str) -> bytes:
        p = self.path(asset_id)
        return p.read_bytes() if p.is_file() else self._placeholder
