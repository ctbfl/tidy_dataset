from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ASSET_ROOT = REPO_ROOT / "RoboTwin" / "assets"
GSO_ASSET_ROOT = Path("~/assets/gso").expanduser()
ASSET_LIBRARY_ROOT = (Path(__file__).resolve().parents[1] / "SAMPLE_ASSET_LIBRARY").resolve()
ASSET_LIBRARY_CATALOG = ASSET_LIBRARY_ROOT / "catalog.json"
ASSET_ROOTS = {
    "asset_library": ASSET_LIBRARY_ROOT,
    "repo": REPO_ROOT,
    "robotwin": ROBOTWIN_ASSET_ROOT,
    "gso": GSO_ASSET_ROOT,
}
ASSET_PAGE_SIZE = 20
MAX_ASSET_PAGE_SIZE = 50

# Previews cache inside the asset library (named by asset_id, overwritten when an
# asset's config changes) so the sim/handcraft tools can read them too. Reassigned
# for the real library in configure_asset_library_root(); browser is the sole writer.
ASSET_PREVIEW_CACHE_DIR = ASSET_LIBRARY_ROOT / ".preview_cache"

SCENE_TAGS = ("Kitchen", "Tools", "Desk")
DEFAULT_OBJECT_TAGS = tuple(sorted(("bowl", "plate", "cup"), key=str.lower))


def configure_asset_library_root(path: str | Path) -> None:
    global ASSET_LIBRARY_ROOT, ASSET_LIBRARY_CATALOG, ASSET_ROOTS, ASSET_PREVIEW_CACHE_DIR

    ASSET_LIBRARY_ROOT = Path(path).expanduser().resolve()
    ASSET_LIBRARY_CATALOG = ASSET_LIBRARY_ROOT / "catalog.json"
    ASSET_PREVIEW_CACHE_DIR = ASSET_LIBRARY_ROOT / ".preview_cache"
    ASSET_ROOTS = {
        "asset_library": ASSET_LIBRARY_ROOT,
        "repo": REPO_ROOT,
        "robotwin": ROBOTWIN_ASSET_ROOT,
        "gso": GSO_ASSET_ROOT,
    }
