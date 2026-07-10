from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, Response
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent
SIMULATIONS_DIR = HERE.parent / "simulations"
if str(SIMULATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATIONS_DIR))

from preview import PreviewRenderer
from scene import LIBRARY

DATASET_DIR = HERE.parent / "data" / "organize_it_dataset_v2"
CONFIG_NAME = "available_assets.json"
GPU = threading.Lock()

previews = PreviewRenderer()
SOURCES = sorted({a.source for a in LIBRARY})
TAGS = sorted({t for a in LIBRARY for t in a.tags})


def _variation_dir(scenario: str, variation: str) -> Path:
    path = (DATASET_DIR / Path(scenario).stem / Path(variation).stem).resolve()
    if path.parent.parent != DATASET_DIR.resolve():
        raise ValueError(f"invalid scenario/variation: {scenario}/{variation}")
    return path


def _config_path(scenario: str, variation: str) -> Path:
    return _variation_dir(scenario, variation) / "template" / CONFIG_NAME


def _list_scenarios() -> list[dict]:
    if not DATASET_DIR.is_dir():
        return []
    out = []
    for scenario_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        variations = [
            p.name for p in sorted(scenario_dir.iterdir())
            if p.is_dir() and (p / "template").is_dir()
        ]
        out.append({"scenario": scenario_dir.name, "variations": variations})
    return out


def _normalize_config(payload: dict) -> dict:
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
            if len(set(clean)) != len(clean):
                raise ValueError(f"{category_id}: duplicate asset_id inside one set")
            key = tuple(clean)
            if key in seen_entries:
                raise ValueError(f"{category_id}: duplicate asset set")
            seen_entries.add(key)
            entries.append(clean)

        categories[category_id] = {"slot_count": slot_count, "entries": entries}
    return {"version": 1, "available_assets": categories}


app = FastAPI(title="available assets editor")


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "available_assets.html").read_text()


@app.get("/meta")
def meta():
    return {"sources": SOURCES, "tags": TAGS, "scenarios": _list_scenarios()}


@app.get("/assets")
def assets(search: str = "", tag: str = "", source: str = ""):
    search = search.lower()
    out = []
    for a in LIBRARY:
        search_blob = " ".join([a.id, a.label, *a.tags]).lower()
        if search and search not in search_blob:
            continue
        if tag and tag not in a.tags:
            continue
        if source and a.source != source:
            continue
        out.append({"id": a.id, "label": a.label, "source": a.source, "tags": list(a.tags)})
    return out


@app.get("/preview")
def preview(asset_id: str):
    with GPU:
        body = previews.image_bytes(asset_id)
    return Response(body, media_type="image/png")


@app.get("/available_assets")
def load_available_assets(scenario: str, variation: str):
    path = _config_path(scenario, variation)
    if not path.is_file():
        return {"version": 1, "available_assets": {}}
    return _normalize_config(json.loads(path.read_text()))


@app.post("/available_assets")
def save_available_assets(scenario: str, variation: str, payload: dict = Body(...)):
    data = _normalize_config(payload)
    path = _config_path(scenario, variation)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"saved": str(path), "category_count": len(data["available_assets"])}


if __name__ == "__main__":
    print("[available assets] http://127.0.0.1:8103")
    uvicorn.run(app, host="127.0.0.1", port=8103, log_level="warning")
