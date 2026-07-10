"""Candidate-category annotation server.

A curation UI for `candidate_categories.json` (the leaf categories mined from the
asset library). Lets a human review each leaf category and add / remove member
assets by eye, backed by the same on-the-fly SAPIEN thumbnail renderer the
`available_assets` editor uses.

Run directly (needs the SAPIEN-enabled env, like the other handcraft servers):

    python handcraft/candidate_category_server.py
    # -> http://127.0.0.1:8106

File it edits:  data/organize_it_dataset_v2/candidate_categories.json
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent
SIMULATIONS_DIR = HERE.parent / "simulations"
if str(SIMULATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATIONS_DIR))

from preview import PreviewRenderer  # noqa: E402
from scene import LIBRARY  # noqa: E402

DATASET_DIR = Path(
    os.environ.get("TIDY_DATASET_DIR", HERE.parent / "data" / "organize_it_dataset_v2")
).resolve()
CANDIDATE_PATH = Path(
    os.environ.get("TIDY_CANDIDATE_CATEGORIES", DATASET_DIR / "candidate_categories.json")
).resolve()
AVAILABLE_PATH = DATASET_DIR / "available_assets.json"

GPU = threading.Lock()
previews = PreviewRenderer()

SOURCES = sorted({a.source for a in LIBRARY})
TAGS = sorted({t for a in LIBRARY for t in a.tags})


def _sizes() -> dict[str, list[float] | None]:
    out: dict[str, list[float] | None] = {}
    for a in LIBRARY:
        try:
            rec = json.loads(LIBRARY.asset_json_path(a.id).read_text())
            out[a.id] = [round(float(v), 3) for v in rec["geometry"]["aabb_m"]["size"]]
        except Exception:
            out[a.id] = None
    return out


SIZES = _sizes()


def _used_in_available() -> set[str]:
    if not AVAILABLE_PATH.is_file():
        return set()
    try:
        data = json.loads(AVAILABLE_PATH.read_text())
    except Exception:
        return set()
    return {
        aid
        for cat in data.get("available_assets", {}).values()
        for entry in cat.get("entries", [])
        for aid in entry
    }


USED_IN_AVAILABLE = _used_in_available()


def _load() -> dict:
    if not CANDIDATE_PATH.is_file():
        return {"version": 1, "note": "", "category_count": 0, "candidate_categories": {}}
    return json.loads(CANDIDATE_PATH.read_text())


def _normalize(payload: dict) -> dict:
    raw = payload.get("candidate_categories", {})
    if not isinstance(raw, dict):
        raise HTTPException(400, "candidate_categories must be an object")
    categories: dict[str, dict] = {}
    for name, cat in raw.items():
        name = str(name).strip()
        if not name:
            raise HTTPException(400, "category name cannot be empty")
        if name in categories:
            raise HTTPException(400, f"duplicate category name: {name}")
        description = str((cat or {}).get("description", "")).strip()
        ids: list[str] = []
        seen: set[str] = set()
        for aid in (cat or {}).get("recommended_asset_ids", []):
            aid = str(aid).strip()
            if not aid or aid in seen:
                continue
            if aid not in LIBRARY.assets:
                raise HTTPException(400, f"{name}: unknown asset_id {aid}")
            seen.add(aid)
            ids.append(aid)
        categories[name] = {"description": description, "recommended_asset_ids": ids}
    return {
        "version": 1,
        "note": str(payload.get("note", "")),
        "category_count": len(categories),
        "candidate_categories": categories,
    }


app = FastAPI(title="candidate category editor")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "candidate_category.html").read_text()


@app.get("/meta")
def meta() -> dict:
    return {
        "sources": SOURCES,
        "tags": TAGS,
        "library_count": len(LIBRARY),
        "used_in_available_count": len(USED_IN_AVAILABLE),
        "path": str(CANDIDATE_PATH),
    }


@app.get("/categories")
def categories() -> dict:
    return _load()


@app.post("/categories")
def save_categories(payload: dict = Body(...)) -> dict:
    data = _normalize(payload)
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CANDIDATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(CANDIDATE_PATH)
    assigned = sum(len(c["recommended_asset_ids"]) for c in data["candidate_categories"].values())
    return {
        "saved": str(CANDIDATE_PATH),
        "category_count": data["category_count"],
        "assigned_assets": assigned,
    }


@app.get("/assets")
def assets(search: str = "", tag: str = "", source: str = "") -> list[dict]:
    search = search.lower().strip()
    out = []
    for a in LIBRARY:
        if source and a.source != source:
            continue
        if tag and tag not in a.tags:
            continue
        if search:
            blob = " ".join([a.id, a.label, *a.tags]).lower()
            if search not in blob:
                continue
        out.append(
            {
                "id": a.id,
                "label": a.label,
                "source": a.source,
                "tags": list(a.tags),
                "size": SIZES.get(a.id),
                "in_available": a.id in USED_IN_AVAILABLE,
            }
        )
    return out


@app.get("/preview")
def preview(asset_id: str) -> Response:
    if asset_id not in LIBRARY.assets:
        raise HTTPException(404, "unknown asset_id")
    with GPU:
        body = previews.image_bytes(asset_id)
    return Response(body, media_type="image/png")


if __name__ == "__main__":
    print(f"[candidate categories] editing {CANDIDATE_PATH}")
    print("[candidate categories] http://127.0.0.1:8106")
    uvicorn.run(app, host="127.0.0.1", port=8106, log_level="warning")
