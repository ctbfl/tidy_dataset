# Web front-end for hand-crafting organize_it_dataset_v2 samples.
#   middle/left : live scene view (drag assets onto the table, WASD/QE to adjust)
#   right       : asset browser (search + tag/source filter, thumbnail grid)
#   floating    : list of placed objects (click to select, x to delete)
#
# Run:  /home/hjs/miniforge3/envs/RoboTwin/bin/python handcraft/server_v2.py

from __future__ import annotations

import base64
import io
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

from editor import SceneEditor
from objects import asset_json_backup_dir, asset_json_backup_path, scene_asset_ids, write_asset_json_backup
from preview import PreviewRenderer
from scene import LIBRARY
from robotwin_utils import curated_textures

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data" / "organize_it_dataset_v2"
ARRANGEMENTS = ("messy", "tidy")
GPU = threading.Lock()  # all SAPIEN rendering is single-threaded

editor = SceneEditor()
previews = PreviewRenderer()

SOURCES = sorted({a.source for a in LIBRARY})
TAGS = sorted({t for a in LIBRARY for t in a.tags})


def _frame() -> dict:
    with GPU:
        rgb = editor.render()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return {"type": "frame", "image": base64.b64encode(buf.getvalue()).decode(), "state": editor.state()}


# -- organize_it_dataset_v2: <scenario>/<variation>/<scene>/{messy,tidy}.json #
def _variation_dir(scenario: str, variation: str) -> Path:
    path = (DATASET_DIR / Path(scenario).stem / Path(variation).stem).resolve()
    if path.parent.parent != DATASET_DIR.resolve():
        raise ValueError(f"invalid scenario/variation: {scenario}/{variation}")
    return path


def _list_scenarios() -> list[str]:
    if not DATASET_DIR.is_dir():
        return []
    return sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())


def _scene_dir(scenario: str, variation: str, scene: str) -> Path:
    path = (_variation_dir(scenario, variation) / Path(scene).stem).resolve()
    if path.parent != _variation_dir(scenario, variation):
        raise ValueError(f"invalid scene: {scene}")
    if path.name == "template":
        raise ValueError("template is not a data scene")
    return path


def _arrangement_path(scenario: str, variation: str, scene: str, arrangement: str) -> Path:
    if arrangement not in ARRANGEMENTS:
        raise ValueError(f"invalid arrangement: {arrangement}")
    return _scene_dir(scenario, variation, scene) / f"{arrangement}.json"


def _read_info(scenario: str, variation: str) -> dict:
    path = _variation_dir(scenario, variation) / "template" / "info.json"
    if not path.is_file():
        return {"scene_description": "", "user_note": ""}
    data = json.loads(path.read_text())
    return {
        "scene_description": data.get("scene_description", ""),
        "user_note": data.get("user_note", ""),
    }


def _role_from_asset_id(asset_id: str) -> str:
    parts = asset_id.split(":")
    token = parts[-2] if len(parts) > 1 and parts[-1].isdigit() else parts[-1]
    role = "".join(c.lower() if c.isalnum() else "_" for c in token).strip("_")
    return role or "object"


def _next_slot(role: str, used_slots: set[str]) -> str:
    i = 1
    while f"{role}-{i}" in used_slots:
        i += 1
    return f"{role}-{i}"


def _manifest_entries(data: dict) -> list[dict]:
    out = []
    for entry in data.get("manifest", []):
        slot = entry.get("slot")
        asset_id = entry.get("asset_id")
        if slot and asset_id:
            out.append({
                "slot": slot,
                "role": entry.get("role") or _role_from_asset_id(asset_id),
                "asset_id": asset_id,
            })
    return out


def _asset_counts(data: dict) -> Counter:
    items = Counter(item.get("asset_id") for item in data.get("items", []) if item.get("asset_id"))
    manifest = Counter(entry["asset_id"] for entry in _manifest_entries(data))
    return Counter({asset_id: max(items[asset_id], manifest[asset_id]) for asset_id in set(items) | set(manifest)})


def _manifest_union(*datasets: dict) -> list[dict]:
    used_slots = set()
    by_asset = defaultdict(list)
    out = []
    asset_order = []

    def remember_asset(asset_id: str) -> None:
        if asset_id not in asset_order:
            asset_order.append(asset_id)

    for data in datasets:
        for entry in _manifest_entries(data):
            remember_asset(entry["asset_id"])
            if entry["slot"] in used_slots:
                continue
            out.append(entry)
            used_slots.add(entry["slot"])
            by_asset[entry["asset_id"]].append(entry["slot"])
        for item in data.get("items", []):
            if item.get("asset_id"):
                remember_asset(item["asset_id"])

    desired = Counter()
    for data in datasets:
        desired |= _asset_counts(data)

    for asset_id in asset_order:
        role = _role_from_asset_id(asset_id)
        while len(by_asset[asset_id]) < desired[asset_id]:
            slot = _next_slot(role, used_slots)
            entry = {"slot": slot, "role": role, "asset_id": asset_id}
            out.append(entry)
            used_slots.add(slot)
            by_asset[asset_id].append(slot)
    return out


def _apply_manifest_to_items(data: dict, manifest: list[dict]) -> dict:
    out = dict(data)
    out["manifest"] = [dict(entry) for entry in manifest]
    manifest_slots = {entry["slot"] for entry in manifest}
    slots_by_asset = defaultdict(list)
    for entry in manifest:
        slots_by_asset[entry["asset_id"]].append(entry["slot"])

    used_slots = set()
    items = []
    for item in data.get("items", []):
        item = dict(item)
        asset_id = item.get("asset_id")
        slot = item.get("slot")
        if slot not in manifest_slots or slot in used_slots:
            slot = next((s for s in slots_by_asset[asset_id] if s not in used_slots), None)
            item["slot"] = slot
        if slot:
            used_slots.add(slot)
        items.append(item)
    out["items"] = items
    return out


def _read_arrangement(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _backup_warnings(paths: list[Path], data: dict) -> list[str]:
    warnings = []
    seen = set()
    for asset_id in scene_asset_ids(data):
        for path in paths:
            backup_path = asset_json_backup_path(asset_json_backup_dir(path), asset_id)
            if not backup_path.is_file():
                continue
            key = (asset_id, backup_path)
            if key in seen:
                continue
            seen.add(key)
            backup = json.loads(backup_path.read_text())
            current = json.loads(LIBRARY.asset_json_path(asset_id).read_text())
            if backup != current:
                warnings.append(f"{asset_id}: saved backup differs from the current asset library")
                break
    return warnings


def _write_pending_asset_scales() -> list[str]:
    changed = []
    for asset_id, scale in editor.pending_asset_scales.items():
        path = LIBRARY.asset_json_path(asset_id)
        data = json.loads(path.read_text())
        old_scale = data["geometry"]["scale"]
        factor = float(sum(scale[i] / old_scale[i] for i in range(3)) / 3)
        data["geometry"]["scale"] = [float(v) for v in scale]
        aabb = data["geometry"].get("aabb_m")
        if aabb:
            for key in ("min", "max", "aabb_center", "size"):
                aabb[key] = [float(v) * factor for v in aabb[key]]
            aabb["bottom_z"] = float(aabb["bottom_z"]) * factor
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        changed.append(asset_id)
    if changed:
        LIBRARY.load_asset_json_backup(None)
        editor.pending_asset_scales.clear()
    return changed


def _scene_arrangements(scenario: str, variation: str, scene: str) -> list[tuple[str, Path, dict | None]]:
    return [
        (name, _arrangement_path(scenario, variation, scene, name),
         _read_arrangement(_arrangement_path(scenario, variation, scene, name)))
        for name in ARRANGEMENTS
    ]


def _arrangement_progress(name: str, path: Path, manifest: list[dict]) -> dict:
    if not path.is_file():
        return {"name": name, "placed": 0, "total": len(manifest), "saved": False}
    data = json.loads(path.read_text())
    data = _apply_manifest_to_items(data, manifest)
    placed = sum(1 for item in data.get("items", []) if item.get("slot"))
    total = len(manifest)
    return {"name": name, "placed": placed, "total": total, "saved": True}


def _list_data_scenes(variation_dir: Path) -> list[str]:
    scenes = [p.name for p in variation_dir.iterdir() if p.is_dir() and p.name != "template"]
    return sorted(scenes, key=lambda name: (name != "example", name))


def _list_scenario_scenes(scenario: str) -> list[dict]:
    out = []
    base = DATASET_DIR / Path(scenario).stem
    if not base.is_dir():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        info = _read_info(scenario, d.name)
        out.append({
            "scene": d.name,
            "scene_description": info["scene_description"],
            "user_note": info["user_note"],
            "samples": [
                _sample_info(scenario, d.name, sample)
                for sample in _list_data_scenes(d)
            ],
        })
    return out


def _sample_info(scenario: str, variation: str, sample: str) -> dict:
    arrangements = _scene_arrangements(scenario, variation, sample)
    manifest = _manifest_union(*(data for _, _, data in arrangements if data))
    return {
        "name": sample,
        "arrangements": [
            _arrangement_progress(name, path, manifest)
            for name, path, _ in arrangements
        ],
    }


def _blank_arrangement(scenario: str, variation: str, scene: str, arrangement: str) -> dict:
    return {
        "version": 2,
        "scenario": Path(scenario).stem,
        "scene_id": Path(scene).stem,
        "arrangement": arrangement,
        "template": Path(variation).stem,
        "manifest": [],
        "items": [],
    }


def _copy_background(dst: dict, src: dict) -> None:
    for key in ("table", "table_texture", "wall_texture"):
        if key in src:
            dst[key] = src[key]


def _sync_sibling_background(scene: dict) -> None:
    for name, path, data in _scene_arrangements(scene["scenario"], scene["template"], scene["scene_id"]):
        if name == scene["arrangement"] or not data:
            continue
        _copy_background(data, scene)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _load_scenario_scene(scenario: str, variation: str, scene: str, arrangement: str) -> list[str]:
    path = _arrangement_path(scenario, variation, scene, arrangement)
    arrangements = _scene_arrangements(scenario, variation, scene)
    datasets = [data for _, _, data in arrangements if data]
    manifest = _manifest_union(*datasets)
    current_data = _read_arrangement(path)
    data = current_data or _blank_arrangement(scenario, variation, scene, arrangement)
    if not current_data:
        sibling = next((d for name, _, d in arrangements if name != arrangement and d), None)
        if sibling:
            _copy_background(data, sibling)
    data = _apply_manifest_to_items(data, manifest)

    backup_paths = [p for _, p, existing in arrangements if existing]
    LIBRARY.load_asset_json_backup(None)
    data["scenario"] = Path(scenario).stem
    data["template"] = Path(variation).stem
    data["scene_id"] = Path(scene).stem
    data["arrangement"] = arrangement
    warnings = _backup_warnings(backup_paths, data)
    with GPU:
        editor.load_scene_dict(data)
        if warnings:
            editor.settle_all()
    return warnings


def _save_scenario_scene() -> str:
    if not editor.scenario or not editor.template or not editor.scene_id or not editor.arrangement:
        raise ValueError("no organize_it_dataset_v2 variation is loaded")
    path = _arrangement_path(editor.scenario, editor.template, editor.scene_id, editor.arrangement)
    path.parent.mkdir(parents=True, exist_ok=True)
    changed_assets = _write_pending_asset_scales()
    scene = editor.scene_dict()
    sibling_data = [
        data for name, _, data in _scene_arrangements(editor.scenario, editor.template, editor.scene_id)
        if name != editor.arrangement and data
    ]
    scene = _apply_manifest_to_items(scene, _manifest_union(scene, *sibling_data))
    path.write_text(json.dumps(scene, indent=2, ensure_ascii=False))
    _sync_sibling_background(scene)
    write_asset_json_backup(path, scene, LIBRARY)
    name = f"{editor.scenario}/{editor.template}/{editor.scene_id}/{path.name}"
    return name + (f" (updated scale: {', '.join(changed_assets)})" if changed_assets else "")


app = FastAPI(title="organize_it_dataset_v2 handcraft")


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index_v2.html").read_text()


@app.get("/meta")
def meta():
    return {"sources": SOURCES, "tags": TAGS}


@app.get("/scenarios")
def scenarios():
    return {"scenarios": _list_scenarios()}


@app.get("/scenario_scenes")
def scenario_scenes(scenario: str):
    return {"scenario": scenario, "scenes": _list_scenario_scenes(scenario)}


@app.get("/textures")
def textures():
    return {"table": [t["id"] for t in curated_textures("table")],
            "wall": [t["id"] for t in curated_textures("wall")]}


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
    # Read from the shared cache; on a miss, SAPIEN-render it (serialized with the
    # editor render via GPU lock) and write it back into the cache.
    with GPU:
        body = previews.image_bytes(asset_id)
    return Response(body, media_type="image/png")


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    await socket.send_json(_frame())
    try:
        while True:
            msg = await socket.receive_json()
            kind = msg["type"]
            if kind == "place":
                if msg.get("slot"):  # manifest card -> place that slot's asset
                    editor.place_slot(msg["slot"], msg["x"], msg["y"])
                else:                # full-library browser -> an extra (no slot)
                    editor.place(msg["asset_id"], msg["x"], msg["y"])
            elif kind == "select":
                editor.select(msg["scene_id"])
            elif kind == "select_at":
                editor.select_at(msg["x"], msg["y"])
            elif kind == "context_select":
                editor.select_at(msg["x"], msg["y"])
                await socket.send_json({"type": "size_target", "target": editor.size_target(),
                                        "client_x": msg["client_x"], "client_y": msg["client_y"]})
            elif kind == "select_rect":
                editor.select_rect(msg["x0"], msg["y0"], msg["x1"], msg["y1"])
            elif kind == "special_start":
                editor.start_special_relation(msg["relation"])
                await socket.send_json(
                    {"type": "special_relation", "relation": msg["relation"], "done": False,
                     "message": "Pen in Holder: first select the object to move, then select the holder/container."}
                )
            elif kind == "special_cancel":
                editor.cancel_special_relation()
            elif kind == "special_pick":
                try:
                    result = editor.special_pick(msg["relation"], msg["x"], msg["y"])
                    await socket.send_json({"type": "special_relation", "relation": msg["relation"], **result})
                except ValueError as exc:
                    editor.cancel_special_relation()
                    await socket.send_json(
                        {"type": "special_relation", "relation": msg["relation"], "done": True, "error": str(exc)}
                    )
            elif kind == "delete":
                editor.delete(msg["scene_id"])
            elif kind == "key":
                editor.key(msg["name"], msg.get("fine", False))
            elif kind == "clear":
                editor.clear()
            elif kind == "set_scale":
                editor.set_selected_asset_scale_factor(float(msg["factor"]))
            elif kind == "save":
                name = _save_scenario_scene()
                await socket.send_json({"type": "saved", "name": name})
            elif kind == "randomize_bg":
                with GPU:
                    editor.randomize_background()
            elif kind == "set_bg":
                with GPU:
                    editor.set_background(msg.get("table_texture") or None,
                                          msg.get("wall_texture") or None)
            elif kind == "load_scene":
                warnings = _load_scenario_scene(msg["scenario"], msg["variation"], msg["scene"], msg["arrangement"])
                await socket.send_json(
                    {"type": "loaded",
                     "name": f'{msg["scenario"]}/{msg["variation"]}/{msg["scene"]}/{msg["arrangement"]}'})
                if warnings:
                    await socket.send_json({"type": "warning", "warnings": warnings})
            await socket.send_json(_frame())
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    print("[handcraft v2] http://127.0.0.1:8101")
    uvicorn.run(app, host="127.0.0.1", port=8101, log_level="warning")
