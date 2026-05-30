# Web front-end for hand-crafting tidy scenes.
#   middle/left : live scene view (drag assets onto the table, WASD/QE to adjust)
#   right       : asset browser (search + tag/source filter, thumbnail grid)
#   floating    : list of placed objects (click to select, x to delete)
#
# Run:  /home/hjs/miniforge3/envs/RoboTwin/bin/python handcraft/server.py

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

from editor import SceneEditor
from preview import PreviewRenderer
from scene import LIBRARY
from robotwin_utils import curated_textures

HERE = Path(__file__).resolve().parent
SAVE_DIR = HERE.parent / "data" / "tidy_scene_v0"
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


def _save_scene() -> str:
    n = max((int(p.stem) for p in SAVE_DIR.glob("*.json") if p.stem.isdigit()), default=0) + 1
    path = SAVE_DIR / f"{n:04d}.json"
    path.write_text(json.dumps(editor.scene_dict(), indent=2))
    return path.name


def _list_scenes() -> list[str]:
    return sorted(p.name for p in SAVE_DIR.glob("*.json"))


def _load_scene(name: str) -> None:
    # Resolve strictly inside SAVE_DIR (the file dialog lives on the server host).
    path = (SAVE_DIR / name).resolve()
    if path.parent != SAVE_DIR.resolve() or not path.is_file():
        raise FileNotFoundError(name)
    with GPU:
        editor.load_scene_dict(json.loads(path.read_text()))


app = FastAPI(title="tidy handcraft")


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()


@app.get("/meta")
def meta():
    return {"sources": SOURCES, "tags": TAGS}


@app.get("/scenes")
def scenes():
    return {"dir": str(SAVE_DIR), "scenes": _list_scenes()}


@app.get("/textures")
def textures():
    return {"table": [t["id"] for t in curated_textures("table")],
            "wall": [t["id"] for t in curated_textures("wall")]}


@app.get("/assets")
def assets(search: str = "", tag: str = "", source: str = ""):
    search = search.lower()
    out = []
    for a in LIBRARY:
        if search and search not in a.id.lower():
            continue
        if tag and tag not in a.tags:
            continue
        if source and a.source != source:
            continue
        out.append({"id": a.id, "source": a.source, "tags": list(a.tags)})
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
                editor.place(msg["asset_id"], msg["x"], msg["y"])
            elif kind == "select":
                editor.select(msg["scene_id"])
            elif kind == "select_at":
                editor.select_at(msg["x"], msg["y"])
            elif kind == "delete":
                editor.delete(msg["scene_id"])
            elif kind == "key":
                editor.key(msg["name"])
            elif kind == "save":
                await socket.send_json({"type": "saved", "name": _save_scene()})
            elif kind == "randomize_bg":
                with GPU:
                    editor.randomize_background()
            elif kind == "set_bg":
                with GPU:
                    editor.set_background(msg.get("table_texture") or None,
                                          msg.get("wall_texture") or None)
            elif kind == "load":
                _load_scene(msg["name"])
                await socket.send_json({"type": "loaded", "name": msg["name"]})
            await socket.send_json(_frame())
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    print("[handcraft] http://127.0.0.1:8099")
    uvicorn.run(app, host="127.0.0.1", port=8099, log_level="warning")
