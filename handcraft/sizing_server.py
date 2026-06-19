# Web tool for uniformly rescaling assets back to a sensible real-world size.
#   left   : live SAPIEN view -- target asset on a metric grid floor, between three
#            calibrated reference objects; live W*D*H readout in cm.
#   right  : asset browser (search) + scale controls (slider / factor / "set axis to N cm")
#            + editable references + Save.
#
# Save writes geometry.scale and geometry.aabb_m (both * factor) back into the shared
# asset.json under organize_it_v2/data/asset_library -- IN PLACE, no backup.
#
# Run:  /home/hjs/miniforge3/envs/RoboTwin/bin/python handcraft/sizing_server.py
#       http://127.0.0.1:8100

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

from sizing import SizingStudio
from preview import PreviewRenderer
from scene import LIBRARY

HERE = Path(__file__).resolve().parent
GPU = threading.Lock()  # all SAPIEN rendering is single-threaded

studio = SizingStudio()
previews = PreviewRenderer()

SOURCES = sorted({a.source for a in LIBRARY})


def _frame() -> dict:
    with GPU:
        rgb = studio.render()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return {"type": "frame", "image": base64.b64encode(buf.getvalue()).decode(),
            "state": studio.current_state()}


app = FastAPI(title="tidy asset sizing")


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "sizing.html").read_text()


@app.get("/sources")
def sources():
    return {"sources": SOURCES}


@app.get("/assets")
def assets(search: str = "", source: str = "", limit: int = 0):
    """All matching assets (no cap by default). search matches the id OR any tag."""
    search = search.lower()
    out = []
    for a in LIBRARY:
        if source and a.source != source:
            continue
        if search and search not in a.id.lower() and not any(search in t.lower() for t in a.tags):
            continue
        out.append({"id": a.id, "source": a.source, "tags": list(a.tags)})
        if limit and len(out) >= limit:
            break
    return out


@app.get("/preview")
def preview(asset_id: str):
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
            with GPU:
                if kind == "load_target":
                    studio.load_target(msg["asset_id"])
                elif kind == "set_factor":
                    studio.set_factor(float(msg["factor"]))
                elif kind == "set_abs":
                    studio.set_absolute(msg["axis"], float(msg["cm"]))
                elif kind == "reset":
                    studio.reset()
                elif kind == "set_refs":
                    studio.set_refs(msg["refs"])
            if kind == "save":
                try:
                    with GPU:
                        info = studio.save()
                    await socket.send_json({"type": "saved", "info": info})
                except Exception as exc:  # nothing loaded / write failed
                    await socket.send_json({"type": "error", "message": str(exc)})
            await socket.send_json(_frame())
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    print("[sizing] http://127.0.0.1:8100")
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
