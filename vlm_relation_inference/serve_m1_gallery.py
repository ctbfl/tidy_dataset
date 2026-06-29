from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, send_from_directory

from render_goal_segmentation import (
    color_map,
    draw_missing_list,
    draw_segmentation_labels,
    ensure_hw_mask,
    ensure_rgb_uint8,
    find_scene_path,
    load_scene,
)
from run_exp1_current_goal_ablation import best_goal_matches


DATA_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2")
M1_REL = Path("debug_vlm_relation_extraction/exp1/m1")
GOAL_OBJECT_ID_IMAGE = "goal_segmentation_object_id.png"

app = Flask(__name__)


def case_dirs() -> list[Path]:
    return [
        path
        for path in sorted(DATA_ROOT.iterdir())
        if path.is_dir() and path.name.isdigit() and len(path.name) == 3
    ]


def full_ok_text(case_dir: Path) -> str:
    path = case_dir / M1_REL / "validation_result.json"
    if not path.is_file():
        return "missing"
    data = json.loads(path.read_text())
    objects = data.get("objects", {})
    full_ok = data.get("ok") is True and all(
        fields.get("x") == "ok" and fields.get("y") == "ok" and fields.get("rotation") == "ok"
        for fields in objects.values()
    )
    return "full-ok" if full_ok else "not-full-ok"


def file_url(path: Path) -> str:
    rel = path.relative_to(DATA_ROOT).as_posix()
    return f"/file/{html.escape(rel, quote=True)}?v={path.stat().st_mtime_ns}"


def image_panel(label: str, path: Path, status: str = "") -> str:
    title = html.escape(label)
    status_html = f"<span>{html.escape(status)}</span>" if status else ""
    if path.is_file():
        src = file_url(path)
        return f"""
        <section class="panel">
          <div class="caption"><strong>{title}</strong>{status_html}</div>
          <a href="{src}" target="_blank"><img src="{src}" loading="lazy" alt="{title}"></a>
        </section>
        """
    return f"""
    <section class="panel missing">
      <div class="caption"><strong>{title}</strong>{status_html}</div>
      <div class="placeholder">missing</div>
    </section>
    """


def render_goal_object_ids(case_dir: Path) -> Path:
    out_dir = case_dir / M1_REL
    scene = load_scene(find_scene_path(case_dir))
    if getattr(scene, "goal_image", None) is None:
        raise ValueError(f"scene has no goal_image: {case_dir}")

    image_rgb = ensure_rgb_uint8(scene.goal_image)
    height, width = image_rgb.shape[:2]
    object_ids = sorted(str(object_id) for object_id in scene.objects)
    numbers = {object_id: str(index) for index, object_id in enumerate(object_ids, start=1)}
    best_by_raw = best_goal_matches(case_dir, numbers)

    groups = []
    missing = []
    for object_id in object_ids:
        obj = scene.objects[object_id]
        raw_id = getattr(obj, "raw_goal_mask_id", None)
        mask = getattr(obj, "goal_mask", None)
        if mask is None or raw_id is None:
            missing.append(object_id)
            continue
        raw_id = int(raw_id)
        if best_by_raw.get(raw_id) != object_id:
            missing.append(object_id)
            continue
        groups.append({"raw_id": raw_id, "label": object_id, "mask": mask})

    overlay = image_rgb.copy().astype(np.float32)
    labels = []
    for color, group in zip(color_map(len(groups)), sorted(groups, key=lambda item: int(item["raw_id"]))):
        mask = ensure_hw_mask(group["mask"], height, width)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
        labels.append((group["label"], mask, color))

    output = overlay.clip(0, 255).astype(np.uint8)
    draw_segmentation_labels(output, labels)
    draw_missing_list(output, missing)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / GOAL_OBJECT_ID_IMAGE
    cv2.imwrite(str(save_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    return save_path


def render_all_goal_object_ids() -> None:
    for case_dir in case_dirs():
        render_goal_object_ids(case_dir)


def progress_text() -> str:
    cases = case_dirs()
    relations = sum(1 for case_dir in cases if (case_dir / M1_REL / "relations.json").is_file())
    rendered = sum(1 for case_dir in cases if (case_dir / M1_REL / "topdown_layout.png").is_file())
    full_ok = sum(1 for case_dir in cases if full_ok_text(case_dir) == "full-ok")
    return f"relations: {relations}/{len(cases)} | full-ok: {full_ok}/{len(cases)} | rendered: {rendered}/{len(cases)}"


def build_rows() -> str:
    rows = []
    for case_dir in case_dirs():
        case_id = html.escape(case_dir.name)
        out_dir = case_dir / M1_REL
        rows.append(f"""
        <article class="case-row" id="case-{case_id}">
          <h2>{case_id}</h2>
          <div class="grid">
            {image_panel("m1 goal segmentation object_id", out_dir / GOAL_OBJECT_ID_IMAGE)}
            {image_panel("m1 layout", out_dir / "topdown_layout.png", full_ok_text(case_dir))}
          </div>
        </article>
        """)
    return "\n".join(rows)


@app.get("/")
def index() -> Response:
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="20">
  <title>m1 goal segmentation / layout</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #111;
      background: #fff;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 12px 18px;
      border-bottom: 1px solid #111;
      background: #fff;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .progress {{
      font-size: 12px;
      color: #333;
    }}
    main {{
      padding: 16px 18px 36px;
    }}
    .case-row {{
      padding: 14px 0 18px;
      border-bottom: 1px solid #ccc;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 16px;
      line-height: 1.2;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: start;
    }}
    .panel {{ min-width: 0; }}
    .caption {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 5px;
      font-size: 12px;
      line-height: 1.2;
    }}
    .caption span {{ color: #444; }}
    img {{
      display: block;
      width: 100%;
      height: 420px;
      object-fit: contain;
      border: 1px solid #111;
      background: #fafafa;
    }}
    .placeholder {{
      display: grid;
      place-items: center;
      width: 100%;
      height: 420px;
      border: 1px solid #111;
      background: #f5f5f5;
      color: #555;
      font-size: 14px;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      img, .placeholder {{
        height: auto;
        min-height: 260px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>m1 goal segmentation / layout</h1>
    <div class="progress">{html.escape(progress_text())}</div>
  </header>
  <main>
    {build_rows()}
  </main>
</body>
</html>
"""
    return Response(html_text, mimetype="text/html")


@app.get("/file/<path:filename>")
def serve_file(filename: str):
    path = (DATA_ROOT / filename).resolve()
    try:
        path.relative_to(DATA_ROOT)
    except ValueError:
        abort(404)
    if not path.is_file():
        abort(404)
    return send_from_directory(DATA_ROOT, filename)


if __name__ == "__main__":
    render_all_goal_object_ids()
    app.run(host="127.0.0.1", port=8001)
