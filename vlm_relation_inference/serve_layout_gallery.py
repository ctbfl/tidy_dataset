from __future__ import annotations

import html
import json
from pathlib import Path

from flask import Flask, Response, abort, send_from_directory


DATA_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2")
EXP_REL = Path("debug_vlm_relation_extraction/exp1")
METHODS = ("m1", "m2", "m3", "m4")
GOAL_OBJECT_ID_IMAGE = "goal_segmentation_object_id.png"

app = Flask(__name__)


def case_dirs() -> list[Path]:
    return [
        path
        for path in sorted(DATA_ROOT.iterdir())
        if path.is_dir() and path.name.isdigit() and len(path.name) == 3
    ]


def full_ok_text(case_dir: Path, method: str) -> str:
    path = case_dir / EXP_REL / method / "validation_result.json"
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
    status_html = f"<span>{html.escape(status)}</span>" if status else ""
    title = html.escape(label)
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


def progress_text() -> str:
    parts = []
    total = len(case_dirs())
    for method in METHODS:
        count = sum(
            1
            for case_dir in case_dirs()
            if (case_dir / EXP_REL / method / "topdown_layout.png").is_file()
        )
        parts.append(f"{method}: {count}/{total}")
    return " | ".join(parts)


def build_rows() -> str:
    rows = []
    for case_dir in case_dirs():
        case_id = html.escape(case_dir.name)
        exp_dir = case_dir / EXP_REL
        rows.append(f"""
        <article class="case-row" id="case-{case_id}">
          <h2>{case_id}</h2>
          <div class="grid">
            {image_panel("goal object_id", exp_dir / "m1" / GOAL_OBJECT_ID_IMAGE)}
            {image_panel("m1", exp_dir / "m1/topdown_layout.png", full_ok_text(case_dir, "m1"))}
            {image_panel("m2", exp_dir / "m2/topdown_layout.png", full_ok_text(case_dir, "m2"))}
            {image_panel("m3", exp_dir / "m3/topdown_layout.png", full_ok_text(case_dir, "m3"))}
            {image_panel("m4", exp_dir / "m4/topdown_layout.png", full_ok_text(case_dir, "m4"))}
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
  <title>layout comparison</title>
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
      grid-template-columns: repeat(5, minmax(0, 1fr));
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
      height: 260px;
      object-fit: contain;
      border: 1px solid #111;
      background: #fafafa;
    }}
    .placeholder {{
      display: grid;
      place-items: center;
      width: 100%;
      height: 260px;
      border: 1px solid #111;
      background: #f5f5f5;
      color: #555;
      font-size: 14px;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 700px) {{
      .grid {{ grid-template-columns: 1fr; }}
      img, .placeholder {{
        height: auto;
        min-height: 220px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>goal object_id / m1 / m2 / m3 / m4</h1>
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
    app.run(host="127.0.0.1", port=8000)
