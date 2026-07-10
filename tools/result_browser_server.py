from __future__ import annotations

import argparse
import re
from pathlib import Path

from flask import Flask, abort, request, send_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "sgbot" / "exp0"

MEDIA_OPTIONS = {
    "current": ("current.png", Path("current.png"), "image"),
    "goal": ("goal.png", Path("goal.png"), "image"),
    "our": ("our_result.png", Path("our_result.png"), "image"),
    "teleport": ("teleport.mp4", Path("our_output") / "teleport.mp4", "video"),
    "filled": (
        "filled_layout_render.png",
        Path("step_I_dof_filling") / "filled_layout_render.png",
        "image",
    ),
    "sgbot": ("sgbot_output/result.png", Path("sgbot_output") / "result.png", "image"),
}
VERIFIED_MARKER = Path("our_output") / "RESULT_VERIFIED"
VERIFIED_STAMP_KEYS = {"our", "teleport"}

SG_BOT_ID_RE = re.compile(r"^([A-Za-z0-9]+)_(\d{4})_(\d)(?:_|$)")


def short_id(name: str) -> str:
    match = SG_BOT_ID_RE.search(name)
    if match:
        prefix, index, count = match.groups()
        return f"{prefix[:6]}_{index}_{count}"
    return name


def id_class(name: str) -> str:
    return "numeric-id" if name.isdigit() else "text-id"


def build_app(data_root: Path) -> Flask:
    app = Flask(__name__)
    root = data_root.resolve()

    def scene_dirs() -> list[Path]:
        if not root.is_dir():
            raise FileNotFoundError(root)
        return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)

    @app.get("/")
    def index():
        raw_selected = request.args.getlist("show")
        selected = [key for key in raw_selected if key in MEDIA_OPTIONS]
        if "submitted" not in request.args:
            selected = list(MEDIA_OPTIONS)

        column_count = min(max(len(selected), 1), 4)
        rows = []
        for scene_dir in scene_dirs():
            media = []
            verified = (scene_dir / VERIFIED_MARKER).is_file()
            for key in selected:
                label, rel_path, kind = MEDIA_OPTIONS[key]
                full_path = scene_dir / rel_path
                media.append(
                    {
                        "key": key,
                        "kind": kind,
                        "label": label,
                        "exists": full_path.is_file(),
                        "verified": verified and key in VERIFIED_STAMP_KEYS,
                    }
                )
            rows.append(
                {
                    "id": scene_dir.name,
                    "short_id": short_id(scene_dir.name),
                    "id_class": id_class(scene_dir.name),
                    "media": media,
                }
            )

        return render_page(
            root=root,
            options=MEDIA_OPTIONS,
            selected=set(selected),
            rows=rows,
            column_count=column_count,
        )

    @app.get("/media/<key>/<scene_id>")
    @app.get("/image/<key>/<scene_id>")
    def media(key: str, scene_id: str):
        if key not in MEDIA_OPTIONS:
            abort(404)
        if "/" in scene_id or scene_id in {"", ".", ".."}:
            abort(404)
        _, rel_path, _ = MEDIA_OPTIONS[key]
        scene_dir = (root / scene_id).resolve()
        try:
            scene_dir.relative_to(root)
        except ValueError:
            abort(404)
        media_path = scene_dir / rel_path
        if not media_path.is_file():
            abort(404)
        return send_file(media_path)

    return app


def render_page(
    *,
    root: Path,
    options: dict[str, tuple[str, Path, str]],
    selected: set[str],
    rows: list[dict],
    column_count: int,
) -> str:
    option_html = "\n".join(
        f"""
        <label class="check">
          <input type="checkbox" name="show" value="{key}" {"checked" if key in selected else ""}>
          <span>{label}</span>
        </label>
        """
        for key, (label, _, _) in options.items()
    )

    row_html = []
    for row in rows:
        if row["media"]:
            media_html = "\n".join(render_media_item(item, row["id"]) for item in row["media"])
        else:
            media_html = '<div class="empty-selection">No media type selected.</div>'

        row_html.append(
            f"""
            <section class="entry">
              <div class="entry-id {row["id_class"]}" title="{row["id"]}">{row["short_id"]}</div>
              <div class="media-grid" style="grid-template-columns: repeat({column_count}, minmax(180px, 1fr));">
                {media_html}
              </div>
            </section>
            """
        )

    rows_html = "\n".join(row_html)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Result Browser</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d9dee7;
      --text: #1f2328;
      --muted: #656d76;
      --accent: #1f6feb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 12px 16px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
    }}
    .root {{
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 34vw;
    }}
    form {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin: 0;
    }}
    .check {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      user-select: none;
    }}
    .check input {{ margin: 0; }}
    main {{
      padding: 14px 16px 28px;
    }}
    .entry {{
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      padding: 12px 0;
      border-bottom: 1px solid var(--border);
    }}
    .entry-id {{
      position: sticky;
      left: 0;
      padding: 7px 8px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      line-height: 1.2;
      word-break: break-word;
    }}
    .numeric-id {{
      min-width: 68px;
      text-align: center;
      font-size: 22px;
      font-weight: 700;
    }}
    .text-id {{
      max-width: 180px;
      font-size: 13px;
    }}
    .media-grid {{
      display: grid;
      gap: 10px;
      min-width: 0;
    }}
    .media-card {{
      min-width: 0;
      margin: 0;
      overflow: hidden;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    figcaption {{
      padding: 6px 8px;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .media-frame {{
      position: relative;
      background: #eef1f5;
    }}
    .media-frame a {{
      display: block;
    }}
    img, video {{
      display: block;
      width: 100%;
      height: auto;
      background: #eef1f5;
    }}
    video {{
      aspect-ratio: 4 / 3;
      max-height: 420px;
      object-fit: contain;
    }}
    .verified-stamp {{
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 2;
      padding: 5px 10px;
      color: #d1242f;
      background: rgba(255, 255, 255, 0.24);
      border: 3px solid #d1242f;
      border-radius: 8px;
      font-weight: 900;
      font-size: 16px;
      line-height: 1;
      letter-spacing: 0;
      text-transform: uppercase;
      transform: rotate(7deg);
      pointer-events: none;
      box-shadow: 0 0 0 1px rgba(209, 36, 47, 0.18) inset;
    }}
    .missing, .empty-selection {{
      display: grid;
      place-items: center;
      min-height: 160px;
      color: var(--muted);
      background: repeating-linear-gradient(
        -45deg,
        #f1f3f6,
        #f1f3f6 10px,
        #e8ecf2 10px,
        #e8ecf2 20px
      );
    }}
    @media (max-width: 1100px) {{
      .media-grid {{
        grid-template-columns: repeat(2, minmax(160px, 1fr)) !important;
      }}
    }}
    @media (max-width: 720px) {{
      .toolbar {{ align-items: flex-start; flex-direction: column; gap: 8px; }}
      .root {{ max-width: 100%; }}
      .entry {{ grid-template-columns: 1fr; }}
      .entry-id {{ position: static; max-width: none; }}
      .media-grid {{
        grid-template-columns: 1fr !important;
      }}
    }}
  </style>
</head>
<body>
  <header class="toolbar">
    <div class="root" title="{root}">{root}</div>
    <form method="get" id="selector">
      <input type="hidden" name="submitted" value="1">
      {option_html}
    </form>
  </header>
  <main>
    {rows_html}
  </main>
  <script>
    document.querySelectorAll('#selector input').forEach((input) => {{
      input.addEventListener('change', () => input.form.submit());
    }});
  </script>
</body>
</html>
"""


def render_media_item(item: dict, scene_id: str) -> str:
    if not item["exists"]:
        body = '<div class="missing">missing</div>'
    else:
        url = f'/media/{item["key"]}/{scene_id}'
        stamp = '<div class="verified-stamp">Verified</div>' if item["verified"] else ""
        if item["kind"] == "video":
            body = f'<div class="media-frame"><video src="{url}" controls preload="metadata"></video>{stamp}</div>'
        else:
            body = (
                f'<div class="media-frame"><a href="{url}" target="_blank">'
                f'<img src="{url}" loading="lazy" alt="{item["label"]}"></a>{stamp}</div>'
            )

    return f"""
                <figure class="media-card">
                  <figcaption>{item["label"]}</figcaption>
                  {body}
                </figure>
                """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args.root)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
