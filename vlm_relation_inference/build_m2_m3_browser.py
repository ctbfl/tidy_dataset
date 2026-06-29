from __future__ import annotations

import html
import json
from pathlib import Path


DATA_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2")
OUT_HTML = DATA_ROOT / "m2_m3_m4_browser.html"


def case_dirs() -> list[Path]:
    return [
        path
        for path in sorted(DATA_ROOT.iterdir())
        if path.is_dir() and path.name.isdigit() and len(path.name) == 3
    ]


def rel(path: Path) -> str:
    return html.escape(path.relative_to(DATA_ROOT).as_posix(), quote=True)


def img_url(path: Path) -> str:
    return f"{rel(path)}?v={path.stat().st_mtime_ns}"


def full_ok_text(case_dir: Path, method: str) -> str:
    path = case_dir / "debug_vlm_relation_extraction/exp1" / method / "validation_result.json"
    if not path.is_file():
        return "missing"
    data = json.loads(path.read_text())
    objects = data.get("objects", {})
    full_ok = data.get("ok") is True and all(
        fields.get("x") == "ok" and fields.get("y") == "ok" and fields.get("rotation") == "ok"
        for fields in objects.values()
    )
    return "full-ok" if full_ok else "not-full-ok"


def image_panel(label: str, path: Path, status: str = "") -> str:
    title = html.escape(label)
    status_html = f"<span>{html.escape(status)}</span>" if status else ""
    if path.is_file():
        src = img_url(path)
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


def build_rows() -> str:
    rows = []
    for case_dir in case_dirs():
        case_id = html.escape(case_dir.name)
        goal = case_dir / "goal.png"
        exp_dir = case_dir / "debug_vlm_relation_extraction/exp1"
        m2 = exp_dir / "m2/topdown_layout.png"
        m3 = exp_dir / "m3/topdown_layout.png"
        m4 = exp_dir / "m4/topdown_layout.png"
        rows.append(f"""
        <article class="case-row" id="case-{case_id}">
          <h2>{case_id}</h2>
          <div class="triplet">
            {image_panel("goal.png", goal)}
            {image_panel("m2", m2, full_ok_text(case_dir, "m2"))}
            {image_panel("m3", m3, full_ok_text(case_dir, "m3"))}
            {image_panel("m4", m4, full_ok_text(case_dir, "m4"))}
          </div>
        </article>
        """)
    return "\n".join(rows)


def main() -> None:
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>m2 / m3 / m4 layout comparison</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
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
      padding: 14px 18px;
      border-bottom: 1px solid #111;
      background: #fff;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
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
    .triplet {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      align-items: start;
    }}
    .panel {{
      min-width: 0;
    }}
    .caption {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 5px;
      font-size: 12px;
      line-height: 1.2;
    }}
    .caption span {{
      color: #444;
    }}
    img {{
      display: block;
      width: 100%;
      height: 280px;
      object-fit: contain;
      border: 1px solid #111;
      background: #fafafa;
    }}
    .placeholder {{
      display: grid;
      place-items: center;
      width: 100%;
      height: 280px;
      border: 1px solid #111;
      background: #f5f5f5;
      color: #555;
      font-size: 14px;
    }}
    @media (max-width: 900px) {{
      .triplet {{
        grid-template-columns: 1fr;
      }}
      img,
      .placeholder {{
        height: auto;
        min-height: 220px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>goal.png / m2 / m3 / m4</h1>
  </header>
  <main>
    {build_rows()}
  </main>
</body>
</html>
"""
    OUT_HTML.write_text(html_text)
    print(OUT_HTML)


if __name__ == "__main__":
    main()
