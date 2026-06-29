from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data" / "organize_it_dataset_v2"
HOST = "127.0.0.1"
PORT = 8010

MEDIA = {
    "current.png": "image/png",
    "goal.png": "image/png",
    "reference_goal.png": "image/png",
    "teleport.mp4": "video/mp4",
}

app = FastAPI(title="organize-it dataset viewer")


def _child_dir(parent: Path, name: str) -> Path:
    path = (parent / name).resolve()
    if path.parent != parent.resolve() or not path.is_dir():
        raise HTTPException(status_code=404, detail=name)
    return path


def _example_dir(scenario: str, variation: str, example: str) -> Path:
    scenario_dir = _child_dir(DATASET_DIR, scenario)
    variation_dir = _child_dir(scenario_dir, variation)
    return _child_dir(variation_dir, example)


def _list_dirs(parent: Path) -> list[str]:
    return sorted(p.name for p in parent.iterdir() if p.is_dir())


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/tree")
def tree():
    if not DATASET_DIR.is_dir():
        raise HTTPException(status_code=500, detail=f"missing dataset: {DATASET_DIR}")
    scenarios = []
    for scenario in _list_dirs(DATASET_DIR):
        scenario_dir = DATASET_DIR / scenario
        variations = [v for v in _list_dirs(scenario_dir) if v not in {"example", "template"}]
        scenarios.append({"name": scenario, "variations": variations})
    return {"dataset": str(DATASET_DIR), "scenarios": scenarios}


@app.get("/api/examples")
def examples(scenario: str, variation: str):
    variation_dir = _child_dir(_child_dir(DATASET_DIR, scenario), variation)
    rows = []
    for path in _list_dirs(variation_dir):
        if not path.isdigit():
            continue
        example_dir = variation_dir / path
        rows.append({
            "name": path,
            "current": (example_dir / "current.png").is_file(),
            "goal": (example_dir / "goal.png").is_file(),
            "reference_goal": (example_dir / "reference_goal.png").is_file(),
            "teleport": (example_dir / "teleport.mp4").is_file(),
        })
    return {"scenario": scenario, "variation": variation, "examples": rows}


@app.get("/file")
def file(scenario: str, variation: str, example: str, name: str):
    if name not in MEDIA:
        raise HTTPException(status_code=404, detail=name)
    path = _example_dir(scenario, variation, example) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{example}/{name}")
    return FileResponse(path, media_type=MEDIA[name])


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #17202a;
      --muted: #64748b;
      --accent: #1f7a6d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 56px;
      padding: 10px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0 10px 0 0;
      font-size: 18px;
      font-weight: 650;
      white-space: nowrap;
    }
    label {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      white-space: nowrap;
    }
    select, button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    select {
      min-width: 190px;
      padding: 0 30px 0 10px;
    }
    button {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    main { padding: 16px; }
    .meta {
      margin-bottom: 12px;
      color: var(--muted);
    }
    .table {
      display: grid;
      gap: 10px;
    }
    .row {
      display: grid;
      grid-template-columns: 72px repeat(4, minmax(220px, 1fr));
      gap: 10px;
      align-items: center;
      padding: 10px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .head {
      position: sticky;
      top: 57px;
      z-index: 4;
      padding: 8px 10px;
      background: #eef2f7;
      color: #334155;
      font-weight: 650;
    }
    .id {
      font-variant-numeric: tabular-nums;
      color: #334155;
      font-weight: 650;
    }
    .cell {
      position: relative;
      display: grid;
      place-items: center;
      width: 100%;
      aspect-ratio: 16 / 9;
      min-height: 130px;
      overflow: hidden;
      background: #e9edf2;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .cell img, .cell video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #111827;
    }
    .missing {
      color: #9f1239;
      font-weight: 650;
    }
    .video-preview {
      padding: 0;
      width: 100%;
      height: 100%;
      border: 0;
      border-radius: 0;
      background: #111827;
    }
    .video-preview img {
      opacity: 0.86;
    }
    .play {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      display: grid;
      place-items: center;
      width: 54px;
      height: 54px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.92);
      color: #0f172a;
      font-size: 22px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.25);
    }
    @media (max-width: 980px) {
      header { flex-wrap: wrap; }
      .row {
        grid-template-columns: 54px 1fr;
      }
      .head { display: none; }
      .cell::before {
        position: absolute;
        left: 8px;
        top: 6px;
        z-index: 1;
        padding: 2px 6px;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 4px;
        color: #334155;
        font-size: 12px;
        font-weight: 650;
      }
      .current::before { content: "current.png"; }
      .goal::before { content: "goal.png"; }
      .reference::before { content: "reference_goal.png"; }
      .teleport::before { content: "teleport.mp4"; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Dataset Viewer</h1>
    <label>Scenario <select id="scenario"></select></label>
    <label>Variation <select id="variation"></select></label>
    <button id="refresh" title="Refresh current variation">↻ Refresh</button>
  </header>
  <main>
    <div class="meta" id="meta"></div>
    <div class="table" id="table"></div>
  </main>
  <script>
    const scenarioSelect = document.querySelector("#scenario");
    const variationSelect = document.querySelector("#variation");
    const refreshButton = document.querySelector("#refresh");
    const table = document.querySelector("#table");
    const meta = document.querySelector("#meta");
    let tree = [];

    function params() {
      return new URLSearchParams(location.search);
    }

    function mediaURL(example, name) {
      const q = new URLSearchParams({
        scenario: scenarioSelect.value,
        variation: variationSelect.value,
        example,
        name,
      });
      return `/file?${q}`;
    }

    function setURL() {
      const q = new URLSearchParams({
        scenario: scenarioSelect.value,
        variation: variationSelect.value,
      });
      history.replaceState(null, "", `${location.pathname}?${q}`);
    }

    function optionList(select, values, selected) {
      select.replaceChildren(...values.map((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === selected;
        return option;
      }));
    }

    function selectedScenario() {
      return tree.find((item) => item.name === scenarioSelect.value);
    }

    function renderMedia(row, name, className, exists) {
      const cell = document.createElement("div");
      cell.className = `cell ${className}`;
      if (!exists) {
        cell.innerHTML = `<span class="missing">missing ${name}</span>`;
        return cell;
      }
      const img = document.createElement("img");
      img.loading = "lazy";
      img.decoding = "async";
      img.src = mediaURL(row.name, name);
      img.alt = `${row.name} ${name}`;
      cell.appendChild(img);
      return cell;
    }

    function renderVideo(row) {
      const cell = document.createElement("div");
      cell.className = "cell teleport";
      if (!row.teleport) {
        cell.innerHTML = '<span class="missing">missing teleport.mp4</span>';
        return cell;
      }
      const button = document.createElement("button");
      button.className = "video-preview";
      button.title = "Play teleport.mp4";
      if (row.current) {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.decoding = "async";
        img.src = mediaURL(row.name, "current.png");
        img.alt = `${row.name} teleport preview`;
        button.appendChild(img);
      }
      const play = document.createElement("span");
      play.className = "play";
      play.textContent = "▶";
      button.appendChild(play);
      button.addEventListener("click", () => {
        const video = document.createElement("video");
        video.controls = true;
        video.autoplay = true;
        video.playsInline = true;
        video.src = mediaURL(row.name, "teleport.mp4");
        cell.replaceChildren(video);
        video.play();
      }, { once: true });
      cell.appendChild(button);
      return cell;
    }

    async function loadExamples() {
      setURL();
      table.textContent = "";
      const q = new URLSearchParams({
        scenario: scenarioSelect.value,
        variation: variationSelect.value,
      });
      const data = await fetch(`/api/examples?${q}`).then((r) => r.json());
      meta.textContent = `${data.scenario} / ${data.variation}: ${data.examples.length} examples`;
      const head = document.createElement("div");
      head.className = "row head";
      head.innerHTML = "<div>Example</div><div>current.png</div><div>reference_goal.png</div><div>goal.png</div><div>teleport.mp4</div>";
      table.appendChild(head);
      for (const row of data.examples) {
        const el = document.createElement("div");
        el.className = "row";
        const id = document.createElement("div");
        id.className = "id";
        id.textContent = row.name;
        el.append(
          id,
          renderMedia(row, "current.png", "current", row.current),
          renderMedia(row, "reference_goal.png", "reference", row.reference_goal),
          renderMedia(row, "goal.png", "goal", row.goal),
          renderVideo(row),
        );
        table.appendChild(el);
      }
    }

    function loadVariations() {
      const scenario = selectedScenario();
      const current = variationSelect.value;
      const initial = scenario.variations.includes(current) ? current : scenario.variations[0];
      optionList(variationSelect, scenario.variations, initial);
    }

    async function init() {
      const data = await fetch("/api/tree").then((r) => r.json());
      tree = data.scenarios;
      const q = params();
      const scenarios = tree.map((item) => item.name);
      const scenario = scenarios.includes(q.get("scenario")) ? q.get("scenario") : scenarios[0];
      optionList(scenarioSelect, scenarios, scenario);
      const scenarioData = selectedScenario();
      const variation = scenarioData.variations.includes(q.get("variation")) ? q.get("variation") : scenarioData.variations[0];
      optionList(variationSelect, scenarioData.variations, variation);
      scenarioSelect.addEventListener("change", () => {
        loadVariations();
        loadExamples();
      });
      variationSelect.addEventListener("change", loadExamples);
      refreshButton.addEventListener("click", loadExamples);
      loadExamples();
    }

    init();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
