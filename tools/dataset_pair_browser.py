#!/usr/bin/env python3
"""Live quality-inspection browser for the organize_it_dataset_v2 dataset.

Layout on disk: <scenario>/<variation>/<sample>/{current.png, reference_goal.png}

The page lets you focus on one scenario/variation and shows every sample as a
pair of images (left: current.png, right: reference_goal.png), 4 pairs per row.
The focused directory is polled once per second, so newly added samples/images,
changed images, and removed samples show up live without a manual refresh.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file


REPO_ROOT = Path(__file__).resolve().parents[1]
# Target dataset dir; override with TIDY_DATASET_DIR (or pass --dataset at runtime).
DEFAULT_ROOT = Path(os.environ.get("TIDY_DATASET_DIR", REPO_ROOT / "data" / "organize_it_dataset_v2"))

# key -> filename inside a sample directory
TARGETS = {"current": "current.png", "reference": "reference_goal.png"}


def _validate_name(name: str) -> str:
    """Reject anything that could escape the dataset root."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        abort(400, description=f"invalid path segment: {name!r}")
    return name


def _sample_sort_key(path: Path):
    name = path.name
    return (0, int(name), "") if name.isdigit() else (1, 0, name)


def _stat_entry(path: Path) -> dict | None:
    if not path.is_file():
        return None
    st = path.stat()
    # millisecond mtime doubles as a cache-buster for the <img> src
    return {"mtime": int(st.st_mtime_ns // 1_000_000), "size": st.st_size}


def build_app(data_root: Path) -> Flask:
    app = Flask(__name__)
    root = data_root.resolve()

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/tree")
    def api_tree():
        if not root.is_dir():
            abort(404, description=f"dataset root not found: {root}")
        scenarios = []
        for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
            variations = sorted(v.name for v in scenario.iterdir() if v.is_dir())
            scenarios.append({"name": scenario.name, "variations": variations})
        return jsonify({"root": str(root), "scenarios": scenarios})

    @app.get("/api/samples")
    def api_samples():
        scenario = _validate_name(request.args.get("scenario", ""))
        variation = _validate_name(request.args.get("variation", ""))
        vdir = root / scenario / variation
        if not vdir.is_dir():
            return jsonify({"exists": False, "samples": [], "summary": {}})

        samples = []
        totals = {"samples": 0, "current": 0, "reference": 0, "complete": 0}
        for sub in sorted((p for p in vdir.iterdir() if p.is_dir()), key=_sample_sort_key):
            entry = {"name": sub.name}
            present = 0
            for key, fname in TARGETS.items():
                info = _stat_entry(sub / fname)
                entry[key] = info
                if info is not None:
                    present += 1
                    totals[key] += 1
            if present == 0:
                continue  # skip helper dirs like `template` that have no pair images
            totals["samples"] += 1
            if present == len(TARGETS):
                totals["complete"] += 1
            samples.append(entry)

        return jsonify({"exists": True, "samples": samples, "summary": totals})

    @app.get("/img")
    def img():
        scenario = _validate_name(request.args.get("scenario", ""))
        variation = _validate_name(request.args.get("variation", ""))
        sample = _validate_name(request.args.get("sample", ""))
        which = request.args.get("which", "")
        fname = TARGETS.get(which)
        if fname is None:
            abort(400, description=f"unknown image: {which!r}")
        path = (root / scenario / variation / sample / fname).resolve()
        if root not in path.parents or not path.is_file():
            abort(404)
        return send_file(path, max_age=0)

    return app


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Pair Browser</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #0e1014; color: #e8ebf0; }
  header { position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
           padding: 10px 16px; background: #161a21; border-bottom: 1px solid #262c36; }
  header h1 { font-size: 15px; margin: 0 8px 0 0; font-weight: 700; color: #cfd6e2; white-space: nowrap; }
  label { font-size: 12px; color: #9aa4b4; display: inline-flex; align-items: center; gap: 6px; }
  select { background: #0e1014; color: #e8ebf0; border: 1px solid #333b47; border-radius: 6px; padding: 6px 8px;
           font-size: 13px; }
  select:hover { border-color: #4a556a; }
  #summary { font-size: 12px; color: #9aa4b4; margin-left: auto; white-space: nowrap; }
  #summary b { color: #dfe5ee; }
  .warn { color: #f0a35e !important; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #3a4150; display: inline-block; }
  .dot.live { background: #46c266; box-shadow: 0 0 6px #46c266; }
  .dot.err { background: #d15b4c; box-shadow: 0 0 6px #d15b4c; }
  #grid { display: grid; grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr)); gap: 12px; padding: 16px; }
  .card { background: #151922; border: 1px solid #242b36; border-radius: 8px; overflow: hidden; }
  .card-head { padding: 5px 9px; font-size: 12px; font-weight: 700; color: #cbd3e0; background: #1b212c;
               border-bottom: 1px solid #242b36; display: flex; justify-content: space-between; align-items: center; }
  .card-head .miss { color: #e0714f; font-weight: 700; font-size: 11px; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: #0a0c10; }
  .cell { position: relative; display: block; background: #0a0c10; }
  .cell img { display: block; width: 100%; height: var(--imgh, 168px); object-fit: contain; background: #060709; }
  .cell .placeholder { display: flex; align-items: center; justify-content: center; width: 100%; height: var(--imgh, 168px);
                       background: #241416; color: #e0714f; font-size: 12px; text-align: center; padding: 6px; }
  .cell .tag { position: absolute; left: 4px; bottom: 4px; font-size: 10px; padding: 1px 5px; border-radius: 4px;
               background: rgba(6,7,9,0.78); color: #aeb7c6; pointer-events: none; }
  #empty { padding: 40px; text-align: center; color: #8a93a3; font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>Dataset Pairs</h1>
  <label>scenario
    <select id="scenario"></select>
  </label>
  <label>variation
    <select id="variation"></select>
  </label>
  <label>cols
    <select id="cols">
      <option>2</option><option>3</option><option selected>4</option><option>6</option><option>8</option>
    </select>
  </label>
  <label>size
    <select id="size">
      <option value="120">S</option><option value="168" selected>M</option><option value="240">L</option><option value="340">XL</option>
    </select>
  </label>
  <button id="reload" style="background:#232b38;color:#cbd3e0;border:1px solid #333b47;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer">Rescan scenarios</button>
  <span id="summary"><span class="dot" id="dot"></span> <span id="summaryText">loading…</span></span>
</header>
<div id="grid"></div>
<div id="empty" style="display:none"></div>

<script>
const $ = (id) => document.getElementById(id);
const grid = $("grid");
const dot = $("dot");
let tree = { scenarios: [] };
let focus = { scenario: "", variation: "" };
const cards = new Map();       // name -> { el, headMiss, cells: {current, reference} }
let polling = false;

function enc(v) { return encodeURIComponent(v); }

function setDot(state) { dot.className = "dot" + (state ? " " + state : ""); }

async function loadTree(preserve = true) {
  const prev = { ...focus };
  try {
    const res = await fetch("/api/tree");
    tree = await res.json();
  } catch (e) {
    $("summaryText").textContent = "tree error: " + e.message;
    setDot("err");
    return;
  }
  const scenarioSel = $("scenario");
  scenarioSel.innerHTML = "";
  for (const s of tree.scenarios) {
    const opt = document.createElement("option");
    opt.value = s.name; opt.textContent = s.name;
    scenarioSel.appendChild(opt);
  }
  if (preserve && tree.scenarios.some((s) => s.name === prev.scenario)) {
    scenarioSel.value = prev.scenario;
  }
  populateVariations(preserve ? prev.variation : "");
}

function currentScenario() {
  return tree.scenarios.find((s) => s.name === $("scenario").value) || null;
}

function populateVariations(preferred = "") {
  const variationSel = $("variation");
  variationSel.innerHTML = "";
  const scenario = currentScenario();
  const variations = scenario ? scenario.variations : [];
  for (const v of variations) {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    variationSel.appendChild(opt);
  }
  if (preferred && variations.includes(preferred)) variationSel.value = preferred;
  applyFocus();
}

function applyFocus() {
  const nextScenario = $("scenario").value;
  const nextVariation = $("variation").value;
  if (nextScenario !== focus.scenario || nextVariation !== focus.variation) {
    focus = { scenario: nextScenario, variation: nextVariation };
    grid.innerHTML = "";
    cards.clear();
  }
  poll();
}

function makeCell(sampleName, which, state) {
  const cell = document.createElement("a");
  cell.className = "cell";
  cell.dataset.which = which;
  cell.dataset.sample = sampleName;
  fillCell(cell, which, state);
  return cell;
}

function cellStateKey(state) { return state ? String(state.mtime) : "x"; }

function fillCell(cell, which, state) {
  cell.dataset.state = cellStateKey(state);
  cell.innerHTML = "";
  if (state) {
    const url = `/img?scenario=${enc(focus.scenario)}&variation=${enc(focus.variation)}`
              + `&sample=${enc(cell.dataset.sample)}&which=${which}&v=${state.mtime}`;
    cell.href = url;
    cell.target = "_blank";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = url;
    cell.appendChild(img);
  } else {
    cell.removeAttribute("href");
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.textContent = TARGET_LABEL[which] + "\nmissing";
    cell.appendChild(ph);
  }
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = TARGET_LABEL[which];
  cell.appendChild(tag);
}

const TARGET_LABEL = { current: "current.png", reference: "reference_goal.png" };

function makeCard(sample) {
  const el = document.createElement("div");
  el.className = "card";
  el.dataset.name = sample.name;

  const head = document.createElement("div");
  head.className = "card-head";
  const nm = document.createElement("span");
  nm.textContent = sample.name;
  const miss = document.createElement("span");
  miss.className = "miss";
  head.append(nm, miss);
  el.appendChild(head);

  const pair = document.createElement("div");
  pair.className = "pair";
  const cells = {};
  for (const which of ["current", "reference"]) {
    const cell = makeCell(sample.name, which, sample[which]);
    pair.appendChild(cell);
    cells[which] = cell;
  }
  el.appendChild(pair);

  const card = { el, miss, cells };
  updateMiss(card, sample);
  return card;
}

function updateMiss(card, sample) {
  const missing = ["current", "reference"].filter((w) => !sample[w]);
  card.miss.textContent = missing.length ? "no " + missing.map((w) => w).join(" & ") : "";
}

function reconcile(samples) {
  const seen = new Set();
  for (const sample of samples) {
    seen.add(sample.name);
    let card = cards.get(sample.name);
    if (!card) {
      card = makeCard(sample);
      cards.set(sample.name, card);
    } else {
      for (const which of ["current", "reference"]) {
        const cell = card.cells[which];
        if (cell.dataset.state !== cellStateKey(sample[which])) {
          fillCell(cell, which, sample[which]);
        }
      }
      updateMiss(card, sample);
    }
  }
  // remove vanished samples
  for (const [name, card] of cards) {
    if (!seen.has(name)) {
      card.el.remove();
      cards.delete(name);
    }
  }
  // enforce order (moving nodes does not reload their <img>)
  for (const sample of samples) {
    const card = cards.get(sample.name);
    if (card) grid.appendChild(card.el);
  }
}

function renderSummary(data) {
  if (!data.exists) {
    $("summaryText").textContent = `${focus.scenario}/${focus.variation} — directory not found`;
    $("summaryText").className = "warn";
    return;
  }
  const s = data.summary;
  const incomplete = s.samples - s.complete;
  const parts = [
    `<b>${s.samples}</b> samples`,
    `<b>${s.current}</b> current`,
    `<b>${s.reference}</b> reference`,
  ];
  let text = parts.join(" · ");
  if (incomplete > 0) text += ` · <span class="warn">${incomplete} incomplete</span>`;
  $("summaryText").innerHTML = text;
  $("summaryText").className = "";
}

async function poll() {
  if (polling) return;
  if (!focus.scenario || !focus.variation) {
    grid.innerHTML = "";
    cards.clear();
    $("empty").style.display = "block";
    $("empty").textContent = "No scenario/variation selected.";
    setDot("");
    return;
  }
  polling = true;
  try {
    const res = await fetch(`/api/samples?scenario=${enc(focus.scenario)}&variation=${enc(focus.variation)}`);
    const data = await res.json();
    if (!data.exists) {
      grid.innerHTML = "";
      cards.clear();
      $("empty").style.display = "block";
      $("empty").textContent = `${focus.scenario}/${focus.variation} no longer exists.`;
    } else if (data.samples.length === 0) {
      grid.innerHTML = "";
      cards.clear();
      $("empty").style.display = "block";
      $("empty").textContent = "No samples with current.png / reference_goal.png here yet.";
    } else {
      $("empty").style.display = "none";
      reconcile(data.samples);
    }
    renderSummary(data);
    setDot("live");
  } catch (e) {
    $("summaryText").textContent = "poll error: " + e.message;
    setDot("err");
  } finally {
    polling = false;
  }
}

$("scenario").addEventListener("change", () => populateVariations(""));
$("variation").addEventListener("change", applyFocus);
$("reload").addEventListener("click", () => loadTree(true));
$("cols").addEventListener("change", () => document.documentElement.style.setProperty("--cols", $("cols").value));
$("size").addEventListener("change", () => document.documentElement.style.setProperty("--imgh", $("size").value + "px"));

document.documentElement.style.setProperty("--cols", "4");
document.documentElement.style.setProperty("--imgh", "168px");

loadTree(false).then(poll);
setInterval(poll, 1000);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_ROOT), help="dataset root directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8130)
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    print(f"[dataset pair browser] dataset={root}")
    print(f"[dataset pair browser] http://{args.host}:{args.port}")
    app = build_app(root)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
