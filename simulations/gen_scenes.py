#!/usr/bin/env python3
"""Generate scene manifests for a scenario from a template.

For each scene, sample concrete assets per role -> write a v2 scene file with
`manifest` filled and `items` empty, ready for hand-annotation in handcraft.

Run (RoboTwin env):
    /home/hjs/miniforge3/envs/RoboTwin/bin/python simulations/gen_scenes.py --template office_desk --n 20
    .../python simulations/gen_scenes.py --template office_desk --n 5 --start 21 --seed 7

Writes to data/scenarios/<scenario>/<NNN>/tidy.json (scenario defaults to the
template id); each scene is a folder so other arrangements (messy, ...) can sit
beside tidy later. Existing scene files are skipped unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from objects import AssetLibrary, write_asset_json_backup
from templates import load_template, sample_manifest

REPO = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO / "data" / "scenarios"


def build_scene(template: dict, manifest: list[dict], scenario: str, scene_id: str,
                arrangement: str = "tidy") -> dict:
    return {
        "version": 2,
        "scenario": scenario,
        "scene_id": scene_id,
        "arrangement": arrangement,
        "template": template["template_id"],
        "table": template["table"],
        "table_texture": None,
        "wall_texture": None,
        "manifest": manifest,
        "items": [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", required=True, help="Template id (templates/<id>.json) or path.")
    p.add_argument("--n", type=int, default=10, help="Number of scenes to generate.")
    p.add_argument("--start", type=int, default=1, help="First scene number (zero-padded to 3 digits).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing scene files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    template = load_template(args.template)
    scenario = template.get("scenario", template["template_id"])
    library = AssetLibrary()
    rng = random.Random(args.seed)

    out_dir = SCENARIOS_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(args.n):
        scene_id = f"{args.start + k:03d}"
        path = out_dir / scene_id / "tidy.json"  # one folder per scene, tidy.json to hand-annotate
        manifest = sample_manifest(template, library, rng)  # draw before skip-check so the seed stays aligned
        if path.exists() and not args.overwrite:
            print(f"[skip]  {path.relative_to(REPO)} exists (use --overwrite)")
            continue
        scene = build_scene(template, manifest, scenario, scene_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scene, indent=2, ensure_ascii=False))
        write_asset_json_backup(path, scene, library)
        roles = ", ".join(m["slot"] for m in manifest)
        print(f"[write] {path.relative_to(REPO)}  {len(manifest)} items: {roles}")


if __name__ == "__main__":
    main()
