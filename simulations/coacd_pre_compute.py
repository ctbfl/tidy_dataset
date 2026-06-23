#!/usr/bin/env python3
"""Precompute a CoACD compound-convex collision mesh for one asset.

Run in the asset preprocessing environment:
    python simulations/coacd_pre_compute.py <asset_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

from objects import AssetLibrary

THRESHOLD = 0.01
MAX_CONVEX_HULL = 32
REAL_METRIC = True
OUTPUT_SUFFIX = "_coacd"
OUTPUT_EXT = ".obj"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python simulations/coacd_pre_compute.py <asset_id>")
    asset_id = sys.argv[1]

    try:
        import coacd
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency: install coacd in the preprocessing env") from exc

    library = AssetLibrary()
    asset = library[asset_id]
    asset_json_path = library.asset_json_path(asset_id)
    data = json.loads(asset_json_path.read_text())
    geometry = data["geometry"]
    collision_ref = geometry["collision_mesh"]
    if geometry.get("collision_shape") == "compound_convex":
        raise SystemExit(f"{asset_id} already uses compound_convex collision")

    src_mesh = asset.collision_mesh
    if src_mesh.stem.endswith(OUTPUT_SUFFIX):
        raise SystemExit(f"{asset_id} already points at a CoACD-looking mesh: {src_mesh}")
    out_mesh = src_mesh.with_name(f"{src_mesh.stem}{OUTPUT_SUFFIX}{OUTPUT_EXT}")
    mesh = _load_mesh(src_mesh)
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32)),
        threshold=THRESHOLD,
        max_convex_hull=MAX_CONVEX_HULL,
        real_metric=REAL_METRIC,
    )
    if not parts:
        raise RuntimeError(f"CoACD produced no parts for {asset_id}")

    _write_obj(out_mesh, parts)
    geometry["collision_mesh"] = _sibling_ref(collision_ref, out_mesh)
    geometry["collision_shape"] = "compound_convex"
    asset_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"{asset_id}")
    print(f"  input : {src_mesh}")
    print(f"  output: {out_mesh}")
    print(f"  parts : {len(parts)}")
    print(f"  asset : {asset_json_path}")


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"empty mesh: {path}")
    return loaded


def _write_obj(path: Path, parts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CoACD compound convex collision mesh"]
    vertex_offset = 0
    for i, (vertices, faces) in enumerate(parts):
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        lines.append(f"o coacd_{i:03d}")
        for x, y, z in vertices:
            lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
        for face in faces:
            a, b, c = face + vertex_offset + 1
            lines.append(f"f {a} {b} {c}")
        vertex_offset += len(vertices)
    path.write_text("\n".join(lines) + "\n")


def _sibling_ref(old_ref: dict, out_mesh: Path) -> dict:
    base = old_ref["base"]
    old_path = Path(old_ref["path"])
    new_path = old_path.with_name(out_mesh.name)
    if base == "absolute":
        return {"base": "absolute", "path": str(out_mesh.resolve())}
    return {"base": base, "path": new_path.as_posix()}


if __name__ == "__main__":
    main()
