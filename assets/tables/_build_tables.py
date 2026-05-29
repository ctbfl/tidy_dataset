#!/usr/bin/env python3
"""Repackage the 13 curated support tables into self-contained, simply-named folders.

For each table we emit assets/tables/<name>/ containing:
  visual.obj   (mtllib rewritten to ./visual.mtl)
  visual.mtl   (map_* rewritten to ./texture.png)
  texture.png  (the table's own copy of its texture)
  table.urdf   (mesh -> visual.obj, texture -> texture.png; physics params preserved)
  meta.json    (source hash, category, tabletop surface info, scale)

Idempotent: rerunning overwrites the output folders.
"""
import json
import re
import shutil
from pathlib import Path

SRC = Path("/home/hjs/Datasets/sgbot/sgbot_dataset/models/support_table")
OUT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/assets/tables")

# hash -> (category, semantic name).  Mapping established by reverse-lookup in the
# shapeNet_support category folders.
TABLES = {
    "61fe7cce3cc4b7f2f1783a44a88d6274": ("DiningTable", "dining_table_01"),
    "647692d3858790a1f1783a44a88d6274": ("DiningTable", "dining_table_02"),
    "bf886e6f28740776f1783a44a88d6274": ("DiningTable", "dining_table_03"),
    "c0ec7cca02bd2225f1783a44a88d6274": ("DiningTable", "dining_table_04"),
    "da2f2572b10c0ed8f1783a44a88d6274": ("DiningTable", "dining_table_05"),
    "e6a188bbf8315d11f1783a44a88d6274": ("DiningTable", "dining_table_06"),
    "2362ec480b3e9baa4fd5721982c508ad": ("EndTable", "end_table_01"),
    "466ed029d1acae054199ce3660e593e":  ("EndTable", "end_table_02"),
    "7b4acb843fd4b0b335836c728d324152": ("EndTable", "end_table_03"),
    "90a157c1a98305296061b394380b4b5a": ("CoffeeTable", "coffee_table_01"),
    "faa5d5ba2a002922511e5b9dc733c75c": ("CoffeeTable", "coffee_table_02"),
    "c6e30f76334e5872822a33e080d0e71c": ("RoundTable", "round_table_01"),
    "dccb87aacbcb40a4f1783a44a88d6274": ("RoundTable", "round_table_02"),
}

tabletop = json.loads((SRC / "tabletop_area.json").read_text())


def compute_tabletop(obj_path: Path, eps: float = 0.01) -> dict:
    """Fallback tabletop estimate: x/y bbox of the vertices at the top z-slab."""
    vs = []
    for ln in obj_path.read_text().splitlines():
        if ln.startswith("v "):
            p = ln.split()
            vs.append((float(p[1]), float(p[2]), float(p[3])))
    zmax = max(v[2] for v in vs)
    top = [v for v in vs if v[2] >= zmax - eps]
    xs = [v[0] for v in top]
    ys = [v[1] for v in top]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, d = round(x1 - x0, 5), round(y1 - y0, 5)
    return {
        "top_z": round(zmax, 5),
        "x_range": [round(x0, 5), round(x1, 5)],
        "y_range": [round(y0, 5), round(y1, 5)],
        "width": w, "depth": d, "area_m2": round(w * d, 5),
    }


def texture_for(obj_mtl: Path) -> str:
    """Return the texture filename referenced by an .obj.mtl (first map_Kd)."""
    for line in obj_mtl.read_text().splitlines():
        m = re.match(r"\s*map_K[ad]\s+(\S+)", line, re.I)
        if m:
            return m.group(1)
    raise RuntimeError(f"no texture in {obj_mtl}")


def build(hash_id: str, category: str, name: str) -> dict:
    dst = OUT / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    tex_name = texture_for(SRC / f"{hash_id}.obj.mtl")
    tex_src = SRC / tex_name
    if not tex_src.exists():  # fall back to same-stem png
        tex_src = SRC / f"{Path(tex_name).stem}.png"
    shutil.copy(tex_src, dst / "texture.png")

    # visual.mtl: point every texture map at the local copy
    mtl = SRC.joinpath(f"{hash_id}.obj.mtl").read_text()
    mtl = re.sub(r"(map_K[ads]\s+)\S+", r"\1texture.png", mtl, flags=re.I)
    (dst / "visual.mtl").write_text(mtl)

    # visual.obj: repoint mtllib at visual.mtl
    obj = SRC.joinpath(f"{hash_id}.obj").read_text()
    obj = re.sub(r"mtllib\s+\S+", "mtllib ./visual.mtl", obj)
    (dst / "visual.obj").write_text(obj)

    # table.urdf: preserve original physics, rewrite mesh + texture + robot name
    urdf = SRC.joinpath(f"{hash_id}.urdf").read_text()
    urdf = re.sub(r'filename="[^"]*\.obj"', 'filename="visual.obj"', urdf)
    urdf = re.sub(r'filename="[^"]*\.png"', 'filename="texture.png"', urdf)
    urdf = re.sub(r'<robot name="[^"]*"', f'<robot name="{name}"', urdf)
    (dst / "table.urdf").write_text(urdf)

    tt = tabletop.get(hash_id)
    tt_source = "provided"
    if tt is None:
        tt = compute_tabletop(dst / "visual.obj")
        tt_source = "computed"

    meta = {
        "name": name,
        "category": category,
        "source_hash": hash_id,
        "source_dataset": "sgbot/sgbot_dataset/models/support_table",
        "scale": 1.0,
        "tabletop": tt,
        "tabletop_source": tt_source,
    }
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    rows = []
    for hash_id, (category, name) in TABLES.items():
        meta = build(hash_id, category, name)
        tt = meta["tabletop"] or {}
        rows.append((name, category, tt.get("width"), tt.get("depth"), tt.get("top_z")))
    print(f"Built {len(rows)} tables into {OUT}\n")
    print(f"{'name':<18}{'category':<14}{'width':>8}{'depth':>8}{'top_z':>8}")
    for name, cat, w, d, z in rows:
        print(f"{name:<18}{cat:<14}{w!s:>8}{d!s:>8}{z!s:>8}")


if __name__ == "__main__":
    main()
