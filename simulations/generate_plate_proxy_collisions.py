from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh


SGBOT_ROOT = Path("/home/hjs/Datasets/sgbot/sgbot_dataset")
ASSET_LIBRARY_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/data/asset_library")
SCENE_ROOT = Path(
    "/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup"
)

PLATES = (1, 2, 4, 5, 6, 11)
PIECES = 24
BASE_RADIUS_RATIO = 0.478
BASE_TOP_HEIGHT_RATIO = 0.0904
RIM_BOTTOM_HEIGHT_RATIO = 0.824


def main() -> None:
    for plate_id in PLATES:
        proxy = write_proxy(plate_id)
        update_asset_jsons(plate_id, proxy)


def write_proxy(plate_id: int) -> Path:
    visual = SGBOT_ROOT / "models" / "plate" / f"plate_{plate_id}.obj"
    mesh = trimesh.load(visual, force="mesh", process=False)
    verts = np.asarray(mesh.vertices)
    z_min = float(verts[:, 2].min())
    z_max = float(verts[:, 2].max())
    height = z_max - z_min
    outer_r = float(np.linalg.norm(verts[:, :2], axis=1).max())
    base_r = outer_r * BASE_RADIUS_RATIO
    base_top_z = z_min + height * BASE_TOP_HEIGHT_RATIO
    rim_bottom_z = z_min + height * RIM_BOTTOM_HEIGHT_RATIO

    out_path = SGBOT_ROOT / "collision" / "plate" / f"plate_{plate_id}_proxy_low.obj"
    lines: list[str] = [
        f"# Low-complexity compound convex proxy for sgbot:obj:plate:{plate_id}",
        "# Generated from visual bounds; raw units, asset scale remains 0.2",
        f"# pieces: 1 base cylinder + {PIECES} sloped rim wedges",
    ]
    vid = 0

    def add_vertex(x: float, y: float, z: float) -> int:
        nonlocal vid
        vid += 1
        lines.append(f"v {_fmt(x)} {_fmt(y)} {_fmt(z)}")
        return vid

    def add_face(*idx: int) -> None:
        lines.append("f " + " ".join(str(i) for i in idx))

    angles = [2.0 * math.pi * i / PIECES for i in range(PIECES + 1)]
    unit = [(math.cos(a), math.sin(a)) for a in angles]

    lines.append(f"o plate{plate_id}_proxy_base")
    bottom_center = add_vertex(0.0, 0.0, z_min)
    top_center = add_vertex(0.0, 0.0, base_top_z)
    bottom = [add_vertex(base_r * x, base_r * y, z_min) for x, y in unit[:-1]]
    top = [add_vertex(base_r * x, base_r * y, base_top_z) for x, y in unit[:-1]]
    for i in range(PIECES):
        j = (i + 1) % PIECES
        add_face(bottom_center, bottom[j], bottom[i])
        add_face(top_center, top[i], top[j])
        add_face(bottom[i], bottom[j], top[j], top[i])

    for i in range(PIECES):
        j = i + 1
        xi, yi = unit[i]
        xj, yj = unit[j]
        lines.append(f"o plate{plate_id}_proxy_rim_{i}")
        a = add_vertex(base_r * xi, base_r * yi, base_top_z)
        b = add_vertex(base_r * xj, base_r * yj, base_top_z)
        c = add_vertex(outer_r * xj, outer_r * yj, z_max)
        d = add_vertex(outer_r * xi, outer_r * yi, z_max)
        e = add_vertex(base_r * xi, base_r * yi, z_min)
        f = add_vertex(base_r * xj, base_r * yj, z_min)
        g = add_vertex(outer_r * xj, outer_r * yj, rim_bottom_z)
        h = add_vertex(outer_r * xi, outer_r * yi, rim_bottom_z)
        add_face(a, b, c, d)
        add_face(e, h, g, f)
        add_face(a, e, f, b)
        add_face(d, c, g, h)
        add_face(a, d, h, e)
        add_face(b, f, g, c)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def update_asset_jsons(plate_id: int, proxy: Path) -> None:
    asset_id = f"sgbot:obj:plate:{plate_id}"
    rel_path = f"collision/plate/plate_{plate_id}_proxy_low.obj"
    paths = [
        ASSET_LIBRARY_ROOT / "assets" / f"sgbot_obj_plate_{plate_id}" / "asset.json",
        *sorted(SCENE_ROOT.glob(f"*/asset_json_backup/{asset_id}.json")),
    ]
    bounds = _scaled_bounds(proxy, scale=0.2)
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        geom = data["geometry"]
        geom["collision_mesh"] = {"base": "source_root", "path": rel_path}
        geom["pybullet_collision_mesh"] = {"base": "absolute", "path": str(proxy)}
        geom["collision_shape"] = "compound_convex"
        geom["aabb_m"] = bounds
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"plate_{plate_id}: wrote {proxy} and updated {len(paths)} json files")


def _scaled_bounds(path: Path, scale: float) -> dict:
    mesh = trimesh.load(path, force="mesh", process=False)
    bounds = np.asarray(mesh.bounds, dtype=float) * scale
    center = (bounds[0] + bounds[1]) * 0.5
    size = bounds[1] - bounds[0]
    return {
        "frame": "stable",
        "unit": "m",
        "source": "pybullet_collision_mesh",
        "source_mesh": str(path),
        "min": bounds[0].round(12).tolist(),
        "max": bounds[1].round(12).tolist(),
        "aabb_center": center.round(12).tolist(),
        "size": size.round(12).tolist(),
        "bottom_z": float(round(bounds[0, 2], 12)),
    }


def _fmt(value: float) -> str:
    return f"{value:.9g}"


if __name__ == "__main__":
    main()
