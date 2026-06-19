#!/usr/bin/env python3
"""Generate a messy arrangement for each hand-annotated tidy scene.

For every data/scenarios/<scenario>/<NNN>/tidy.json this keeps the large
"anchor" objects (taller than --height-thresh, e.g. displays and lamps) exactly
where the tidy annotation put them, and randomly scatters every other object
upright on the table. Each candidate layout is validated by rendering a
segmentation image from the same camera the handcraft editor uses, and is only
accepted when:

  * every object is fully inside the frame,
  * every object shows at least --min-visible of its silhouette unoccluded, and
  * nothing intrudes into the cropped bottom-left / bottom-right corners.

The result is written next to tidy.json as messy.json with the *same* manifest
(same objects), differing only in `items` (placement) and `arrangement`.

Run (RoboTwin env):
    /home/hjs/miniforge3/envs/RoboTwin/bin/python simulations/gen_messy.py --scenario office_desk
    .../python simulations/gen_messy.py --scenario office_desk --overwrite --seed 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import sapien.core as sapien
from sapien import render

from scene import LIBRARY, add_camera, create_scene
from objects import spawn

REPO = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO / "data" / "scenarios"

CAM_W, CAM_H = 1024, 768
AWAY = sapien.Pose([0, 0, -50])  # park entities here to render one object in isolation


# -- asset geometry (stable-frame AABB, already scaled, in meters) --------- #
def asset_aabb(asset_id: str):
    g = json.loads(LIBRARY.asset_json_path(asset_id).read_text())["geometry"]["aabb_m"]
    return np.array(g["min"], float), np.array(g["max"], float), float(g["bottom_z"]), np.array(g["size"], float)


def footprint_radius(mn, mx) -> float:
    """Max distance from the stable-frame origin to a footprint corner, so any
    yaw keeps the whole object within this radius of its (x, y)."""
    return max(math.hypot(x, y) for x in (mn[0], mx[0]) for y in (mn[1], mx[1]))


def is_anchor(size, height_thresh: float, area_thresh: float | None) -> bool:
    if size[2] >= height_thresh:
        return True
    return bool(area_thresh and size[0] * size[1] >= area_thresh)


def yaw_pose(x: float, y: float, z: float, yaw: float) -> sapien.Pose:
    """Upright object: pure z-translation + yaw about world z (stable frame)."""
    return sapien.Pose([x, y, z], [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


# -- rendering ------------------------------------------------------------- #
def corner_mask(corner_frac: float) -> np.ndarray:
    """Quarter-circle exclusion zones in the bottom-left and bottom-right
    image corners, radius = corner_frac * min(W, H)."""
    r = corner_frac * min(CAM_W, CAM_H)
    ys, xs = np.mgrid[0:CAM_H, 0:CAM_W]
    bl = (xs - 0) ** 2 + (ys - (CAM_H - 1)) ** 2 < r * r
    br = (xs - (CAM_W - 1)) ** 2 + (ys - (CAM_H - 1)) ** 2 < r * r
    return bl | br


def segmentation(ts, cam) -> np.ndarray:
    ts.scene.update_render()
    cam.take_picture()
    return cam.get_picture("Segmentation")[..., 1]  # per-actor per_scene_id


def silhouette(ts, cam, objs: dict, key: str):
    """Unoccluded pixel area of one object (everything else parked out of view)
    plus which frame edges it touches: (top, bottom, left, right)."""
    saved = {k: o.entity.get_pose() for k, o in objs.items()}
    for k, o in objs.items():
        if k != key:
            o.entity.set_pose(AWAY)
    seg = segmentation(ts, cam)
    for k, o in objs.items():
        o.entity.set_pose(saved[k])
    mask = seg == objs[key].entity.per_scene_id
    edges = (bool(mask[0, :].any()), bool(mask[-1, :].any()),
             bool(mask[:, 0].any()), bool(mask[:, -1].any()))
    return int(mask.sum()), edges


# -- per-scene generation -------------------------------------------------- #
def clear_objects(ts) -> None:
    for o in list(ts.objects.values()):
        ts.scene.remove_entity(o.entity)
    ts.objects.clear()


def sample_layout(movers, anchors_xy, table, rng, gap, tries=200):
    """Random (x, y, yaw) per mover with footprints kept on the table and a
    loose min-gap from anchors and each other (a cheap pre-filter before the
    expensive render check). Returns a dict key -> (x, y, yaw) or None."""
    hx, hy = table["length"] / 2, table["width"] / 2
    for _ in range(tries):
        placed = list(anchors_xy)  # [(x, y, radius)]
        layout = {}
        ok = True
        for key, rad in movers:
            ix, iy = max(0.0, hx - rad), max(0.0, hy - rad)
            for _ in range(tries):
                x, y = rng.uniform(-ix, ix), rng.uniform(-iy, iy)
                if all(math.hypot(x - px, y - py) >= gap * (rad + pr) for px, py, pr in placed):
                    break
            else:
                ok = False
                break
            placed.append((x, y, rad))
            layout[key] = (x, y, rng.uniform(0, 2 * math.pi))
        if ok:
            return layout
    return None


def generate(ts, cam, tidy: dict, args, rng, cmask, obj_ids):
    table = tidy["table"]
    top = table["height"]

    clear_objects(ts)
    ts.random_background = False
    ts.robotwin_create_table_and_wall(
        no_wall=True, table_length=table["length"], table_width=table["width"],
        table_height=top, table_thickness=table["thickness"],
        table_texture_id=tidy.get("table_texture"), wall_texture_id=tidy.get("wall_texture"))

    # spawn every tidy item; classify anchor (fixed pose) vs mover (scattered)
    objs, meta, anchors, movers = {}, {}, [], []
    for i, item in enumerate(tidy.get("items", [])):
        key = str(i)
        mn, mx, bottom_z, size = asset_aabb(item["asset_id"])
        obj = spawn(ts.scene, LIBRARY[item["asset_id"]], f'{item["asset_id"]}#{key}')
        ts.objects[key] = objs[key] = obj
        meta[key] = {"slot": item.get("slot"), "asset_id": item["asset_id"],
                     "bottom_z": bottom_z, "radius": footprint_radius(mn, mx), "aabb_xy": (mn[:2] + mx[:2]) / 2}
        if is_anchor(size, args.height_thresh, args.area_thresh):
            obj.set_pose(sapien.Pose(np.asarray(item["transform"], float)))
            anchors.append(key)
        else:
            movers.append(key)

    if not objs:
        return None, 0, "no items"
    obj_ids[:] = [o.entity.per_scene_id for o in objs.values()]

    # Anchors keep their (trusted) tidy pose, so we don't reject them for
    # touching a frame edge (a tall lamp legitimately reaches the top); we only
    # need their unoccluded area, cached once since they never move.
    anchor_full = {}
    for key in anchors:
        full, _ = silhouette(ts, cam, objs, key)
        if full == 0:
            return None, 0, f"anchor {meta[key]['slot'] or key} not visible in tidy"
        anchor_full[key] = full

    anchors_xy = [(*objs[key].get_pose().p[:2], meta[key]["radius"]) for key in anchors]
    mover_radii = [(key, meta[key]["radius"]) for key in movers]

    for attempt in range(1, args.max_attempts + 1):
        layout = sample_layout(mover_radii, anchors_xy, table, rng, args.gap)
        if layout is None:
            continue
        for key, (x, y, yaw) in layout.items():
            objs[key].set_pose(yaw_pose(x, y, top - meta[key]["bottom_z"], yaw))

        seg = segmentation(ts, cam)
        if np.isin(seg[cmask], obj_ids).any():          # something in a cropped corner
            continue
        full = {**anchor_full}
        cut = False
        for key in movers:
            full[key], (_, bottom, left, right) = silhouette(ts, cam, objs, key)
            cut = cut or full[key] == 0 or bottom or left or right  # mover off the view edge
        if cut:
            continue
        visible = {k: int((seg == objs[k].entity.per_scene_id).sum()) for k in objs}
        if all(visible[k] / full[k] >= args.min_visible for k in objs):
            items = [{"slot": meta[k]["slot"], "asset_id": meta[k]["asset_id"],
                      "transform": objs[k].get_pose().to_transformation_matrix().tolist()}
                     for k in objs]
            return items, attempt, "ok"
    return None, args.max_attempts, "no valid layout"


def messy_dict(tidy: dict, items: list) -> dict:
    out = dict(tidy)               # same manifest / table / textures / template
    out["arrangement"] = "messy"
    out["items"] = items
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="office_desk", help="data/scenarios/<scenario>/")
    p.add_argument("--scenes", nargs="*", help="Limit to these scene folders (e.g. 001 002).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (offset per scene for independence).")
    p.add_argument("--overwrite", action="store_true", help="Regenerate scenes that already have messy.json.")
    p.add_argument("--height-thresh", type=float, default=0.30, help="Anchor objects at least this tall (m).")
    p.add_argument("--area-thresh", type=float, default=None, help="Also anchor footprints >= this (m^2); off by default.")
    p.add_argument("--min-visible", type=float, default=0.80, help="Min unoccluded fraction per object.")
    p.add_argument("--corner-frac", type=float, default=0.25, help="Cropped corner radius as a fraction of min(W,H).")
    p.add_argument("--gap", type=float, default=0.5, help="Min center gap as a fraction of summed footprint radii.")
    p.add_argument("--max-attempts", type=int, default=300, help="Render-validated layouts to try before skipping.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = SCENARIOS_DIR / args.scenario
    scene_dirs = sorted(d for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
    if args.scenes:
        wanted = set(args.scenes)
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
    if not scene_dirs:
        print(f"no scenes under {base.relative_to(REPO)}")
        return

    ts = create_scene(headless=True, table_length=1.2, table_width=0.7, table_height=0.74)
    cam = add_camera(ts, width=CAM_W, height=CAM_H)
    render.set_ray_tracing_samples_per_pixel(1)  # segmentation is geometric; skip the costly color samples
    render.set_ray_tracing_path_depth(1)
    cmask = corner_mask(args.corner_frac)

    written = skipped = 0
    for idx, d in enumerate(scene_dirs):
        tidy_path, out_path = d / "tidy.json", d / "messy.json"
        if not tidy_path.is_file():
            print(f"[skip]  {d.name}: no tidy.json")
            continue
        if out_path.exists() and not args.overwrite:
            print(f"[skip]  {d.name}: messy.json exists (use --overwrite)")
            continue
        tidy = json.loads(tidy_path.read_text())
        rng = random.Random(args.seed + idx)
        items, attempts, status = generate(ts, cam, tidy, args, rng, cmask, [])
        if items is None:
            print(f"[FAIL]  {d.name}: {status} ({attempts} attempts)")
            skipped += 1
            continue
        out_path.write_text(json.dumps(messy_dict(tidy, items), indent=2, ensure_ascii=False))
        n_anchor = sum(1 for it in tidy.get("items", [])
                       if is_anchor(asset_aabb(it["asset_id"])[3], args.height_thresh, args.area_thresh))
        print(f"[write] {args.scenario}/{d.name}/messy.json  "
              f"{len(items)} items ({n_anchor} anchored)  attempt {attempts}")
        written += 1

    print(f"\ndone: {written} written, {skipped} failed")


if __name__ == "__main__":
    main()
