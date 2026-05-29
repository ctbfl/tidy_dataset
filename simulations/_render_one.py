"""Offscreen-render one texture set for manual review. Run per material in its own
process (avoids GPU renderer accumulation). Usage: python _render_one.py <wall|table> <id>"""
import sys
import numpy as np
import sapien.core as sapien
from robotwin_migration import TidyScene, _tile_repeat
from robotwin_utils import create_box, create_table, curated_textures
from scene import look_at

OUT = "/home/hjs/Projects/table_arrangement/tidy_dataset/tmp"
kind, mid = sys.argv[1], sys.argv[2]
walls = {w["id"]: w for w in curated_textures("wall")}
tables = {t["id"]: t for t in curated_textures("table")}

ts = TidyScene(random_background=False)
ts.render_freq = 0
ts.random_light = False
ts.setup_scene(have_ground=True, no_default_light=False, timestep=1 / 100,
               ambient_light=[0.5, 0.5, 0.5], shadow=True,
               direction_lights=[[[0, 0.5, -1], [0.5, 0.5, 0.5]]],
               point_lights=[[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]])

# Material under test gets its real texture; the other surface stays neutral.
if kind == "wall":
    w = walls[mid]
    create_box(ts.scene, sapien.Pose(p=[0, 1, 1.5]), half_size=[3, 0.6, 1.5], color=(1, 0.9, 0.9),
               name="wall", texture=w, texture_repeat=_tile_repeat((6, 3), w["tile_m"]), is_static=True)
    create_table(ts.scene, sapien.Pose(p=[0, 0, 0.74]), length=1.2, width=0.74, height=0.74,
                 thickness=0.05, is_static=True, color=(0.8, 0.8, 0.8))
else:
    t = tables[mid]
    create_box(ts.scene, sapien.Pose(p=[0, 1, 1.5]), half_size=[3, 0.6, 1.5], color=(0.9, 0.9, 0.9),
               name="wall", is_static=True)
    create_table(ts.scene, sapien.Pose(p=[0, 0, 0.74]), length=1.2, width=0.74, height=0.74,
                 thickness=0.05, is_static=True, texture=t, texture_repeat=_tile_repeat((1.2, 0.74), t["tile_m"]))

cam = ts.scene.add_camera("c", 1024, 768, fovy=0.85, near=0.05, far=100)
cam.set_local_pose(look_at((0.0, -0.85, 1.6), (0.0, 0.0, 0.74)))
ts.scene.step()
ts.scene.update_render()
cam.take_picture()
rgb = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
from PIL import Image
Image.fromarray(rgb).save(f"{OUT}/{kind}_{mid}.png")
print(f"{kind}/{mid}")
