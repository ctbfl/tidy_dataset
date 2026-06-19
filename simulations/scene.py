# open a empty scene with a random/assigned color, and with a random/assigned background.
# also table size is configurable.
# the scene does not have a robot, but you can add it later on.

import sys
import json
import numpy as np
from sapien import render
import sapien.core as sapien
from pathlib import Path
from sapien.render import set_global_config

OUR_OTHER_ASSETS_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/assets")

from robotwin_migration import TidyScene # Base_Task class from RoboTwin, include lots of scene initialization functions.
from objects import AssetLibrary, spawn

LIBRARY = AssetLibrary()


# only configure the table and background. As the mininum layer.
def create_scene(
    headless=False,
    use_hdri=False,
    table_length=1.2,
    table_width=0.7,
    table_height=0.74,
    table_thickness=0.05,
    table_texture_id=None,
    wall_texture_id=None,
    random_background=False,
    random_light = False,
    scene_fps=100.0,
    render_freq=10,
):
    tidy_scene = TidyScene(random_background=random_background)

    tidy_scene.random_light = random_light
    if headless==True and render_freq!=0:
        tidy_scene.render_freq = 0
        print("[GUI] render freq is for GUI rendering, force to zero in headless mode.")
    else:
        tidy_scene.render_freq = render_freq

    no_wall=True  # walls removed by default; pass wall_texture_id has no effect while this holds
    if use_hdri==True:
        have_ground=False
        no_default_light=True
    else:
        have_ground=True
        no_default_light=False

    tidy_scene.setup_scene(
        have_ground = have_ground,
        no_default_light = no_default_light,
        timestep = 1/scene_fps,
        ground_height = 0.0,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0,
        ambient_light=[0.5, 0.5, 0.5],
        shadow=True,
        direction_lights=[[[0, 0.5, -1], [0.5, 0.5, 0.5]]],
        point_lights=[[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]],
        camera_xyz_x=0.4,
        camera_xyz_y=0.22,
        camera_xyz_z=1.5,
        camera_rpy_r=0,
        camera_rpy_p=-0.8,
        camera_rpy_y=2.45
    )

    # HDRI environment support
    if use_hdri:
        tidy_scene.scene.set_environment_map(str(OUR_OTHER_ASSETS_ROOT / "hdri" / "sundowner_overlook_1k.exr"))

    # load table
    # after table loading we will have a self.tabletop_2D_area to use.
    tidy_scene.robotwin_create_table_and_wall(
        no_wall=no_wall,
        no_table=False,
        table_length=table_length,
        table_width=table_width,
        table_height=table_height,
        table_thickness=table_thickness,
        table_texture_id=table_texture_id,
        wall_texture_id=wall_texture_id,
    )
    tidy_scene.table = {"length": table_length, "width": table_width,
                        "height": table_height, "thickness": table_thickness}

    tidy_scene.objects = {}  # scene-unique id -> SceneObject

    return tidy_scene # contains .scene (SAPIEN scene) and .viewer (SAPIEN viewer) and other settings

def load_item(tidy_scene: TidyScene, item) -> str:
    asset = LIBRARY[item["asset_id"]]
    scene_id = f"{asset.id}#{len(tidy_scene.objects)}"
    obj = spawn(tidy_scene.scene, asset, scene_id)
    obj.set_pose(sapien.Pose(item["transform"]))
    tidy_scene.objects[scene_id] = obj
    return scene_id

def load_items(tidy_scene: TidyScene, item_list):
    for item in item_list:
        load_item(tidy_scene, item)


TIDY_SCENE_DIR = Path(__file__).resolve().parents[1] / "data" / "tidy_scene_v0"


def open_scene(json_path=TIDY_SCENE_DIR / "0001.json", headless=False, use_hdri=False, random_background=True) -> TidyScene:
    data = json.loads(Path(json_path).read_text())
    table = data["table"]
    tidy_scene = create_scene(
        headless=headless, use_hdri=use_hdri,
        table_length=table["length"], table_width=table["width"],
        table_height=table["height"], table_thickness=table["thickness"],
        random_background=True
    )
    load_items(tidy_scene, data["items"])
    return tidy_scene

def load_arm(tidy_scene: TidyScene, arms_json):
    # NOTE: may not be necessary in dataset creation
    pass

def look_at(eye, target, up=(0, 0, 1)) -> sapien.Pose:
    """Camera pose looking from `eye` to `target` (SAPIEN camera faces +x, +y left, +z up)."""
    eye, target = np.array(eye, float), np.array(target, float)
    forward = target - eye; forward /= np.linalg.norm(forward)
    left = np.cross(up, forward); left /= np.linalg.norm(left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    matrix[:3, 3] = eye
    return sapien.Pose(matrix)


def add_camera(tidy_scene: TidyScene, width=1024, height=768,
               eye=(0.0, -0.85, 1.6), target=(0.0, 0.0, 0.74), fovy=0.85):
    camera = tidy_scene.scene.add_camera("editor_cam", width, height, fovy=fovy, near=0.05, far=100)
    camera.set_local_pose(look_at(eye, target))
    tidy_scene.camera = camera
    return camera


# if __name__ == "__main__":
#     tidy_scene = create_scene(headless=False, random_background=True, use_hdri=False)
#     print("tidy_scene.scene: ", tidy_scene.scene)
#     print("tidy_scene.viewer: ", tidy_scene.viewer)
#     print("----------------------")
#     for name, value in vars(tidy_scene).items():
#         print(f"{name}: {value}")

#     i=0
#     while True:
#         tidy_scene.scene.step()
#         if tidy_scene.render_freq and i % tidy_scene.render_freq == 0:
#             tidy_scene._update_render()
#             tidy_scene.viewer.render()
#         i+=1
#         if i>=10000:
#             i=0

if __name__=="__main__":
    ts = open_scene(headless=False, use_hdri=False, random_background=True)
    i=0
    while True:
        ts.scene.step()
        if ts.render_freq and i % ts.render_freq == 0:
            ts._update_render()
            ts.viewer.render()
        i+=1
        if i>=10000:
            i=0