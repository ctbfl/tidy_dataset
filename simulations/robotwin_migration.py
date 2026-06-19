from pathlib import Path

import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from robotwin_utils import create_box, create_table, curated_textures, curated_texture


def _tile_repeat(surface_wh, tile_m):
    """How many times a tile_m-sized texture repeats across a surface (w, h) in meters."""
    return (max(1, round(surface_wh[0] / tile_m)), max(1, round(surface_wh[1] / tile_m)))


# Tabletop can be a plain solid colour instead of a PBR texture, at this rate.
TABLE_SOLID_RATE = 0.30
TABLE_SOLID_COLORS = [(0.94, 0.94, 0.94), (0.91, 0.88, 0.81), (0.78, 0.78, 0.78),
                      (0.29, 0.29, 0.29), (0.85, 0.75, 0.63)]




class TidyScene:
    """
    Clean migration from RoboTwin.envs.Base_Task
    """
    def __init__(self, random_background):
        self.crazy_random_light = False # TODO: add randomization fundation
        self.random_background = random_background
        self.clean_background_rate = 0 # if random number smaller than this threshold, will be default white background
        self.table_z_bias = 0
        self.eval_mode = False

        # crazy_random_light = (0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate)

    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.
        """
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("optix")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        # set simulation timestep
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        # add ground to scene
        have_ground = kwargs.get("have_ground", True)
        if have_ground:
            self.scene.add_ground(kwargs.get("ground_height", 0))
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        no_default_light = kwargs.get("no_default_light", False)
        if not no_default_light:
            print("default lighting loaded")
            # give some white ambient light of moderate intensity
            self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
            # default enable shadow unless specified otherwise
            shadow = kwargs.get("shadow", True)
            # default spotlight angle and intensity
            direction_lights = kwargs.get("direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]])
            self.direction_light_lst = []
            for direction_light in direction_lights:
                if self.random_light:
                    direction_light[1] = [
                        np.random.rand(),
                        np.random.rand(),
                        np.random.rand(),
                    ]
                self.direction_light_lst.append(
                    self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=shadow))
            # default point lights position and intensity
            point_lights = kwargs.get("point_lights", [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]])
            self.point_light_lst = []
            for point_light in point_lights:
                if self.random_light:
                    point_light[1] = [np.random.rand(), np.random.rand(), np.random.rand()]
                self.point_light_lst.append(self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow))
        else:
            print("No default lighting set, make sure you have other lighting like hdri")
        # initialize viewer with camera position and orientation
        if self.render_freq:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            # GUI starts at the handcraft/add_camera view: eye=(0,-0.85,1.6) looking
            # at (0,0,0.74). rpy precomputed from that look-at for SAPIEN's FPS camera.
            self.viewer.set_camera_xyz(x=0.0, y=-0.85, z=1.6)
            self.viewer.set_camera_rpy(r=0.0, p=-0.7912, y=-1.5708)
    
    def robotwin_create_table_and_wall(self,
        table_xy_bias=[0, 0],
        table_length=1.2,
        table_width=0.74,
        table_height=0.74,
        table_thickness=0.05,
        no_wall=True,
        no_table=False,
        table_texture_id=None,
        wall_texture_id=None,
        ):
        self.table_xy_bias = table_xy_bias
        table_height += self.table_z_bias

        # Drop any previously built table/wall so this can be called again to rebuild.
        for attr in ("table_entity", "wall_entity"):
            ent = getattr(self, attr, None)
            if ent is not None:
                self.scene.remove_entity(ent)
                setattr(self, attr, None)

        if self.random_background:
            # Random background takes priority over any manual texture ids.
            walls = curated_textures("wall")
            tables = curated_textures("table")
            self.wall_texture = walls[np.random.randint(len(walls))] if walls else None
            # Table: mostly a PBR set, sometimes a plain solid colour.
            if tables and np.random.rand() > TABLE_SOLID_RATE:
                self.table_texture = tables[np.random.randint(len(tables))]
                self.table_color = (1, 1, 1)
            else:
                self.table_texture = None
                self.table_color = TABLE_SOLID_COLORS[np.random.randint(len(TABLE_SOLID_COLORS))]

            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
        else:
            # Manual mode: use the requested texture ids (None -> plain).
            self.wall_texture = curated_texture("wall", wall_texture_id) if wall_texture_id else None
            self.table_texture = curated_texture("table", table_texture_id) if table_texture_id else None
            self.table_color = (1, 1, 1)

        # Resolved ids (None = plain) so scenes can record/restore their look.
        self.wall_texture_id = self.wall_texture["id"] if self.wall_texture else None
        self.table_texture_id = self.table_texture["id"] if self.table_texture else None

        if no_wall == False:
            wall_half = [3, 0.6, 1.5]
            # Tile by physical size: the visible 6x3 m face repeats the texture every tile_m.
            wall_repeat = (_tile_repeat((wall_half[0] * 2, wall_half[2] * 2), self.wall_texture["tile_m"])
                           if self.wall_texture else (1, 1))
            self.wall_entity = create_box(
                self.scene,
                sapien.Pose(p=[0, 1, 1.5]),
                half_size=wall_half,
                color=(1, 0.9, 0.9),
                name="wall",
                texture=self.wall_texture,
                texture_repeat=wall_repeat,
                is_static=True,
            ).actor

        if no_table == False:
            table_repeat = (_tile_repeat((table_length, table_width), self.table_texture["tile_m"])
                            if self.table_texture else (1, 1))
            self.table_entity = create_table(
                self.scene,
                sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
                length=table_length,
                width=table_width,
                height=table_height,
                thickness=table_thickness,
                is_static=True,
                color=self.table_color,
                texture=self.table_texture,
                texture_repeat=table_repeat,
            )

        self.tabletop_2D_area = [
            table_xy_bias[0] - table_length/2,  # x_min
            table_xy_bias[1] - table_width/2, # y_min
            table_xy_bias[0] + table_length/2,  # x_max
            table_xy_bias[1] + table_width/2  # y_max
        ]


    # =========================================================== Sapien ===========================================================

    def _update_render(self):
        """
        Update rendering to refresh the camera's RGBD information
        (rendering must be updated even when disabled, otherwise data cannot be collected).
        """
        if self.crazy_random_light:
            for renderColor in self.point_light_lst:
                renderColor.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            for renderColor in self.direction_light_lst:
                renderColor.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            now_ambient_light = self.scene.ambient_light
            now_ambient_light = np.clip(np.array(now_ambient_light) + np.random.rand(3) * 0.2 - 0.1, 0, 1)
            self.scene.set_ambient_light(now_ambient_light)
        # self.cameras.update_wrist_camera(self.robot.left_camera.get_pose(), self.robot.right_camera.get_pose())
        self.scene.update_render()