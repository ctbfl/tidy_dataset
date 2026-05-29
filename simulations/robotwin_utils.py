import sapien.core as sapien
from pathlib import Path
import numpy as np
from PIL import Image
from utils.actor_utils import Actor

# Curated CC0 PBR texture sets: assets/textures/<kind>/<id>/ holding
# albedo.jpg + normal.jpg + roughness.jpg + meta.json {"tile_m": <real-world size m>}.
TEXTURE_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/assets/textures")


def curated_textures(kind: str) -> list[dict]:
    """PBR texture sets under assets/textures/<kind>/ (kind = 'wall' or 'table')."""
    import json
    sets = []
    for d in sorted((TEXTURE_ROOT / kind).iterdir()):
        if not d.is_dir():
            continue
        tile_m = json.loads((d / "meta.json").read_text())["tile_m"]
        sets.append({"id": d.name, "tile_m": tile_m, "albedo": d / "albedo.jpg",
                     "normal": d / "normal.jpg", "roughness": d / "roughness.jpg"})
    return sets


def _tiled_texture(path: Path, repeat, srgb):
    """RenderTexture2D of `path` repeated (rx, ry) times. Seamless CC0 source, so
    plain tiling joins cleanly. SAPIEN box UVs are fixed 0..1, so we bake the tiling."""
    rx, ry = repeat
    img = np.asarray(Image.open(path).convert("RGBA"))
    if rx > 1 or ry > 1:
        img = np.ascontiguousarray(np.tile(img, (ry, rx, 1)))
    return sapien.render.RenderTexture2D(img, "R8G8B8A8Unorm", address_mode="repeat", srgb=srgb)


def _textured_material(texture: dict, texture_repeat=(1, 1)) -> sapien.render.RenderMaterial:
    """PBR material (albedo + normal + roughness) tiled to physical size."""
    m = sapien.render.RenderMaterial()
    m.set_base_color_texture(_tiled_texture(texture["albedo"], texture_repeat, srgb=True))
    m.set_normal_texture(_tiled_texture(texture["normal"], texture_repeat, srgb=False))
    m.set_roughness_texture(_tiled_texture(texture["roughness"], texture_repeat, srgb=False))
    m.base_color = [1, 1, 1, 1]
    m.metallic = 0.0
    m.roughness = 1.0
    return m


# convinient function so you can directly add objects using TidyScene and workspaces coordinates.
def preprocess(scene, pose: sapien.Pose) -> tuple[sapien.Scene, sapien.Pose]:
    """Add entity to scene. Add bias to z axis if scene is not sapien.Scene."""
    if isinstance(scene, sapien.Scene):
        return scene, pose
    else:
        return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)

def create_table(
        scene,
        pose: sapien.Pose,
        length: float,
        width: float,
        height: float,
        thickness=0.1,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture=None,
        texture_repeat=(1, 1),
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    # Tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])  # Center the tabletop at z=0
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(
        pose=tabletop_pose,
        half_size=tabletop_half_size,
        material=scene.default_physical_material,
    )

    # Add texture
    if texture is not None:
        material = _textured_material(texture, texture_repeat)
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
    else:
        builder.add_box_visual(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=color,
        )

    # Table legs (x4)
    leg_spacing = 0.1
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_spacing / 2)
            y = j * (width / 2 - leg_spacing / 2)
            table_leg_pose = sapien.Pose([x, y, -height / 2 - 0.002])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2 - 0.002]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    return table



# create box
def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture=None,
    texture_repeat=(1, 1),
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))

    # Add texture
    if texture is not None:
        material = _textured_material(texture, texture_repeat)
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity




def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture=None,
    texture_repeat=(1, 1),
    boxtype="default",
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture=texture,
        texture_repeat=texture_repeat,
    )
    if boxtype == "default":
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data)