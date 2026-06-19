# use the organizeit asset schema
# see here /home/hjs/Projects/table_arrangement/organize_it_v2/data/asset_library

# Recognize all the items inside the asset library
# And load then onto the RoboTwin/Pybullet simulator.

# 1. object register
#    put all assets
# 2. object fetching (parse the asset information and file path)
# 3. object self-transform wrapper (use a more user friendly axis definition to use the assets)
#    pose = tidy_scene.get_pose(obj), already in stable transform coordination system.
#    tidy_scene.set_pose(obj), will automatically use the stable transform coordination system, and covert to raw SAPIEN pose.


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sapien.core as sapien

ASSET_LIBRARY_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/data/asset_library")

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class Asset:
    id: str
    tags: tuple[str, ...]  # object-class tags, drawn from the catalog vocabulary
    source: str  # robotwin / objaverse / lightwheel / sgbot / gso
    scale: Vec3
    stable_rotation: Mat3
    visual_mesh: Path
    collision_mesh: Path
    pybullet_collision_mesh: Path


@dataclass(frozen=True)
class SceneObject:
    """A spawned asset. Poses are read/written in the asset's stable-transform frame,
    so callers never touch the raw mesh frame."""
    id: str  # scene-unique instance id
    asset: Asset
    entity: sapien.Entity
    stable: sapien.Pose  # raw mesh frame -> stable frame

    def set_pose(self, pose: sapien.Pose) -> None:
        self.entity.set_pose(pose * self.stable)

    def get_pose(self) -> sapien.Pose:
        return self.entity.get_pose() * self.stable.inv()


def spawn(scene: sapien.Scene, asset: Asset, id: str) -> SceneObject:
    if asset.visual_mesh.suffix == ".urdf":  # urdf bakes its own meshes + scale
        loader = scene.create_urdf_loader()
        loader.scale = asset.scale[0]
        loader.fix_root_link = False
        entity = loader.load_multiple(str(asset.visual_mesh))[1][0]
        entity.set_name(id)
    else:
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_multiple_convex_collisions_from_file(filename=str(asset.collision_mesh), scale=asset.scale)
        builder.add_visual_from_file(filename=str(asset.visual_mesh), scale=asset.scale)
        entity = builder.build(name=id)
    return SceneObject(id, asset, entity, _rotation_pose(asset.stable_rotation))


class AssetLibrary:
    def __init__(self, root: Path = ASSET_LIBRARY_ROOT) -> None:
        self.root = root
        catalog = _read(root / "catalog.json")
        self._source_roots = {name: Path(value).expanduser() for name, value in catalog["source_roots"].items()}
        self._object_tags = set(catalog["object_tags"])
        index = _read(root / "assets.json")["assets"]
        self.assets = {entry["asset_id"]: self._parse(root / entry["asset_json"]) for entry in index}
        self._asset_json = {entry["asset_id"]: root / entry["asset_json"] for entry in index}

    def _parse(self, asset_json: Path) -> Asset:
        record = _read(asset_json)
        geometry = record["geometry"]
        source_root = self._source_roots[record["source"]]
        mesh = lambda ref: _resolve(ref, asset_dir=asset_json.parent, source_root=source_root)
        return Asset(
            id=record["asset_id"],
            tags=tuple(tag for tag in record["semantics"]["tags"] if tag in self._object_tags),
            source=record["source"],
            scale=tuple(geometry["scale"]),
            stable_rotation=tuple(tuple(row) for row in geometry["stable_rotation"]),
            visual_mesh=mesh(geometry["visual_mesh"]),
            collision_mesh=mesh(geometry["collision_mesh"]),
            pybullet_collision_mesh=mesh(geometry["pybullet_collision_mesh"] or geometry["collision_mesh"]),
        )

    def is_enabled(self, asset_id: str) -> bool:
        """Read the asset's CURRENT semantics.enabled straight from disk (live, not a
        startup snapshot), so toggling an asset off takes effect immediately. Unknown
        ids and read errors count as disabled / enabled-by-default respectively."""
        path = self._asset_json.get(asset_id)
        if path is None:
            return False
        try:
            return _read(path).get("semantics", {}).get("enabled", True) is not False
        except Exception:
            return True

    def __getitem__(self, asset_id: str) -> Asset:
        return self.assets[asset_id]

    def __iter__(self):
        return iter(self.assets.values())

    def __len__(self) -> int:
        return len(self.assets)

    def by_tag(self, tag: str) -> list[Asset]:
        return [asset for asset in self if tag in asset.tags]

    def by_source(self, source: str) -> list[Asset]:
        return [asset for asset in self if asset.source == source]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _rotation_pose(rotation: Mat3) -> sapien.Pose:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return sapien.Pose(matrix)


def _resolve(ref: dict, *, asset_dir: Path, source_root: Path) -> Path:
    base, path = ref["base"], Path(ref["path"]).expanduser()
    if base == "absolute":
        return path.resolve()
    return ((asset_dir if base == "asset_dir" else source_root) / path).resolve()


if __name__ == "__main__":
    library = AssetLibrary()
    print(f"{len(library)} assets loaded")
    sample = library["gso:obj:45oz_RAMEKIN_ASST_DEEP_COLORS"]
    print(sample.id, sample.tags, sample.source, sample.scale)
    print(sample.visual_mesh)
    print(f"bottles: {len(library.by_tag('bottle'))}")
