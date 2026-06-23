from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sapien.core as sapien

ORGANIZE_IT_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
ORGANIZE_IT_SRC = ORGANIZE_IT_ROOT / "src"
if str(ORGANIZE_IT_SRC) not in sys.path:
    sys.path.insert(0, str(ORGANIZE_IT_SRC))

from organize_it.assets.registry import AssetHandle, AssetRegistry  # noqa: E402

ASSET_LIBRARY_ROOT = ORGANIZE_IT_ROOT / "data" / "asset_library"
ASSET_JSON_BACKUP_DIR = "asset_json_backup"
NONCONVEX_CONTAINER_TAGS = {"holder"}

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class Asset:
    handle: AssetHandle
    scale: Vec3 | None = None

    def __post_init__(self) -> None:
        if self.scale is None:
            object.__setattr__(self, "scale", tuple(float(v) for v in self.handle.record.geometry.scale))

    @property
    def id(self) -> str:
        return self.handle.asset_id

    @property
    def label(self) -> str:
        return self.handle.record.label

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(self.handle.record.semantics.tags)

    @property
    def source(self) -> str:
        return self.handle.record.source

    @property
    def stable_rotation(self) -> Mat3:
        return tuple(tuple(float(v) for v in row) for row in self.handle.record.geometry.stable_rotation)

    @property
    def visual_mesh(self) -> Path:
        return self.handle.visual_mesh_path()

    @property
    def collision_mesh(self) -> Path:
        return self.handle.collision_mesh_path()

    @property
    def collision_shape(self) -> str:
        data = json.loads(self.handle.asset_json_path.read_text())
        return str(data.get("geometry", {}).get("collision_shape") or "")

    @property
    def pybullet_collision_mesh(self) -> Path:
        return self.handle.pybullet_mesh_path()


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
    scale = tuple(float(v) for v in asset.scale)
    if asset.visual_mesh.suffix == ".urdf":  # urdf bakes its own meshes + scale
        loader = scene.create_urdf_loader()
        loader.scale = scale[0]
        loader.fix_root_link = False
        entity = loader.load_multiple(str(asset.visual_mesh))[1][0]
        entity.set_name(id)
    else:
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        shape = asset.collision_shape
        if shape == "compound_convex" or (not shape and not _needs_nonconvex_collision(asset)):
            builder.add_multiple_convex_collisions_from_file(filename=str(asset.collision_mesh), scale=scale)
        elif shape == "nonconvex" or (not shape and _needs_nonconvex_collision(asset)):
            builder.add_nonconvex_collision_from_file(filename=str(asset.collision_mesh), scale=scale)
        else:
            raise ValueError(f"unsupported collision_shape for {asset.id}: {shape!r}")
        builder.add_visual_from_file(filename=str(asset.visual_mesh), scale=scale)
        entity = builder.build(name=id)
    return SceneObject(id, asset, entity, _rotation_pose(asset.stable_rotation))


def _needs_nonconvex_collision(asset: Asset) -> bool:
    tags = {tag.lower() for tag in asset.tags}
    return bool(tags & NONCONVEX_CONTAINER_TAGS)


class AssetLibrary:
    def __init__(self, root: Path = ASSET_LIBRARY_ROOT, backup_dir: Path | None = None) -> None:
        self.root = Path(root)
        self.catalog_path = self.root / "catalog.json"
        self.load_asset_json_backup(backup_dir)

    def load_asset_json_backup(self, backup_dir: Path | None) -> None:
        overwrite_dir = Path(backup_dir) if backup_dir is not None and Path(backup_dir).is_dir() else None
        self.registry = AssetRegistry.load(self.catalog_path, asset_json_overwrite_dir=overwrite_dir)
        self.assets = {handle.asset_id: Asset(handle) for handle in self.registry.list(enabled_only=False)}

    def asset_json_path(self, asset_id: str) -> Path:
        return self.registry.get(asset_id).asset_json_path

    def is_enabled(self, asset_id: str) -> bool:
        return self.registry.get(asset_id).record.semantics.enabled

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


def asset_json_backup_dir(scene_json_path: Path) -> Path:
    return Path(scene_json_path).parent / ASSET_JSON_BACKUP_DIR


def asset_json_backup_path(backup_dir: Path, asset_id: str) -> Path:
    if "/" in asset_id:
        raise ValueError(f"asset_id cannot be used as a backup filename: {asset_id}")
    return Path(backup_dir) / f"{asset_id}.json"


def scene_asset_ids(scene_data: dict) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for section in ("manifest", "items"):
        for item in scene_data.get(section, []):
            asset_id = item.get("asset_id")
            if asset_id and asset_id not in seen:
                seen.add(asset_id)
                ids.append(asset_id)
    return ids


def write_asset_json_backup(scene_json_path: Path, scene_data: dict, library: AssetLibrary) -> int:
    backup_dir = asset_json_backup_dir(scene_json_path)
    backup_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for asset_id in scene_asset_ids(scene_data):
        record = json.loads(library.asset_json_path(asset_id).read_text())
        asset_json_backup_path(backup_dir, asset_id).write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        count += 1
    return count


def _rotation_pose(rotation: Mat3) -> sapien.Pose:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return sapien.Pose(matrix)


if __name__ == "__main__":
    library = AssetLibrary()
    print(f"{len(library)} assets loaded")
    sample = library["gso:obj:45oz_RAMEKIN_ASST_DEEP_COLORS"]
    print(sample.id, sample.tags, sample.source, sample.scale)
    print(sample.visual_mesh)
    print(f"bottles: {len(library.by_tag('bottle'))}")
