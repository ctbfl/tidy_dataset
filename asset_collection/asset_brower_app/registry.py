from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import CATALOG_SCHEMA_VERSION, AssetRecord


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_source_root(root_text: str, *, library_root: Path) -> Path:
    root = Path(str(root_text)).expanduser()
    return root.resolve() if root.is_absolute() else (library_root / root).resolve()


def _resolve_path(path_ref: Any, *, asset_dir: Path, source: str, source_roots: Mapping[str, Path]) -> Path:
    if not isinstance(path_ref, Mapping):
        raise ValueError(f"asset_v2 path reference must be an object, got: {path_ref!r}")
    base = str(path_ref["base"])
    path = Path(str(path_ref["path"])).expanduser()
    if base == "absolute":
        return path.resolve()
    if base == "asset_dir":
        return (asset_dir / path).resolve()
    if base == "source_root":
        return (source_roots[str(source)] / path).resolve()
    raise ValueError(f"Unsupported asset path base: {base!r}")


@dataclass(frozen=True)
class AssetHandle:
    record: AssetRecord
    asset_dir: Path
    library_root: Path
    source_roots: Mapping[str, Path]

    @property
    def asset_id(self) -> str:
        return self.record.asset_id

    def resolve_path(self, path_ref: Any) -> Path:
        if not path_ref:
            raise ValueError(f"Empty path for asset {self.asset_id}")
        return _resolve_path(path_ref, asset_dir=self.asset_dir, source=self.record.source, source_roots=self.source_roots)

    def visual_mesh_path(self) -> Path:
        return self.resolve_path(self.record.geometry.visual_mesh)

    def collision_mesh_path(self) -> Path:
        path = self.record.geometry.collision_mesh or self.record.geometry.visual_mesh
        return self.resolve_path(path)

    def pybullet_mesh_path(self) -> Path:
        path = self.record.geometry.pybullet_collision_mesh or self.record.geometry.collision_mesh or self.record.geometry.visual_mesh
        return self.resolve_path(path)


class AssetRegistry:
    def __init__(self, handles: Iterable[AssetHandle], *, catalog_path: Path) -> None:
        self.catalog_path = catalog_path
        self._handles = {handle.asset_id: handle for handle in handles}

    @classmethod
    def load(cls, catalog_path: str | Path) -> "AssetRegistry":
        path = Path(catalog_path).expanduser().resolve()
        data = _read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Asset catalog must be a JSON object: {path}")
        version = str(data.get("schema_version", ""))
        if version != CATALOG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported asset catalog schema {version!r}: {path}")

        library_root = path.parent.resolve()
        source_roots = {
            str(key): _resolve_source_root(str(value), library_root=library_root)
            for key, value in dict(data["source_roots"]).items()
        }

        assets_index_path = library_root / "assets.json"
        assets_data = _read_json(assets_index_path)
        if not isinstance(assets_data, dict) or not isinstance(assets_data.get("assets"), list):
            raise ValueError(f"Asset index must be a JSON object with an assets list: {assets_index_path}")

        handles: list[AssetHandle] = []
        for item in assets_data["assets"]:
            if not isinstance(item, dict) or "asset_json" not in item:
                raise ValueError(f"assets.json entries must contain asset_json: {item!r}")
            asset_json_path = Path(str(item["asset_json"])).expanduser()
            if not asset_json_path.is_absolute():
                asset_json_path = (library_root / asset_json_path).resolve()
            record = AssetRecord.from_dict(_read_json(asset_json_path))
            handles.append(
                AssetHandle(
                    record=record,
                    asset_dir=asset_json_path.parent,
                    library_root=library_root,
                    source_roots=source_roots,
                )
            )
        return cls(handles, catalog_path=path)

    def list(self, *, category: str | None = None, enabled_only: bool = True) -> list[AssetHandle]:
        category_l = category.lower() if category is not None else None
        out: list[AssetHandle] = []
        for handle in self._handles.values():
            sem = handle.record.semantics
            if enabled_only and not sem.enabled:
                continue
            if category_l is not None and sem.category.lower() != category_l:
                continue
            out.append(handle)
        return sorted(out, key=lambda handle: handle.asset_id)
