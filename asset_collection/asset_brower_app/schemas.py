from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "asset_v2"
CATALOG_SCHEMA_VERSION = "asset_catalog_v3"


def _list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _category_from_tags(tags: list[str]) -> str:
    for tag in tags:
        if tag.lower() == "kitchen":
            return "Kitchen"
        if tag.lower() == "tools":
            return "Tools"
        if tag.lower() == "desk":
            return "Desk"
    raise ValueError("asset_v2 semantics.tags must include at least one scene tag: Kitchen, Tools, or Desk")


def _float3(value: Any, default: tuple[float, float, float]) -> list[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, (int, float)):
        v = float(value)
        return [v, v, v]
    raw = list(value)
    if len(raw) == 1:
        v = float(raw[0])
        return [v, v, v]
    if len(raw) >= 3:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    return [float(v) for v in default]


def _matrix3(value: Any, default: list[list[float]] | None = None) -> list[list[float]]:
    if default is None:
        default = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if value is None:
        return [[float(v) for v in row] for row in default]
    rows = [list(row) for row in value]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        return [[float(v) for v in row] for row in default]
    return [[float(v) for v in row] for row in rows]


def _path_ref(value: Any) -> dict[str, str] | str:
    if not value:
        return ""
    if not isinstance(value, Mapping):
        raise ValueError(f"asset_v2 path reference must be an object, got: {value!r}")
    base = str(value["base"])
    if base not in {"asset_dir", "source_root", "absolute"}:
        raise ValueError(f"Unsupported asset_v2 path base: {base!r}")
    return {"base": base, "path": str(value["path"])}


@dataclass(frozen=True)
class AssetSemantics:
    category: str = "Unknown"
    tags: list[str] = field(default_factory=list)
    yaw_sensitive: bool = True
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssetSemantics":
        tags = _list_str(data.get("tags"))
        return cls(
            category=_category_from_tags(tags),
            tags=tags,
            yaw_sensitive=bool(data.get("yaw_sensitive", True)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True)
class AssetGeometry:
    visual_mesh: dict[str, str] | str = ""
    collision_mesh: dict[str, str] | str = ""
    pybullet_collision_mesh: dict[str, str] | str = ""
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    stable_rotation: list[list[float]] = field(default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssetGeometry":
        return cls(
            visual_mesh=_path_ref(data.get("visual_mesh")),
            collision_mesh=_path_ref(data.get("collision_mesh")),
            pybullet_collision_mesh=_path_ref(data.get("pybullet_collision_mesh")),
            scale=_float3(data.get("scale"), (1.0, 1.0, 1.0)),
            stable_rotation=_matrix3(data.get("stable_rotation")),
        )


@dataclass(frozen=True)
class ContactInfo:
    lower_points_local: dict[str, str] | str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContactInfo":
        return cls(lower_points_local=_path_ref(data.get("lower_points_local")))


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    source: str
    model_type: str
    semantics: AssetSemantics
    geometry: AssetGeometry
    source_specific_terms: dict[str, Any] = field(default_factory=dict)
    contacts: ContactInfo = field(default_factory=ContactInfo)
    schema_version: str = SCHEMA_VERSION
    notes: str = ""
    label: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssetRecord":
        version = str(data.get("schema_version", ""))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported asset schema {version!r}; expected {SCHEMA_VERSION!r}")
        raw_source = data["source"]
        if not isinstance(raw_source, str):
            raise ValueError("asset_v2 source must be a string")
        return cls(
            asset_id=str(data["asset_id"]),
            label=str(data.get("label", "")),
            source=raw_source,
            model_type=str(data["model_type"]),
            source_specific_terms=dict(data.get("source_specific_terms", {}) or {}),
            semantics=AssetSemantics.from_dict(data["semantics"]),
            geometry=AssetGeometry.from_dict(data["geometry"]),
            contacts=ContactInfo.from_dict(data.get("contacts", {}) or {}),
        )
