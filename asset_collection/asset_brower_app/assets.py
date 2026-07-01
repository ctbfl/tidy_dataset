from __future__ import annotations

import hashlib
import json
import math
import queue
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import config as _config
from .config import (
    ASSET_LIBRARY_CATALOG,
    ASSET_LIBRARY_ROOT,
    ASSET_ROOTS,
    DEFAULT_OBJECT_TAGS,
    GSO_ASSET_ROOT,
    REPO_ROOT,
    SCENE_TAGS,
)
from .deps import AssetRegistry, CATALOG_SCHEMA_VERSION, compute_stable_aabb_m

# Guards concurrent writes to asset metadata (tag/enable saves) from the
# threaded HTTP handlers. Preview rendering uses its own dedicated thread/queue.
_ASSET_LOCK = threading.Lock()
_SCENE_TAG_KEYS = {tag.lower() for tag in SCENE_TAGS}


def configure_asset_library_root(path: str | Path) -> None:
    global ASSET_LIBRARY_CATALOG, ASSET_LIBRARY_ROOT, ASSET_ROOTS

    _config.configure_asset_library_root(path)
    ASSET_LIBRARY_ROOT = _config.ASSET_LIBRARY_ROOT
    ASSET_LIBRARY_CATALOG = _config.ASSET_LIBRARY_CATALOG
    ASSET_ROOTS = {**_config.ASSET_ROOTS, **_catalog_source_roots()}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _available_object_tags() -> list[str]:
    tags: list[str] = []
    data = _catalog_data()
    raw_tags = data.get("object_tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError(f"catalog object_tags must be a list: {ASSET_LIBRARY_CATALOG}")
    for raw_tag in raw_tags:
        text = str(raw_tag).strip()
        if text and text.lower() not in _SCENE_TAG_KEYS:
            _append_unique(tags, text)
    return sorted(tags, key=str.lower)


def _available_tags() -> list[str]:
    return [*SCENE_TAGS, *_available_object_tags()]


def _tag_label_by_key() -> dict[str, str]:
    return {tag.lower(): tag for tag in _available_tags()}


def _editable_tag_keys() -> set[str]:
    return set(_tag_label_by_key())


def _canonical_tag(value: Any) -> str:
    text = str(value or "").strip()
    return _tag_label_by_key().get(text.lower(), text)


def _path_label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("path", ""))
    return str(value or "")


def _asset_editable_tags(item: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for raw_tag in [item.get("category"), *list(item.get("extra_categories") or [])]:
        canonical = _canonical_tag(raw_tag)
        if canonical.lower() in _editable_tag_keys():
            _append_unique(tags, canonical)
    return tags


def _normalize_editable_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("tags must be a list")
    tags: list[str] = []
    for raw_tag in value:
        canonical = _canonical_tag(raw_tag)
        if canonical.lower() not in _editable_tag_keys():
            raise ValueError(f"Unknown tag: {raw_tag}")
        _append_unique(tags, canonical)
    if not any(tag.lower() in _SCENE_TAG_KEYS for tag in tags):
        raise ValueError("Select at least one scene tag: Kitchen, Tools, or Desk")
    return tags


def _scene_category_from_tags(tags: list[str]) -> str:
    selected = {tag.lower() for tag in tags}
    for scene_tag in SCENE_TAGS:
        if scene_tag.lower() in selected:
            return scene_tag
    raise ValueError("Select at least one scene tag: Kitchen, Tools, or Desk")


def _write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _default_asset_label(source: str, asset_key: str, model_name: str, model_id: str, asset_id: str = "") -> str:
    if source == "GSO" and asset_key.startswith("gso:"):
        return asset_key.removeprefix("gso:")
    name = str(model_name or "").strip()
    mid = str(model_id or "").strip()
    if not name:
        return mid or asset_id
    if not mid or mid == name:
        return name
    return f"{name} · {mid}"


def _catalog_data() -> dict[str, Any]:
    if not ASSET_LIBRARY_CATALOG.is_file():
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "source_roots": {
                "robotwin": str(REPO_ROOT / "RoboTwin"),
                "gso": str(GSO_ASSET_ROOT),
            },
            "object_tags": list(DEFAULT_OBJECT_TAGS),
        }
    data = json.loads(ASSET_LIBRARY_CATALOG.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Asset catalog must be a JSON object: {ASSET_LIBRARY_CATALOG}")
    if data.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"Asset catalog must use {CATALOG_SCHEMA_VERSION}: {ASSET_LIBRARY_CATALOG}")
    if "source_roots" not in data:
        raise ValueError(f"Asset catalog missing source_roots: {ASSET_LIBRARY_CATALOG}")
    if "object_tags" not in data:
        raise ValueError(f"Asset catalog missing object_tags: {ASSET_LIBRARY_CATALOG}")
    return data


def _assets_index_path() -> Path:
    return ASSET_LIBRARY_CATALOG.parent / "assets.json"


def _assets_index_data() -> dict[str, Any]:
    path = _assets_index_path()
    if not path.is_file():
        return {"assets": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("assets"), list):
        raise ValueError(f"Asset index must be a JSON object with an assets list: {path}")
    return data


def _catalog_library_root(catalog: dict[str, Any]) -> Path:
    return ASSET_LIBRARY_CATALOG.parent.resolve()


def _catalog_source_roots(catalog: dict[str, Any] | None = None) -> dict[str, Path]:
    catalog = _catalog_data() if catalog is None else catalog
    root = _catalog_library_root(catalog)
    out: dict[str, Path] = {}
    for key, value in dict(catalog["source_roots"]).items():
        path = Path(str(value)).expanduser()
        out[str(key)] = path.resolve() if path.is_absolute() else (root / path).resolve()
    return out


def _load_asset_catalog() -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if not ASSET_LIBRARY_CATALOG.is_file():
        raise FileNotFoundError(f"Asset library catalog not found: {ASSET_LIBRARY_CATALOG}")

    registry = AssetRegistry.load(ASSET_LIBRARY_CATALOG)
    items: list[dict[str, Any]] = []
    categories: list[str] = []
    sources: list[str] = []
    for index, handle in enumerate(registry.list(enabled_only=False)):
        record = handle.record
        raw_record: dict[str, Any] = {}
        raw_asset_json = handle.asset_dir / "asset.json"
        if raw_asset_json.is_file():
            try:
                loaded = json.loads(raw_asset_json.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw_record = loaded
            except (OSError, json.JSONDecodeError):
                raw_record = {}
        visual_path = handle.visual_mesh_path()
        collision_path = handle.collision_mesh_path()
        pybullet_path = handle.pybullet_mesh_path()
        lower_path = handle.resolve_path(record.contacts.lower_points_local) if record.contacts.lower_points_local else None
        upper_path = None
        category = record.semantics.category
        for tag in record.semantics.tags:
            if str(tag).strip().lower() in _SCENE_TAG_KEYS:
                _append_unique(categories, _canonical_tag(tag))
        source_key = str(record.source).lower()
        source_root = handle.source_roots[str(record.source)]
        source_label = "GSO" if source_key == "gso" else "RoboTwin" if source_key == "robotwin" else record.source
        is_gso = source_key == "gso"
        terms = record.source_specific_terms
        model_id = str(terms.get("model_id") or terms.get("source_asset_id") or record.asset_id)
        model_name = str(terms.get("model_name") or model_id)
        asset_key = f"gso:{model_id}" if is_gso else record.asset_id
        raw_label = raw_record.get("label")
        label = str(raw_label).strip() if raw_label else _default_asset_label(source_label, asset_key, model_name, model_id, record.asset_id)
        _append_unique(categories, category)
        _append_unique(sources, source_label)
        item = {
            "uid": record.asset_id,
            "asset_id": record.asset_id,
            "source": source_label,
            "source_key": source_key,
            "asset_root": str(source_root),
            "category": category,
            "asset_key": asset_key,
            "label": label,
            "model_name": model_name,
            "model_id": model_id,
            "file_format": record.model_type.lower(),
            "asset_path": _path_label(record.geometry.visual_mesh),
            "resolved_path": str(visual_path),
            "visual_path": str(visual_path),
            "collision_path": str(collision_path),
            "pybullet_path": str(pybullet_path),
            "lower_contact_path": str(lower_path) if lower_path is not None and lower_path.is_file() else "",
            "upper_contact_path": str(upper_path) if upper_path is not None and upper_path.is_file() else "",
            "scale": list(record.geometry.scale),
            "stable_rotation": record.geometry.stable_rotation,
            "thumbnail_path": "",
            "resolved_thumbnail_path": "",
            "description": "" if is_gso else record.notes,
            "extra_categories": list(record.semantics.tags),
            "source_tier": "",
            "score": None,
            "exists": visual_path.is_file(),
            "enabled": bool(record.semantics.enabled),
            "index": index,
        }
        item["search_blob"] = " ".join(
            [
                item["asset_id"],
                item["source"],
                item["category"],
                item["label"],
                item["model_name"],
                item["model_id"],
                item["file_format"],
                item["asset_path"],
                item["collision_path"],
                " ".join(item["extra_categories"]),
            ]
        ).lower()
        items.append(item)
    return items, categories, sources


def _asset_public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": item["uid"],
        "asset_id": item["asset_id"],
        "label": item.get("label") or _default_asset_label(item["source"], item["asset_key"], item["model_name"], item["model_id"], item["asset_id"]),
        "source": item["source"],
        "file_format": item["file_format"],
        "category": item["category"],
        "asset_key": item["asset_key"],
        "model_name": item["model_name"],
        "model_id": item["model_id"],
        "asset_path": item["asset_path"],
        "description": item["description"],
        "extra_categories": item["extra_categories"],
        "tags": _asset_editable_tags(item),
        "available_tags": _available_tags(),
        "scene_tags": list(SCENE_TAGS),
        "enabled": bool(item.get("enabled", True)),
        "source_tier": item["source_tier"],
        "score": item["score"],
        "exists": item["exists"],
        "contacts": {
            "lower": bool(item.get("lower_contact_path")),
            "upper": bool(item.get("upper_contact_path")),
        },
        "preview_url": _asset_preview_url(item),
        "viewer": _asset_viewer_payload(item),
    }


def _find_asset(uid: str) -> dict[str, Any]:
    items, _, _ = _load_asset_catalog()
    for asset in items:
        if asset["uid"] == uid:
            return asset
    raise FileNotFoundError(f"Asset not found: {uid}")


def _asset_json_path_for_id(asset_id: str) -> Path:
    catalog = _catalog_data()
    assets_index = _assets_index_data()
    library_root = _catalog_library_root(catalog)
    for item in assets_index["assets"]:
        if not isinstance(item, dict) or "asset_json" not in item:
            raise ValueError(f"assets.json entries must contain asset_json: {item!r}")
        path = Path(str(item["asset_json"])).expanduser()
        if not path.is_absolute():
            path = (library_root / path).resolve()
        else:
            path = path.resolve()
        if item.get("asset_id") == asset_id:
            return path
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("asset_id") == asset_id:
                return path
    raise FileNotFoundError(f"Asset JSON not found for {asset_id}")


def _save_asset_tags(asset_id: str, tags: Any) -> dict[str, Any]:
    asset_json_path = _asset_json_path_for_id(asset_id)
    record = json.loads(asset_json_path.read_text(encoding="utf-8"))
    result = _apply_asset_tags(record, asset_json_path, asset_id, tags)
    asset_json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _save_asset_enabled(asset_id: str, enabled: Any) -> dict[str, Any]:
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    asset_json_path = _asset_json_path_for_id(asset_id)
    record = json.loads(asset_json_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"Asset JSON must be an object: {asset_json_path}")
    if record.get("asset_id") != asset_id:
        raise ValueError(f"Asset JSON id mismatch: {asset_json_path}")
    semantics = record.setdefault("semantics", {})
    if not isinstance(semantics, dict):
        raise ValueError(f"Asset semantics must be an object: {asset_json_path}")
    semantics["enabled"] = enabled
    asset_json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "asset_id": asset_id,
        "asset_json": str(asset_json_path),
        "enabled": enabled,
    }


def _apply_asset_tags(record: dict[str, Any], asset_json_path: Path, asset_id: str, tags: Any) -> dict[str, Any]:
    normalized_tags = _normalize_editable_tags(tags)
    scene_category = _scene_category_from_tags(normalized_tags)
    if not isinstance(record, dict):
        raise ValueError(f"Asset JSON must be an object: {asset_json_path}")
    if record.get("asset_id") != asset_id:
        raise ValueError(f"Asset JSON id mismatch: {asset_json_path}")
    semantics = record.setdefault("semantics", {})
    if not isinstance(semantics, dict):
        raise ValueError(f"Asset semantics must be an object: {asset_json_path}")

    existing_tags = semantics.get("tags", [])
    if not isinstance(existing_tags, list):
        existing_tags = []
    preserved_tags = [
        str(tag).strip()
        for tag in existing_tags
        if str(tag).strip() and str(tag).strip().lower() not in _editable_tag_keys()
    ]
    next_tags: list[str] = []
    for tag in [*normalized_tags, *preserved_tags]:
        _append_unique(next_tags, tag)

    semantics["tags"] = next_tags

    return {
        "asset_id": asset_id,
        "asset_json": str(asset_json_path),
        "category": scene_category,
        "tags": normalized_tags,
        "all_tags": next_tags,
    }


def _asset_tags_from_record(record: dict[str, Any]) -> list[str]:
    semantics = record.get("semantics", {})
    if not isinstance(semantics, dict):
        semantics = {}
    tags = list(semantics.get("tags") if isinstance(semantics.get("tags"), list) else [])
    normalized: list[str] = []
    for raw_tag in tags:
        canonical = _canonical_tag(raw_tag)
        if canonical.lower() in _editable_tag_keys():
            _append_unique(normalized, canonical)
    return normalized


def _save_asset_tag_batch(tag: Any, states: Any) -> dict[str, Any]:
    canonical_tag = _canonical_tag(tag)
    if canonical_tag.lower() not in _editable_tag_keys():
        raise ValueError(f"Unknown tag: {tag}")
    if not isinstance(states, list):
        raise ValueError("states must be a list")
    if len(states) > 200:
        raise ValueError("Batch tag request is too large")

    pending: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for raw_state in states:
        if not isinstance(raw_state, dict):
            raise ValueError("Each batch state must be an object")
        asset_id = str(raw_state.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required for every batch state")
        if asset_id in seen:
            raise ValueError(f"Duplicate asset in batch: {asset_id}")
        seen.add(asset_id)
        enabled = bool(raw_state.get("enabled", False))
        asset_json_path = _asset_json_path_for_id(asset_id)
        record = json.loads(asset_json_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"Asset JSON must be an object: {asset_json_path}")
        current_tags = _asset_tags_from_record(record)
        next_tags = [tag for tag in current_tags if tag.lower() != canonical_tag.lower()]
        if enabled:
            _append_unique(next_tags, canonical_tag)
        result = _apply_asset_tags(record, asset_json_path, asset_id, next_tags)
        pending.append((asset_json_path, record, result))

    for asset_json_path, record, _ in pending:
        asset_json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "tag": canonical_tag,
        "updated_count": len(pending),
        "items": [result for _, _, result in pending],
    }


def _validate_new_object_tag(value: Any) -> str:
    tag = str(value or "").strip()
    if not tag:
        raise ValueError("tag is required")
    if len(tag) > 40:
        raise ValueError("tag is too long")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]*", tag):
        raise ValueError("tag may only contain letters, numbers, spaces, underscore, dash, and dot")
    if tag.lower() in _SCENE_TAG_KEYS:
        raise ValueError("Scene tags are fixed: Kitchen, Tools, Desk")
    return tag


def _save_new_object_tag(value: Any) -> dict[str, Any]:
    tag = _validate_new_object_tag(value)
    existing = _available_object_tags()
    canonical = {item.lower(): item for item in existing}
    if tag.lower() in canonical:
        tag = canonical[tag.lower()]
    else:
        existing.append(tag)
        catalog = _catalog_data()
        catalog["object_tags"] = sorted(existing, key=str.lower)
        _write_json_file(ASSET_LIBRARY_CATALOG, catalog)
    return {
        "tag": tag,
        "object_tags": _available_object_tags(),
        "available_tags": _available_tags(),
    }


def _normalize_rotation_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("stable_rotation must be a 3x3 matrix")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("stable_rotation must be a 3x3 matrix")
        out_row = []
        for raw in row:
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError("stable_rotation contains a non-finite value")
            out_row.append(number)
        matrix.append(out_row)

    def dot(a: list[float], b: list[float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    rows = matrix
    max_err = 0.0
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            max_err = max(max_err, abs(dot(rows[i], rows[j]) - expected))
    det = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if max_err > 1e-6:
        raise ValueError(f"stable_rotation is not orthonormal enough: {max_err:.2e}")
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"stable_rotation must be right-handed with det=1, got {det:.6f}")

    return [[round(value, 12) for value in row] for row in matrix]


def _placeholder_svg(title: str, detail: str = "") -> bytes:
    def esc(value: str) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="360" height="270" viewBox="0 0 360 270">
  <rect width="360" height="270" fill="#f0f2f5"/>
  <rect x="18" y="18" width="324" height="234" rx="6" fill="#ffffff" stroke="#d8dde5"/>
  <text x="180" y="124" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#667085">{esc(title)}</text>
  <text x="180" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#667085">{esc(detail[:72])}</text>
</svg>""".encode("utf-8")


def _asset_content_key(asset: dict[str, Any]) -> str:
    """Fingerprint of everything that affects an asset's preview render.

    Stored next to the cached PNG; the browser re-renders when this changes, so the
    PNG filename can stay stable (asset_id) while still invalidating on config edits.
    """
    source_path = Path(asset["resolved_path"])
    stat_bits = "missing"
    if source_path.exists():
        stat = source_path.stat()
        stat_bits = f"{stat.st_mtime_ns}:{stat.st_size}"
    transform_bits = json.dumps(
        {
            "scale": asset.get("scale", [1.0, 1.0, 1.0]),
            "stable_rotation": asset.get("stable_rotation"),
            "camera": [-1.0, -1.0, 1.0],
            "mesh": "visual",
            "renderer": "open3d_visual_models",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    key = f"v7:{asset['uid']}:{asset['source']}:{asset['file_format']}:{asset['resolved_path']}:{stat_bits}:{transform_bits}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _asset_cache_path(asset: dict[str, Any]) -> Path:
    # Stable, human-readable name so the sim/handcraft tools can read by asset_id.
    safe_id = asset["uid"].replace(":", "_").replace("/", "_")
    return _config.ASSET_PREVIEW_CACHE_DIR / f"{safe_id}.png"


def _asset_preview_url(item: dict[str, Any]) -> str:
    # v = content fingerprint so the browser refetches when an asset's config changes.
    return f"/asset-preview?uid={quote(item['uid'])}&v={_asset_content_key(item)}"


def _relative_asset_path(path: Path, source_key: str) -> str:
    resolved = path.expanduser().resolve()
    if source_key not in ASSET_ROOTS:
        raise ValueError(f"Unknown asset source root: {source_key}")
    asset_root = ASSET_ROOTS[source_key].expanduser().resolve()
    try:
        return resolved.relative_to(asset_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Asset path is outside {source_key} asset root: {resolved}") from exc


def _asset_static_url(path: Path, source_key: str) -> str:
    return f"/asset-static/{quote(source_key)}/{quote(_relative_asset_path(path, source_key), safe='/')}"


def _asset_static_url_for_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    for source_key in ASSET_ROOTS:
        try:
            return _asset_static_url(resolved, source_key)
        except ValueError:
            continue
    raise ValueError(f"Asset path is outside known asset roots: {resolved}")


def _obj_material_path(mesh_path: Path) -> Path | None:
    if mesh_path.suffix.lower() != ".obj":
        return None
    try:
        with mesh_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(80):
                line = handle.readline()
                if not line:
                    break
                if line.lower().startswith("mtllib "):
                    material_name = line.split(maxsplit=1)[1].strip()
                    material_path = (mesh_path.parent / material_name).resolve()
                    return material_path if material_path.is_file() else None
    except OSError:
        return None
    return None


def _obj_texture_path(mesh_path: Path) -> Path | None:
    material_path = _obj_material_path(mesh_path)
    if material_path is None:
        return None
    try:
        with material_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.lower().lstrip().startswith("map_kd "):
                    continue
                texture_name = line.split(maxsplit=1)[1].strip()
                if not texture_name:
                    continue
                texture_path = Path(texture_name)
                candidates = []
                if texture_path.is_absolute():
                    candidates.append(texture_path)
                else:
                    candidates.extend(
                        [
                            material_path.parent / texture_path,
                            mesh_path.parent / texture_path,
                            mesh_path.parent.parent / "materials" / "textures" / texture_path.name,
                        ]
                    )
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if resolved.is_file():
                        return resolved
                return candidates[0].resolve()
    except OSError:
        return None
    return None


def _first_urdf_mesh(asset: dict[str, Any], tag_name: str) -> dict[str, Any] | None:
    source_path = Path(asset["resolved_path"])
    if not source_path.is_file():
        return None
    root = ET.parse(source_path).getroot()
    for elem in root.findall(f".//{tag_name}"):
        mesh_node = elem.find(".//mesh")
        if mesh_node is None:
            continue
        filename = mesh_node.attrib.get("filename")
        if not filename:
            continue
        mesh_path = _resolve_urdf_mesh_path(source_path, filename, Path(asset["asset_root"]))
        if not mesh_path.is_file():
            continue
        origin = elem.find("origin")
        return {
            "mesh_path": mesh_path,
            "material_path": _obj_material_path(mesh_path),
            "texture_path": _obj_texture_path(mesh_path),
            "scale": _parse_float_triplet(mesh_node.attrib.get("scale"), (1.0, 1.0, 1.0)),
            "xyz": _parse_float_triplet(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
            "rpy": _parse_float_triplet(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
        }
    return None


def _first_urdf_visual(asset: dict[str, Any]) -> dict[str, Any] | None:
    return _first_urdf_mesh(asset, "visual")


def _viewer_payload_for_mesh_path(asset: dict[str, Any], mesh_path: Path, *, kind: str) -> dict[str, Any] | None:
    if not mesh_path.is_file():
        return None
    suffix = mesh_path.suffix.lower().lstrip(".")
    scale = list(asset.get("scale") or [1.0, 1.0, 1.0])
    if suffix in {"glb", "gltf"}:
        return {
            "format": suffix,
            "model_url": _asset_static_url_for_path(mesh_path),
            "scale": scale,
            "stable_rotation": asset.get("stable_rotation"),
            "xyz": [0.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0],
        }
    if suffix == "obj":
        material_path = _obj_material_path(mesh_path)
        texture_path = _obj_texture_path(mesh_path)
        return {
            "format": "obj",
            "model_url": _asset_static_url_for_path(mesh_path),
            "material_url": _asset_static_url_for_path(material_path) if material_path else None,
            "texture_url": _asset_static_url_for_path(texture_path) if texture_path else None,
            "scale": scale,
            "stable_rotation": asset.get("stable_rotation"),
            "xyz": [0.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0],
        }
    if suffix == "urdf":
        mesh_info = _first_urdf_mesh(asset, "collision" if kind == "collision" else "visual")
        if mesh_info is None:
            return None
        material_path = mesh_info["material_path"]
        texture_path = mesh_info["texture_path"]
        return {
            "format": mesh_info["mesh_path"].suffix.lower().lstrip(".") or "mesh",
            "model_url": _asset_static_url_for_path(mesh_info["mesh_path"]),
            "material_url": _asset_static_url_for_path(material_path) if material_path else None,
            "texture_url": _asset_static_url_for_path(texture_path) if texture_path else None,
            "scale": list(mesh_info["scale"]),
            "stable_rotation": asset.get("stable_rotation"),
            "xyz": list(mesh_info["xyz"]),
            "rpy": list(mesh_info["rpy"]),
        }
    return None


def _asset_viewer_payload(asset: dict[str, Any]) -> dict[str, Any] | None:
    if not asset["exists"]:
        return None
    visual = _viewer_payload_for_mesh_path(asset, Path(asset["visual_path"]), kind="visual")
    collision_source = Path(asset.get("pybullet_path") or asset.get("collision_path") or "")
    collision = _viewer_payload_for_mesh_path(asset, collision_source, kind="collision") if str(collision_source) else None
    if visual is None and collision is None:
        return None
    return {
        "visual": visual,
        "collision": collision,
        "stable_rotation": asset.get("stable_rotation"),
        "lower_contact_url": f"/asset-contact?uid={quote(asset['uid'])}&kind=lower" if asset.get("lower_contact_path") else "",
        "upper_contact_url": f"/asset-contact?uid={quote(asset['uid'])}&kind=upper" if asset.get("upper_contact_path") else "",
    }


def _parse_float_triplet(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    parts = value.split()
    if len(parts) != 3:
        return default
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return default


def _rpy_xyz_matrix(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> Any:
    import numpy as np

    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rz @ ry @ rx
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    return matrix


def _scale_matrix(scale: tuple[float, float, float]) -> Any:
    import numpy as np

    matrix = np.eye(4, dtype=float)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale
    return matrix


def _resolve_urdf_mesh_path(urdf_path: Path, filename: str, asset_root: Path) -> Path:
    clean = filename.strip()
    if clean.startswith("package://"):
        clean = clean[len("package://") :]
    if clean.startswith("file://"):
        clean = clean[len("file://") :]
    path = Path(clean)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([urdf_path.parent / path, asset_root / path])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].expanduser().resolve()


def _read_triangle_mesh(path: Path) -> Any:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"No triangle mesh loaded from {path}")
    mesh.compute_vertex_normals()
    return mesh


def _stable_rotation_matrix(asset: dict[str, Any]) -> Any:
    import numpy as np

    raw = asset.get("stable_rotation") or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    matrix = np.asarray(raw, dtype=float).reshape(3, 3)
    out = np.eye(4, dtype=float)
    out[:3, :3] = matrix
    return out


def _apply_asset_preview_transform(mesh: Any, asset: dict[str, Any]) -> Any:
    scale = _parse_float_triplet(" ".join(str(v) for v in asset.get("scale", [1.0, 1.0, 1.0])), (1.0, 1.0, 1.0))
    mesh.transform(_stable_rotation_matrix(asset) @ _scale_matrix(scale))
    mesh.compute_vertex_normals()
    return mesh


def _load_triangle_model(path: Path) -> Any | None:
    import open3d as o3d

    model = o3d.io.read_triangle_model(str(path))
    if not getattr(model, "meshes", None):
        return None
    for mesh_info in model.meshes:
        if len(mesh_info.mesh.vertices) == 0:
            continue
        mesh_info.mesh.compute_vertex_normals()
    return model


def _transform_triangle_model(model: Any, matrix: Any) -> Any:
    for mesh_info in model.meshes:
        mesh_info.mesh.transform(matrix)
        mesh_info.mesh.compute_vertex_normals()
    return model


def _triangle_model_bounds(model: Any) -> tuple[Any, Any]:
    import numpy as np

    mins = []
    maxs = []
    for mesh_info in model.meshes:
        if len(mesh_info.mesh.vertices) == 0:
            continue
        bbox = mesh_info.mesh.get_axis_aligned_bounding_box()
        mins.append(np.asarray(bbox.get_min_bound(), dtype=float))
        maxs.append(np.asarray(bbox.get_max_bound(), dtype=float))
    if not mins:
        raise ValueError("Asset model has empty bounds")
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def _mesh_bounds(mesh: Any) -> tuple[Any, Any]:
    import numpy as np

    bbox = mesh.get_axis_aligned_bounding_box()
    return np.asarray(bbox.get_min_bound(), dtype=float), np.asarray(bbox.get_max_bound(), dtype=float)


def _preview_object_bounds(obj: dict[str, Any]) -> tuple[Any, Any]:
    if obj["kind"] == "model":
        return _triangle_model_bounds(obj["geometry"])
    return _mesh_bounds(obj["geometry"])


def _transform_preview_object(obj: dict[str, Any], matrix: Any) -> dict[str, Any]:
    if obj["kind"] == "model":
        _transform_triangle_model(obj["geometry"], matrix)
    else:
        obj["geometry"].transform(matrix)
        obj["geometry"].compute_vertex_normals()
    return obj


def _make_preview_object_from_mesh_path(mesh_path: Path, transform: Any) -> dict[str, Any]:
    model = _load_triangle_model(mesh_path)
    if model is not None:
        return {"kind": "model", "geometry": _transform_triangle_model(model, transform)}
    mesh = _read_triangle_mesh(mesh_path)
    mesh.transform(transform)
    mesh.compute_vertex_normals()
    return {"kind": "mesh", "geometry": mesh}


def _asset_scale_matrix(asset: dict[str, Any]) -> Any:
    scale = _parse_float_triplet(" ".join(str(v) for v in asset.get("scale", [1.0, 1.0, 1.0])), (1.0, 1.0, 1.0))
    return _scale_matrix(scale)


def _load_asset_preview_objects(asset: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    source_path = Path(asset["resolved_path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Asset file not found: {source_path}")

    asset_transform = _stable_rotation_matrix(asset) @ _asset_scale_matrix(asset)
    if asset["file_format"] in {"glb", "gltf", "obj"}:
        return [_make_preview_object_from_mesh_path(source_path, asset_transform)]

    if asset["file_format"] != "urdf":
        raise ValueError(f"Unsupported asset file format: {asset['file_format']}")

    root = ET.parse(source_path).getroot()
    objects: list[dict[str, Any]] = []
    for visual in root.findall(".//visual"):
        mesh_node = visual.find(".//mesh")
        if mesh_node is None:
            continue
        filename = mesh_node.attrib.get("filename")
        if not filename:
            continue
        mesh_path = _resolve_urdf_mesh_path(source_path, filename, Path(asset["asset_root"]))
        scale = _parse_float_triplet(mesh_node.attrib.get("scale"), (1.0, 1.0, 1.0))
        origin = visual.find("origin")
        xyz = _parse_float_triplet(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_float_triplet(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        visual_transform = asset_transform @ _rpy_xyz_matrix(xyz, rpy) @ _scale_matrix(scale)
        objects.append(_make_preview_object_from_mesh_path(mesh_path, visual_transform))

    if not objects:
        raise ValueError(f"No visual mesh found in URDF: {source_path}")
    return objects


def _load_asset_mesh(asset: dict[str, Any]) -> Any:
    import open3d as o3d

    source_path = Path(asset["resolved_path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Asset file not found: {source_path}")

    if asset["file_format"] in {"glb", "gltf", "obj"}:
        return _read_triangle_mesh(source_path)

    if asset["file_format"] != "urdf":
        raise ValueError(f"Unsupported asset file format: {asset['file_format']}")

    root = ET.parse(source_path).getroot()
    combined = o3d.geometry.TriangleMesh()
    visual_count = 0
    for visual in root.findall(".//visual"):
        mesh_node = visual.find(".//mesh")
        if mesh_node is None:
            continue
        filename = mesh_node.attrib.get("filename")
        if not filename:
            continue
        mesh_path = _resolve_urdf_mesh_path(source_path, filename, Path(asset["asset_root"]))
        mesh = _read_triangle_mesh(mesh_path)
        scale = _parse_float_triplet(mesh_node.attrib.get("scale"), (1.0, 1.0, 1.0))
        origin = visual.find("origin")
        xyz = _parse_float_triplet(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_float_triplet(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        mesh.transform(_rpy_xyz_matrix(xyz, rpy) @ _scale_matrix(scale))
        combined += mesh
        visual_count += 1

    if visual_count == 0 or len(combined.vertices) == 0 or len(combined.triangles) == 0:
        raise ValueError(f"No visual mesh found in URDF: {source_path}")
    combined.compute_vertex_normals()
    return combined


def _normalize_asset_mesh(mesh: Any) -> Any:
    import numpy as np

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=float)
    max_extent = float(np.max(extent)) if extent.size else 0.0
    if max_extent <= 0:
        raise ValueError("Asset mesh has empty bounds")
    mesh.translate(-bbox.get_center())
    mesh.scale(1.55 / max_extent, center=(0.0, 0.0, 0.0))
    if not mesh.has_vertex_colors() and not mesh.has_textures():
        mesh.paint_uniform_color((0.46, 0.62, 0.78))
    return mesh


def _normalize_asset_preview_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np

    if not objects:
        raise ValueError("Asset preview has no geometry")
    mins = []
    maxs = []
    for obj in objects:
        min_bound, max_bound = _preview_object_bounds(obj)
        mins.append(min_bound)
        maxs.append(max_bound)
    min_bound = np.min(np.stack(mins, axis=0), axis=0)
    max_bound = np.max(np.stack(maxs, axis=0), axis=0)
    extent = max_bound - min_bound
    max_extent = float(np.max(extent)) if extent.size else 0.0
    if max_extent <= 0:
        raise ValueError("Asset preview has empty bounds")
    center = (min_bound + max_bound) * 0.5
    translate = np.eye(4, dtype=float)
    translate[:3, 3] = -center
    scale = np.eye(4, dtype=float)
    scale[0, 0] = scale[1, 1] = scale[2, 2] = 1.55 / max_extent
    normalize = scale @ translate
    for obj in objects:
        _transform_preview_object(obj, normalize)
        if obj["kind"] == "mesh":
            mesh = obj["geometry"]
            if not mesh.has_vertex_colors() and not mesh.has_textures():
                mesh.paint_uniform_color((0.46, 0.62, 0.78))
    return objects


def _asset_preview_material(mesh: Any) -> Any:
    import numpy as np
    import open3d as o3d

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = (1.0, 1.0, 1.0, 1.0)
    if mesh.has_textures() and mesh.has_triangle_uvs() and len(mesh.textures) > 0:
        material.albedo_img = o3d.geometry.Image(np.asarray(mesh.textures[0]).copy())
    return material


# All Open3D rendering happens on this single dedicated thread holding one
# persistent OffscreenRenderer. Creating/destroying a renderer per preview across
# the HTTP server's worker threads leaks the Filament/GL context and segfaults
# after a page of thumbnails; one renderer on one thread is the supported pattern.
_RENDER_QUEUE: "queue.Queue" = queue.Queue()
_RENDER_THREAD: threading.Thread | None = None
_RENDER_THREAD_LOCK = threading.Lock()


def _ensure_render_thread() -> None:
    global _RENDER_THREAD
    with _RENDER_THREAD_LOCK:
        if _RENDER_THREAD is None:
            _RENDER_THREAD = threading.Thread(target=_render_worker, name="asset-preview", daemon=True)
            _RENDER_THREAD.start()


def _render_worker() -> None:
    import numpy as np
    import open3d as o3d

    cache_dir = _config.ASSET_PREVIEW_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / ".gitignore").write_text("*\n")  # cache is regenerable, never versioned

    renderer = o3d.visualization.rendering.OffscreenRenderer(360, 270)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.scene.set_indirect_light_intensity(45000)
    renderer.scene.scene.set_sun_light([-1.0, -1.0, -1.5], [1.0, 1.0, 1.0], 65000)
    renderer.scene.scene.enable_sun_light(True)

    while True:
        asset, cache_path, key_path, content_key, reply = _RENDER_QUEUE.get()
        try:
            preview_objects = _normalize_asset_preview_objects(_load_asset_preview_objects(asset))
            for idx, obj in enumerate(preview_objects):
                name = f"asset_{idx}"
                if obj["kind"] == "model":
                    renderer.scene.add_model(name, obj["geometry"])
                else:
                    renderer.scene.add_geometry(name, obj["geometry"], _asset_preview_material(obj["geometry"]))
            renderer.setup_camera(
                35.0,
                np.asarray([0.0, 0.0, 0.0], dtype=float),
                np.asarray([-2.15, -2.15, 2.15], dtype=float),
                np.asarray([0.0, 0.0, 1.0], dtype=float),
            )
            image = renderer.render_to_image()
            o3d.io.write_image(str(cache_path), image)
            key_path.write_text(content_key)
            reply.put((None, cache_path))
        except Exception as exc:  # propagate to the requesting HTTP thread
            reply.put((exc, None))
        finally:
            renderer.scene.clear_geometry()  # leave the shared scene empty for the next job


def _render_asset_preview(asset: dict[str, Any]) -> Path:
    cache_path = _asset_cache_path(asset)
    key_path = cache_path.with_suffix(".key")
    content_key = _asset_content_key(asset)
    if cache_path.is_file() and key_path.is_file() and key_path.read_text() == content_key:
        return cache_path
    _ensure_render_thread()
    reply: "queue.Queue" = queue.Queue(maxsize=1)
    _RENDER_QUEUE.put((asset, cache_path, key_path, content_key, reply))
    error, result = reply.get()
    if error is not None:
        raise error
    return result
