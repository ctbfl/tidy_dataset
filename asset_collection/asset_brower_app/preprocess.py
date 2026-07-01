from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _float3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        raw = value.split()
    else:
        raw = list(value)
    if len(raw) == 1:
        v = float(raw[0])
        return v, v, v
    if len(raw) >= 3:
        return float(raw[0]), float(raw[1]), float(raw[2])
    return default


def _matrix3(value: Any) -> Any:
    import numpy as np

    rows = [list(row) for row in value]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("stable_rotation must be a 3x3 matrix")
    return np.asarray(rows, dtype=float).reshape(3, 3)


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


def _resolve_urdf_mesh_path(urdf_path: Path, filename: str, asset_dir: Path, library_root: Path) -> Path:
    clean = filename.strip()
    if clean.startswith("package://"):
        clean = clean[len("package://") :]
    if clean.startswith("file://"):
        clean = clean[len("file://") :]
    path = Path(clean)
    candidates = [path] if path.is_absolute() else [urdf_path.parent / path, asset_dir / path, library_root / path]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].expanduser().resolve()


def _read_mesh_vertices(mesh_path: Path) -> Any:
    import numpy as np
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    points = np.asarray(mesh.vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(f"No vertices loaded from {mesh_path}")
    return points


def _urdf_vertices(urdf_path: Path, *, asset_dir: Path, library_root: Path, preferred_tag: str = "collision") -> tuple[Any, str, str]:
    import numpy as np

    root = ET.parse(urdf_path).getroot()
    for tag_name in (preferred_tag, "visual" if preferred_tag == "collision" else "collision"):
        chunks = []
        source_meshes = []
        for elem in root.findall(f".//{tag_name}"):
            mesh_node = elem.find(".//mesh")
            if mesh_node is None:
                continue
            filename = mesh_node.attrib.get("filename")
            if not filename:
                continue
            mesh_path = _resolve_urdf_mesh_path(urdf_path, filename, asset_dir, library_root)
            points = _read_mesh_vertices(mesh_path)
            scale = np.asarray(_float3(mesh_node.attrib.get("scale"), (1.0, 1.0, 1.0)), dtype=float)
            origin = elem.find("origin")
            xyz = _float3(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _float3(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            points = points * scale.reshape(1, 3)
            transform = _rpy_xyz_matrix(xyz, rpy)
            points = points @ transform[:3, :3].T + transform[:3, 3].reshape(1, 3)
            chunks.append(points)
            source_meshes.append(str(mesh_path))
        if chunks:
            source_mesh = source_meshes[0] if len(source_meshes) == 1 else f"{len(source_meshes)} meshes"
            return np.concatenate(chunks, axis=0), tag_name, source_mesh
    raise ValueError(f"No mesh vertices found in URDF: {urdf_path}")


def _resolve_record_path(path_ref: Any, *, asset_dir: Path, source: str, source_roots: dict[str, Path]) -> Path:
    if not isinstance(path_ref, dict):
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


def _portable_mesh_path(path_ref: Any) -> str:
    if not isinstance(path_ref, dict):
        raise ValueError(f"asset_v2 path reference must be an object, got: {path_ref!r}")
    return str(path_ref.get("path", ""))


def compute_stable_aabb_m(record_data: dict[str, Any], *, asset_dir: Path, library_root: Path, source_roots: dict[str, Path] | None = None) -> dict[str, Any]:
    import numpy as np

    geometry = record_data.get("geometry", {}) or {}
    source_path_ref = geometry.get("pybullet_collision_mesh") or geometry.get("collision_mesh") or geometry.get("visual_mesh") or ""
    if not source_path_ref:
        raise ValueError(f"Asset has no mesh path: {record_data.get('asset_id', '')}")
    if source_roots is None:
        raise ValueError("asset_catalog_v3 source_roots are required to compute stable AABB")

    source_path = _resolve_record_path(
        source_path_ref,
        asset_dir=asset_dir,
        source=str(record_data.get("source", "")),
        source_roots=source_roots,
    )
    source_kind = "pybullet_collision_mesh" if geometry.get("pybullet_collision_mesh") else "collision_mesh" if geometry.get("collision_mesh") else "visual_mesh"
    if source_path.suffix.lower() == ".urdf":
        points, urdf_tag, source_mesh = _urdf_vertices(source_path, asset_dir=asset_dir, library_root=library_root, preferred_tag="collision")
        source_kind = f"urdf_{urdf_tag}"
    else:
        points = _read_mesh_vertices(source_path)
        source_mesh = _portable_mesh_path(source_path_ref)

    scale = np.asarray(_float3(geometry.get("scale"), (1.0, 1.0, 1.0)), dtype=float)
    points = points * scale.reshape(1, 3)
    stable_rotation = _matrix3(geometry.get("stable_rotation"))
    stable_points = points @ stable_rotation.T
    min_xyz = stable_points.min(axis=0)
    max_xyz = stable_points.max(axis=0)
    size = max_xyz - min_xyz
    center = (min_xyz + max_xyz) * 0.5
    return {
        "frame": "stable",
        "unit": "m",
        "source": source_kind,
        "source_mesh": source_mesh,
        "min": [_round_float(v) for v in min_xyz.tolist()],
        "max": [_round_float(v) for v in max_xyz.tolist()],
        "aabb_center": [_round_float(v) for v in center.tolist()],
        "size": [_round_float(v) for v in size.tolist()],
        "bottom_z": _round_float(float(min_xyz[2])),
    }
