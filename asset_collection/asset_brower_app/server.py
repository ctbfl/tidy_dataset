from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .assets import (
    ASSET_LIBRARY_CATALOG,
    ASSET_LIBRARY_ROOT,
    ASSET_ROOTS,
    _ASSET_LOCK,
    _asset_json_path_for_id,
    _asset_public_item,
    _catalog_data,
    _catalog_source_roots,
    _find_asset,
    _json_bytes,
    _load_asset_catalog,
    _normalize_rotation_matrix,
    _placeholder_svg,
    _render_asset_preview,
    _save_asset_enabled,
    _save_asset_tag_batch,
    _save_asset_tags,
    _save_new_object_tag,
    configure_asset_library_root,
)
from .config import ASSET_PAGE_SIZE, MAX_ASSET_PAGE_SIZE
from .deps import compute_stable_aabb_m
from .pages import STATIC_DIR, page_text

ASSETS_PAGE = page_text("assets.html")


class AssetBrowserRequestHandler(BaseHTTPRequestHandler):
    server_version = "AssetBrowser/1.0"
    _head_only = False

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _send_bytes(
        self,
        body: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "application/octet-stream",
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        if not self._head_only:
            self.wfile.write(body)

    def _send_text(self, text: str, *, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/plain; charset=utf-8") -> None:
        self._send_bytes(text.encode("utf-8"), status=status, content_type=content_type)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(_json_bytes(payload), status=status, content_type="application/json; charset=utf-8")

    def _send_error_text(self, status: HTTPStatus, message: str) -> None:
        self._send_text(message, status=status)

    def _handle_static_file(self, relative_url_path: str) -> None:
        if not relative_url_path:
            raise FileNotFoundError("Missing static path")
        relative_path = Path(unquote(relative_url_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Invalid static path")
        static_path = (STATIC_DIR / relative_path).resolve()
        try:
            static_path.relative_to(STATIC_DIR.resolve())
        except ValueError as exc:
            raise ValueError("Invalid static path") from exc
        if not static_path.is_file():
            raise FileNotFoundError(f"Static file not found: {relative_path.as_posix()}")

        content_type = mimetypes.guess_type(str(static_path))[0] or "application/octet-stream"
        if static_path.suffix.lower() == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif static_path.suffix.lower() == ".css":
            content_type = "text/css; charset=utf-8"
        self._send_bytes(static_path.read_bytes(), content_type=content_type, cache_control="public, max-age=30")

    def do_GET(self) -> None:
        self._head_only = False
        self._route_request()

    def do_HEAD(self) -> None:
        self._head_only = True
        self._route_request()

    def do_POST(self) -> None:
        self._head_only = False
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/assets/stable-rotation":
                self._handle_asset_stable_rotation_save()
            elif parsed.path == "/api/assets/enabled":
                self._handle_asset_enabled_save()
            elif parsed.path == "/api/assets/tags":
                self._handle_asset_tags_save()
            elif parsed.path == "/api/assets/tags/batch":
                self._handle_asset_tags_batch_save()
            elif parsed.path == "/api/tags":
                self._handle_tag_create()
            else:
                self._send_error_text(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError as exc:
            self._send_error_text(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _route_request(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/assets"}:
                self._send_text(ASSETS_PAGE, content_type="text/html; charset=utf-8")
            elif parsed.path.startswith("/static/"):
                self._handle_static_file(parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/assets":
                self._handle_assets(parse_qs(parsed.query))
            elif parsed.path == "/asset-preview":
                self._handle_asset_preview(parse_qs(parsed.query))
            elif parsed.path == "/asset-contact":
                self._handle_asset_contact(parse_qs(parsed.query))
            elif parsed.path.startswith("/asset-static/"):
                self._handle_asset_static(parsed.path.removeprefix("/asset-static/"))
            else:
                self._send_error_text(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError as exc:
            self._send_error_text(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _handle_assets(self, query: dict[str, list[str]]) -> None:
        page = max(1, int(query.get("page", ["1"])[0] or "1"))
        page_size = max(1, min(MAX_ASSET_PAGE_SIZE, int(query.get("page_size", [str(ASSET_PAGE_SIZE)])[0] or ASSET_PAGE_SIZE)))
        category = query.get("category", [""])[0]
        source = query.get("source", query.get("model_type", [""]))[0].lower()
        search = query.get("q", [""])[0].strip().lower()
        tag_filters = [tag.strip().lower() for tag in query.get("tags", []) if tag.strip()]

        items, categories, sources = _load_asset_catalog()
        filtered = items
        if category:
            category_key = category.lower()
            filtered = [
                item
                for item in filtered
                if item["category"].lower() == category_key
                or any(str(tag).lower() == category_key for tag in item.get("extra_categories", []))
            ]
        if source:
            filtered = [item for item in filtered if item["source"].lower() == source or item["source_key"] == source]
        if tag_filters:
            filtered = [
                item
                for item in filtered
                if any(tag in {str(value).lower() for value in item.get("extra_categories", [])} for tag in tag_filters)
            ]
        if search:
            filtered = [item for item in filtered if search in item["search_blob"]]

        total = len(filtered)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, total_pages)
        start = (page - 1) * page_size
        end = start + page_size

        self._send_json(
            {
                "catalog_path": f"Asset library: {ASSET_LIBRARY_CATALOG}",
                "catalog_paths": {"AssetLibrary": str(ASSET_LIBRARY_CATALOG)},
                "asset_roots": {name: str(path.expanduser().resolve()) for name, path in ASSET_ROOTS.items()},
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
                "categories": categories,
                "sources": sources,
                "types": sources,
                "items": [_asset_public_item(item) for item in filtered[start:end]],
            }
        )

    def _handle_asset_preview(self, query: dict[str, list[str]]) -> None:
        uid = query.get("uid", [""])[0]
        if not uid:
            raise ValueError("uid query parameter is required")
        asset = _find_asset(uid)
        if not asset["exists"]:
            body = _placeholder_svg("Asset file missing", asset["asset_path"])
            self._send_bytes(body, content_type="image/svg+xml; charset=utf-8", cache_control="public, max-age=30")
            return
        thumbnail_path = Path(asset.get("resolved_thumbnail_path") or "")
        if thumbnail_path.is_file():
            content_type = mimetypes.guess_type(str(thumbnail_path))[0] or "image/jpeg"
            self._send_bytes(thumbnail_path.read_bytes(), content_type=content_type, cache_control="public, max-age=3600")
            return
        try:
            image_path = _render_asset_preview(asset)
            self._send_bytes(image_path.read_bytes(), content_type="image/png", cache_control="public, max-age=3600")
        except Exception as exc:
            body = _placeholder_svg("Preview unavailable", f"{type(exc).__name__}: {exc}")
            self._send_bytes(body, content_type="image/svg+xml; charset=utf-8", cache_control="no-store")

    def _handle_asset_stable_rotation_save(self) -> None:
        payload = self._read_json_body(max_bytes=64_000)
        asset_id = str(payload.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        stable_rotation = _normalize_rotation_matrix(payload.get("stable_rotation"))

        with _ASSET_LOCK:
            asset_json_path = _asset_json_path_for_id(asset_id)
            record = json.loads(asset_json_path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError(f"Asset JSON must be an object: {asset_json_path}")
            if record.get("asset_id") != asset_id:
                raise ValueError(f"Asset JSON id mismatch: {asset_json_path}")
            geometry = record.setdefault("geometry", {})
            if not isinstance(geometry, dict):
                raise ValueError(f"Asset geometry must be an object: {asset_json_path}")
            geometry["stable_rotation"] = stable_rotation
            catalog = _catalog_data()
            geometry["aabb_m"] = compute_stable_aabb_m(
                record,
                asset_dir=asset_json_path.parent,
                library_root=ASSET_LIBRARY_ROOT,
                source_roots=_catalog_source_roots(catalog),
            )
            asset_json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        self._send_json(
            {
                "asset_id": asset_id,
                "asset_json": str(asset_json_path),
                "stable_rotation": stable_rotation,
                "aabb_m": geometry["aabb_m"],
            }
        )

    def _handle_asset_tags_save(self) -> None:
        payload = self._read_json_body(max_bytes=64_000)
        asset_id = str(payload.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required")

        with _ASSET_LOCK:
            result = _save_asset_tags(asset_id, payload.get("tags"))

        self._send_json(result)

    def _handle_asset_enabled_save(self) -> None:
        payload = self._read_json_body(max_bytes=16_000)
        asset_id = str(payload.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required")

        with _ASSET_LOCK:
            result = _save_asset_enabled(asset_id, payload.get("enabled"))

        self._send_json(result)

    def _handle_asset_tags_batch_save(self) -> None:
        payload = self._read_json_body(max_bytes=512_000)
        with _ASSET_LOCK:
            result = _save_asset_tag_batch(payload.get("tag"), payload.get("states"))
        self._send_json(result)

    def _handle_tag_create(self) -> None:
        payload = self._read_json_body(max_bytes=16_000)
        with _ASSET_LOCK:
            result = _save_new_object_tag(payload.get("tag"))
        self._send_json(result)

    def _handle_asset_contact(self, query: dict[str, list[str]]) -> None:
        uid = query.get("uid", [""])[0]
        kind = query.get("kind", [""])[0].strip().lower()
        if not uid:
            raise ValueError("uid query parameter is required")
        if kind not in {"lower", "upper"}:
            raise ValueError("kind must be lower or upper")
        asset = _find_asset(uid)
        path_text = asset.get("lower_contact_path") if kind == "lower" else asset.get("upper_contact_path")
        if not path_text:
            self._send_json({"asset_id": asset["asset_id"], "kind": kind, "points": [], "count": 0})
            return
        contact_path = Path(path_text).expanduser().resolve()
        if not contact_path.is_file():
            raise FileNotFoundError(f"Contact points not found: {kind} for {uid}")

        import numpy as np

        points = np.load(contact_path).astype(float).reshape(-1, 3)
        self._send_json({"asset_id": asset["asset_id"], "kind": kind, "count": int(points.shape[0]), "points": points.tolist()})

    def _handle_asset_static(self, relative_url_path: str) -> None:
        if not relative_url_path:
            raise FileNotFoundError("Missing asset path")
        parts = relative_url_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Asset static URL must include a source key")
        source_key = unquote(parts[0])
        if source_key not in ASSET_ROOTS:
            raise ValueError(f"Unknown asset source: {source_key}")
        relative_path = Path(unquote(parts[1]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Invalid asset path")
        root = ASSET_ROOTS[source_key].expanduser().resolve()
        asset_path = (root / relative_path).resolve()
        try:
            asset_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Invalid asset path") from exc
        if not asset_path.is_file():
            raise FileNotFoundError(f"Asset file not found: {relative_path.as_posix()}")

        content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
        suffix = asset_path.suffix.lower()
        if suffix == ".glb":
            content_type = "model/gltf-binary"
        elif suffix == ".gltf":
            content_type = "model/gltf+json"
        elif suffix in {".obj", ".mtl"}:
            content_type = "text/plain; charset=utf-8"
        self._send_bytes(asset_path.read_bytes(), content_type=content_type, cache_control="public, max-age=3600")

    def _read_json_body(self, *, max_bytes: int) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            raise ValueError("Missing request body")
        if content_length > max_bytes:
            raise ValueError("Request body is too large")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload


def run_server(host: str, port: int, asset_lib_root: str | Path = ASSET_LIBRARY_ROOT) -> None:
    global ASSET_LIBRARY_CATALOG, ASSET_LIBRARY_ROOT, ASSET_ROOTS

    configure_asset_library_root(asset_lib_root)
    from . import assets as asset_state

    ASSET_LIBRARY_ROOT = asset_state.ASSET_LIBRARY_ROOT
    ASSET_LIBRARY_CATALOG = asset_state.ASSET_LIBRARY_CATALOG
    ASSET_ROOTS = asset_state.ASSET_ROOTS

    server = ThreadingHTTPServer((host, port), AssetBrowserRequestHandler)
    print(f"Asset Browser running at http://{host}:{port}", flush=True)
    print(f"Asset library: {ASSET_LIBRARY_CATALOG}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    # Open3D's Filament/EGL context segfaults during interpreter teardown; the
    # preview render thread has already flushed every image to disk, so skip the
    # destructors entirely for a clean exit.
    os._exit(0)
    # Open3D's Filament/EGL context segfaults during interpreter teardown; the
    # preview render thread has already flushed every image to disk, so skip the
    # destructors entirely for a clean exit.
    os._exit(0)
