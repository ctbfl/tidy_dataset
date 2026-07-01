#!/usr/bin/env python3
"""Entry point for the standalone Asset Viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from asset_brower_app.server import run_server


def _resolve_asset_library(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the standalone local Asset Viewer.")
    parser.add_argument(
        "asset_library",
        nargs="?",
        default="../handcraft_bundle/asset_library",
        help="Asset library directory, default ../handcraft_bundle/asset_library",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=8767, help="HTTP port, default 8767")
    args = parser.parse_args()

    run_server(args.host, args.port, _resolve_asset_library(args.asset_library))
