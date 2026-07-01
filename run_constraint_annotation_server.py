#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path


ASSET_LIBRARY_DIR = "handcraft_bundle/asset_library"
DATASET_DIR = "handcraft_bundle/data/organize_it_dataset_v2"
PORT = 8104


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default=ASSET_LIBRARY_DIR)
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    asset_library = Path(args.assets).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    port = args.port

    for path in (asset_library / "catalog.json", asset_library / "assets.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (dataset / "available_assets.json").is_file():
        raise FileNotFoundError(dataset / "available_assets.json")

    bundle_src = root / "handcraft_bundle" / "src"
    if bundle_src.is_dir():
        os.environ["TIDY_ORGANIZE_IT_SRC"] = str(bundle_src)
        sys.path.insert(0, str(bundle_src))

    os.environ["TIDY_ASSET_LIBRARY_ROOT"] = str(asset_library)
    os.environ["TIDY_DATASET_DIR"] = str(dataset)
    sys.path.insert(0, str(root / "handcraft"))
    sys.path.insert(0, str(root / "simulations"))

    import uvicorn
    from constrain_annotation_server import app

    print(f"[constraint annotation] asset_library={asset_library}")
    print(f"[constraint annotation] dataset={dataset}")
    print(f"[constraint annotation] http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
