#!/usr/bin/env python3
"""Resolve scene ids in need_fix_ids.txt to dataset scene directories."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NEED_FIX_IDS = REPO_ROOT / "sgbot_comparison" / "logs" / "need_fix_ids.txt"
DATA_ROOT = REPO_ROOT / "data" / "sgbot" / "exp0"
SCENE_ID_RE = re.compile(r"^\s*([0-9a-fA-F]+_\d+_\d+)")


def parse_need_fix_ids(path: Path = NEED_FIX_IDS) -> list[str]:
    scene_ids: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        match = SCENE_ID_RE.match(line)
        if match is None:
            raise ValueError(f"{path}:{line_no}: cannot parse scene id from {line!r}")
        scene_ids.append(match.group(1).lower())
    return scene_ids


def resolve_scene_path(scene_id: str, data_root: Path = DATA_ROOT) -> Path:
    hash_prefix, index, count = scene_id.split("_")
    matches = sorted(data_root.glob(f"{hash_prefix}*_{index}_{count}_scene*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{scene_id}: expected 1 scene dir, found {len(matches)}")
    return matches[0].resolve()


def resolve_need_fix_scene_paths(
    path: Path = NEED_FIX_IDS,
    data_root: Path = DATA_ROOT,
) -> list[Path]:
    return [resolve_scene_path(scene_id, data_root) for scene_id in parse_need_fix_ids(path)]


def main() -> None:
    for scene_path in resolve_need_fix_scene_paths():
        print(scene_path)


if __name__ == "__main__":
    main()
