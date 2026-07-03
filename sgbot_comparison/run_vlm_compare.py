#!/usr/bin/env python3
"""VLM compare SG-Bot output against our SG-Bot-rollout result."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ORGANIZE_IT_V2_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
VLM_COMPARISON_PATH = ORGANIZE_IT_V2_ROOT / "src" / "organize_it" / "streamline_v2" / "vlm_comparison.py"
sys.path.insert(0, str(ORGANIZE_IT_V2_ROOT / "src"))


USER_INTENT = "Set up this table to be an immediately usable single-person dining setup."
DEBUG_DIR = "debug_comparison"
WITH_OBSTACLE_NAME = "sg_bot_vs_ours_with_obstacle.json"
WITHOUT_OBSTACLE_NAME = "sg_bot_vs_ours_without_obstacle.json"


def load_run_vlm_comparison():
    spec = importlib.util.spec_from_file_location("streamline_v2_vlm_comparison", VLM_COMPARISON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {VLM_COMPARISON_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_vlm_comparison


def run(scene_dir: Path, *, no_obstacle: bool) -> dict:
    from organize_it.modules.vlm import codex

    run_vlm_comparison = load_run_vlm_comparison()

    scene_dir = scene_dir.expanduser().resolve()
    sgbot_image = scene_dir / "sgbot_output" / "result.png"
    ours_image = scene_dir / ("our_result_no_obstacle.png" if no_obstacle else "our_result.png")
    out_name = WITHOUT_OBSTACLE_NAME if no_obstacle else WITH_OBSTACLE_NAME
    out_path = scene_dir / DEBUG_DIR / out_name

    return run_vlm_comparison(
        scene_dir,
        codex=codex,
        out_path=out_path,
        reference_image=sgbot_image,
        candidate_image=ours_image,
        reference_label="sg_bot",
        candidate_label="ours",
        user_prompt_override=USER_INTENT,
        preset=out_name.removesuffix(".json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument(
        "--no-obstacle",
        action="store_true",
        help="compare against our_result_no_obstacle.png and write the no-obstacle JSON",
    )
    args = parser.parse_args()
    run(args.scene_dir, no_obstacle=args.no_obstacle)


if __name__ == "__main__":
    main()
