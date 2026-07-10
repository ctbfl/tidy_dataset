#!/usr/bin/env python3
"""Compare SG-Bot output against ours with the open-source VLM server."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from open_source_vlm import (
    MODEL_IDS,
    bedrock_converse,
    chat_completions,
    codex,
    curl_chat_completions,
    gemini,
    model_config_for_id,
)


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


def run(scene_dir: Path, *, model_id: str, no_obstacle: bool) -> dict:
    run_vlm_comparison = load_run_vlm_comparison()
    model_config = model_config_for_id(model_id)

    def selected_codex(prompt: str, images=None) -> str:
        if model_config["provider"] == "gemini":
            return gemini(prompt, images, model=model_config["model"])
        if model_config["provider"] == "bedrock_converse":
            return bedrock_converse(
                prompt,
                images,
                region=model_config["region"],
                model=model_config["model"],
            )
        if model_config["provider"] == "chat_completions":
            return chat_completions(
                prompt,
                images,
                base_url=model_config["base_url"],
                model=model_config["model"],
                api_key_env=model_config.get("api_key_env"),
            )
        if model_config["provider"] == "curl_chat_completions":
            return curl_chat_completions(
                prompt,
                images,
                base_url=model_config["base_url"],
                model=model_config["model"],
                api_key_env=model_config.get("api_key_env"),
            )
        return codex(
            prompt,
            images,
            base_url=model_config["base_url"],
            model=model_config["model"],
            api_key_env=model_config.get("api_key_env"),
        )

    scene_dir = scene_dir.expanduser().resolve()
    sgbot_image = scene_dir / "sgbot_output" / "result.png"
    ours_image = scene_dir / ("our_result_no_obstacle.png" if no_obstacle else "our_result.png")
    base_name = WITHOUT_OBSTACLE_NAME if no_obstacle else WITH_OBSTACLE_NAME
    out_name = f"{model_id}_{base_name}"
    out_path = scene_dir / DEBUG_DIR / out_name

    return run_vlm_comparison(
        scene_dir,
        codex=selected_codex,
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
    parser.add_argument("--model-id", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument(
        "--no-obstacle",
        action="store_true",
        help="compare against our_result_no_obstacle.png and write the no-obstacle JSON",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    run(args.scene_dir, model_id=args.model_id, no_obstacle=args.no_obstacle)
    print(f"elapsed_seconds={time.perf_counter() - start:.3f}", flush=True)


if __name__ == "__main__":
    main()
