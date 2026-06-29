from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from extract_relations_from_current_goal import (
    GUIDE_PATH,
    PROMPT_NAME,
    RAW_RESPONSE_NAME,
    RELATIONS_NAME,
    VALIDATION_NAME,
    load_codex,
    load_object_id_map,
    parse_relation_response,
)
from relation_state import evaluate_relations
from render_goal_segmentation import render_current_segmentation
from run_exp1_current_goal_ablation import raw_goal_to
from visualize_exp1_topdown import load_object_dims, load_table, render_if_full_ok


MAX_CONCURRENT_VLM_CALLS = 3
EXP_REL = Path("debug_vlm_relation_extraction/exp1/m3")
SUMMARY_NAME = "m3_batch_summary.json"


def case_dirs(parent_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(parent_dir.iterdir())
        if path.is_dir() and re.fullmatch(r"\d{3}", path.name)
    ]


def object_size_prior(data_dir: Path) -> dict[str, list[float]]:
    table = load_table(data_dir)
    table_size = (float(table["size"][0]), float(table["size"][1]))
    size_reference = max(table_size)
    dims = load_object_dims(data_dir, table_size)
    out = {}
    for object_id, dim in sorted(dims.items()):
        sx, sy = dim["size_m"]
        sx = sx / size_reference * 2.0
        sy = sy / size_reference * 2.0
        long_side, short_side = sorted((float(sx), float(sy)), reverse=True)
        out[object_id] = [round(long_side, 4), round(short_side, 4)]
    return out


def build_prompt(object_id_map: dict[str, str], size_prior: dict[str, list[float]]) -> str:
    guide = GUIDE_PATH.read_text()
    return (
        f"{guide}\n\n"
        "Now infer layout constraints using TWO attached images.\n\n"
        "Image 1 is the real current scene segmentation. It labels the REAL input objects with numeric ids. "
        "Use it to establish which object_id each number refers to and what each real object looks like. "
        "Do not copy the current scene's messy positions as the target layout.\n\n"
        "Image 2 is a tidy-scene reference image without segmentation labels. Use it as the target "
        "arrangement style and layout prior. "
        "It may miss some real objects, merge multiple real objects into one visible object, or contain extra objects "
        "that are not in the real inventory. The output must always arrange the REAL objects from Image 1 / the map below.\n\n"
        "Important reasoning rules:\n"
        "- If the tidy reference has one object but the real scene has multiple similar objects, arrange all real similar objects "
        "as an analogous group, preferably using same-line or evenly-spaced relations.\n"
        "- If the tidy reference has extra objects not present in the real inventory, ignore those extra objects.\n"
        "- If the tidy reference is missing a real object, infer a reasonable position from similar objects and common table layout.\n"
        "- Always output constraints for object_id values, never numeric ids.\n"
        "- Prefer structure priors such as in_same_horizontal_line, in_same_vertical_line, evenly_spaced_from_anchor, "
        "on_top_of, and in_holder when visible or inferable.\n\n"
        "Numeric-id to object_id map for the real current scene:\n"
        f"{json.dumps(object_id_map, ensure_ascii=False, indent=2)}\n\n"
        "Approximate top-down object AABB sizes. Both dimensions use the tabletop long side / horizontal width "
        "as the single reference length, so the two numbers are directly comparable. "
        "Each value is [long_side, short_side]; use these only as rough scale hints for spacing and grouping:\n"
        f"{json.dumps(size_prior, ensure_ascii=False, indent=2)}\n\n"
        "Return only the final JSON object with the top-level key constraints."
    )


def run_case(data_dir: Path) -> dict[str, Any]:
    out_dir = data_dir / EXP_REL
    relations_path = out_dir / RELATIONS_NAME
    if relations_path.is_file():
        return {
            "case": data_dir.name,
            "status": "skipped_existing",
            "relations": str(relations_path),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    current_segmentation = render_current_segmentation(data_dir, str(EXP_REL))
    object_id_map = load_object_id_map(out_dir)
    goal_image = raw_goal_to(data_dir, out_dir)
    size_prior = object_size_prior(data_dir)

    prompt = build_prompt(object_id_map, size_prior)
    (out_dir / PROMPT_NAME).write_text(prompt)

    codex = load_codex()
    vlm_start = time.perf_counter()
    raw = codex(prompt, [str(current_segmentation), str(goal_image)], reasoning_effort="low")
    vlm_seconds = time.perf_counter() - vlm_start
    (out_dir / RAW_RESPONSE_NAME).write_text(raw)

    relations = parse_relation_response(raw)
    relations_path.write_text(json.dumps(relations, indent=2, ensure_ascii=False))

    validation = evaluate_relations(object_id_map, relations["constraints"])
    (out_dir / VALIDATION_NAME).write_text(json.dumps(validation, indent=2, ensure_ascii=False))

    render_result = render_if_full_ok(data_dir, out_dir)
    return {
        "case": data_dir.name,
        "status": "completed",
        "current_segmentation": str(current_segmentation),
        "goal_image": str(goal_image),
        "relations": str(relations_path),
        "validation": str(out_dir / VALIDATION_NAME),
        "vlm_seconds": vlm_seconds,
        **render_result,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: python {Path(__file__).name} DATA_PARENT_DIR")

    parent_dir = Path(sys.argv[1])
    if not parent_dir.is_dir():
        raise FileNotFoundError(parent_dir)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM_CALLS) as executor:
        futures = [executor.submit(run_case, data_dir) for data_dir in case_dirs(parent_dir)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    results.sort(key=lambda item: item["case"])
    summary = {
        "parent_dir": str(parent_dir),
        "max_concurrent_vlm_calls": MAX_CONCURRENT_VLM_CALLS,
        "total": len(results),
        "completed": sum(1 for item in results if item["status"] == "completed"),
        "skipped_existing": sum(1 for item in results if item["status"] == "skipped_existing"),
        "full_ok": sum(1 for item in results if item.get("full_ok") is True),
        "rendered": sum(1 for item in results if item.get("rendered") is True),
        "results": results,
    }
    summary_path = parent_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(summary_path)


if __name__ == "__main__":
    main()
