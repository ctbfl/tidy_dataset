from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from relation_state import evaluate_relations
from run_exp1_current_goal_ablation import (
    PROMPT_NAME,
    RAW_RESPONSE_NAME,
    RELATIONS_NAME,
    VALIDATION_NAME,
    build_prompt,
    load_codex,
    load_object_id_map,
    parse_relation_response,
    render_goal_segmentation_to,
)
from render_goal_segmentation import render_current_segmentation
from visualize_exp1_topdown import render_if_full_ok


MAX_CONCURRENT_VLM_CALLS = 6
EXP_REL = Path("debug_vlm_relation_extraction/exp1/m1")
SUMMARY_NAME = "m1_batch_summary.json"


def case_dirs(parent_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(parent_dir.iterdir())
        if path.is_dir() and re.fullmatch(r"\d{3}", path.name)
    ]


def run_case(data_dir: Path) -> dict[str, Any]:
    out_dir = data_dir / EXP_REL
    relations_path = out_dir / RELATIONS_NAME
    if relations_path.is_file():
        render_result = render_if_full_ok(data_dir, out_dir)
        return {
            "case": data_dir.name,
            "status": "skipped_existing",
            "relations": str(relations_path),
            **render_result,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    current_segmentation = render_current_segmentation(data_dir, str(EXP_REL))
    object_id_map = load_object_id_map(out_dir)
    goal_segmentation = render_goal_segmentation_to(data_dir, out_dir)

    prompt = build_prompt(object_id_map, "m1")
    (out_dir / PROMPT_NAME).write_text(prompt)

    codex = load_codex()
    vlm_start = time.perf_counter()
    raw = codex(prompt, [str(current_segmentation), str(goal_segmentation)], reasoning_effort="low")
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
        "goal_segmentation": str(goal_segmentation),
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
