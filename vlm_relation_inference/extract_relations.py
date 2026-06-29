from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relation_state import evaluate_relations
from render_goal_segmentation import DEBUG_DIR_NAME, OBJECT_ID_MAP_NAME, render_goal_segmentation


ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
GUIDE_PATH = Path(__file__).resolve().parent / "layout_constraint_vlm_guide.md"
RAW_RESPONSE_NAME = "vlm_relation_response.txt"
PROMPT_NAME = "vlm_relation_prompt.txt"
RELATIONS_NAME = "relations.json"
VALIDATION_NAME = "validation_result.json"


def load_codex():
    src = str(ORGANIZE_IT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from organize_it.modules.vlm import codex

    return codex


def build_prompt(object_id_map: dict[str, str]) -> str:
    guide = GUIDE_PATH.read_text()
    return (
        f"{guide}\n\n"
        "Now infer layout constraints for the attached segmentation image.\n"
        "The image labels are numeric ids. Use this numeric-id to object_id map:\n"
        f"{json.dumps(object_id_map, ensure_ascii=False, indent=2)}\n\n"
        "Return only the final JSON object with the top-level key constraints."
    )


def load_object_id_map(debug_dir: Path) -> dict[str, str]:
    path = debug_dir / OBJECT_ID_MAP_NAME
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def parse_relation_response(raw: str) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")
    constraints = data.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError("VLM response must contain constraints list")
    return data


def extract_relations(data_dir: Path) -> dict:
    segmentation_path = render_goal_segmentation(data_dir)
    debug_dir = segmentation_path.parent
    object_id_map = load_object_id_map(debug_dir)
    prompt = build_prompt(object_id_map)
    (debug_dir / PROMPT_NAME).write_text(prompt)

    codex = load_codex()
    raw = codex(prompt, str(segmentation_path), reasoning_effort="low")
    (debug_dir / RAW_RESPONSE_NAME).write_text(raw)

    relations = parse_relation_response(raw)
    (debug_dir / RELATIONS_NAME).write_text(json.dumps(relations, indent=2, ensure_ascii=False))

    validation = evaluate_relations(object_id_map, relations["constraints"])
    (debug_dir / VALIDATION_NAME).write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    return {
        "segmentation": str(segmentation_path),
        "object_id_map": str(debug_dir / OBJECT_ID_MAP_NAME),
        "prompt": str(debug_dir / PROMPT_NAME),
        "raw_response": str(debug_dir / RAW_RESPONSE_NAME),
        "relations": str(debug_dir / RELATIONS_NAME),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    result = extract_relations(Path(args.data_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
