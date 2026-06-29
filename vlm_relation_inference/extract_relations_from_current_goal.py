from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relation_state import evaluate_relations
from render_goal_segmentation import OBJECT_ID_MAP_NAME, render_current_segmentation


ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
GUIDE_PATH = Path(__file__).resolve().parent / "layout_constraint_vlm_guide.md"
DEBUG_DIR_NAME = "debug_vlm_relation_extraction_current_goal"
PROMPT_NAME = "vlm_relation_prompt.txt"
RAW_RESPONSE_NAME = "vlm_relation_response.txt"
RELATIONS_NAME = "relations.json"
VALIDATION_NAME = "validation_result.json"


def load_codex():
    src = str(ORGANIZE_IT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from organize_it.modules.vlm import codex

    return codex


def goal_image_path(data_dir: Path) -> Path:
    path = data_dir / "goal.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing goal.png in {data_dir}")
    return path


def load_object_id_map(debug_dir: Path) -> dict[str, str]:
    path = debug_dir / OBJECT_ID_MAP_NAME
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def build_prompt(object_id_map: dict[str, str]) -> str:
    guide = GUIDE_PATH.read_text()
    return (
        f"{guide}\n\n"
        "Now infer layout constraints using TWO attached images.\n\n"
        "Image 1 is the real current scene segmentation. It labels the REAL input objects with numeric ids. "
        "Use it to establish which object_id each number refers to and what each real object looks like. "
        "Do not copy the current scene's messy positions as the target layout.\n\n"
        "Image 2 is a tidy-scene reference image. Use it as the target arrangement style and layout prior. "
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
        "Return only the final JSON object with the top-level key constraints."
    )


def parse_relation_response(raw: str) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")
    constraints = data.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError("VLM response must contain constraints list")
    return data


def extract_relations(data_dir: Path) -> dict:
    current_segmentation_path = render_current_segmentation(data_dir, DEBUG_DIR_NAME)
    debug_dir = current_segmentation_path.parent
    object_id_map = load_object_id_map(debug_dir)
    goal_path = goal_image_path(data_dir)

    prompt = build_prompt(object_id_map)
    (debug_dir / PROMPT_NAME).write_text(prompt)

    codex = load_codex()
    raw = codex(prompt, [str(current_segmentation_path), str(goal_path)], reasoning_effort="low")
    (debug_dir / RAW_RESPONSE_NAME).write_text(raw)

    relations = parse_relation_response(raw)
    (debug_dir / RELATIONS_NAME).write_text(json.dumps(relations, indent=2, ensure_ascii=False))

    validation = evaluate_relations(object_id_map, relations["constraints"])
    (debug_dir / VALIDATION_NAME).write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    return {
        "current_segmentation": str(current_segmentation_path),
        "goal_image": str(goal_path),
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
