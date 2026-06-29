from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from relation_state import evaluate_relations
from render_goal_segmentation import (
    CURRENT_OUTPUT_NAME,
    OBJECT_ID_MAP_NAME,
    color_map,
    draw_missing_list,
    draw_segmentation_labels,
    ensure_hw_mask,
    ensure_rgb_uint8,
    find_scene_path,
    load_scene,
    render_current_segmentation,
    write_object_id_map,
)


DATA_ROOT = Path("/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/dining_table/after_meal_cleanup_v2")
ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
GUIDE_PATH = Path(__file__).resolve().parent / "layout_constraint_vlm_guide.md"
CASES = ("001", "011", "021", "031")
EXP_ROOT = Path("debug_vlm_relation_extraction/exp1")

GOAL_SEGMENTATION_NAME = "goal_segmentation_for_relation_detection.png"
GOAL_RAW_NAME = "goal_raw.png"
PROMPT_NAME = "vlm_relation_prompt.txt"
RAW_RESPONSE_NAME = "vlm_relation_response.txt"
RELATIONS_NAME = "relations.json"
VALIDATION_NAME = "validation_result.json"
SUMMARY_NAME = "summary.json"


def load_codex():
    src = str(ORGANIZE_IT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from organize_it.modules.vlm import codex

    return codex


def best_goal_matches(data_dir: Path, numbers: dict[str, str]) -> dict[int, str]:
    path = data_dir / "goal_matching" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing goal matching summary: {path}")
    data = json.loads(path.read_text())
    matches = data.get("matches")
    if not isinstance(matches, list):
        raise ValueError(f"{path} must contain matches list")

    best: dict[int, tuple[float, str]] = {}
    for match in matches:
        object_id = match.get("obj_id")
        raw_id = match.get("raw_goal_mask_id")
        score = match.get("score")
        if object_id not in numbers:
            continue
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        prev = best.get(raw_id)
        if prev is None or float(score) > prev[0]:
            best[raw_id] = (float(score), str(object_id))
    return {raw_id: object_id for raw_id, (_, object_id) in best.items()}


def grouped_goal_masks_best_match(data_dir: Path, scene) -> tuple[list[dict], list[str], dict[str, str]]:
    numbers = {object_id: str(index) for index, object_id in enumerate(sorted(scene.objects), start=1)}
    best_by_raw = best_goal_matches(data_dir, numbers)
    groups: dict[int, dict] = {}
    missing: list[str] = []

    for object_id, obj in scene.objects.items():
        raw_id = getattr(obj, "raw_goal_mask_id", None)
        mask = getattr(obj, "goal_mask", None)
        if mask is None or raw_id is None:
            missing.append(numbers[str(object_id)])
            continue
        raw_id = int(raw_id)
        if best_by_raw.get(raw_id) != str(object_id):
            continue
        groups[raw_id] = {
            "raw_id": raw_id,
            "numbers": [numbers[str(object_id)]],
            "mask": mask,
        }

    missing.sort(key=int)
    out = sorted(groups.values(), key=lambda group: int(group["raw_id"]))
    return out, missing, numbers


def render_goal_segmentation_to(data_dir: Path, out_dir: Path) -> Path:
    scene_path = find_scene_path(data_dir)
    scene = load_scene(scene_path)
    if getattr(scene, "goal_image", None) is None:
        raise ValueError(f"scene has no goal_image: {scene_path}")

    image_rgb = ensure_rgb_uint8(scene.goal_image)
    height, width = image_rgb.shape[:2]
    groups, missing, numbers = grouped_goal_masks_best_match(data_dir, scene)

    overlay = image_rgb.copy().astype(np.float32)
    labels = []
    for color, group in zip(color_map(len(groups)), groups):
        mask = ensure_hw_mask(group["mask"], height, width)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
        labels.append((group["numbers"][0], mask, color))

    output = overlay.clip(0, 255).astype(np.uint8)
    draw_segmentation_labels(output, labels)
    draw_missing_list(output, missing)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_object_id_map(out_dir, numbers)
    save_path = out_dir / GOAL_SEGMENTATION_NAME
    cv2.imwrite(str(save_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    return save_path


def load_object_id_map(debug_dir: Path) -> dict[str, str]:
    data = json.loads((debug_dir / OBJECT_ID_MAP_NAME).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{debug_dir / OBJECT_ID_MAP_NAME} must contain a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def build_prompt(object_id_map: dict[str, str], method: str) -> str:
    goal_description = {
        "m1": (
            "Image 2 is a tidy-scene reference segmentation image. It labels visible target-layout regions "
            "with numeric ids for high-confidence explicit goal matches. If multiple real objects matched the same "
            "goal region, only the highest-score match is labeled there. Use labeled goal regions as strong anchors, "
            "but still arrange every real object from Image 1."
        ),
        "m2": (
            "Image 2 is a tidy-scene reference image without segmentation labels. Use it as the target "
            "arrangement style and layout prior."
        ),
    }[method]
    guide = GUIDE_PATH.read_text()
    return (
        f"{guide}\n\n"
        "Now infer layout constraints using TWO attached images.\n\n"
        "Image 1 is the real current scene segmentation. It labels the REAL input objects with numeric ids. "
        "Use it to establish which object_id each number refers to and what each real object looks like. "
        "Do not copy the current scene's messy positions as the target layout.\n\n"
        f"{goal_description} "
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
    if not isinstance(data.get("constraints"), list):
        raise ValueError("VLM response must contain constraints list")
    return data


def raw_goal_to(data_dir: Path, out_dir: Path) -> Path:
    source = data_dir / "goal.png"
    if not source.is_file():
        raise FileNotFoundError(f"missing goal.png in {data_dir}")
    target = out_dir / GOAL_RAW_NAME
    shutil.copyfile(source, target)
    return target


def run_one(codex, case_id: str, method: str) -> dict:
    data_dir = DATA_ROOT / case_id
    out_dir = data_dir / EXP_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    current_segmentation = render_current_segmentation(data_dir, str(EXP_ROOT / method))
    object_id_map = load_object_id_map(out_dir)
    if method == "m1":
        goal_image = render_goal_segmentation_to(data_dir, out_dir)
    elif method == "m2":
        goal_image = raw_goal_to(data_dir, out_dir)
    else:
        raise ValueError(f"unknown method: {method}")

    prompt = build_prompt(object_id_map, method)
    (out_dir / PROMPT_NAME).write_text(prompt)

    raw = codex(prompt, [str(current_segmentation), str(goal_image)], reasoning_effort="low")
    (out_dir / RAW_RESPONSE_NAME).write_text(raw)

    relations = parse_relation_response(raw)
    (out_dir / RELATIONS_NAME).write_text(json.dumps(relations, indent=2, ensure_ascii=False))

    validation = evaluate_relations(object_id_map, relations["constraints"])
    (out_dir / VALIDATION_NAME).write_text(json.dumps(validation, indent=2, ensure_ascii=False))

    return {
        "case": case_id,
        "method": method,
        "current_segmentation": str(out_dir / CURRENT_OUTPUT_NAME),
        "goal_image": str(goal_image),
        "object_id_map": str(out_dir / OBJECT_ID_MAP_NAME),
        "prompt": str(out_dir / PROMPT_NAME),
        "raw_response": str(out_dir / RAW_RESPONSE_NAME),
        "relations": str(out_dir / RELATIONS_NAME),
        "validation": validation,
    }


def main() -> None:
    codex = load_codex()
    results = []
    for case_id in CASES:
        for method in ("m1", "m2"):
            result = run_one(codex, case_id, method)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary_path = DATA_ROOT / SUMMARY_NAME
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(summary_path)


if __name__ == "__main__":
    main()
