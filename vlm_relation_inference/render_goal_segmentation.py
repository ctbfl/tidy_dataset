from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
OUTPUT_NAME = "segmentation_for_relation_detection.png"
CURRENT_OUTPUT_NAME = "current_segmentation_for_relation_detection.png"
DEBUG_DIR_NAME = "debug_vlm_relation_extraction"
OBJECT_ID_MAP_NAME = "object_id_map.json"


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_scene(path: Path) -> Any:
    if not ORGANIZE_IT_SRC.is_dir():
        raise FileNotFoundError(f"missing organize_it source: {ORGANIZE_IT_SRC}")
    src = str(ORGANIZE_IT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return _NumpyCompatUnpickler(f).load()


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if image.dtype.kind == "f" and image.max() <= 1.0 + 1e-6:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"goal_image must be HxWx3, got {image.shape}")
    return image[..., :3]


def ensure_hw_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    m = m.astype(np.uint8)
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def color_map(n: int) -> list[np.ndarray]:
    colors = []
    for i in range(max(0, n)):
        hue = int(180 * i / max(1, n))
        color = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(color.astype(np.float32))
    return colors


def label_candidates(mask: np.ndarray, max_candidates: int = 80) -> list[tuple[int, int, float]]:
    m = np.asarray(mask).astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if num <= 1:
        return []
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = labels == best
    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.where(dist > 0)
    if len(px) == 0:
        return []
    scores = dist[py, px]
    keep = min(max_candidates * 12, len(scores))
    idx = np.argpartition(scores, -keep)[-keep:] if keep < len(scores) else np.arange(len(scores))
    idx = idx[np.argsort(scores[idx])[::-1]]
    min_sep = max(6.0, float(min(mask.shape)) / 80.0)
    out: list[tuple[int, int, float]] = []
    for i in idx:
        x, y, score = int(px[i]), int(py[i]), float(scores[i])
        if any((x - ox) ** 2 + (y - oy) ** 2 < min_sep ** 2 for ox, oy, _ in out):
            continue
        out.append((x, y, score))
        if len(out) >= max_candidates:
            break
    return out


def auto_text_color(image_rgb: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x0, y0, x1, y1 = rect
    patch = image_rgb[y0:y1, x0:x1, :3]
    if patch.size == 0:
        return (255, 255, 255)
    rgb = patch.astype(np.float32)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return (0, 0, 0) if float(np.mean(luminance)) > 150.0 else (255, 255, 255)


def text_mask(
    image_shape: tuple[int, int],
    text: str,
    origin: tuple[int, int],
    font: int,
    scale: float,
    thickness: int,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.putText(mask, text, origin, font, scale, 255, thickness, cv2.LINE_AA)
    return mask > 0


def label_rect(
    center_x: int,
    center_y: int,
    text_w: int,
    text_h: int,
    baseline: int,
    image_w: int,
    image_h: int,
    pad: int,
) -> tuple[int, int, int, int]:
    box_w = text_w + pad * 2
    box_h = text_h + baseline + pad * 2
    x0 = max(0, min(image_w - box_w, int(center_x - box_w / 2)))
    y0 = max(0, min(image_h - box_h, int(center_y - box_h / 2)))
    return x0, y0, x0 + box_w, y0 + box_h


def draw_segmentation_labels(
    image_rgb: np.ndarray,
    labels: list[tuple[str, np.ndarray, np.ndarray]],
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    masks = [ensure_hw_mask(mask, height, width) for _, mask, _ in labels]
    occupied_text = np.zeros((height, width), dtype=bool)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 4

    for label_idx, (label, _, _) in enumerate(labels):
        mask = masks[label_idx]
        other_objects = np.zeros((height, width), dtype=bool)
        for other_idx, other_mask in enumerate(masks):
            if other_idx != label_idx:
                other_objects |= other_mask

        candidates = label_candidates(mask)
        if not candidates:
            continue
        scale = 0.6
        (text_w, text_h), baseline = cv2.getTextSize(str(label), font, scale, 2)
        if text_w > width - pad * 4:
            scale = max(0.35, scale * (width - pad * 4) / max(1, text_w))
            (text_w, text_h), baseline = cv2.getTextSize(str(label), font, scale, 2)

        best_rect = None
        best_text_mask = None
        best_score = None
        for cx, cy, dist_score in candidates:
            rect = label_rect(cx, cy, text_w, text_h, baseline, width, height, pad)
            x0, y0, _, _ = rect
            origin = (x0 + pad, y0 + pad + text_h)
            mask_text = text_mask((height, width), str(label), origin, font, scale, 2)
            text_pixels = max(1, int(mask_text.sum()))
            self_overlap = int(np.logical_and(mask_text, mask).sum())
            if self_overlap == 0:
                continue
            score = (
                int(np.logical_and(mask_text, occupied_text).sum()) / text_pixels,
                int(np.logical_and(mask_text, other_objects).sum()) / text_pixels,
                (text_pixels - self_overlap) / text_pixels,
                -dist_score,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_rect = rect
                best_text_mask = mask_text
        if best_rect is None or best_text_mask is None:
            continue

        x0, y0, _, _ = best_rect
        text_color = auto_text_color(image_rgb, best_rect)
        cv2.putText(
            image_rgb,
            str(label),
            (x0 + pad, y0 + pad + text_h),
            font,
            scale,
            text_color,
            2,
            cv2.LINE_AA,
        )
        occupied_text |= best_text_mask
    return image_rgb


def object_number_map(scene: Any) -> dict[str, str]:
    ordered = sorted(scene.objects)
    return {object_id: str(index) for index, object_id in enumerate(ordered, start=1)}


def grouped_goal_masks(scene: Any) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    numbers = object_number_map(scene)
    groups: dict[Any, dict[str, Any]] = {}
    missing: list[str] = []
    for object_id, obj in scene.objects.items():
        mask = getattr(obj, "goal_mask", None)
        raw_id = getattr(obj, "raw_goal_mask_id", None)
        if mask is None or raw_id is None:
            missing.append(numbers[str(object_id)])
            continue
        group = groups.setdefault(raw_id, {"raw_id": raw_id, "numbers": [], "masks": []})
        group["numbers"].append(numbers[str(object_id)])
        group["masks"].append(mask)

    out = []
    for group in groups.values():
        mask_union = None
        for mask in group["masks"]:
            m = np.asarray(mask).astype(bool)
            mask_union = m if mask_union is None else np.logical_or(mask_union, m)
        group["mask"] = mask_union
        out.append(group)
    out.sort(key=lambda group: (0, int(group["raw_id"])) if isinstance(group["raw_id"], (int, np.integer)) else (1, str(group["raw_id"])))
    missing.sort(key=int)
    return out, missing, numbers


def grouped_current_masks(scene: Any) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    numbers = object_number_map(scene)
    groups = []
    missing: list[str] = []
    for object_id, obj in sorted(scene.objects.items()):
        mask = getattr(obj, "mask", None)
        if mask is None:
            missing.append(numbers[str(object_id)])
            continue
        groups.append({
            "raw_id": numbers[str(object_id)],
            "numbers": [numbers[str(object_id)]],
            "mask": mask,
        })
    missing.sort(key=int)
    return groups, missing, numbers


def label_for_group(group: dict[str, Any]) -> str:
    numbers = sorted(group["numbers"], key=int)
    return "/".join(numbers)


def draw_missing_list(image_rgb: np.ndarray, missing: list[str]) -> None:
    if not missing:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = ["missing:", *missing]
    scale = 0.55
    thickness = 2
    pad = 8
    line_h = 22
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + pad * 2
    height = line_h * len(lines) + pad * 2
    x0, y0 = 8, 8
    x1, y1 = min(image_rgb.shape[1] - 1, x0 + width), min(image_rgb.shape[0] - 1, y0 + height)
    cv2.rectangle(image_rgb, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(image_rgb, (x0, y0), (x1, y1), (0, 0, 0), 1)
    for i, line in enumerate(lines):
        cv2.putText(image_rgb, line, (x0 + pad, y0 + pad + 16 + i * line_h), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def find_scene_path(data_dir: Path) -> Path:
    direct = data_dir / "ta_real_scene.pkl"
    if direct.is_file():
        return direct
    matches = sorted(data_dir.rglob("ta_real_scene.pkl"))
    if not matches:
        raise FileNotFoundError(f"missing ta_real_scene.pkl under {data_dir}")
    if len(matches) > 1:
        raise ValueError(f"found multiple ta_real_scene.pkl files under {data_dir}; pass one scene data directory")
    return matches[0]


def write_object_id_map(debug_dir: Path, numbers: dict[str, str]) -> Path:
    path = debug_dir / OBJECT_ID_MAP_NAME
    by_number = {number: object_id for object_id, number in sorted(numbers.items(), key=lambda item: int(item[1]))}
    path.write_text(json.dumps(by_number, indent=2, ensure_ascii=False))
    return path


def render_goal_segmentation(data_dir: Path) -> Path:
    scene_path = find_scene_path(data_dir)
    scene = load_scene(scene_path)
    if getattr(scene, "goal_image", None) is None:
        raise ValueError(f"scene has no goal_image: {scene_path}")

    image_rgb = ensure_rgb_uint8(scene.goal_image)
    height, width = image_rgb.shape[:2]
    groups, missing, numbers = grouped_goal_masks(scene)

    overlay = image_rgb.copy().astype(np.float32)
    labels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for color, group in zip(color_map(len(groups)), groups):
        mask = ensure_hw_mask(group["mask"], height, width)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
        labels.append((label_for_group(group), mask, color))

    output = overlay.clip(0, 255).astype(np.uint8)
    draw_segmentation_labels(output, labels)
    draw_missing_list(output, missing)

    debug_dir = scene_path.parent / DEBUG_DIR_NAME
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_object_id_map(debug_dir, numbers)
    save_path = debug_dir / OUTPUT_NAME
    cv2.imwrite(str(save_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    return save_path


def render_current_segmentation(data_dir: Path, debug_dir_name: str = DEBUG_DIR_NAME) -> Path:
    scene_path = find_scene_path(data_dir)
    scene = load_scene(scene_path)
    if getattr(scene, "rgb", None) is None:
        raise ValueError(f"scene has no rgb: {scene_path}")

    image_rgb = ensure_rgb_uint8(scene.rgb)
    height, width = image_rgb.shape[:2]
    groups, missing, numbers = grouped_current_masks(scene)

    overlay = image_rgb.copy().astype(np.float32)
    labels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for color, group in zip(color_map(len(groups)), groups):
        mask = ensure_hw_mask(group["mask"], height, width)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
        labels.append((label_for_group(group), mask, color))

    output = overlay.clip(0, 255).astype(np.uint8)
    draw_segmentation_labels(output, labels)
    draw_missing_list(output, missing)

    debug_dir = scene_path.parent / debug_dir_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_object_id_map(debug_dir, numbers)
    save_path = debug_dir / CURRENT_OUTPUT_NAME
    cv2.imwrite(str(save_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    save_path = render_goal_segmentation(Path(args.data_dir))
    print(save_path)


if __name__ == "__main__":
    main()
