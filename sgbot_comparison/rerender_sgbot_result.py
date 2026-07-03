#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


ORGANIZE_IT_V2_ROOT = Path("/home/hjs/Projects/table_arrangement/organize_it_v2")
MAIN_SGBOT_COMPARE = ORGANIZE_IT_V2_ROOT / "experiments" / "main_sgbot_compare"
CUSTOM_SGBOT_DIR = MAIN_SGBOT_COMPARE / "custom_sgbot"
SGBOT_DIR = ORGANIZE_IT_V2_ROOT / "SG-Bot"
ORACLE_RESULTS = SGBOT_DIR / "sg_bot_oracle_results"

sys.path.insert(0, str(CUSTOM_SGBOT_DIR))
sys.path.insert(0, str(SGBOT_DIR))

import run_custom_sgbot as rcs  # noqa: E402


def render_one(scene_dir: Path) -> Path:
    os.chdir(SGBOT_DIR)
    scene_dir = scene_dir.resolve()
    random_scene_path = scene_dir / "sgbot_input" / "random_scene.json"
    result_pkl_path = ORACLE_RESULTS / scene_dir.name / "result.pkl"
    if not random_scene_path.is_file():
        raise FileNotFoundError(random_scene_path)
    if not result_pkl_path.is_file():
        raise FileNotFoundError(result_pkl_path)

    scene = json.loads(random_scene_path.read_text())
    rcs._uniquify_scene_object_names(scene)

    out_dir = scene_dir / "sgbot_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_view = out_dir / "synthetic_view_for_tidy_render.json"
    synthetic_view.write_text(json.dumps(scene, ensure_ascii=False, indent=2))

    import pybullet as pb
    import pybullet_utils.transformations as trans
    from initial import RearrangeEnv
    from PIL import Image

    try:
        env = RearrangeEnv(
            [(str(synthetic_view), str(synthetic_view))],
            rcs.MODELS_ROOT,
            None,
            5,
        )
        env.reset_rearrange(0)

        with result_pkl_path.open("rb") as f:
            result = pickle.load(f)
        final_poses = result["pred"]["final_poses_bullet"]
        for name, pose in final_poses.items():
            if name not in env.pybullet_id_dict:
                continue
            pose = np.asarray(pose, dtype=np.float64)
            quat = trans.quaternion_from_matrix(pose)
            pb.resetBasePositionAndOrientation(
                env.pybullet_id_dict[name],
                pose[:3, 3],
                quat,
            )

        output = out_dir / "result_tidy_render.png"
        Image.fromarray(rcs.render_with_tiny(pb, scene["camera_data"])).save(output)
    finally:
        pb.disconnect()
    return output


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} SCENE_DIR [SCENE_DIR ...]")
    for arg in sys.argv[1:]:
        output = render_one(Path(arg))
        print(output, flush=True)


if __name__ == "__main__":
    main()
