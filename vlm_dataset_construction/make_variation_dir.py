"""从 DB 的 scenario+variation 生成"数据集需要的 variation 文件夹"。

产出结构(constrain_annotation_server 会自动发现):
    <parent_dir>/<variation_slug>/template/
        info.json                         # {scene_description, user_note=叙事}
        available_assets.json             # {version, assets_group=各 count_template 类目并集}  ← 左边栏
        constraints/
            t<template_id>.json           # 每个 count_template 一份 DSL(初始化用)

DSL 由 count_template 初始化(无双向绑定):
  - object_sets:  按 count 展开(count=2 → 两条 {category})
  - sample_entry_index: 每个 set 默认用第 0 个类内资产([0]*count),你可自行调整
  - user_prompt:  = count_template.instructions
  - selection_constraints / constraints: 留空(relation 你在标注工具里画)

用法:
    from make_variation_dir import make_variation_dir, make_all
    make_variation_dir("coffee_table", "watch_variety_snacking",
                       "/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2/living_room_ai")
    make_all("coffee_table", ".../living_room_ai")

CLI:
    python make_variation_dir.py --scenario coffee_table --parent .../living_room_ai --all
    python make_variation_dir.py --scenario coffee_table --parent .../living_room_ai --variation watch_variety_snacking
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from db import ScenarioDB  # noqa: E402


def _resolve_variation(db: ScenarioDB, scenario, ident) -> dict:
    variations = db.list_variations(scenario)
    key = str(ident)
    for v in variations:
        if str(v["id"]) == key or v["slug"] == key:
            return v
    raise ValueError(f"scenario {scenario!r} 下找不到 variation {ident!r}")


def _dsl_from_template(tpl: dict, *, scenario: str, variation: str) -> dict:
    """一个 count_template → 一份初始化 DSL(relation 空,资产 id 默认第 0 个)。"""
    object_sets = []
    per_cat_count: "OrderedDict[str,int]" = OrderedDict()
    for o in tpl["objects"]:
        cat, cnt = o["category"], int(o["count"])
        per_cat_count[cat] = per_cat_count.get(cat, 0) + cnt
        object_sets.extend({"category": cat} for _ in range(cnt))
    sample_entry_index = {cat: [0] * n for cat, n in per_cat_count.items()}
    return {
        "version": 1,
        "scenario": scenario,
        "variation": variation,
        "user_prompt": tpl.get("instructions", ""),
        "object_sets": object_sets,
        "sample_entry_index": sample_entry_index,
        "selection_constraints": [],
        "constraints": [],
    }


def make_variation_dir(scenario, variation_ident, parent_dir, *,
                       db: ScenarioDB | None = None, overwrite: bool = False) -> Path:
    """产出一个 variation 文件夹,返回 template 目录路径。"""
    db = db or ScenarioDB()
    var = _resolve_variation(db, scenario, variation_ident)
    slug = (var["slug"] or "").strip()
    if not slug:
        raise ValueError(f"variation {var['id']} 没有 slug(简写英文名),先在 DB 里补上")

    parent = Path(parent_dir)
    dataset_scenario = parent.name
    template_dir = parent / slug / "template"
    if template_dir.exists() and not overwrite:
        raise FileExistsError(f"{template_dir} 已存在(加 overwrite=True 覆盖)")
    (template_dir / "constraints").mkdir(parents=True, exist_ok=True)

    templates = db.list_count_templates(var["id"])

    # info.json —— 记录该 variation 层级信息
    (template_dir / "info.json").write_text(json.dumps(
        {"scene_description": "", "user_note": var["variation"]},
        ensure_ascii=False, indent=2))

    # available_assets.json —— assets_group = 所有 count_template 类目并集(左边栏)
    assets_group = sorted({o["category"] for t in templates for o in t["objects"]})
    (template_dir / "available_assets.json").write_text(json.dumps(
        {"version": 1, "assets_group": assets_group}, ensure_ascii=False, indent=2))

    # constraints/ —— 每个 count_template 一份初始化 DSL
    for t in templates:
        dsl = _dsl_from_template(t, scenario=dataset_scenario, variation=slug)
        (template_dir / "constraints" / f"t{t['id']}.json").write_text(
            json.dumps(dsl, ensure_ascii=False, indent=2))

    print(f"  {slug}: {len(templates)} DSL, {len(assets_group)} 类目 -> {template_dir}")
    return template_dir


def make_all(scenario, parent_dir, *, overwrite: bool = False, db: ScenarioDB | None = None) -> list[Path]:
    db = db or ScenarioDB()
    out = []
    for v in db.list_variations(scenario):
        if not (v["slug"] or "").strip():
            print(f"  跳过 v{v['id']}(无 slug)")
            continue
        out.append(make_variation_dir(scenario, v["id"], parent_dir, db=db, overwrite=overwrite))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="从 DB 生成数据集 variation 文件夹")
    ap.add_argument("--scenario", required=True, help="DB scenario slug")
    ap.add_argument("--parent", required=True, help="输出父目录(数据集 scenario 目录,如 .../living_room_ai)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--variation", help="单个 variation(id 或 slug)")
    g.add_argument("--all", action="store_true", help="该 scenario 全部 variation")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.all:
        made = make_all(args.scenario, args.parent, overwrite=args.overwrite)
        print(f"done: {len(made)} variations")
    else:
        make_variation_dir(args.scenario, args.variation, args.parent, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
