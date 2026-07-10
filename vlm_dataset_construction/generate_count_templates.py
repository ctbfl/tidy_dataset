"""第三层闭环:基于某条 variation 的叙事,自动调 VLM 产出物体数量清单并写回 DB。

variation(narrative) --build_count_template_prompt--> VLM(codex) --parse--> add_count_template --> DB
prompt / 原始返回 / 解析结果存到 result/<exp>/ 追溯。

用法(函数):
    from generate_count_templates import generate_count_template, generate_for_scenario
    tid = generate_count_template(1)                       # 给 variation 1 生成一份清单
    res = generate_for_scenario("coffee_table", only_missing=True)   # 批量:给没清单的叙事各生成一份

用法(命令行):
    python generate_count_templates.py --variation 1
    python generate_count_templates.py --scenario coffee_table --only-missing
    python generate_count_templates.py --scenario coffee_table --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
for p in (str(HERE), str(ORGANIZE_IT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_count_template_prompt import build_count_template_prompt  # noqa: E402
from db import ScenarioDB                                            # noqa: E402
from organize_it.modules import vlm                                 # noqa: E402

RESULT_DIR = HERE / "result"


def _clean_counts(mapping) -> list[dict] | None:
    """{"key": count, ...} 或 [{"category":k,"count":n},...] → [{category,count}, ...]。"""
    pairs = []
    if isinstance(mapping, dict):
        for k, v in mapping.items():
            pairs.append((str(k).strip(), v))
    elif isinstance(mapping, list):
        for o in mapping:
            if not (isinstance(o, dict) and "category" in o and "count" in o):
                return None
            pairs.append((str(o["category"]).strip(), o["count"]))
    else:
        return None
    out = []
    for cat, cnt in pairs:
        try:
            out.append({"category": cat, "count": int(cnt)})
        except (ValueError, TypeError):
            return None
    return out or None


def _clean_template(item) -> dict | None:
    """一份桌面 → {"instructions": str, "objects": [{category,count},...]}。

    主形态: {"instructions": "...", "objects": {"key": count, ...}}
    兼容:   裸的 {"key": count} 或 [{category,count}](无 instructions)。
    """
    if isinstance(item, dict) and "objects" in item:
        objs = _clean_counts(item["objects"])
        instr = str(item.get("instructions", "")).strip()
    else:
        objs = _clean_counts(item)
        instr = ""
    if objs is None:
        return None
    return {"instructions": instr, "objects": objs}


def extract_templates(text: str) -> list[dict] | None:
    """从模型输出里抽出 K 份桌面: [{"instructions","objects"}, ...]。

    优先 ```json 代码块;兼容单份(退化为 1 份)。
    """
    blocks = re.findall(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, flags=re.DOTALL)
    candidates = list(blocks) or re.findall(r"([\[{].*[\]}])", text, flags=re.DOTALL)
    for chunk in reversed(candidates):
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):                              # 单份 → 1 元素
            one = _clean_template(parsed)
            return [one] if one else None
        if isinstance(parsed, list) and parsed:
            tpls = [_clean_template(x) for x in parsed]
            if all(t is not None for t in tpls):
                return tpls
    return None


def generate_count_template(
    variation_id: int,
    *,
    k: int = 8,
    n_min: int = 4,
    n_max: int = 9,
    total_min: int = 5,
    total_max: int = 12,
    exp_name: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    timeout: float = 300,
    validate: bool = True,
    db: ScenarioDB | None = None,
) -> list[int]:
    """给一条 variation 一次生成 k 份 count_template 并写库。返回新 template id 列表(解析失败为空)。"""
    db = db or ScenarioDB()
    var = db.get_variation(variation_id)
    if not var:
        raise ValueError(f"variation 不存在: {variation_id}")
    scn = db.get_scenario(var["scenario_id"])
    direction = scn["direction"] or scn["slug"]
    model = model or vlm.OPENAI_MODEL

    prompt = build_count_template_prompt(direction, var["variation"], k=k, n_min=n_min,
                                         n_max=n_max, total_min=total_min, total_max=total_max)

    exp_name = exp_name or f"counttpl_{scn['slug']}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"v{variation_id}"
    (out_dir / f"{stem}_prompt.md").write_text(prompt)

    response = vlm.codex(prompt, model=model, reasoning_effort=effort, timeout=timeout)
    (out_dir / f"{stem}_response.md").write_text(response)

    templates = extract_templates(response)
    (out_dir / f"{stem}_objects.json").write_text(
        json.dumps(templates, ensure_ascii=False, indent=2) if templates is not None else "null"
    )
    if not templates:
        print(f"  [v{variation_id}] WARNING: 无法解析桌面清单")
        return []

    tids = []
    for i, tpl in enumerate(templates, 1):
        try:
            tid = db.add_count_template(variation_id, tpl["objects"],
                                        instructions=tpl["instructions"],
                                        note=f"auto:{exp_name}#{i}", validate=validate)
            tids.append(tid)
        except ValueError as e:  # 某份含非法类目:跳过该份,不阻断其余
            print(f"  [v{variation_id}] 第{i}份清单被拒: {e}")
    return tids


def generate_for_scenario(
    scenario,
    *,
    only_missing: bool = True,
    **kwargs,
) -> list[dict]:
    """批量:为 scenario 下的 variation 各生成一份 count_template。

    only_missing=True 只处理还没有任何 count_template 的叙事。
    共用一个 exp_name(默认按 scenario+时间戳),所有产物落在同一 result 文件夹。
    """
    db = kwargs.pop("db", None) or ScenarioDB()
    scn = db.get_scenario(scenario)
    if not scn:
        raise ValueError(f"scenario 不存在: {scenario!r}")
    exp_name = kwargs.pop("exp_name", None) or f"counttpl_{scn['slug']}_{time.strftime('%Y%m%d_%H%M%S')}"
    results = []
    for v in db.list_variations(scn["slug"]):
        if only_missing and db.list_count_templates(v["id"]):
            continue
        try:
            tids = generate_count_template(v["id"], exp_name=exp_name, db=db, **kwargs)
        except Exception as e:  # noqa: BLE001  一条失败不阻断整批
            tids = []
            print(f"  [v{v['id']}] ERROR: {e}")
        results.append({"variation_id": v["id"], "template_ids": tids, "narrative": v["variation"]})
        print(f"  v{v['id']} -> {len(tids)} templates: {tids}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="基于 variation 叙事自动生成 count_template 写入 DB")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--variation", type=int, help="单条 variation id")
    g.add_argument("--scenario", help="批量:该 scenario 下的 variation")
    ap.add_argument("--all", action="store_true", help="批量时:即使已有清单也再生成")
    ap.add_argument("--only-missing", action="store_true", help="批量时:只处理没有清单的(默认行为)")
    ap.add_argument("--k", type=int, default=8, help="每条 variation 产出几份清单(默认 8)")
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=9)
    ap.add_argument("--total-min", type=int, default=5)
    ap.add_argument("--total-max", type=int, default=12)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()

    kw = dict(k=args.k, n_min=args.n_min, n_max=args.n_max, total_min=args.total_min,
              total_max=args.total_max, model=args.model, effort=args.effort)

    if args.variation is not None:
        tids = generate_count_template(args.variation, **kw)
        print(f"variation {args.variation} -> {len(tids)} templates: {tids}")
        return 0 if tids else 1

    res = generate_for_scenario(args.scenario, only_missing=not args.all, **kw)
    total = sum(len(r["template_ids"]) for r in res)
    ok = sum(1 for r in res if r["template_ids"])
    print(f"\ndone: {ok}/{len(res)} variations, {total} count_templates written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
