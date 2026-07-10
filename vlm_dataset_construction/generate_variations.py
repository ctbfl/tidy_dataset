"""基于 DB 里某个 scenario 条目,自动调 VLM 生成 N 条 variation 并写回 DB。

闭环: scenario(direction) --build_prompt--> VLM(codex) --parse--> add_variation(去重) --> DB
同时把 prompt / 原始返回 / 解析结果 存到 result/<exp>/ 便于追溯。

用法(函数):
    from generate_variations import generate_variations
    r = generate_variations("coffee_table")           # 基于该 scenario 再生成 5 条(自动避开已有)
    print(r["added"], r["skipped"])

用法(命令行):
    python generate_variations.py --scenario coffee_table
    python generate_variations.py --scenario coffee_table --n 5 --no-dedup --exp coffee_table_run2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
for p in (str(HERE), str(ORGANIZE_IT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_scenario_prompt import build_prompt          # noqa: E402
from run_brainstorm import extract_narratives           # noqa: E402  (复用解析器)
from name_variations import fill_missing_slugs          # noqa: E402
from db import ScenarioDB                                # noqa: E402
from organize_it.modules import vlm                      # noqa: E402

RESULT_DIR = HERE / "result"


def generate_variations(
    scenario,
    *,
    n: int = 5,
    dedup_from_db: bool = True,
    exp_name: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    timeout: float = 300,
    name_slugs: bool = True,
    db: ScenarioDB | None = None,
) -> dict:
    """基于 scenario(slug 或 id)自动生成并写入 N 条 variation。

    返回 {exp, added:[{id,narrative}], skipped:[narrative], raw_count, out_dir}。
    dedup_from_db=True 时:prompt 里让模型避开 DB 已有叙事,且写库时再去重一次(双保险)。
    """
    db = db or ScenarioDB()
    scn = db.get_scenario(scenario)
    if not scn:
        raise ValueError(f"scenario 不存在于 DB: {scenario!r}(请先 get_or_create_scenario / import-exp)")
    slug = scn["slug"]
    direction = scn["direction"] or slug
    model = model or vlm.OPENAI_MODEL

    dedup = db.existing_narratives(slug) if dedup_from_db else None
    prompt = build_prompt(direction, dedup=dedup, n=n)

    exp_name = exp_name or f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.md").write_text(prompt)

    t0 = time.time()
    response = vlm.codex(prompt, model=model, reasoning_effort=effort, timeout=timeout)
    dt = time.time() - t0
    (out_dir / "response.md").write_text(response)

    narratives = extract_narratives(response) or []
    (out_dir / "narratives.json").write_text(json.dumps(narratives, ensure_ascii=False, indent=2))

    added, skipped = [], []
    for nar in narratives:
        vid = db.add_variation(slug, nar, source={"exp": exp_name, "model": model}, dedup=True)
        if vid is not None:
            added.append({"id": vid, "narrative": nar})
        else:
            skipped.append(nar)

    meta = {
        "exp": exp_name, "scenario": slug, "direction": direction, "model": model,
        "reasoning_effort": effort, "n_requested": n, "dedup_from_db": dedup_from_db,
        "raw_count": len(narratives), "added": len(added), "skipped": len(skipped),
        "elapsed_sec": round(dt, 1), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": vlm.OPENAI_BASE_URL,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # 补英文名:检查该 scenario 下所有还没 slug 的 variation,单独一次 VLM 调用补上
    # (英文名没必要占用上面第一次叙事生成的思考)。补名失败不影响主流程。
    slugs = {}
    if name_slugs:
        try:
            slugs = fill_missing_slugs(slug, db=db, model=model)
        except Exception as e:  # noqa: BLE001
            print(f"[slug] WARNING: 补英文名失败(可稍后 python name_variations.py --scenario {slug}): {e}")

    return {"exp": exp_name, "added": added, "skipped": skipped,
            "raw_count": len(narratives), "out_dir": str(out_dir), "slugs": slugs}


def main() -> int:
    ap = argparse.ArgumentParser(description="基于 scenario 自动生成 variation 并写入 DB")
    ap.add_argument("--scenario", required=True, help="DB 里的 scenario slug 或 id")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--no-dedup", action="store_true", help="不基于 DB 已有叙事去重")
    ap.add_argument("--exp", default=None, help="实验名(默认 <slug>_<时间戳>)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--no-name", action="store_true", help="跑完不自动补英文名")
    args = ap.parse_args()

    r = generate_variations(args.scenario, n=args.n, dedup_from_db=not args.no_dedup,
                            exp_name=args.exp, model=args.model, effort=args.effort,
                            name_slugs=not args.no_name)
    print(f"[exp] {r['exp']}  (raw {r['raw_count']} -> added {len(r['added'])}, skipped {len(r['skipped'])})")
    print(f"[artifacts] {r['out_dir']}")
    for a in r["added"]:
        slug = r["slugs"].get(a["id"], "")
        print(f"  + v{a['id']}  [{slug}]  {a['narrative']}")
    for s in r["skipped"]:
        print(f"  ~ (dup) {s}")
    if r["slugs"]:
        print(f"[slug] 补了 {len(r['slugs'])} 个英文名")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
