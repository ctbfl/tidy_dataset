"""给 variation 叙事批量起简写英文名(snake_case),并补全 DB 里缺名的 variation。

英文名只用来做数据集文件夹名,没必要占用第一次(叙事生成)模型的思考,所以单独一步补。

用法:
    from name_variations import name_variations, fill_missing_slugs
    slugs = name_variations(["下班后年轻人瘫沙发看综艺…", "小学生写作业家长陪着…"])
    # -> ["watch_variety_snacking", "kid_homework_parent"]

    fill_missing_slugs("coffee_table")   # 给该 scenario 下所有没 slug 的 variation 补名并写库

CLI:
    python name_variations.py --scenario coffee_table
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
for p in (str(HERE), str(ORGANIZE_IT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_brainstorm import extract_narratives   # noqa: E402  (解析 JSON 字符串数组)
from db import ScenarioDB                        # noqa: E402
from organize_it.modules import vlm              # noqa: E402


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s or "variation"


def _unique(slug: str, taken: set[str]) -> str:
    if slug not in taken:
        return slug
    i = 2
    while f"{slug}_{i}" in taken:
        i += 1
    return f"{slug}_{i}"


def build_naming_prompt(narratives: list[str]) -> str:
    listing = "\n".join(f"{i+1}. {n}" for i, n in enumerate(narratives))
    return f"""给你 {len(narratives)} 条"桌面整理"场景的生活叙事,请为每一条起一个简短的**英文文件夹名**。

要求:
- 全小写英文,单词间用下划线连接(snake_case)。
- 2–4 个词,概括这条叙事的核心活动/主体(如看综艺吃零食 → watch_variety_snacking)。
- 只用小写字母和下划线,**不要**数字、空格、标点、中文。
- {len(narratives)} 个名字彼此不同。

严格输出一个 **JSON 字符串数组**,长度正好 {len(narratives)},顺序与下面输入一一对应,不要输出数组以外的任何内容。

输入(按序):
{listing}

示例输出格式(请勿照抄内容):
```json
["watch_variety_snacking", "kid_homework_parent"]
```
"""


def name_variations(narratives: list[str], *, model: str | None = None,
                    effort: str | None = None, timeout: float = 180) -> list[str]:
    """一批叙事 -> 一批 snake_case 英文名(与输入等长、同序)。"""
    if not narratives:
        return []
    prompt = build_naming_prompt(narratives)
    resp = vlm.codex(prompt, model=model or vlm.OPENAI_MODEL, reasoning_effort=effort, timeout=timeout)
    slugs = extract_narratives(resp)
    if not slugs or len(slugs) != len(narratives):
        raise RuntimeError(f"起名返回 {len(slugs) if slugs else 0} 个,期望 {len(narratives)} 个")
    return [_slugify(s) for s in slugs]


def fill_missing_slugs(scenario, *, db: ScenarioDB | None = None,
                       model: str | None = None) -> dict[int, str]:
    """给 scenario 下所有缺 slug 的 variation 补英文名并写库。返回 {variation_id: slug}。"""
    db = db or ScenarioDB()
    variations = db.list_variations(scenario)
    missing = [v for v in variations if not (v["slug"] or "").strip()]
    if not missing:
        return {}
    slugs = name_variations([v["variation"] for v in missing], model=model)
    taken = {(v["slug"] or "").strip() for v in variations if (v["slug"] or "").strip()}
    out: dict[int, str] = {}
    for v, slug in zip(missing, slugs):
        slug = _unique(slug, taken)
        taken.add(slug)
        db.set_variation_slug(v["id"], slug)
        out[v["id"]] = slug
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="给缺名的 variation 补 snake_case 英文名")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    filled = fill_missing_slugs(args.scenario, model=args.model)
    if not filled:
        print("没有缺名的 variation")
    else:
        print(f"补了 {len(filled)} 个英文名:")
        for vid, slug in filled.items():
            print(f"  v{vid} -> {slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
