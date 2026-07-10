"""Build a scenario-brainstorm prompt for VLM/LLM dataset construction.

The prompt is assembled from four blocks so it generalises across big scenes
(coffee table / study desk / dining table / vanity …):

  1. direction   —— 大方向,由调用方给(这次要头脑风暴哪个大场景)。
  2. categories  —— 可用物体类目,**现场**从 data/organize_it_dataset_v2 解析成文本。
  3. dedup       —— 需要避开的已有情景,可选,由调用方给。
  4. workflow    —— 固定的三阶段工作流(发散 → 核对资产 → 定稿 5 个)。

模型最终只输出「一句话生活叙事」;每个情景具体用哪些物体 / 怎么摆 / 怎么弄乱,
由后续的其它 prompt 处理。

用法(库内调用):
    from build_scenario_prompt import build_prompt
    text = build_prompt("客厅茶几 coffee_table")
    text = build_prompt("书桌 study_desk", dedup=["研究生边吃外卖边看网课…"], n=5)

用法(命令行):
    python build_scenario_prompt.py --direction "客厅茶几 coffee_table"
    python build_scenario_prompt.py --direction "书桌 study_desk" --dedup old.txt --n 5 --out prompt.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATASET_DIR = Path(
    "/home/hjs/Projects/table_arrangement/tidy_dataset/data/organize_it_dataset_v2"
)


def load_categories(dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> dict[str, str]:
    """现场解析 available_assets.json,返回 {category_key: description}(按字母序)。"""
    path = Path(dataset_dir) / "available_assets.json"
    data = json.loads(path.read_text())
    cats = data.get("available_assets", data)
    out = {k: (v.get("description", "") if isinstance(v, dict) else "") for k, v in cats.items()}
    return {k: out[k] for k in sorted(out)}


def format_categories(categories: dict[str, str]) -> str:
    return "\n".join(f"- `{k}`: {v}" if v else f"- `{k}`" for k, v in categories.items())


def _load_dedup(dedup) -> list[str]:
    """dedup 可以是: None / list[str] / 一个文件路径(.json 数组 或 每行一条)。"""
    if not dedup:
        return []
    if isinstance(dedup, (list, tuple)):
        return [str(x).strip() for x in dedup if str(x).strip()]
    p = Path(dedup)
    if p.is_file():
        text = p.read_text()
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [str(dedup).strip()]


def build_prompt(
    direction: str,
    dedup=None,
    n: int = 5,
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
) -> str:
    """组装并返回完整 prompt 文本。

    direction: 本次头脑风暴的大方向,例如 "客厅茶几 coffee_table"。
    dedup:     需要避开的已有情景,list[str] 或文件路径,可选。
    n:         最终要输出几个新情景(默认 5)。
    """
    if not direction or not direction.strip():
        raise ValueError("direction 不能为空")
    direction = direction.strip()
    categories = load_categories(dataset_dir)
    cat_block = format_categories(categories)
    dedup_items = _load_dedup(dedup)

    if dedup_items:
        dedup_block = "\n".join(f"- {s}" for s in dedup_items)
    else:
        dedup_block = "（本次没有需要避开的情景。）"

    return f"""# 情景头脑风暴 · 大方向:{direction}

> 用途:为"桌面整理机器人"数据集头脑风暴真实生活情景 (variation)。本轮**只产出一句话生活叙事**;每个情景具体用哪些物体、怎么摆、怎么弄乱,由后续的其它 prompt 处理。
> 使用方法:把 `===== PROMPT 开始 =====` 到 `===== PROMPT 结束 =====` 之间的内容整段发给模型。可用物体类目已从数据集现场解析嵌入(共 {len(categories)} 类)。

===== PROMPT 开始 =====

## 你的角色
你是一位为"桌面整理机器人"设计数据集的生活场景编剧。你擅长捕捉普通人日常里一个个具体的瞬间,并把它们还原成一张桌面上真实会出现的物体布置。

## 本次大方向
{direction}

围绕这个大方向,想象**真实、日常、会实际发生**的生活情景。每个情景的灵魂是一句话生活叙事:**谁 + 在什么时刻 + 在干什么**。例如:"研究生边吃外卖边看网课,笔记本摊开记着重点。"

## 可用物体类目(你的想象边界,category key: 描述)
这是我们资产库里真实拥有的物体类目。你构思的情景,其中会出现的物体应基本都能在此找到对应项——它决定了"这个情景我们到底搭不搭得出来"。
{cat_block}

## 需要避开的已有情景(不要与这些重复或高度相似)
{dedup_block}

## 工作流(请严格按三阶段进行)

### 阶段一 · 发散
在「{direction}」这个大方向下,现实中会出现哪些真实场景?尽量多地列出候选(谁 / 何时 / 在干什么),先不管资产够不够,只求真实和多样(不同的人、不同时刻、不同活动)。

### 阶段二 · 核对资产
逐个检查阶段一的候选:这个场景需要哪些物体?对照上面的"可用物体类目",我们的资产**能不能满足**?
- 能满足 → 保留。
- 关键物体缺失、搭不出来 → 剔除,或改造成一个能用现有类目表达的近似情景。
- 同时剔除与"需要避开的已有情景"重复或高度相似的候选。

### 阶段三 · 定稿
从通过阶段二的候选里,挑出 **{n} 个**最真实、彼此差异明显、且资产能搭出来的情景,写成最终的一句话生活叙事。

## 输出要求
阶段一、阶段二可以自由书写你的思考过程。**最后**,把阶段三定稿的 {n} 句叙事放进一个 ```json 代码块,作为一个只含字符串的 JSON 数组。示例格式(请勿照抄内容):
```json
[
  "周末傍晚一个人窝沙发看球赛,啤酒零食摊了一茶几,遥控器就在手边。",
  "妈妈追剧时敷着面膜,茶几上一杯热茶配一小盘水果。"
]
```

===== PROMPT 结束 =====
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="生成情景头脑风暴 prompt")
    ap.add_argument("--direction", required=True, help='大方向,如 "客厅茶几 coffee_table"')
    ap.add_argument("--dedup", default=None, help="需要避开的已有情景:文件路径(.json 数组或每行一条)")
    ap.add_argument("--n", type=int, default=5, help="最终输出几个情景(默认 5)")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="dataset 目录")
    ap.add_argument("--out", default=None, help="写入文件;不给则打印到 stdout")
    args = ap.parse_args()

    prompt = build_prompt(args.direction, dedup=args.dedup, n=args.n, dataset_dir=args.dataset)
    if args.out:
        Path(args.out).write_text(prompt)
        print(f"wrote {args.out} ({len(prompt)} chars)")
    else:
        print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
