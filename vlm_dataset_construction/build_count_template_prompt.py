"""第三层 prompt:把一条叙事 (variation) 落成"物体数量清单" count_template。

给定 scenario 大方向 + 一条具体叙事 + 可用物体类目,让模型判断:这一刻这张桌面上
**会出现哪几类物体、每类几个**,只用库里真实存在的类目 key 作答。

与第二层的差别:这里类目清单是**输出词表**(必须从中选 key),不再只是想象边界。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_scenario_prompt import load_categories, format_categories, DEFAULT_DATASET_DIR  # noqa: E402


def build_count_template_prompt(
    direction: str,
    narrative: str,
    *,
    k: int = 8,
    n_min: int = 4,
    n_max: int = 9,
    total_min: int = 5,
    total_max: int = 12,
    categories: dict[str, str] | None = None,
    dataset_dir=DEFAULT_DATASET_DIR,
) -> str:
    """组装并返回 count_template prompt。

    direction: scenario 大方向(上下文),如 "客厅茶几 coffee_table"。
    narrative: 具体这条叙事。
    k:         一次产出多少份互不相同的清单(默认 8)。
    n_min/n_max:   每份清单里**不同类目数**的建议区间。
    total_min/total_max: 每份清单里**物体总件数**(各 count 之和)的建议区间。
    """
    if not narrative or not narrative.strip():
        raise ValueError("narrative 不能为空")
    cats = categories if categories is not None else load_categories(dataset_dir)
    cat_block = format_categories(cats)

    return f"""# 叙事 → 物体数量清单 (count_template) ×{k}

> 用途:把一条生活叙事,落成这张桌面上**会出现哪几类物体、每类几个**的具体清单(不含摆放位置)。产物用于后续搭建桌面整理场景。
> 使用方法:把 `===== PROMPT 开始 =====` 到 `===== PROMPT 结束 =====` 之间的内容整段发给模型。可用物体类目已从数据集现场解析嵌入(共 {len(cats)} 类)。

===== PROMPT 开始 =====

## 你的角色
你是一位布景师。给你一个生活瞬间,你要产出两样东西:①此刻这张桌面上真实会摆着哪些东西、各有几件(`objects`);②这个人此刻会对整理机器人说的一句**需求语句**(`instructions`)。这些桌面之后会交给整理机器人去收拾。

## 场景大方向
{direction}

## 本条叙事(就为这一句布景)
「{narrative}」

## 可用物体类目(**只能从这些 key 里选**,category key: 描述)
{cat_block}

## 规则(每一份清单都要满足)
1. **只能输出上面清单里存在的 category key**,一个都不能虚构或改写。
2. 只放**这条叙事此刻真实会出现**的物体:每一件都应能从叙事里找到理由(正在用、刚用完、随手放着)。不要为了凑数塞无关物品。
3. **数量要真实**:值是该类目在桌面上的件数(如"两罐饮料"→ `"can_drink": 2`)。绝大多数类目 1–2 件即可,只有成组出现的(如零食、餐具)才可能更多。
4. **规模约束**:每份清单出现 **{n_min}–{n_max} 种**不同类目,物体**总件数(各数量之和)控制在 {total_min}–{total_max} 件**。别堆成杂货铺。
5. 若叙事需要某个关键物体、但类目清单里确实没有,就用一个能表达同样意思的近似类目替代;不要输出不存在的 key。

## 关于 `instructions`(用户需求语句)——重要
每份清单都配一句 `instructions`:模拟这个人此刻会对整理机器人说的话,第一人称、口语化。它表达的是**用户的需求和情景**,不是给机器人的摆放说明。
- ✅ 可以说:他想干什么(如"我想看世界杯")、以及**定义情景所必需的基础信息**——比如就坐位置("我陪孩子写作业,我坐左边、孩子坐右边")、或高层意图("帮我把面前空出一大块,我要摊开复习")。
- ❌ 绝不能包含任何**具体整理/摆放指令**:不要指定某个物体该放哪("把遥控器放右边、饮料放左边"),不要教机器人怎么整理("按大小排好、摆整齐")。摆放是机器人的活,不是用户说的。
- 简洁即可。基础情景比如看球就一句"我想看会儿球赛"就够;只有当情景确实需要(如双人、要空出某区域)才补一句基础空间/就坐信息。

## 任务:产出 {k} 份**互不相同**的桌面
针对**同一条叙事**,给出 {k} 份合理但**彼此有差异**的桌面:它们都符合这条叙事,但各自的物品取舍/数量不同——有人桌上东西多、有人少,有人多摆了个杯垫、有人把零食换成果盘。{k} 份之间不要几乎雷同,要覆盖这条叙事下真实可能的多种桌面样子。每份桌面各自配一句贴合它的 `instructions`。

## 工作流(先想后答)
1. 先还原画面:这个人此刻在干什么?手边、桌上一定会有什么(必选物)?哪些是可有可无的(可选物)?这个人会怎么对机器人说出自己的需求?
2. 把桌面物体对应到 category key。
3. 通过"必选物 + 不同的可选物组合 / 不同件数",构造出 {k} 份互不相同、且都满足上面规则的桌面,并为每份写好 `instructions`。

## 输出要求
可以先简短写出你的思考。**最后**输出一个 ```json 代码块,是一个**长度为 {k} 的数组**;数组每个元素是一份桌面,形如:
`{{"instructions": "<用户需求语句>", "objects": {{"category_key": 数量, ...}}}}`
其中 `objects` 用 **`{{key: 数量}}` 字典**表示(键是 category key,值是正整数件数,**不要**重复写 "category"/"count" 字段)。示例格式(此处只示意 2 份,你要给满 {k} 份;请勿照抄内容):
```json
[
  {{"instructions": "我想一个人窝着看会儿球赛。", "objects": {{"can_drink": 2, "snack_box": 1, "tv_remote": 1, "coaster": 1}}}},
  {{"instructions": "看球呢,给我把面前腾出点地方放吃的。", "objects": {{"chip_can": 1, "can_drink": 1, "tv_remote": 1, "phone": 1}}}}
]
```

===== PROMPT 结束 =====
"""
