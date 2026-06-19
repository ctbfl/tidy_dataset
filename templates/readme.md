# 场景模版 (templates)

每个 `<id>.json` 是一个桌面**模版**：声明这张桌子上有哪些**角色 (role)**、每个角色
摆几个、候选资产从哪来。生成脚本据此为每个 scene 采样出一份 manifest（清单）。

## 字段

```json
{
  "template_id": "office_desk",
  "scenario": "office_desk",          // 生成时写到 data/scenarios/<scenario>/，默认 = template_id
  "name": "办公学习桌",
  "table": { "length": 1.2, "width": 0.7, "height": 0.74, "thickness": 0.05 },
  "roles": [
    { "role": "laptop", "count": 1,      "tags": ["laptop"] },
    { "role": "pen",    "count": [1, 2], "tags": ["pen"] },
    { "role": "mouse",  "count": 1,      "asset_ids": ["robotwin:glb:047_mouse:2"] }
  ]
}
```

- **role**：角色名，slot 命名为 `<role>-1`、`<role>-2`…
- **count**：`1` = 固定；`[min,max]` = 每个 scene 在闭区间内随机。
- **候选池** = 命中 `tags` 里任一标签的资产 ∪ `tags` 之外显式列出的 `asset_ids`，去重。
  两者可只给其一，也可并用（tags 自动拉、asset_ids 精选补充/覆盖）。
- 采样在候选池内**不重复**抽 count 个；count 超过池大小时取整池。池为空或抽到 0 个则该角色不出现。

> 标签词表见 `organize_it_v2/data/asset_library/catalog.json` 的 `object_tags`。
> 注意部分类目很薄（如 `mouse` 仅 1 个、`keyboard` 4 个），多样性受限。

## 生成 scene

```bash
/home/hjs/miniforge3/envs/RoboTwin/bin/python simulations/gen_scenes.py --template office_desk --n 20
# --start N 起始编号  --seed S 复现采样  --overwrite 覆盖已存在文件
```

输出到 `data/scenarios/<scenario>/001.json …`，每个文件 `manifest` 已填、`items` 为空，
等待在 handcraft 里手工摆放标注。
