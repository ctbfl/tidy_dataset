# scenarios

按 scenario 组织的场景数据，**每个 scene 一个文件夹**，文件夹里每个 `<arrangement>.json`
是同一组物体的一种摆放：

```
<scenario>/<scene>/<arrangement>.json     例: office_desk/001/tidy.json
                                               office_desk/001/messy.json   ← 之后由脚本生成
```

- 同一个 scene 文件夹下，`tidy` 与 `messy`（以及将来可能的 `messy_2`…）**共享同一份
  `manifest`**（同一组物体），只是 `items`（摆放）不同。arrangement 名 = 文件名 stem，新增
  变体不需要改代码。
- `tidy.json` 由 `simulations/gen_scenes.py` 从 `templates/<id>.json` 生成（manifest 已填、
  items 空），再在 handcraft 里手工标注摆放；`messy.json` 之后用专门的脚本随机生成。

## 场景文件格式 v2（向后兼容 v1）

```jsonc
{
  "version": 2,
  "scenario": "office_desk",
  "scene_id": "001",
  "arrangement": "tidy",                         // == 文件名 (tidy / messy / ...)
  "template": "office_desk",
  "table": { "length": 1.2, "width": 0.7, "height": 0.74, "thickness": 0.05 },
  "table_texture": null,
  "wall_texture": null,
  "manifest": [                                  // 要放什么（生成时写好，标注不改，tidy/messy 一致）
    { "slot": "laptop-1", "role": "laptop", "asset_id": "robotwin:glb:015_laptop:5" }
  ],
  "items": [                                     // 放在哪（该 arrangement 的摆放）
    { "slot": "laptop-1", "asset_id": "robotwin:glb:015_laptop:5", "transform": [[...4x4...]] }
    // slot=null 的 item 是清单外的额外物体 (extra)
  ]
}
```

- `manifest` 与 `items` 通过 `slot` 关联；标注器据此在清单卡片上打 ✅。
- 下游 `simulations/open_scene` / `tools/export_to_organize_it.py` 只读 `table` + `items`
  的 `asset_id`/`transform`，忽略 v2 新增字段，**v1 旧场景仍可直接加载**。

## 标注流程

```bash
/home/hjs/miniforge3/envs/RoboTwin/bin/python handcraft/server.py   # http://127.0.0.1:8099
```

右上选 scenario / scene / arrangement → 右侧清单显示该 scene 需要摆的物体（小预览图）→ 把卡片
拖到桌面，放好的物体卡片右上出现绿色 ✅ → WASD 移动 / QE 旋转微调 → Save 写回该 arrangement
文件的 `items`。清单不够用时，可展开「Add extra object」从全资产库补放（记为无 slot 的 extra）。
