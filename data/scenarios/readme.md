# scenarios

按 scenario 组织的场景数据：`<scenario>/<NNN>.json`（如 `office_desk/001.json`）。
每个文件由 `simulations/gen_scenes.py` 从 `templates/<id>.json` 生成，再在 handcraft 里手工标注摆放。

## 场景文件格式 v2（向后兼容 v1）

```jsonc
{
  "version": 2,
  "scenario": "office_desk",
  "scene_id": "001",
  "template": "office_desk",
  "table": { "length": 1.2, "width": 0.7, "height": 0.74, "thickness": 0.05 },
  "table_texture": null,
  "wall_texture": null,
  "manifest": [                                  // 要放什么（生成时写好，标注不改）
    { "slot": "laptop-1", "role": "laptop", "asset_id": "robotwin:glb:015_laptop:5" }
  ],
  "items": [                                     // 放在哪（标注时填）
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

右上选 scenario / scene → 右侧清单显示该 scene 需要摆的物体（小预览图）→ 把卡片拖到桌面，
放好的物体卡片右上出现绿色 ✅ → WASD 移动 / QE 旋转微调 → Save 写回该 scene 文件的 `items`。
清单不够用时，可展开「Add extra object」从全资产库补放（记为无 slot 的 extra）。
