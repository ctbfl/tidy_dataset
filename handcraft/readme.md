# handcraft

手工构建整齐桌面数据的交互标注器（FastAPI + SAPIEN + WebSocket）。

```bash
/home/hjs/miniforge3/envs/RoboTwin/bin/python handcraft/server.py   # http://127.0.0.1:8099
```

- **scenario 标注模式**：右上选 scenario / scene，右侧按该 scene 的 manifest 显示待摆物体清单
  （小预览图）；拖卡片到桌面摆放，放好的卡片打 ✅；Save 写回 `data/scenarios/<scenario>/<NNN>.json`
  的 `items`（保留 manifest）。先用 `simulations/gen_scenes.py` 生成 scene，见 `templates/readme.md`。
- **额外物体**：展开右侧「Add extra object」从全资产库补放（记为无 slot 的 extra）。
- **背景**：可选桌面/墙体纹理或随机化。
- **操作**：拖资产落桌 · 点击选中 · WASD 平移 · Q/E 偏航 · Clear 清空已摆物体（保留清单）。

文件：`server.py`（HTTP/WS 路由）· `editor.py`（场景编辑 + slot 记账 + v1/v2 读写）·
`preview.py`（缩略图缓存）· `index.html`（前端）。

## 资产尺寸标定（sizing studio）

把尺寸不对劲的资产**等比**缩放回合理大小的小工具。

```bash
/home/hjs/miniforge3/envs/RoboTwin/bin/python handcraft/sizing_server.py   # http://127.0.0.1:8100
```

- 左侧实时视图：目标物摆在 5/10cm 刻度网格地板上，旁边一排三个**已标定参照物**（小→大），
  画面左上角实时显示当前 `W×D×H`（cm）。
- 右侧：搜索/缩略图选物（◀▶ 在结果里快速翻）→ 载入为目标物；用滑块/factor 数字目测对齐参照物，
  或「set 某轴 to N cm」按绝对尺寸一键标定；references 三槽可改成任意 asset + 真实 cm。
- **Save**：把 `geometry.scale` 与 `geometry.aabb_m` 同乘缩放因子，**原地写回**共享资产库
  `organize_it_v2/data/asset_library/.../asset.json`（无备份）。等比缩放下这两者皆精确，不重跑 mesh/pybullet。
  不改 contacts / mass。

文件：`sizing_server.py`（HTTP/WS）· `sizing.py`（`SizingStudio`：网格地板 + 参照物 + 目标物 + 存盘，
复用 `objects.spawn` 加载器）· `sizing.html`（前端）。
