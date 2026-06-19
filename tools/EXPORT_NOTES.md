# tidy → organize_it 数据导出：约定与经验

本目录两个脚本：

- `export_to_organize_it.py`：把手搓的 v0 场景（`data/tidy_scene_v0/*.json`）转成 organize_it 的
  `scene.json` + `tabletop_area.json`。
- `render_organize_it_scene.py`：用我们自己的 SAPIEN loader 渲染该场景，产出 pipeline 能直接消费的
  采集文件（RGB / 深度 / 内外参 / GT 分割）。

下面是「怎么做才对」的核心约定，最后附常见易错点。

---

## 1. 坐标系：统一用「桌面中心帧」（桌面顶面 z=0）

**一个场景里所有几何量必须在同一个帧里**：物体位姿、相机外参 `T_world_from_cam`、
`workspace_bounds`、`tabletop_area.json`。这个帧就是：

> 把手搓世界（SAPIEN world，桌面在 XY 原点、顶面 z≈table_height）整体沿 z 平移，使**桌面顶面 = z=0**。

- 纯 z 平移，无旋转；桌子本来就在原点。转换只需要 `table["height"]`，**不依赖 UR 标定**。
- z 朝上，相机轴定义 x前/y左/z上 —— 和 pipeline 的反投影 / 桌面检测约定一致。
- 为什么是 z=0：`layout_optimization.py` 把工作区四角当作 **z=0** 的点，用 `T_world_from_cam` 投影
  到图像；step2 又从点云重新检测桌面 z。所以「桌面在 z=0、各量同帧」是硬要求，绝对原点无所谓。

## 2. 字段契约对齐「真机采集」，不要造别名

输出文件的字段以 `organize_it_v2/test_data/*`（真机采集）为准，而不是 RoboTwin collect 脚本里
`ur_base_as_workspace` 那套带 `sapien_scene_debug` 别名的写法。

- `current_intrinsics.yaml`：`width, height, fov_vertical_deg, fx, fy, cx, cy`（无 `K` / `depth_unit`）。
- `current_extrinsics.yaml`：只有三个 key —— `camera_pose_world {p,q}`、`T_world_from_cam`、`T_cam_from_world`。
  - `T_world_from_cam` = **桌面帧下的相机位姿**（不是 ur_base，也不是 sapien_world）。
- `current_depth.pkl`：`ndarray (H, W, 1) float32`，单位 **毫米**（保留通道维）。
- 下游只认契约里的 key 名。**别在核心 pipeline 加别名识别**——该改的是数据生产方。

## 3. tabletop_area.json 由数据生产方产出

pipeline 读取顺序：先找 `tabletop_area.json`，没有才回退到从 `scene.json` 的
`generation.workspace_bounds_ur_base` 推导。**我们直接产出它**，schema 与 pipeline 推导出来的一致：

```json
{ "frame": "table_centered_z0",
  "min": [x_min, y_min, 0.0], "max": [x_max, y_max, 0.0],
  "size": [dx, dy], "source": ".../scene.json" }
```

> 注：`scene.json` 里仍叫 `generation.workspace_bounds_ur_base`（pipeline 写死的 key 名），
> 但里面装的是桌面帧的值——pipeline 本就把这个 key 当作 table/world 帧解释。

## 4. 位姿转换：stable 帧 → raw mesh 世界位姿

v0 的 `transform` 是物体在**稳定帧（stable）**下的位姿；organize_it 的 loader 直接设的是
**raw mesh** 实体位姿。所以：

```
T_world_from_raw = T_world_from_stable @ stable_rotation     # stable_rotation 来自资产记录
pos/quat(wxyz)   = 平移 / 旋转   （再按第 1 节平移到桌面帧）
```

很多资产 `stable_rotation` 非单位阵，这步不能省。可用「反算回 v0 transform」做往返校验。

## 5. 场景构建用我们自己的 loader，采集/GT 复用参考

- **建场景（含桌面纹理、PBR 材质、物体）用 `simulations/scene.py`**
  （`create_scene(table_texture_id=...)` + `load_items`）。它原生支持纹理和稳定帧位姿，
  不要去给 organize_it 的渲染器打补丁补纹理。
- 相机位姿换算、`capture_camera`、GT 分割写出（`save_streamline_gt_seg_outputs` →
  `current_pybullet_segmentation.npy` + `extract_meta.json`）复用 collect 脚本的函数，保证格式一致。
- 渲染时 `settle_steps=0`，保留手搓摆放；墙默认关闭（已是默认）。

## 6. 资产 ID 两库共享

我们的 `AssetLibrary` 和 organize_it 的 `AssetRegistry` 读同一个
`organize_it_v2/data/asset_library`，`asset_id` 直接通用，**无需重映射**。

---

## 常见易错点（速查）

- **帧不一致**：`workspace_bounds` / `tabletop_area` 在桌面帧（z=0），但外参或物体位姿留在 sapien_world
  （桌面 z=0.74）→ step6 工作区投影全错。务必同帧、桌面 z=0。
- **造字段别名**：写 `T_sapien_world_from_cam` 之类下游不认。用契约名 `T_world_from_cam`，值放桌面帧位姿。
- **深度丢通道**：契约是 `(H,W,1) float32 mm`，别 squeeze 成 `(H,W)`。
- **给渲染器打补丁**：纹理用我们 loader，不要在 organize_it 渲染链里硬塞材质。
- **忘了 stable_rotation**：直接拿 v0 transform 当 raw 位姿会让有非单位 stable_rotation 的物体姿态错。
- **漏产 tabletop_area.json**：数据生产方的责任，缺了下游 Step6 直接报错。

## 校验清单

1. `tabletop_area.json` 的 min/max 是桌面帧、桌面在 z=0。
2. 物体 `pos.z` ≈ 略大于 0（贴桌面）。
3. 用 pipeline 同款投影（`CV_FROM_PROJECT_CAM @ inv(T_world_from_cam)` + 内参）把工作区四角和物体中心
   投到图像：四角在相机前方、物体落点与 RGB 渲染位置一致。
