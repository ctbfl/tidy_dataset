# VLM Layout Constraint Guide

本文档面向一个从目标图推理可执行 layout constraint 的 VLM。

VLM 看到的是一张目标桌面图，以及一组固定物体资产。图中物体用数字 id 标注；物体列表里同时给出
数字 id 和英文 id。VLM 的任务是把目标图中的几何布局推理成一组物体 id 级别的 constraint，用于
后续执行和验证。

这不是离线标注任务。所有几何关系都只绑定到输入物体列表中的 `object_id`。

## 输入

VLM 会收到：

1. 一张目标桌面图，图中可见物体带数字 id。
2. 一份物体字典。key 是图上的数字标号，value 是输出 constraint 时使用的稳定 `object_id`。
   系统也可能额外提供物体描述，例如类别、尺寸、形状、可抓取信息。
3. 有些物体可能在图中不明显，甚至没有直接标出。它们仍然属于任务的一部分，需要 VLM 根据场景关系推理合理位置。

示例物体字典：

```json
{
  "1": "dinner_plate",
  "2": "fork",
  "3": "knife",
  "4": "cup"
}
```

数字 id 只用于看图定位。最终 constraint 必须使用 `object_id`。

## 输出格式

只输出一个 JSON object，不要输出解释文字、Markdown、代码块或注释。

JSON 顶层必须只有 `constraints` 字段：


```json
{
  "constraints": [
    {
      "type": "table_xy",
      "object": "dinner_plate",
      "x": 0.0,
      "y": 0.0
    },
    {
      "type": "xy_offset_from",
      "object": "cup",
      "anchor": "dinner_plate",
      "dx": 0.25,
      "dy": 0.2
    }
  ]
}
```

每个 constraint 都描述一个可执行的几何关系。所有物体 id 必须来自输入物体列表中的
`object_id`，不要使用图上的数字 id 作为输出 id。

约束图需要满足：

- 每个物体都能得到桌面平面上的位置。
- 每个物体都能得到 `rotation` 状态；如果朝向不重要，使用 `align_axis` 的 `"any"`。
- 优先提取结构化 layout prior，而不是只让字段变成可执行状态。
- 看不清或无法关系化的对象使用绝对位置锚定。
- 不要创造物体，不要漏掉输入物体。

## 坐标和字段

每个物体最终需要这些执行字段：

- `x`: 桌面归一化 x 坐标。桌面中心是 `0`，左侧为负，右侧为正。
- `y`: 桌面归一化 y 坐标。桌面中心是 `0`，前侧/下侧为负，后侧/上侧为正。
- `rotation`: 桌面法线方向的朝向状态。

`x/y` 约在 `[-1, 1]` 范围内。边缘附近接近 `-1` 或 `1`，中心附近接近 `0`。

VLM 不需要估计毫米级精度。核心是构建稳定、可执行、可验证的几何关系。

不要直接输出 `rotation` 数值字段。朝向必须通过 `align_axis` relation 定义。

## 单赋值规则

每个字段只能由一个 constraint 定义。

错误示例：

```json
[
  {"type": "table_xy", "object": "cup", "x": 0.2, "y": 0.1},
  {"type": "xy_offset_from", "object": "cup", "anchor": "plate", "dx": 0.3, "dy": 0.0}
]
```

这里 `cup.x` 和 `cup.y` 被定义了两次。

正确做法是二选一：

```json
[
  {"type": "xy_offset_from", "object": "cup", "anchor": "plate", "dx": 0.3, "dy": 0.0}
]
```

或者如果只想共享一条轴：

```json
[
  {"type": "in_same_vertical_line", "objects": ["plate", "cup"], "anchor": "plate"},
  {"type": "table_y", "object": "cup", "y": 0.25}
]
```

## 依赖顺序

constraint 按顺序执行。读取 anchor 的 relation 必须放在 anchor 已经被定义之后。

推荐顺序：

1. 用 `table_xy` 固定少量基准物体。
2. 用 `align_axis` 定义明显朝向。
3. 优先用 same-line、spacing、support/special relation 表达结构化 prior。
4. 再用 offset relation 补足 still undefined 的轴。
5. 对剩余未定义物体补绝对位置和朝向。

## 构建策略

VLM 应先识别结构，再补全细节：

1. 找出图中主要基准物体，例如盘子、键盘、托盘、碗、显示器。
2. 用 `table_xy` 给这些基准物体锚定大致位置。
3. 找出明显结构关系，按优先级选择 relation：
   - 同一列或同一行。
   - 左右/前后偏移。
   - 等距排列。
   - 堆叠。
   - 插入容器。
4. 对图中没明显出现但在物体列表中的对象，按场景常识补齐：
   - 餐具通常在盘子两侧。
   - 杯子通常在盘子右上或附近。
   - 笔通常在笔筒内、笔记本旁，或和其它笔成组。
   - 鼠标通常在键盘右侧或惯用手侧。
5. 对无法确定关系的对象，用 `table_xy` 放到合理空位。
6. 给所有物体补朝向。圆形物体或朝向不重要的物体使用 `align_axis` + `"any"`。

不要为了复杂而输出关系。关系必须对执行有帮助，且能由目标图或稳定常识支持。

### Relation 选择优先级

如果多个物体形成清晰结构，优先使用更结构化的 relation：

1. 同一列：使用 `in_same_vertical_line` 定义共享 `x`。
2. 同一行：使用 `in_same_horizontal_line` 定义共享 `y`。
3. 一排或一列物体间隔相近：使用 `evenly_spaced_from_anchor` 定义该轴。
4. 上下叠放：使用 `on_top_of`，必要时再加 `xy_offset_from` 让中心重合。
5. 放入容器：使用 `in_holder`。
6. 只有两个物体之间有独立二维相对位置时，才使用 `xy_offset_from`。
7. 只需要定义一个轴时，使用 `x_offset_from`、`y_offset_from`、`table_x` 或 `table_y`，不要用
   `xy_offset_from` 同时写两个轴。

不要把一排物体写成连续 `xy_offset_from` chain。如果它们同一条横线，应该用
`in_same_horizontal_line`；如果它们同一条竖线，应该用 `in_same_vertical_line`；如果间距也近似
一致，再加 `evenly_spaced_from_anchor`。

## 可用 Constraint

只允许使用本节列出的 constraint type 和字段名。

不要输出额外字段。不要把 `x/y/rotation` 直接写到普通 object 上；它们必须由 relation 定义。

### table_x

定义单个物体的绝对 `x`。

```json
{
  "type": "table_x",
  "object": "cup",
  "x": 0.3
}
```

用于只需要绝对横向位置、纵向位置由其它关系定义的情况。

### table_y

定义单个物体的绝对 `y`。

```json
{
  "type": "table_y",
  "object": "cup",
  "y": 0.2
}
```

### table_xy

定义单个物体的绝对 `x/y`。

```json
{
  "type": "table_xy",
  "object": "dinner_plate",
  "x": -0.2,
  "y": 0.1
}
```

用于基准物体、孤立物体、或无法从关系推理的物体。

### align_axis

定义单个物体的 `rotation`。

```json
{
  "type": "align_axis",
  "object": "fork",
  "axis": "vertical"
}
```

支持：

- `"horizontal"`: 物体主轴水平。
- `"vertical"`: 物体主轴竖直。
- `"any"`: 朝向不重要。
- `"custom"`: 明确既不是水平也不是竖直的斜向朝向。

VLM 不需要提供具体角度数值。`custom` 的具体角度由后续 pose detection / execution 模块估计。
只有在图中明确能看出物体是斜放，且不能归入 `horizontal` 或 `vertical` 时，才使用 `custom`。

### in_same_vertical_line

让多个物体共享同一个 `x`。读取 `anchor.x`，写其它物体的 `x`。

```json
{
  "type": "in_same_vertical_line",
  "objects": ["plate", "cup", "bowl"],
  "anchor": "plate"
}
```

适合中心对齐、同列摆放。

如果多个物体的中心在图中明显上下对齐，应使用这个 relation，而不是给每个物体写独立的
`xy_offset_from`。

### in_same_horizontal_line

让多个物体共享同一个 `y`。读取 `anchor.y`，写其它物体的 `y`。

```json
{
  "type": "in_same_horizontal_line",
  "objects": ["keyboard", "mouse"],
  "anchor": "keyboard"
}
```

适合同排摆放。

如果多个物体的中心在图中明显左右排成一行，应使用这个 relation，而不是给每个物体写独立的
`xy_offset_from`。

### evenly_spaced_from_anchor

沿某个轴等距排列多个物体。

```json
{
  "type": "evenly_spaced_from_anchor",
  "objects": ["cup_left", "plate", "cup_right"],
  "anchor": "plate",
  "axis": "x",
  "mode": "footprint",
  "order": ["cup_left", "plate", "cup_right"],
  "spacing": 0.08
}
```

参数：

- `objects`: 参与排列的物体。
- `anchor`: 基准物体，必须在 `objects` 内。
- `axis`: `"x"` 或 `"y"`。
- `mode`: `"obj_center"` 或 `"footprint"`。
- `order`: 沿该轴从小到大的物体顺序。
- `spacing`: 间距。

`obj_center` 表示中心点间距。`footprint` 表示外轮廓边缘间距，更适合避免碰撞。

如果一排餐具、杯子或盘子间隔近似一致，优先使用该 relation 表达排列 prior。它只定义一个轴；
另一轴通常用 `in_same_horizontal_line` / `in_same_vertical_line` 或 `table_x/table_y` 补足。

### x_offset_from

定义 `object.x = anchor.x + dx`。

等价地：`dx = object.x - anchor.x`。正数表示 object 在 anchor 右侧，负数表示 object 在 anchor 左侧。

```json
{
  "type": "x_offset_from",
  "object": "knife",
  "anchor": "plate",
  "dx": 0.25
}
```

### y_offset_from

定义 `object.y = anchor.y + dy`。

等价地：`dy = object.y - anchor.y`。正数表示 object 在 anchor 后侧/上侧，负数表示 object 在
anchor 前侧/下侧。

```json
{
  "type": "y_offset_from",
  "object": "cup",
  "anchor": "plate",
  "dy": 0.2
}
```

### xy_offset_from

定义：

- `object.x = anchor.x + dx`
- `object.y = anchor.y + dy`

等价地：

- `dx = object.x - anchor.x`
- `dy = object.y - anchor.y`

```json
{
  "type": "xy_offset_from",
  "object": "cup",
  "anchor": "plate",
  "dx": 0.25,
  "dy": 0.2
}
```

适合表达“在某物右上方”“在某物旁边”“和某物保持固定相对位置”。

不要用 `xy_offset_from` 替代明显的同线或等距结构。比如 5 把餐具排成一行时，应先用
`in_same_horizontal_line` 定义共享 `y`，再用 `evenly_spaced_from_anchor` 或若干 `x_offset_from`
定义 `x`。

### on_top_of

表达上下支撑关系。`anchor` 是下方物体，`object` 是上方物体。

```json
{
  "type": "on_top_of",
  "object": "bowl",
  "anchor": "plate"
}
```

`on_top_of` 只表达上下关系，不定义 `x/y/rotation`。如果两个物体应同中心叠放，需要再加：

```json
{
  "type": "xy_offset_from",
  "object": "bowl",
  "anchor": "plate",
  "dx": 0,
  "dy": 0
}
```

### in_holder

表达一个物体插入或放入容器。

```json
{
  "type": "in_holder",
  "object": "pen",
  "holder": "pen_holder"
}
```

适合笔在笔筒中、勺子在杯中等容器关系。通常还需要 holder 本身已经有位置。

## 缺失物体补齐

如果物体列表中有对象在图中不明显，VLM 仍需输出它的约束。

补齐优先级：

1. 如果它属于某个明显组合，把它放入组合关系中。例如第二把餐具、另一支笔。
2. 如果它有自然容器，用 `in_holder`。
3. 如果它与某个物体强相关，用 `xy_offset_from` 放在旁边。
4. 如果没有明显关系，用 `table_xy` 放到桌面空位。
5. 朝向不确定时用 `align_axis` 的 `"any"`。

补齐时应保守。不要让补齐物体破坏图中已明确的结构。

## Validator Feedback 修正

VLM 应根据验证反馈迭代修改 constraint。

validator 会返回每个物体当前字段状态：

```json
{
  "objects": {
    "plate": {"x": "ok", "y": "ok", "rotation": "undefined"},
    "cup": {"x": "ok", "y": "undefined", "rotation": "undefined"}
  }
}
```

`ok` 表示字段已经被某个 relation 定义。`undefined` 表示字段还没有定义。

常见修正：

- 某物体 `x` 是 `undefined`: 添加 `table_x`，或加入写 x 的 relation。
- 某物体 `y` 是 `undefined`: 添加 `table_y`，或加入写 y 的 relation。
- 某物体 `rotation` 是 `undefined`: 添加 `align_axis`；不确定时用 `"any"`。
- `over_defined`: 同一字段被多个 constraint 定义。删除重复关系，或把 `table_xy` 拆成
  `table_x` / `table_y`。
- `anchor_field_missing`: anchor 的字段还没定义。先定义 anchor，或调整 constraint 顺序。
- `unknown_object_id`: 使用了物体列表外的 id。改回输入中的 `object_id`。
- `object not constrained`: 输入物体没有任何可执行位置。补位置 constraint。

修正原则：

- 优先局部修改，不要重写所有 constraint。
- 保留图中最明确的关系。
- 对不确定物体补简单、可执行的约束。
- 最终每个输入物体都必须有可执行位置。

## 完整输出示例

输入物体：

```json
{
  "1": "plate",
  "2": "fork",
  "3": "knife",
  "4": "cup"
}
```

目标图关系：盘子在中心偏左，叉子在盘子左侧，刀在盘子右侧，杯子在盘子右上。

有效输出：

```json
{
  "constraints": [
    {"type": "table_xy", "object": "plate", "x": -0.1, "y": 0.0},
    {"type": "align_axis", "object": "plate", "axis": "any"},

    {"type": "xy_offset_from", "object": "fork", "anchor": "plate", "dx": -0.25, "dy": 0.0},
    {"type": "align_axis", "object": "fork", "axis": "vertical"},

    {"type": "xy_offset_from", "object": "knife", "anchor": "plate", "dx": 0.25, "dy": 0.0},
    {"type": "align_axis", "object": "knife", "axis": "vertical"},

    {"type": "xy_offset_from", "object": "cup", "anchor": "plate", "dx": 0.25, "dy": 0.25},
    {"type": "align_axis", "object": "cup", "axis": "any"}
  ]
}
```

同线结构示例：三个餐具竖直放置并横向排成一行。

```json
{
  "constraints": [
    {"type": "table_xy", "object": "fork", "x": 0.2, "y": -0.1},
    {"type": "align_axis", "object": "fork", "axis": "vertical"},

    {"type": "in_same_horizontal_line", "objects": ["fork", "spoon", "knife"], "anchor": "fork"},
    {
      "type": "evenly_spaced_from_anchor",
      "objects": ["fork", "spoon", "knife"],
      "anchor": "fork",
      "axis": "x",
      "mode": "footprint",
      "order": ["fork", "spoon", "knife"],
      "spacing": 0.06
    },

    {"type": "align_axis", "object": "spoon", "axis": "vertical"},
    {"type": "align_axis", "object": "knife", "axis": "vertical"}
  ]
}
```

这个例子不要写成：

```json
{
  "constraints": [
    {"type": "table_xy", "object": "fork", "x": 0.2, "y": -0.1},
    {"type": "xy_offset_from", "object": "spoon", "anchor": "fork", "dx": 0.06, "dy": 0.0},
    {"type": "xy_offset_from", "object": "knife", "anchor": "spoon", "dx": 0.06, "dy": 0.0}
  ]
}
```
