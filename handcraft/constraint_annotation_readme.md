# Constraint Annotation Design

This tool defines ordered layout templates for one variation. A variation may
have multiple constraint files:

```text
data/organize_it_dataset_v2/<scenario>/<variation>/template/
  available_assets.json
  constraints/
    two_people_left_stack.json
    three_people_front_stack.json
```

Each constraint file references categories from `available_assets.json`. The
annotator adds category sets, and each concrete object is addressed by:

```json
{ "category": "cutlery_a", "set": 0, "slot": 2 }
```

## Object State

Every concrete object must end with:

```text
asset/category defined
x defined
y defined
rotation defined
```

`x` and `y` are table-normalized coordinates in `[-1, 1]`, where table center is
`(0, 0)`. The field refers to the object's actual mesh center on the table plane,
not necessarily the actor origin.

Fields are single-assignment. A constraint must fail if it tries to define a
field that is already defined. There is no over-definition and no implicit
fallback.

## Constraint Execution

Constraints are interpreted in file order.

Each relation declares:

```text
requires: fields that must already be defined
writes: fields it defines
```

If a required field is missing, the relation is invalid. If a written field is
already defined, the relation is invalid. Invalid relations are shown in red in
the annotation UI.

New multi-object relations store the participating object set in `objects`.
Configurable roles such as `anchor` or `holder` are parameters, not part of
relation creation. A relation whose configurable parameters are not filled is
shown in blue and does not write any field until it is complete.

Load and Reapply use deterministic values only. Randomize resamples each object
set and applies jitter, so it can be used to test template stability.

New creates an empty constraint file for the current variation, clears the scene,
and selects the new template name in the UI.

## Relations

### table_x

Single-object relation. Defines `target.x`.

```json
{ "type": "table_x", "target": {"category": "plate", "set": 0, "slot": 0}, "x": -0.4 }
```

`x` may be a number or a `[min, max]` range. Preview uses the midpoint.
`x_jitter` defaults to `0` and is sampled uniformly from
`[-x_jitter, x_jitter]` only during Randomize.

### table_y

Single-object relation. Defines `target.y`.

```json
{ "type": "table_y", "target": {"category": "plate", "set": 0, "slot": 0}, "y": 0.2 }
```

`y` may be a number or a `[min, max]` range. Preview uses the midpoint.
`y_jitter` defaults to `0` and is sampled uniformly from
`[-y_jitter, y_jitter]` only during Randomize.

### table_xy

Single-object relation. Defines `target.x` and `target.y`.

```json
{
  "type": "table_xy",
  "target": {"category": "plate", "set": 0, "slot": 0},
  "x": -0.4,
  "y": 0.2,
  "x_jitter": 0,
  "y_jitter": 0
}
```

### align_axis

Single-object relation. Defines `target.rotation`.

```json
{
  "type": "align_axis",
  "target": {"category": "cutlery_a", "set": 0, "slot": 1},
  "axis": "vertical",
  "jitter_deg": 5
}
```

Supported axes:

```text
horizontal
vertical
any
```

### in_same_vertical_line

Multi-object relation. Reads `anchor.x`, writes each target's `x`.

```json
{
  "type": "in_same_vertical_line",
  "objects": [
    {"category": "plate", "set": 0, "slot": 0},
    {"category": "cutlery_a", "set": 0, "slot": 0},
    {"category": "cutlery_a", "set": 0, "slot": 1}
  ],
  "anchor": {"category": "plate", "set": 0, "slot": 0},
}
```

All non-anchor objects in `objects` are targets.

### in_same_horizontal_line

Multi-object relation. Reads `anchor.y`, writes each target's `y`.

```json
{
  "type": "in_same_horizontal_line",
  "objects": [
    {"category": "plate", "set": 0, "slot": 0},
    {"category": "cup", "set": 0, "slot": 0}
  ],
  "anchor": {"category": "plate", "set": 0, "slot": 0},
}
```

### evenly_spaced_from_anchor

Multi-object relation. Reads one anchor coordinate and writes the same coordinate
for targets in order.

```json
{
  "type": "evenly_spaced_from_anchor",
  "objects": [
    {"category": "cup", "set": 0, "slot": 0},
    {"category": "cup", "set": 1, "slot": 0},
    {"category": "cup", "set": 2, "slot": 0}
  ],
  "anchor": {"category": "cup", "set": 0, "slot": 0},
  "axis": "x",
  "mode": "obj_center",
  "order": "abc",
  "spacing": 0.12,
  "spacing_jitter": 0
}
```

Each object in `objects` gets a local id from `a` to `z` by list position.
`order` stores the stable layout order with those local ids. The UI initializes
`axis` and `order` from the current object centers when the relation is created;
later edits reuse the stored order instead of recomputing it from scene position.

For `axis = x` and `order = abc`:

```text
b.x = a.x + 1 * spacing
c.x = a.x + 2 * spacing
```

For `axis = y`, the same rule applies to `y`. `mode = obj_center` spaces object
centers. `mode = footprint` spaces object XY bounding boxes. `spacing_jitter`
defaults to `0` and is sampled uniformly from `[-spacing_jitter,
spacing_jitter]` only during Randomize. `spacing + sampled_jitter` must be
non-negative.

### x_offset_from

Two-object relation. Reads `anchor.x`, writes `target.x`.

```json
{
  "type": "x_offset_from",
  "objects": [
    {"category": "plate", "set": 0, "slot": 0},
    {"category": "cup", "set": 0, "slot": 0}
  ],
  "anchor": {"category": "plate", "set": 0, "slot": 0},
  "dx": 0.18,
  "dx_jitter": 0
}
```

The non-anchor object in `objects` is the target.
`dx_jitter` defaults to `0` and is sampled uniformly from
`[-dx_jitter, dx_jitter]` only during Randomize.

### y_offset_from

Two-object relation. Reads `anchor.y`, writes `target.y`.

```json
{
  "type": "y_offset_from",
  "objects": [
    {"category": "plate", "set": 0, "slot": 0},
    {"category": "cup", "set": 0, "slot": 0}
  ],
  "anchor": {"category": "plate", "set": 0, "slot": 0},
  "dy": -0.12,
  "dy_jitter": 0
}
```

### xy_offset_from

Two-object relation. Reads `anchor.x` and `anchor.y`, writes `target.x` and
`target.y`.

```json
{
  "type": "xy_offset_from",
  "objects": [
    {"category": "plate", "set": 0, "slot": 0},
    {"category": "cup", "set": 0, "slot": 0}
  ],
  "anchor": {"category": "plate", "set": 0, "slot": 0},
  "dx": 0.18,
  "dy": -0.08,
  "dx_jitter": 0,
  "dy_jitter": 0
}
```

### pen_in_holder

Special two-object relation. Reads holder `x`, `y`, and `rotation`; writes target
`x`, `y`, and `rotation`.

```json
{
  "type": "pen_in_holder",
  "objects": [
    {"category": "holder", "set": 0, "slot": 0},
    {"category": "cutlery_a", "set": 0, "slot": 1}
  ],
  "holder": {"category": "holder", "set": 0, "slot": 0}
}
```

The non-holder object in `objects` is the target.
The generation stage performs the actual physical insertion and settle.
