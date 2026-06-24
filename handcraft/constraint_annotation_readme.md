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

`object_sets` stores only high-level category instances. The preview/debug
sample is stored separately in `sample_entry_index`; it is not a constraint.

Selection constraints are stored in `selection_constraints`, beside the layout
`constraints`. They constrain which `available_assets` entry is sampled and are
evaluated from asset metadata, not from the simulation scene.

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

## Selection Constraints

### same_entry

Set-level relation. All listed set instances of one category must use the same
entry from `available_assets`.

```json
{ "type": "same_entry", "category": "cutlery_a", "sets": [0, 1, 2] }
```

The relation is created by cloning a set in the annotation UI.

### bbox_larger_than

Object-level selection relation. Compares stable-frame asset bounding boxes from
asset metadata before the scene is built. The `larger` object's XY bbox area
must be greater than or equal to the `smaller` object's XY bbox area.

```json
{
  "type": "bbox_larger_than",
  "larger": {"category": "plate", "set": 0, "slot": 0},
  "smaller": {"category": "bowl", "set": 0, "slot": 0}
}
```

## Layout Relations

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
  "axis": "90",
  "jitter_deg": 5
}
```

Supported axes:

```text
0
90
180
270
any
custom
```

When an `align_axis` relation is created, the UI snaps the object's current yaw
to the nearest right angle. `custom` uses `yaw_deg`.

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

### on_top_of

Two-object ordering relation. The anchor is the lower object and the other object
is above it. This relation does not define `x`, `y`, `rotation`, or a separate
`z` field in the annotation preview; it only records stack order for generation.

```json
{
  "type": "on_top_of",
  "objects": [
    {"category": "plate", "set": 0, "slot": 0},
    {"category": "bowl", "set": 0, "slot": 0}
  ],
  "anchor": {"category": "plate", "set": 0, "slot": 0}
}
```

Use `xy_offset_from` with `dx = 0` and `dy = 0` when the stacked objects should
share the same table-plane center.

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
