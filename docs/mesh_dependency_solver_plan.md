# Mesh Dependency Solver Plan

## Scope

This document describes the planned one-shot dependency solver for SG-Bot comparison transport. The solver works in the mesh domain, not in the simulator domain and not with privileged PyBullet geometry.

Input is a standard tidy dataset scene directory containing `ta_real_scene.pkl`. Object geometry and poses are read from the v2 fields:

- `any6D_scaled_mesh`
- `v2_position`
- `v2_rotation`

The solver produces a full plan before execution. Simulation execution should not use closed-loop point cloud feedback or simulator-only geometry to rebuild dependencies.

## State

The planner state is a mapping from object id to item state:

```python
all_items[obj_id] = {
    "current_pose": ...,
    "goal_pose": ...,
    "mesh": ...,
    "is_arrived": bool,
}
```

Arrived objects are committed:

- They are not action candidates.
- They are removed from dependency graph keys.
- They remain passive occupancy objects.
- They may still appear in dependency values as blockers.
- If an arrived object blocks an unarrived object, the current DFS branch should fail and backtrack.

## Dependency Graph

The dependency graph is rebuilt from the current planner state at every DFS node. It only describes whether each unarrived object can directly move to its final goal pose.

Graph edges are triples:

```python
[former, latter, reason]
```

Meaning:

```text
former cannot directly move to final goal because of latter.
```

Supported reasons:

```text
need_latter_one_arrive_first
latter_one_block_former_one_arrive_goal
latter_one_block_former_one_leave
```

Rules:

1. `latter_one_block_former_one_leave`

   The former object's vertical lift path from its current pose is blocked by the latter object.

2. `need_latter_one_arrive_first`

   The former object's final goal is on top of the latter object according to the DSL or goal support relation, and the latter has not arrived yet.

3. `latter_one_block_former_one_arrive_goal`

   The former object's final placement volume is blocked by the latter object.

   If the former has a goal support object, the clear region is checked from the support object's top surface upward over the former object's footprint.

   If the former has no goal support object, the clear region is checked from the tabletop upward over the former object's goal footprint.

If an unarrived object has no dependency edges as `former`, then `goal_move(object)` is valid by definition. No extra validity diagnosis should be repeated during goal action application.

## Actions

The action space contains two action types.

### Goal Move

```python
goal_move(A)
```

This moves `A` to its final goal pose.

It is available when `A` is unarrived and has no dependency edges.

Applying it only updates planner state:

```python
current_pose[A] = goal_pose[A]
is_arrived[A] = True
```

### Buffer Move

```python
buffer_move(A, release=B)
```

This moves `A` to a temporary buffer pose on the tabletop, intended to release object `B` from a dependency caused by `A`.

Buffer moves are generated only from cycle objects. For a cycle object `A` to be a buffer candidate, `A` must not have any outgoing `latter_one_block_former_one_leave` dependency. If `A` itself cannot leave, it cannot be selected as the moving buffer object.

For each object `B` blocked by `A`, analyzer may generate:

```python
buffer_move(A, release=B)
```

The buffer pose is not searched during analysis. It is searched only when the action is applied.

Applying a buffer action:

1. Build available area:

   ```text
   tabletop_area
   - current occupied tabletop area of all objects except A
   - B goal occupied area
   ```

2. Search for a pose where `A` can be placed directly on the tabletop.

3. If a clean tabletop buffer pose is found:

   ```python
   current_pose[A] = buffer_pose
   is_arrived[A] = False
   ```

4. If no clean buffer pose exists, the action fails.

Buffer target must actually be on the tabletop. It is not merely treated as tabletop placement.

## Analyzer

`analyze(dep_graph, all_items)` produces candidate actions. Goal moves are always ordered before buffer moves.

1. Add `goal_move(A)` for every unarrived object `A` with no dependency edges.

2. If cycle dependencies exist, find all objects in cycles.

3. For each cycle object `A`, if `A` has no outgoing leave-block dependency, generate buffer actions for objects blocked by `A`.

4. Objects with dependencies but not involved in available goal or buffer actions are not moved. They wait for blockers to be resolved by other actions.

## DFS Solver

The solver is a one-shot DFS over mesh-domain planner states:

```python
def solve(all_items):
    if all_arrived(all_items):
        return []

    dep_graph = build_dep_graph(all_items)
    available_actions = analyze(dep_graph, all_items)

    if not available_actions:
        raise RuntimeError(
            "objects remain unarrived but no available action exists; inspect dependency builder"
        )

    for action in available_actions:
        success, next_items, info = try_apply(all_items, action)
        if not success:
            continue

        sub_trace = solve(next_items)
        if sub_trace is not None:
            enriched_action = enrich_action(action, info)
            return [enriched_action] + sub_trace

    return None
```

Visited-state pruning is intentionally omitted in the first implementation. Keep the solver simple until repeated-state loops are observed in practice.

## Output Plan

Each enriched action should include enough data for the PyBullet executor:

```python
{
    "obj_id": "A",
    "kind": "goal" | "buffer",
    "target_world_pose": ...,
    "pose_before_move": ...,
    "meta": {
        "to_release": None | "B",
        "buffer_pose": None | ...,
    },
}
```

The external executor interface should remain compatible with `run_sgbot_compare.py`:

- `kind == "goal"` means move to final goal pose.
- `kind == "buffer"` means move to the planned buffer pose.

## Verification

The offline PyBullet verifier is an audit tool, not planner input. It may use SG-Bot URDF/PyBullet collision to check whether exported steps were executable in simulation, but the planner itself must not use simulator-only geometry or point clouds.

Planner correctness is defined in the mesh domain:

```text
dependency-free goal move == legal mesh-domain goal action
```

If the offline verifier finds an invalid step, first inspect whether the mesh-domain dependency builder failed to model the corresponding blocker.
