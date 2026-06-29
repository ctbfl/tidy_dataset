from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


FIELDS = ("x", "y", "rotation")
OK = "ok"
UNDEFINED = "undefined"


class RelationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RelationPlan:
    reads: tuple[tuple[str, str], ...]
    writes: tuple[tuple[str, str], ...]


class RelationState:
    def __init__(self, objects: Iterable[str | dict] | dict):
        object_ids = _object_ids(objects)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("object_ids must not contain duplicates")
        self._fields = {object_id: {field: False for field in FIELDS} for object_id in object_ids}

    def status(self) -> dict[str, dict[str, str]]:
        return {
            object_id: {
                field: OK if defined else UNDEFINED
                for field, defined in fields.items()
            }
            for object_id, fields in self._fields.items()
        }

    def apply_relation(self, relation: dict, relation_index: int | None = None) -> dict:
        try:
            self._apply_relation(relation)
        except RelationError as exc:
            return self._result(False, exc, relation_index)
        return self._result(True, None, relation_index)

    def apply_all(self, relations: Iterable[dict]) -> dict:
        for index, relation in enumerate(relations):
            result = self.apply_relation(relation, index)
            if not result["ok"]:
                return result
        return self._result(True, None, None)

    def _result(self, ok: bool, error: RelationError | None, relation_index: int | None) -> dict:
        result = {
            "ok": ok,
            "objects": self.status(),
        }
        if error is not None:
            result["error"] = {
                "code": error.code,
                "message": error.message,
                "relation_index": relation_index,
            }
        return result

    def _apply_relation(self, relation: dict) -> None:
        if not isinstance(relation, dict):
            raise RelationError("invalid_relation", "relation must be an object")
        plan = self._plan_relation(relation)
        self._validate_plan(plan)
        for object_id, field in plan.writes:
            self._fields[object_id][field] = True

    def _validate_plan(self, plan: RelationPlan) -> None:
        for object_id, field in plan.reads:
            self._require_defined(object_id, field)
        seen_writes = set()
        for object_id, field in plan.writes:
            key = (object_id, field)
            if key in seen_writes:
                raise RelationError("duplicate_write", f"{object_id}.{field} is written more than once by one relation")
            seen_writes.add(key)
            if self._fields[object_id][field]:
                raise RelationError("over_defined", f"{object_id}.{field} is already defined")

    def _plan_relation(self, relation: dict) -> RelationPlan:
        kind = _required_str(relation, "type")
        if kind == "table_x":
            return RelationPlan((), ((_known_object(self._fields, relation, "object"), "x"),))
        if kind == "table_y":
            return RelationPlan((), ((_known_object(self._fields, relation, "object"), "y"),))
        if kind == "table_xy":
            object_id = _known_object(self._fields, relation, "object")
            _required_number(relation, "x")
            _required_number(relation, "y")
            return RelationPlan((), ((object_id, "x"), (object_id, "y")))
        if kind == "align_axis":
            object_id = _known_object(self._fields, relation, "object")
            axis = _required_str(relation, "axis")
            if axis not in {"horizontal", "vertical", "any", "custom"}:
                raise RelationError("invalid_parameter", f"align_axis.axis is invalid: {axis}")
            return RelationPlan((), ((object_id, "rotation"),))
        if kind == "in_same_vertical_line":
            anchor, targets = self._anchor_targets(relation)
            return RelationPlan(((anchor, "x"),), tuple((target, "x") for target in targets))
        if kind == "in_same_horizontal_line":
            anchor, targets = self._anchor_targets(relation)
            return RelationPlan(((anchor, "y"),), tuple((target, "y") for target in targets))
        if kind == "evenly_spaced_from_anchor":
            anchor, targets = self._anchor_targets(relation)
            axis = _required_str(relation, "axis")
            if axis not in {"x", "y"}:
                raise RelationError("invalid_parameter", f"evenly_spaced_from_anchor.axis is invalid: {axis}")
            mode = _required_str(relation, "mode")
            if mode not in {"obj_center", "footprint"}:
                raise RelationError("invalid_parameter", f"evenly_spaced_from_anchor.mode is invalid: {mode}")
            spacing = _required_number(relation, "spacing")
            if spacing < 0:
                raise RelationError("invalid_parameter", "evenly_spaced_from_anchor.spacing must be non-negative")
            self._validate_order(relation)
            return RelationPlan(((anchor, axis),), tuple((target, axis) for target in targets))
        if kind == "x_offset_from":
            object_id = _known_object(self._fields, relation, "object")
            anchor = _known_object(self._fields, relation, "anchor")
            _required_number(relation, "dx")
            _require_distinct(object_id, anchor, "object", "anchor")
            return RelationPlan(((anchor, "x"),), ((object_id, "x"),))
        if kind == "y_offset_from":
            object_id = _known_object(self._fields, relation, "object")
            anchor = _known_object(self._fields, relation, "anchor")
            _required_number(relation, "dy")
            _require_distinct(object_id, anchor, "object", "anchor")
            return RelationPlan(((anchor, "y"),), ((object_id, "y"),))
        if kind == "xy_offset_from":
            object_id = _known_object(self._fields, relation, "object")
            anchor = _known_object(self._fields, relation, "anchor")
            _required_number(relation, "dx")
            _required_number(relation, "dy")
            _require_distinct(object_id, anchor, "object", "anchor")
            return RelationPlan(((anchor, "x"), (anchor, "y")), ((object_id, "x"), (object_id, "y")))
        if kind == "on_top_of":
            object_id = _known_object(self._fields, relation, "object")
            anchor = _known_object(self._fields, relation, "anchor")
            _require_distinct(object_id, anchor, "object", "anchor")
            return RelationPlan((), ())
        if kind == "in_holder":
            object_id = _known_object(self._fields, relation, "object")
            holder = _known_object(self._fields, relation, "holder")
            _require_distinct(object_id, holder, "object", "holder")
            return RelationPlan(((holder, "x"), (holder, "y")), ((object_id, "x"), (object_id, "y"), (object_id, "rotation")))
        raise RelationError("unknown_relation", f"unknown relation type: {kind}")

    def _anchor_targets(self, relation: dict) -> tuple[str, list[str]]:
        anchor = _known_object(self._fields, relation, "anchor")
        objects = _required_object_list(self._fields, relation, "objects")
        if anchor not in objects:
            raise RelationError("invalid_parameter", "anchor must be included in objects")
        targets = [object_id for object_id in objects if object_id != anchor]
        if not targets:
            raise RelationError("missing_target", "relation needs at least one target object")
        return anchor, targets

    def _validate_order(self, relation: dict) -> None:
        objects = _required_object_list(self._fields, relation, "objects")
        order = relation.get("order")
        if not isinstance(order, list) or not order:
            raise RelationError("missing_parameter", "evenly_spaced_from_anchor.order must be a non-empty list")
        order = [_known_id(self._fields, value, "order") for value in order]
        if len(set(order)) != len(order):
            raise RelationError("invalid_parameter", "evenly_spaced_from_anchor.order must not contain duplicates")
        if set(order) != set(objects):
            raise RelationError("invalid_parameter", "evenly_spaced_from_anchor.order must contain exactly objects")

    def _require_defined(self, object_id: str, field: str) -> None:
        if not self._fields[object_id][field]:
            raise RelationError("anchor_field_missing", f"{object_id}.{field} is undefined")


def evaluate_relations(objects: Iterable[str | dict] | dict, relations: Iterable[dict]) -> dict:
    return RelationState(objects).apply_all(relations)


def _object_ids(objects: Iterable[str | dict] | dict) -> list[str]:
    if isinstance(objects, dict):
        return [_known_id(None, value, "object_id") for value in objects.values()]
    return [_object_id(obj) for obj in objects]


def _object_id(obj: str | dict) -> str:
    if isinstance(obj, str) and obj:
        return obj
    if isinstance(obj, dict):
        return _known_id(None, obj.get("object_id"), "object_id")
    raise ValueError("each object must be an object_id string or an object with object_id")


def _known_object(fields: dict[str, dict[str, bool]], relation: dict, name: str) -> str:
    return _known_id(fields, relation.get(name), name)


def _known_id(fields: dict[str, dict[str, bool]] | None, value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationError("missing_parameter", f"{name} is required")
    if fields is not None and value not in fields:
        raise RelationError("unknown_object_id", f"unknown object_id: {value}")
    return value


def _required_str(relation: dict, name: str) -> str:
    value = relation.get(name)
    if not isinstance(value, str) or not value:
        raise RelationError("missing_parameter", f"{name} is required")
    return value


def _required_number(relation: dict, name: str) -> float:
    value = relation.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelationError("missing_parameter", f"{name} must be a number")
    return float(value)


def _required_object_list(fields: dict[str, dict[str, bool]], relation: dict, name: str) -> list[str]:
    values = relation.get(name)
    if not isinstance(values, list) or not values:
        raise RelationError("missing_parameter", f"{name} must be a non-empty list")
    object_ids = [_known_id(fields, value, name) for value in values]
    if len(set(object_ids)) != len(object_ids):
        raise RelationError("invalid_parameter", f"{name} must not contain duplicates")
    return object_ids


def _require_distinct(left: str, right: str, left_name: str, right_name: str) -> None:
    if left == right:
        raise RelationError("invalid_parameter", f"{left_name} and {right_name} must be different")
