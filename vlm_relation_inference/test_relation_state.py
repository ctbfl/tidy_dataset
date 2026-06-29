import unittest

from vlm_relation_inference.relation_state import RelationState, evaluate_relations


class RelationStateTest(unittest.TestCase):
    def test_offsets_define_target_fields(self):
        result = evaluate_relations(
            {"1": "plate", "2": "cup"},
            [
                {"type": "table_xy", "object": "plate", "x": 0.0, "y": 0.0},
                {"type": "xy_offset_from", "object": "cup", "anchor": "plate", "dx": 0.2, "dy": 0.1},
                {"type": "align_axis", "object": "cup", "axis": "any"},
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["objects"]["cup"], {"x": "ok", "y": "ok", "rotation": "ok"})
        self.assertEqual(result["objects"]["plate"]["rotation"], "undefined")

    def test_relation_failure_does_not_mutate_state(self):
        state = RelationState(["plate", "cup"])
        self.assertTrue(state.apply_relation({"type": "table_xy", "object": "cup", "x": 0.2, "y": 0.1})["ok"])
        result = state.apply_relation({"type": "x_offset_from", "object": "cup", "anchor": "plate", "dx": 0.3})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "anchor_field_missing")
        self.assertEqual(result["objects"]["cup"]["x"], "ok")
        self.assertEqual(result["objects"]["plate"]["x"], "undefined")

    def test_over_defined_field_is_reported(self):
        result = evaluate_relations(
            ["plate", "cup"],
            [
                {"type": "table_xy", "object": "plate", "x": 0.0, "y": 0.0},
                {"type": "table_xy", "object": "cup", "x": 0.2, "y": 0.1},
                {"type": "xy_offset_from", "object": "cup", "anchor": "plate", "dx": 0.3, "dy": 0.0},
            ],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "over_defined")

    def test_same_line_plus_absolute_axis(self):
        result = evaluate_relations(
            ["plate", "cup"],
            [
                {"type": "table_xy", "object": "plate", "x": -0.1, "y": 0.0},
                {"type": "in_same_vertical_line", "objects": ["plate", "cup"], "anchor": "plate"},
                {"type": "table_y", "object": "cup", "y": 0.25},
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["objects"]["cup"]["x"], "ok")
        self.assertEqual(result["objects"]["cup"]["y"], "ok")
        self.assertEqual(result["objects"]["cup"]["rotation"], "undefined")

    def test_in_holder_writes_object_pose_fields(self):
        result = evaluate_relations(
            ["pen_holder", "pen"],
            [
                {"type": "table_xy", "object": "pen_holder", "x": 0.4, "y": 0.2},
                {"type": "in_holder", "object": "pen", "holder": "pen_holder"},
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["objects"]["pen"], {"x": "ok", "y": "ok", "rotation": "ok"})

    def test_even_spacing_validates_order_and_defines_axis(self):
        result = evaluate_relations(
            ["cup_left", "plate", "cup_right"],
            [
                {"type": "table_xy", "object": "plate", "x": 0.0, "y": 0.0},
                {
                    "type": "evenly_spaced_from_anchor",
                    "objects": ["cup_left", "plate", "cup_right"],
                    "anchor": "plate",
                    "axis": "x",
                    "mode": "footprint",
                    "order": ["cup_left", "plate", "cup_right"],
                    "spacing": 0.08,
                },
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["objects"]["cup_left"]["x"], "ok")
        self.assertEqual(result["objects"]["cup_left"]["y"], "undefined")
        self.assertEqual(result["objects"]["cup_right"]["x"], "ok")

    def test_align_axis_uses_vlm_axis_names(self):
        result = evaluate_relations(
            ["fork", "pen"],
            [
                {"type": "align_axis", "object": "fork", "axis": "vertical"},
                {"type": "align_axis", "object": "pen", "axis": "custom"},
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["objects"]["fork"]["rotation"], "ok")
        self.assertEqual(result["objects"]["pen"]["rotation"], "ok")

    def test_numeric_align_axis_is_rejected(self):
        result = evaluate_relations(
            ["fork"],
            [{"type": "align_axis", "object": "fork", "axis": "90"}],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "invalid_parameter")

    def test_unknown_object_is_reported(self):
        result = evaluate_relations(
            ["plate"],
            [{"type": "table_xy", "object": "cup", "x": 0.0, "y": 0.0}],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "unknown_object_id")


if __name__ == "__main__":
    unittest.main()
