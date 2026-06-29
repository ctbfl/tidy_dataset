Deprecated tool scripts kept for reference.

The active dataset generation flow is:

1. `tools/1_sample_tidy_goals.py`
2. `tools/2_sample_messy_inputs.py`

Scripts in this directory are older mixed or wrapper entry points. Do not use
them for new dataset generation unless you are intentionally reproducing an old
workflow.

Moved here:

- `export_v1.py`, `render_organize_it_scene.py`, `tidy_to_organize_it.py`:
  old v0/v1 export path.
- `export_v2.py`: mixed tidy/messy batch exporter superseded by the two-stage
  flow.
- `sample_tidy_from_constraint_template.py`: old stage-1 script superseded by
  `tools/1_sample_tidy_goals.py`.
- `gen_messy_from_tidy_root.py`: old stage-2 wrapper superseded by
  `tools/2_sample_messy_inputs.py`.
