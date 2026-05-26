# 2026-03-13 Hybrid UI Plan

## Goal
- [x] Add a lightweight local web UI under `bubbly_flows/tests/` for rapid tuning of the current hybrid pipeline on `img006001`.
- [x] Show the gold annotation in one panel and the current hybrid overlay in another.
- [x] Debounce parameter changes by `5` seconds and rerun only when parameters actually changed.
- [x] Keep the UI as a local research tool inside `bubbly_flows/tests/`, not a reusable app outside the research area.

## Current Project State
- [x] The hybrid experiment harness exists under `bubbly_flows/tests/src/experiments/`.
- [x] One-image tuning exists and now defaults to a baseline-anchored adaptive search.
- [x] The adaptive Slurm run on `img006001` has reached non-zero F1 candidates.
- [x] The hybrid runtime currently depends on the scratch `sam3` environment for SAM3-backed execution.

## Scope
- [ ] Fixed source image only:
  - [ ] `ZeroG_FlightDay_Test_C1S0014_img006001`
- [ ] Fixed pipeline only:
  - [ ] `bubble_frst_sam3_mask.py`
- [ ] Two-panel comparison:
  - [ ] gold reference panel
  - [ ] current hybrid overlay panel
- [ ] Focused control set for high-signal knobs only.
- [ ] No run history in v1.
- [ ] No multi-image browsing in v1.
- [ ] No scheduler integration in v1.

## File Layout
- [x] Add `bubbly_flows/tests/ui/__init__.py`
- [x] Add `bubbly_flows/tests/ui/state.py`
- [x] Add `bubbly_flows/tests/ui/render.py`
- [x] Add `bubbly_flows/tests/ui/server.py`
- [x] Add `bubbly_flows/tests/ui/static/index.html`
- [x] Add `bubbly_flows/tests/run_hybrid_ui.py`
- [x] Add unit tests under `bubbly_flows/tests/unit/`

## Behavior
- [x] Slider and toggle changes update in the browser immediately.
- [x] The backend schedules a rerun `5` seconds after the last parameter change.
- [x] If a run is already in progress, one pending rerun is queued and executed after the current run finishes.
- [x] Reset button restores `baseline_hybrid_original` values.
- [x] Latest overlay replaces the previous overlay in the right panel.
- [x] Gold reference remains fixed in the left panel.

## API
- [x] `GET /`
- [x] `GET /api/state`
- [x] `POST /api/params`
- [x] `POST /api/reset`
- [x] `GET /api/image/gold`
- [x] `GET /api/image/current`

## Controls For V1
- [ ] FRST:
  - [ ] `r_min`
  - [ ] `r_max`
  - [ ] `alpha`
  - [ ] `mag_percentile`
  - [ ] `peak_percentile`
  - [ ] `nms_size`
- [ ] geometry:
  - [ ] `knn_k`
  - [ ] `hex_radius_factor`
  - [ ] `tile_size_factor`
  - [ ] `tile_overlap_factor`
  - [ ] `area_limit_factor`
- [ ] adaptive branch:
  - [ ] `adaptive_area_min`
  - [ ] `adaptive_area_max`
  - [ ] `adaptive_circularity_min`
  - [ ] `adaptive_solidity_min`
  - [ ] `adaptive_intensity_max`
  - [ ] `blackhat_split_fused`
- [ ] postprocess:
  - [ ] `iou_dedup_thresh`
  - [ ] `containment_thresh`
  - [ ] `min_area_px`
  - [ ] `enable_consolidation`
  - [ ] `enable_hole_fill`

## Temporary Output Model
- [x] Use a small working directory under `bubbly_flows/tests/output/ui/`.
- [x] Keep only the latest render artifacts needed by the UI.
- [x] Do not add persistent run-history/provenance requirements to the UI itself.

## Verification
- [x] Unit tests for debounce scheduling and pending-rerun behavior.
- [x] Unit tests for parameter reset to baseline.
- [x] Unit tests for API state payload shape.
- [ ] Lightweight server-start smoke check.
- [ ] One manual run on a GPU compute node with SSH port forwarding.
