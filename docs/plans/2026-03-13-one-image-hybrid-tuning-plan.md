# 2026-03-13 One-Image Hybrid Tuning Plan

## Goal
- [x] Overfit intentionally on a single gold image, `ZeroG_FlightDay_Test_C1S0014_img006001`, to stress-test the hybrid search workflow.
- [x] Keep all provenance and outputs under `bubbly_flows/tests/`.
- [ ] Search only families that currently look useful enough to justify compute:
  - [x] `frst_only`
  - [x] `hybrid_current`
- [x] Exclude `blackhat_only` from the first one-image search pass.

## Why This Slice
- [x] The experiment harness is now working end-to-end under Slurm.
- [x] `frst_only` and `hybrid_current` both execute on gold data with provenance.
- [x] `hybrid_current` no longer depends on the broken facebookresearch big-bubble path in the current runtime.
- [x] What is missing is not infrastructure, but a search space that matches the real high-signal knobs in the current hybrid driver.

## High-Signal Knobs To Expose In This Pass

### FRST center generation
- [x] `r_min`
- [x] `r_max`
- [x] `r_step`
- [x] `alpha`
- [x] `mag_percentile`
- [x] `peak_percentile`
- [x] `nms_size`
- [x] `border`
- [x] `max_peaks`

### Prompt geometry and tile behavior
These are currently hardcoded in `bubble_frst_sam3_mask.py` and need to become tunable.
- [x] `knn_k`
- [x] `hex_radius_factor`
- [x] `tile_size_factor`
- [x] `tile_overlap_factor`
- [x] `area_limit_factor`

### Blackhat adaptive branch
Expose the behavior-shaping parameters of the currently used adaptive branch.
- [x] `adaptive_area_min`
- [x] `adaptive_area_max`
- [x] `adaptive_circularity_min`
- [x] `adaptive_solidity_min`
- [x] `adaptive_intensity_max`
- [x] `blackhat_split_fused`

### Consolidation and postprocess
- [x] `iou_dedup_thresh`
- [x] `containment_thresh`
- [x] `min_area_px`
- [x] `enable_consolidation`
- [x] `enable_hole_fill`

## Search Strategy For This Pass
- [x] Use one fixed image only:
  - [x] `ZeroG_FlightDay_Test_C1S0014_img006001`
- [x] Build a reproducible tuning runner under `bubbly_flows/tests/` that:
  - [x] constructs experiment specs from an explicit search grid
  - [x] writes a run manifest and ranking
  - [x] saves overlays and predicted masks for each candidate
  - [x] evaluates against the gold mask for that one image
- [x] Do not force a coarse-only strategy in this pass.
- [x] Allow direct grid search over the chosen parameter grid for the single image.

## Hough Transforms
- [x] Hough candidate generation exists in the codebase under `bubbly_flows/tests/src/common/bubble_sam3/candidates.py`.
- [x] Hough is not currently wired into the active hybrid driver being benchmarked.
- [x] Treat Hough as the next candidate-family feature, not as part of this implementation slice.

## Implementation Steps
- [x] Add CLI support in `bubble_frst_sam3_mask.py` for the hidden prompt-geometry and adaptive-branch knobs.
- [x] Update the experiment executor so those parameters can be passed from `ExperimentSpec` into the actual command/config.
- [x] Add a one-image tuning search module under `bubbly_flows/tests/src/experiments/`.
- [x] Add a runnable entrypoint/script under `bubbly_flows/tests/` to launch the one-image search through Slurm or directly on a compute node.
- [x] Keep provenance output rooted under `bubbly_flows/tests/output/experiments/`.

## Verification
- [x] Unit tests for parameter plumbing from `ExperimentSpec` into the hybrid command/config.
- [x] Unit tests for the one-image spec builder / search expansion.
- [x] One real validation run on `img006001` with at least:
  - [x] one `frst_only` candidate
  - [x] one `hybrid_current` candidate
- [x] Confirm generated outputs include:
  - [x] overlay image
  - [x] LabelMe JSON
  - [x] prediction mask
  - [x] metrics CSV/JSON
  - [x] ranking summary

## Smoke Run Result
- [x] Slurm smoke run completed successfully:
  - [x] job `712596`
  - [x] output root `bubbly_flows/tests/output/experiments/one_image_tuning_smoke_20260313/results/`
- [x] Current bounded smoke settings do not recover true positives on `img006001` at IoU `0.5`:
  - [x] `frst_only_001`: `TP=0`, `FP=324`, `FN=114`, `F1=0.000`
  - [x] `hybrid_current_001`: `TP=0`, `FP=494`, `FN=114`, `F1=0.000`
- [ ] Next pass should improve candidate quality rather than broaden the search blindly.

## Baseline-Anchored Adaptive Search
- [x] Define `baseline_hybrid_original` explicitly as the default anchor for one-image tuning.
- [x] Use current script defaults as the baseline parameter set:
  - [x] FRST defaults
  - [x] prompt geometry defaults
  - [x] adaptive blackhat defaults
  - [x] postprocess defaults
- [x] Keep the operational backend override that makes the hybrid run executable in the current environment.
- [x] Replace the default one-image execution path with a bounded adaptive search around that baseline.
- [x] Proposal strategy: one-parameter-at-a-time coordinate neighbors around the current best spec.
- [x] Default runner now starts from `baseline_hybrid_original` instead of an arbitrary grid candidate.
- [x] Run a fresh baseline-anchored adaptive search on `img006001`.

## Adaptive Run Status Checkpoint
- [x] Slurm adaptive run submitted:
  - [x] job `712987`
  - [x] output root `bubbly_flows/tests/output/experiments/one_image_tuning_adaptive_20260313/results/`
- [x] Baseline completed:
  - [x] `baseline_hybrid_original`: `TP=0`, `FP=349`, `FN=114`, `F1=0.000`
- [x] Round 1 completed with multiple one-knob neighbors evaluated.
- [x] Round 2 has started and produced the first non-zero F1 results.
- [x] Best result observed so far:
  - [x] `hybrid_current_r2_mag_percentile_92p0`
  - [x] `F1=0.005`
- [x] Other near-best round-2 candidates currently at `F1=0.004`:
  - [x] `hybrid_current_r2_alpha_1p2`
  - [x] `hybrid_current_r2_alpha_1p6`
  - [x] `hybrid_current_r2_mag_percentile_84p0`
  - [x] `hybrid_current_r2_nms_size_9`
- [x] Early conclusion:
  - [x] the baseline-centered adaptive search is functioning
  - [x] the search is no longer stuck at all-zero candidates
