# 2026-05-26 Hybrid Research Retrospective

## Scope Of This Retrospective

This document captures the March 2026 hybrid-research work carried out under
`bubbly_flows/tests/`. It is not a production-pipeline design doc. It records what
we implemented, what we ran on Oscar, what we found, and which working hypotheses
were falsified.

## What We Built

- Reorganized the active hybrid code under `bubbly_flows/tests/src/` into explicit
  families:
  - `hybrid/`
  - `sam3/`
  - `deterministic/`
  - `prompting/`
  - `backends/`
  - `common/`
- Added an experiment inventory and minimal curated metadata so the testbed has a
  machine-readable map of entrypoints, purpose, and expected outputs.
- Added a provenance-first experiment harness under
  `bubbly_flows/tests/src/experiments/`.
- Added a one-image tuning runner centered on the gold image
  `ZeroG_FlightDay_Test_C1S0014_img006001`.
- Added a baseline-anchored adaptive search around the current hybrid defaults.
- Added a lightweight local HTML UI under `bubbly_flows/tests/ui/` for fast manual
  tuning against the same fixed image.

## What We Ran

### 1. Two-image blackhat-only validation

- Slurm job: `698942`
- Output root:
  `bubbly_flows/tests/output/experiments/validate_2img_blackhat_only/`
- Purpose:
  validate the end-to-end harness path on a family that does not depend on the SAM3
  runtime.

Observed aggregate result:
- precision = `0.000`
- recall = `0.000`
- F1 = `0.000`

Interpretation:
- the harness path itself worked
- `blackhat_only` was not a useful standalone detector on this slice

### 2. Two-image three-family validation

- Initial failing Slurm job: `699441`
- Successful rerun: `700150`
- Output root:
  `bubbly_flows/tests/output/experiments/validate_2img_seed_v04/`
- Families compared:
  - `frst_only`
  - `hybrid_current`
  - `blackhat_only`

Successful rerun aggregate ranking:
1. `frst_only` -> `F1 = 0.003`
2. `hybrid_current` -> `F1 = 0.003`
3. `blackhat_only` -> `F1 = 0.000`

What changed between failure and rerun:
- The hybrid runner was switched to use the user-scoped SAM3 environment for
  SAM3-backed families.
- The failing facebookresearch big-bubble backend path was replaced with the HF big
  backend in the experiment runner so `hybrid_current` became executable again.

Interpretation:
- `frst_only` and `hybrid_current` both became runnable end-to-end.
- On this small slice, the hybrid path did not outperform FRST-only.
- The adaptive/blackhat additions were not buying measurable recall on these two
  images.

### 3. One-image smoke search on `img006001`

- Slurm job: `712596`
- Output root:
  `bubbly_flows/tests/output/experiments/one_image_tuning_smoke_20260313/results/`
- Families compared:
  - `frst_only`
  - `hybrid_current`

Observed result:
- `frst_only_001`: `TP=0`, `FP=324`, `FN=114`, `F1=0.000`
- `hybrid_current_001`: `TP=0`, `FP=494`, `FN=114`, `F1=0.000`

Interpretation:
- The one-image tuning infrastructure worked.
- The initial bounded search admitted destructive parameter regimes.
- Hybrid had worse false-positive burden than FRST-only in the tested smoke setting.

### 4. Baseline-anchored adaptive search on `img006001`

- Slurm job: `712987`
- Output root:
  `bubbly_flows/tests/output/experiments/one_image_tuning_adaptive_20260313/results/`
- Search strategy:
  bounded coordinate-neighbor search around `baseline_hybrid_original`

Top ranking from `ranking.csv`:
1. `hybrid_current_r2_mag_percentile_92p0` -> `F1 = 0.005`
2. `hybrid_current_r1_r_max_22` -> `F1 = 0.004`
3. `hybrid_current_r2_alpha_1p2` -> `F1 = 0.004`
4. `hybrid_current_r2_alpha_1p6` -> `F1 = 0.004`
5. `hybrid_current_r2_mag_percentile_84p0` -> `F1 = 0.004`
6. `hybrid_current_r2_nms_size_9` -> `F1 = 0.004`
- Baseline: `baseline_hybrid_original` -> `F1 = 0.000`

Interpretation:
- Anchoring the search on the historical hybrid defaults was better than starting from
  arbitrary smoke-grid points.
- The adaptive search produced non-zero improvements, but the best result remained very
  weak in absolute terms.
- The first useful movement came from FRST-side thresholding, not from a fundamentally
  new fusion behavior.

## What We Found

### Environment and runtime

- The production environment (`bubbly-train-env`) was not the original runtime for the
  SAM3-backed hybrid experiments.
- The historical hybrid work depended on a separate user-scoped SAM3 environment plus
  scratch-scoped assets.
- The original facebookresearch big-bubble path was not runnable in the current runtime
  state, even though the broader hybrid codebase had worked previously.
- The HF SAM3 backend ran in the scratch environment, but emitted a compatibility warning
  about loading a `sam3_video` model as `sam3_tracker`. This did not block execution, but
  it remains a quality concern.

### Algorithmic behavior

- `blackhat_only` was not competitive as a standalone family on the tested slices.
- `hybrid_current` was not automatically better than `frst_only`; on the two-image slice
  it only tied it, and on the smoke run it was worse by false positives.
- The search surface is fragile: naive search over exposed knobs can easily move into
  regimes that wash out useful signal.
- Baseline-centered local search is more defensible than broad arbitrary search, but it
  still did not uncover a strong configuration on `img006001`.

## Hypotheses We Falsified

### Falsified 1: `blackhat_only` is a strong standalone baseline

Status: falsified on the tested 2-image slice.

Evidence:
- `blackhat_only` scored `F1 = 0.000` in the dedicated two-image validation.
- It also ranked last in the three-family two-image comparison.

### Falsified 2: the current hybrid path is automatically stronger than FRST-only

Status: falsified on the tested slices.

Evidence:
- `hybrid_current` only tied `frst_only` at `F1 = 0.003` on the 2-image validation.
- In the one-image smoke run, `hybrid_current` had worse false-positive burden and still
  `F1 = 0.000`.

### Falsified 3: broad one-image tuning can start from arbitrary grid points without a
trustworthy baseline

Status: falsified.

Evidence:
- The initial smoke search explored parameter sets that produced unusable overlays and
  zero true positives.
- Re-centering on `baseline_hybrid_original` produced the first non-zero F1 candidates.

### Falsified 4: the immediate blocker for the hybrid runtime was missing model weights

Status: falsified.

Evidence:
- The actual blocker was environment/runtime drift: missing or mismatched SAM3-side
  Python dependencies and backend assumptions.
- Reusing the user-scoped SAM3 environment restored executability for HF-backed paths.

## What Remains Unresolved

- Whether a true source-aware fusion strategy can beat `frst_only` materially.
- Whether the HF backend warning is harming inference quality in a meaningful way.
- Whether Hough-backed candidate generation is a better family than the current FRST-led
  hybrid path.
- Whether temporal information across adjacent frames can improve the brittle single-frame
  behavior without over-smoothing real breakup events.

## Practical Next Directions

1. Compare the top adaptive-search overlays visually against the baseline, not just by
   F1.
2. Treat Hough as the next family-level experiment rather than just another knob.
3. If the current hybrid remains weak, improve fusion logic before widening the search
   space further.
4. If temporal methods are explored, start with local temporal evidence aggregation rather
   than global identity tracking.
