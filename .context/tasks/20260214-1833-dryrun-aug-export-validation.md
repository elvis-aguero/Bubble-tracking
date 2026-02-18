# Task 20260214-1833-dryrun-aug-export-validation

## Summary
- Create a dry-run ML dataset export with augmentation enabled from an existing Gold version.
- Validate that output counts and image/label pairing are correct.

## Owner
- codex/20260214-1833

## Status
- done

## Plan
- [x] Identify available gold versions and choose one for dry run.
- [x] Run `manage_bubbly.py` export flow into a new dataset name with augmentation enabled.
- [x] Validate output counts (base vs augmented) and stem-level image/label pairing.
- [x] Inspect augmentation metadata for seed/settings/provenance sanity.

## Log
- 2026-02-14 18:33 EST: Found `gold_seed_v00` and `gold_v00` (124 labels each).
- 2026-02-14 18:34 EST: Initial run with system python failed due missing `cv2/numpy`; reran with `/oscar/home/eaguerov/.conda/envs/bubbly-train-env/bin/python` and `_MANAGE_SKIP_ENV_CHECK=1`.
- 2026-02-14 18:35 EST: Created dataset `bubbly_flows/microsam/datasets/dryrun_aug_20260214_1835`.
- 2026-02-14 18:36 EST: Verified counts: 124 originals + 372 augmented = 496 image files and 496 label files.
- 2026-02-14 18:36 EST: Verified stem pairing: 496 image stems == 496 label stems, no mismatches.
- 2026-02-14 18:36 EST: Verified metadata in `augmentation_config.json`: enabled=true, seed=42, sources=124, variants_per_source=3, copy-paste requested=124 and applied=122.

## Messages
- 
