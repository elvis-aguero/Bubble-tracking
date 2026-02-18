# Task 20260214-1818-export-augmentation-gold-to-ml

## Summary
- Implement export-time data augmentation in `manage_bubbly.py` for Gold -> MicroSAM dataset generation.
- Add three techniques: geometric transforms, photometric transforms, and copy-paste with limited overlap.
- Export 4 total samples per source by default (1 original + 3 augmented variants).

## Owner
- codex/20260214-1818

## Status
- done

## Plan
- [x] Add augmentation helper functions for deterministic RNG, geometric transforms, photometric transforms, and copy-paste.
- [x] Integrate augmentation prompts and variant generation into `export_microsam_dataset()`.
- [x] Write augmentation metadata (`augmentation_config.json`) with provenance and settings.
- [x] Validate script syntax and confirm help output remains functional.

## Log
- 2026-02-14 18:18 EST: Added helper functions for deterministic per-image RNG (`sha256(seed:stem)`), geometric transforms, photometric jitter, and limited-overlap copy-paste.
- 2026-02-14 18:18 EST: Updated export flow to prompt for augmentation enable/seed and emit 3 augmented variants per source (`__aug01..03`).
- 2026-02-14 18:18 EST: Added export metadata file `augmentation_config.json` with per-source variant details and transform metadata.
- 2026-02-14 18:18 EST: Ran `python3 -m py_compile bubbly_flows/scripts/manage_bubbly.py` and `python3 bubbly_flows/scripts/manage_bubbly.py --help`.

## Messages
- 
