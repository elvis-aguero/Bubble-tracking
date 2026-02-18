# Task 20260214-1639-full-image-workspace-default

## Summary
- Switch `manage_bubbly.py` workspace creation default from patch pool to full-frame images.
- Keep patch pool available as an explicit source option.
- Make MicroSAM export resolve source images from full-frame directories first.

## Owner
- codex/20260214-1639

## Status
- done

## Plan
- [x] Inspect current workspace source logic in `create_workspace()`.
- [x] Add selectable image source with default full-frame directory.
- [x] Update export image lookup to support full-frame-first workflows.
- [x] Run syntax validation.

## Log
- 2026-02-14 16:39 EST: Confirmed current behavior uses `data/patches_pool/images` in `create_workspace()`.
- 2026-02-14 16:40 EST: Patched `manage_bubbly.py` to default workspace source to `data/frames/images_16bit_png` and expose patch-pool as source option.
- 2026-02-14 16:40 EST: Added multi-directory image lookup helper for export (`images_16bit_png`, `images_clahe`, `images_raw`, `patches_pool/images`).
- 2026-02-14 16:40 EST: Ran `python3 -m py_compile bubbly_flows/scripts/manage_bubbly.py` successfully.

## Messages
- 

