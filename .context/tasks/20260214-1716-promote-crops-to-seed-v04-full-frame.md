# Task 20260214-1716-promote-crops-to-seed-v04-full-frame

## Summary
- Promote confirmed human-edited crop annotations from `seed_v00` into full-frame labels in `seed_v04`.
- Copy corresponding uncropped images into `seed_v04/images`.
- Adapt crop coordinates to full-frame coordinates and write merged JSON labels to `seed_v04/labels`.

## Owner
- codex/20260214-1716

## Status
- done

## Plan
- [x] Confirm user-approved list of human-edited crop JSON files.
- [x] Copy 7 uncropped full-frame PNG files into `workspaces/seed_v04/images`.
- [x] Shift crop annotation points by crop offsets (`__x####_y####`) and write full-frame JSONs.
- [x] Merge both approved `0002` crops into one full-frame JSON.
- [x] Update `seed_v04/manifest.csv` to match current `seed_v04/images` files.
- [x] Validate output JSON metadata and coordinate ranges.

## Log
- 2026-02-14 17:10 EST: Confirmed selected crop list from user, excluding low-confidence `img018008__x0384_y0384`.
- 2026-02-14 17:14 EST: Copied full images from `data/frames/images_16bit_png` to `workspaces/seed_v04/images`.
- 2026-02-14 17:14 EST: Wrote adapted full-frame labels for `018008`, `018351`, `011890`, `1005070`, `1012062`, `1019655`; merged two approved crops for `1000002` into one label file.
- 2026-02-14 17:15 EST: Rewrote `workspaces/seed_v04/manifest.csv` from current image directory contents to remove stale crop-name entries.
- 2026-02-14 17:16 EST: Validated output JSONs: full `imagePath` values and coordinate ranges are within `1024x1024`.

## Messages
- 

