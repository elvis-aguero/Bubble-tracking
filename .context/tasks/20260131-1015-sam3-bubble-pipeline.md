# Task: SAM3 bubble segmentation pipeline script

- status: done
- owner: codex/20260131-1015
- branch: main
- created: 2026-01-31

## Plan
- (done) Inspect existing SAM3 usage and data samples
- (done) Design configurable pipeline structure and defaults
- (done) Implement script in tests/ with CLI+config+debug
- (done) Add debug outputs and finalize script

## Log
- 2026-01-31: Task created and claimed by codex/20260131-1015.
- 2026-01-31: Reviewed sam3 usage in bubbly_flows/tests/test.py; inspected sample images in bubbly_flows/data/frames/images_clahe (1024x1024 grayscale PNGs).
- 2026-01-31: Added tests/bubble_sam3_mask.py implementing SAM3 segmentation pipeline with tiling, candidates, consolidation, and RGBA output.

## Messages
- None
