# Task 20260201-2200-blackhat-tiny-masks

## Summary
- add classical black-hat + threshold + connected-components tiny-mask branch to FRST+SAM3 pipeline
- expose tuning knobs in script/config and log counts

## Plan
- [x] add black-hat config defaults
- [x] add CLI flags + pipeline integration + helper functions
- [x] log output and verify wiring

## Log
- 2026-02-01 21:59 EST: added black-hat config defaults and pipeline integration in bubble_frst_sam3_mask.py; added watershed split support and CLI flags.

## Messages
- 

- 2026-02-01 22:04 EST: updated blackhat defaults (radius=5, percentile=99.0, area 5-120, watershed on). Attempted blackhat test on img6001.png but cv2 is not available in this environment.
- 2026-02-02 08:12 EST: added DoG/LoG blob detection on blackhat response with local patch threshold+CC masking; added config knobs for blob and patch params.
- 2026-02-02 08:24 EST: added standalone black-hat CLI (bubbly_flows/tests/blackhat_mask.py) and fixed Otsu threshold handling.
