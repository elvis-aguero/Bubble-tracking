# Hybrid Experiment Status And Next Directions

## Current Status

- [x] Hybrid research code reorganized under `bubbly_flows/tests/src/`
- [x] Experiment inventory created under `bubbly_flows/tests/`
- [x] Minimal curated experiment metadata added
- [x] Provenance-first experiment harness created under `bubbly_flows/tests/src/experiments/`
- [x] Gold evaluation-set preparation implemented from `annotations/gold/*/labels_json`
- [x] Run-level provenance outputs implemented:
  - [x] `manifest.json`
  - [x] `aggregate_metrics.json`
  - [x] `per_image_metrics.csv`
  - [x] `gallery.md`
  - [x] `ranking.csv`
- [x] Initial experiment families wired into the harness:
  - [x] `frst_only`
  - [x] `blackhat_only`
  - [x] `hybrid_current`
- [x] Existing `evaluate.py` reused as the metric backend
- [x] Lightweight unit-test coverage added for:
  - [x] gold-eval-set prep
  - [x] provenance writing
  - [x] batch ranking
  - [x] search-space generation
  - [x] family-specific command construction
  - [x] evaluation CSV parsing

## Validated So Far

- [x] The harness can describe bounded experiment batches with saved provenance
- [x] The harness can materialize family-specific commands
- [x] The harness can convert prediction JSON into instance-mask files for evaluation
- [x] The harness can rank finished runs by aggregate F1
- [x] A real end-to-end batch has been executed on actual images for `blackhat_only`
- [x] A real end-to-end batch has succeeded for `frst_only`
- [x] A real end-to-end batch has succeeded for `hybrid_current`

## Important Current Limits

- [x] `frst_only` is currently realized by disabling blackhat and PCS within the existing hybrid driver
- [x] `blackhat_only` is currently realized by disabling candidate-driven FRST prompting and PCS while keeping blackhat active
- [x] `hybrid_current` uses the current combined pipeline behavior
- [x] `frst_only` and `hybrid_current` are not reproducible from `bubbly-train-env` alone; they require the separate user-scoped SAM3 runtime used by the hybrid testbed
  - [x] the production environment still lacks the SAM3-side runtime stack
  - [x] the experiment runner now uses the user-scoped SAM3 interpreter for SAM3-backed families
- [ ] `branch_priority` fusion is not yet implemented as distinct algorithm logic
- [ ] Search-stage orchestration is still manual; there is not yet a top-level benchmark CLI
- [ ] The harness has not yet been widened to additional families such as standalone SAM3 or prompting-only variants

## Immediate Next Steps

- [x] Run a small real benchmark batch on 2 gold images to validate the full execution path
- [x] Expose a working SAM3-backed runtime to the experiment runner via the user-scoped SAM3 environment
- [x] Re-run the 2-image validation for:
  - [x] `frst_only`
  - [x] `hybrid_current`
- [ ] Run the first full 14-image baseline comparison for:
  - [ ] `frst_only`
  - [ ] `blackhat_only`
  - [ ] `hybrid_current`
- [ ] Inspect saved overlays and per-image metrics to identify dominant failure modes
- [ ] Decide whether the next algorithmic step is:
  - [ ] true `branch_priority` fusion
  - [ ] stronger conservative dedup / containment behavior
  - [ ] a new family outside the current hybrid driver

## Future Directions

- [ ] Implement true source-aware fusion variants instead of only threshold-based variants
- [ ] Add a benchmark entrypoint script under `bubbly_flows/tests/` for repeatable experiment launches
- [ ] Add adaptive second-stage search on the best-performing family
- [ ] Add richer failure analysis outputs:
  - [ ] duplicate-detection diagnostics
  - [ ] small-bubble recall summaries
  - [ ] large-bubble recall summaries
- [ ] Add optional human-review overlays or galleries optimized for fast side-by-side inspection

## Latest Validation Snapshot

- [x] `blackhat_only` 2-image Slurm validation completed
  - [x] Job: `698942`
  - [x] Outputs written under `bubbly_flows/tests/output/experiments/validate_2img_blackhat_only/`
  - [x] Saved overlays, LabelMe JSON, prediction masks, manifest, gallery, metrics, and ranking
  - [x] Aggregate result on the 2-image slice:
    - [x] `precision = 0.0`
    - [x] `recall = 0.0`
    - [x] `F1 = 0.0`
- [x] Attempted 3-family 2-image validation via Slurm
  - [x] the first attempt exposed environment/runtime drift in the SAM3-backed paths
  - [x] the successful rerun completed after switching SAM3-backed families to the user-scoped SAM3 runtime and replacing the failing facebookresearch big-bubble path with the HF big backend in the experiment runner
- [x] Successful 3-family 2-image validation via Slurm
  - [x] job `700150`
  - [x] `frst_only`: `F1 = 0.003`
  - [x] `hybrid_current`: `F1 = 0.003`
  - [x] `blackhat_only`: `F1 = 0.000`
