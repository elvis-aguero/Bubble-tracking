# Task 20260310-1302-manage-bubbly-restructure

## Summary
- Continue the `manage_bubbly.py` pipeline restructure from the March 9 design/plan docs.
- First establish the exact implementation slice and design approval.
- Then execute the approved slice with disciplined workflow and verification.

## Owner
- codex/20260310-1302

## Status
- open

## Plan
- [ ] Review current repo state against `docs/plans/2026-03-09-pipeline-restructure-*.md`.
- [ ] Brainstorm and confirm the next implementation slice with the user.
- [ ] Implement the approved slice with tests first.
- [ ] Verify behavior and update task log.

## Log
- 2026-03-10 13:02 EST: Created task after reviewing protocol, KB, plan/design docs, and current `manage_bubbly.py` state.
- 2026-03-10 13:02 EST: Observed partial refactor state: `configs/` exists, `pipeline/` rename is present, `train.py` uses `--config`, but `manage_bubbly.py` still uses the legacy 7-option menu and training/evaluation flow is incomplete.
- 2026-03-10 13:02 EST: Observed dirty worktree on `main` with unrelated user-owned changes in `bubbly_flows/scripts/train.py`, `bubbly_flows/scripts/evaluate.py`, plus untracked logs/outputs; these must remain untouched.

## Messages
- 
