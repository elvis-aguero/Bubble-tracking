# 20260130-1345-impl-train-inference Implement Real Training and Inference

## Owner + Lease
- owner_session: Antigravity/20260130-1345
- lease_expires: 2026-01-30 14:45 EST

## Goal / Acceptance Criteria
- Replace the placeholder simulation in `bubbly_flows/scripts/train.py` with actual `micro_sam` training logic.
- Implement a functional inference script (replacing the stub in `manage_bubbly.py`).
- Ensure the system can actually save a model and use it to predict masks on new images.

## Constraints / Non-goals
- We assume `micro_sam` is available or will be installed.
- We will not rewrite the entire dataset management system, only the execution endpoints for train/eval.

## Repo Touchpoints
- `bubbly_flows/scripts/train.py` (currently a simulation)
- `bubbly_flows/scripts/manage_bubbly.py` (contains stubs for inference)
- `bubbly_flows/scripts/inference.py` (new file likely needed)

## Plan
1.  **Analyze `train.py`**: replace the `time.sleep` loop with `micro_sam.training.train_sam_for_instance_segmentation` (or equivalent).
2.  **Create `inference.py`**: implement `run_inference` to load a saved checkpoint and process an image.
3.  **Update `manage_bubbly.py`**: Wire up option 6 to call `inference.py`.
4.  **Verify Dependencies**: Check `requirements.txt` or environment setup.

## Work Log
- [2026-01-30 13:45 EST] Task created.

## Messages
(None)

## Handoff
- Task initialized. Ready to start implementation of `train.py` logic.
