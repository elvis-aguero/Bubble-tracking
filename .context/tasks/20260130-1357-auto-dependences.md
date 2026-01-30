# 20260130-1357-auto-dependences Enhance manage_bubbly.py to auto-install all dependencies

## Owner + Lease
- owner_session: Antigravity/20260130-1357
- lease_expires: 2026-01-30 14:57 EST

## Goal / Acceptance Criteria
- Modify `manage_bubbly.py` to check for ALL dependencies in `requirements.txt` (not just `torch`).
- Specifically ensure `micro_sam` is detected.
- If missing, prompt user to install via the existing self-repair mechanism.
- Fix logic to ensure new requirements are installed even if `numpy` is already present.

## Constraints / Non-goals
- We still respect the `bubbly-train-env`.

## Repo Touchpoints
- `bubbly_flows/scripts/manage_bubbly.py`

## Plan
1.  **Modify `check_training_reqs`** in `manage_bubbly.py `:
    - Instead of just checking `torch`, iterate through critical packages: `micro_sam`, `torch`, `cv2`, `tqdm`.
    - If ANY are missing, prompt for full reinstall/update from `requirements.txt`.
2.  **Startup Check**:
    - Ensure this check runs at startup or when entering training menus.

## Work Log
- [2026-01-30 13:57 EST] Task created.
- [2026-01-30 13:59 EST] Updated `check_training_reqs` to check `micro_sam`, `cv2`, and `tqdm`. If missing, it runs `pip install -r requirements.txt`.

## Messages
(None)

## Handoff
- Implementation complete. The script will now robustly handle missing `micro_sam`.
