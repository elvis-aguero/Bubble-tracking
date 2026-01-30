# 20260130-1357-auto-dependences Enhance manage_bubbly.py to auto-install all dependencies

## Owner + Lease
- owner_session: Antigravity/20260130-1357
- lease_expires: 2026-01-30 14:57 EST

## Goal / Acceptance Criteria
- Modify `manage_bubbly.py` to check for ALL dependencies in `requirements.txt` (not just `torch`).
- Specifically ensure `micro_sam` is detected.
- If missing, prompt user to install via the existing self-repair mechanism.
- Fix logic to ensure new requirements are installed even if `numpy` is already present.
- **Correction**: Run this check on STARTUP, not just before training.

## Constraints / Non-goals
- We still respect the `bubbly-train-env`.

## Repo Touchpoints
- `bubbly_flows/scripts/manage_bubbly.py`

## Plan
1.  **Modify `check_training_reqs`** in `manage_bubbly.py `: (Done)
2.  **Startup Check**: Move the call to `check_training_reqs()` to `if __name__ == "__main__":` before `main_menu()`.

## Work Log
- [2026-01-30 13:57 EST] Task created.
- [2026-01-30 13:59 EST] Updated `check_training_reqs`.
- [2026-01-30 14:01 EST] Moving check to startup.

## Messages
(None)

## Handoff
- Moving check to main entry point.
