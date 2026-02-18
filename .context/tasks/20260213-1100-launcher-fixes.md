# Task 20260213-1100-launcher-fixes

## Summary
- investigate why clickable launchers open editor/empty terminal instead of running labeling flow
- harden Windows launcher runtime detection and add Linux-native click launcher
- align user docs with OS-specific launcher files

## Plan
- [x] inspect `launch_labeling.command`, `launch_labeling.bat`, and target shell launcher
- [x] patch Windows launcher to prefer `bash.exe` and avoid `sh` incompatibility
- [x] add Linux `launch_labeling.sh` and update user guide references

## Log
- 2026-02-13 10:58 EST: updated `launch_labeling.bat` to discover `bash.exe` via PATH/common Git install paths and run Bash script directly.
- 2026-02-13 10:59 EST: added executable `launch_labeling.sh` as Linux click launcher.
- 2026-02-13 11:00 EST: updated `USER_GUIDE.md` launcher instructions to separate macOS/Linux/Windows paths.
- 2026-02-13 11:10 EST: added root `launch_labeling.desktop` that resolves its own directory from `%k` and launches `bubbly_flows/scripts/xanylabel.sh`; switched Linux quick-start docs to `.desktop` first.
- 2026-02-13 11:31 EST: simplified `launch_labeling.desktop` to fixed repo `Path` + `Exec=/bin/bash ./launch_labeling.sh` (removed brittle escaped command); updated `launch_labeling.sh` to pause on non-zero exit so terminal stays open on errors.

## Messages
- 
