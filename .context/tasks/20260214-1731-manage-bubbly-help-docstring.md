# Task 20260214-1731-manage-bubbly-help-docstring

## Summary
- Add a detailed module docstring to `bubbly_flows/scripts/manage_bubbly.py`.
- Ensure `python .../manage_bubbly.py --help` prints that usage text plus concrete example calls.

## Owner
- codex/20260214-1731

## Status
- done

## Plan
- [x] Inspect current script entry flow and confirm how `--help` is handled.
- [x] Add comprehensive usage-focused module docstring and example command block.
- [x] Add argparse-based CLI parsing that exits on `--help` before interactive bootstrap.
- [x] Validate syntax and verify help output.

## Log
- 2026-02-14 17:30 EST: Reviewed `manage_bubbly.py` top-level flow; confirmed no explicit CLI parser was wired.
- 2026-02-14 17:31 EST: Added expanded module docstring and `HELP_EXAMPLES` epilog content.
- 2026-02-14 17:31 EST: Added `parse_cli_args()` with `RawDescriptionHelpFormatter` and early parse call to support help-first behavior.
- 2026-02-14 17:31 EST: Validated with `python3 -m py_compile` and `python3 bubbly_flows/scripts/manage_bubbly.py --help`.

## Messages
- 
