# Hybrid Experiment Metadata Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lightweight curated JSON provenance for top-level hybrid experiment scripts and surface it in the generated experiment index.

**Architecture:** Keep filesystem discovery and human curation separate. `inventory.py` will continue to discover files into `experiment_registry.json`, while a new `experiment_metadata.json` will hold short, hand-maintained experiment notes for top-level entrypoints only. The Markdown index will merge both views so the team can see what exists and what each experiment was trying to test.

**Tech Stack:** Python 3, `json`, `pathlib`, `unittest`, Markdown

---

## Chunk 1: Minimal Metadata Layer

### Task 1: Add a failing metadata merge test

**Files:**
- Modify: `bubbly_flows/tests/unit/test_experiment_inventory.py`
- Test: `bubbly_flows/tests/unit/test_experiment_inventory.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds a small fake registry plus a matching curated metadata file and asserts the rendered Markdown includes the experiment question, components, and status for a top-level entrypoint.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory.ExperimentInventoryTests.test_render_markdown_includes_curated_experiment_metadata -v`
Expected: FAIL because the inventory code does not yet load or render curated metadata.

- [ ] **Step 3: Write minimal implementation**

Add metadata loading and merge helpers in `bubbly_flows/tests/inventory.py`, then update Markdown rendering to show a compact curated section under pipeline entrypoints when metadata exists.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory.ExperimentInventoryTests.test_render_markdown_includes_curated_experiment_metadata -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/unit/test_experiment_inventory.py bubbly_flows/tests/inventory.py

git commit -m "feat: merge curated hybrid experiment metadata"
```

### Task 2: Add the curated metadata JSON and regenerate the index

**Files:**
- Create: `bubbly_flows/tests/experiment_metadata.json`
- Modify: `bubbly_flows/tests/inventory.py`
- Modify: `bubbly_flows/tests/EXPERIMENT_INDEX.md`
- Modify: `bubbly_flows/tests/experiment_registry.json`

- [ ] **Step 1: Add minimal curated JSON entries**

Create metadata entries for top-level experiment scripts only, with these fields:
- `path`
- `label`
- `status`
- `components`
- `question`
- `outputs`
- `notes`

- [ ] **Step 2: Regenerate the inventory outputs**

Run: `python3 bubbly_flows/tests/inventory.py`
Expected: `experiment_registry.json` and `EXPERIMENT_INDEX.md` are rewritten with curated metadata rendered into the index.

- [ ] **Step 3: Run the full inventory test file**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: PASS.

- [ ] **Step 4: Append task log checkpoint**

Update `.context/tasks/20260310-1302-manage-bubbly-restructure.md` with a short checkpoint covering the new metadata file and regenerated index.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/experiment_metadata.json bubbly_flows/tests/inventory.py bubbly_flows/tests/EXPERIMENT_INDEX.md bubbly_flows/tests/experiment_registry.json .context/tasks/20260310-1302-manage-bubbly-restructure.md

git commit -m "docs: add minimal hybrid experiment metadata"
```
