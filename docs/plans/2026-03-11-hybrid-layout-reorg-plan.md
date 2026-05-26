# Hybrid Layout Reorg Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the active hybrid research code under `bubbly_flows/tests/` into a cleaner `src/`-based layout without changing algorithm behavior.

**Architecture:** Move top-level hybrid research scripts into category-based folders under `bubbly_flows/tests/src/`, move the shared `bubble_sam3` package under `src/common/`, and update imports plus the experiment inventory scanner. Keep outputs, logs, and sample data in place for provenance.

**Tech Stack:** Python 3.11, `unittest`, filesystem reorganization, import path updates.

---

## File Map

- Move: `bubbly_flows/tests/bubble_frst_sam3_mask.py` -> `bubbly_flows/tests/src/hybrid/bubble_frst_sam3_mask.py`
- Move: `bubbly_flows/tests/bubble_sam3_mask.py` -> `bubbly_flows/tests/src/sam3/bubble_sam3_mask.py`
- Move: `bubbly_flows/tests/blackhat_mask.py` -> `bubbly_flows/tests/src/deterministic/blackhat_mask.py`
- Move: `bubbly_flows/tests/classical_test.py` -> `bubbly_flows/tests/src/deterministic/classical_test.py`
- Move: `bubbly_flows/tests/detect_bubbles.py` -> `bubbly_flows/tests/src/deterministic/detect_bubbles.py`
- Move: `bubbly_flows/tests/big_bubble_prompt_fb.py` -> `bubbly_flows/tests/src/prompting/big_bubble_prompt_fb.py`
- Move: `bubbly_flows/tests/big_bubble_prompt_hf.py` -> `bubbly_flows/tests/src/prompting/big_bubble_prompt_hf.py`
- Move: `bubbly_flows/tests/frst_point_backend_fb.py` -> `bubbly_flows/tests/src/backends/frst_point_backend_fb.py`
- Move: `bubbly_flows/tests/frst_point_backend_hf.py` -> `bubbly_flows/tests/src/backends/frst_point_backend_hf.py`
- Move: `bubbly_flows/tests/bubble_sam3/` -> `bubbly_flows/tests/src/common/bubble_sam3/`
- Modify: `bubbly_flows/tests/inventory.py`
- Modify: `bubbly_flows/tests/unit/test_experiment_inventory.py`
- Regenerate: `bubbly_flows/tests/EXPERIMENT_INDEX.md`
- Regenerate: `bubbly_flows/tests/experiment_registry.json`
- Update: `.context/tasks/20260310-1302-manage-bubbly-restructure.md`

## Chunk 1: Lock New Layout With Tests

### Task 1: Define the new category-based layout in tests

**Files:**
- Modify: `bubbly_flows/tests/unit/test_experiment_inventory.py`
- Modify: `bubbly_flows/tests/inventory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_registry_detects_src_category_entrypoints(self):
    ...


def test_build_registry_detects_common_modules_under_src(self):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: FAIL because the inventory scanner still assumes the old flat layout.

- [ ] **Step 3: Write minimal implementation**

Teach the inventory scanner to classify `tests/src/<category>/...` as the new source of truth.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/unit/test_experiment_inventory.py bubbly_flows/tests/inventory.py
git commit -m "test: lock hybrid src layout in inventory"
```

## Chunk 2: Move The Active Research Code

### Task 2: Move category-based entrypoints and shared modules

**Files:**
- Move files listed in File Map

- [ ] **Step 1: Move deterministic, prompting, backend, sam3, and hybrid entrypoints**
- [ ] **Step 2: Move the shared `bubble_sam3` package under `src/common/`**
- [ ] **Step 3: Update imports minimally**

Examples:
```python
from bubbly_flows.tests.src.common.bubble_sam3.backend import Sam3ConceptBackend
from bubbly_flows.tests.src.backends.frst_point_backend_fb import FrstPointBackendFB
from bubbly_flows.tests.src.deterministic.classical_test import frst_symmetry_map
```

- [ ] **Step 4: Run targeted syntax/import verification**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/src bubbly_flows/tests/inventory.py bubbly_flows/tests/unit/test_experiment_inventory.py
git commit -m "refactor: reorganize hybrid research code under tests/src"
```

## Chunk 3: Regenerate Inventory And Record Checkpoint

### Task 3: Refresh inventory artifacts and task log

**Files:**
- Regenerate: `bubbly_flows/tests/EXPERIMENT_INDEX.md`
- Regenerate: `bubbly_flows/tests/experiment_registry.json`
- Update: `.context/tasks/20260310-1302-manage-bubbly-restructure.md`

- [ ] **Step 1: Regenerate the inventory**

Run: `python3 bubbly_flows/tests/inventory.py`
Expected: updated paths point into `tests/src/...`

- [ ] **Step 2: Inspect the rendered index**

Run: `sed -n '1,240p' bubbly_flows/tests/EXPERIMENT_INDEX.md`
Expected: clear category-based listing.

- [ ] **Step 3: Record task log update**

Append that the active hybrid code is now organized under `bubbly_flows/tests/src/`.

- [ ] **Step 4: Commit**

```bash
git add bubbly_flows/tests/EXPERIMENT_INDEX.md bubbly_flows/tests/experiment_registry.json .context/tasks/20260310-1302-manage-bubbly-restructure.md docs/plans/2026-03-11-hybrid-layout-reorg-plan.md
git commit -m "chore: checkpoint hybrid layout reorg"
```
