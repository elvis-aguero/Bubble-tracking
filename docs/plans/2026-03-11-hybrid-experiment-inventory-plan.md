# Hybrid Experiment Inventory Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean, up-to-date inventory of the hybrid bubble-identification research workspace under `bubbly_flows/tests/`, covering what pipelines exist and what outputs/logs they produced.

**Architecture:** Keep the hybrid code where it is for now, but add one small scanner that builds a machine-readable manifest from `bubbly_flows/tests/`, and one human-readable Markdown index for the team. The inventory should distinguish pipeline scripts, helper modules, sample inputs, outputs, and logs without moving code yet.

**Tech Stack:** Python 3.11, `unittest`, Markdown, JSON.

---

## File Map

- Create: `bubbly_flows/tests/inventory.py`
  - Small scanner that classifies hybrid research artifacts.
- Create: `bubbly_flows/tests/EXPERIMENT_INDEX.md`
  - Human-readable inventory of tested models/pipelines and known outputs.
- Create: `bubbly_flows/tests/experiment_registry.json`
  - Machine-readable manifest produced by the scanner.
- Create: `bubbly_flows/tests/unit/test_experiment_inventory.py`
  - Tests for scanner classification and manifest structure.
- Update: `.context/tasks/20260310-1302-manage-bubbly-restructure.md`
  - Append a handoff note that the next feature established the hybrid experiment inventory.

## Chunk 1: Scanner And Tests

### Task 1: Define manifest classification in tests

**Files:**
- Create: `bubbly_flows/tests/unit/test_experiment_inventory.py`
- Create: `bubbly_flows/tests/inventory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_registry_groups_hybrid_entrypoints_outputs_and_logs():
    ...


def test_build_registry_detects_known_pipeline_families():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: FAIL because the inventory module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement a scanner that returns a dict with sections like:
- `pipeline_entrypoints`
- `helper_modules`
- `sample_inputs`
- `outputs`
- `logs`
- `notes`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/inventory.py bubbly_flows/tests/unit/test_experiment_inventory.py
git commit -m "feat: add hybrid experiment inventory scanner"
```

## Chunk 2: Inventory Artifacts

### Task 2: Generate and save the current experiment inventory

**Files:**
- Create: `bubbly_flows/tests/EXPERIMENT_INDEX.md`
- Create: `bubbly_flows/tests/experiment_registry.json`
- Modify: `bubbly_flows/tests/inventory.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_markdown_lists_known_hybrid_scripts_and_outputs():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: FAIL because markdown rendering/export does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add export helpers that:
- write `experiment_registry.json`
- render `EXPERIMENT_INDEX.md`
- include current known hybrid families and artifact locations

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/inventory.py bubbly_flows/tests/EXPERIMENT_INDEX.md bubbly_flows/tests/experiment_registry.json bubbly_flows/tests/unit/test_experiment_inventory.py
git commit -m "docs: add hybrid experiment inventory"
```

## Chunk 3: Verification And Task Log

### Task 3: Verify and record the checkpoint

**Files:**
- Update: `.context/tasks/20260310-1302-manage-bubbly-restructure.md`

- [ ] **Step 1: Run focused verification**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`
Expected: PASS.

- [ ] **Step 2: Inspect the generated artifacts**

Run: `sed -n '1,220p' bubbly_flows/tests/EXPERIMENT_INDEX.md`
Expected: clear listing of tested pipelines, outputs, and logs.

- [ ] **Step 3: Append task log update**

Record that the hybrid experiment inventory is now the source of truth for future cleanup/reorg work.

- [ ] **Step 4: Commit**

```bash
git add .context/tasks/20260310-1302-manage-bubbly-restructure.md docs/plans/2026-03-11-hybrid-experiment-inventory-plan.md
git commit -m "chore: checkpoint hybrid experiment inventory"
```
