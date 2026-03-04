# Knowledge Base (Stable Facts Only)
- Repo map (key dirs/modules):
  - `claude/lit_review/` — literature review outputs (references.csv, lit_review.md)
  - `~/Zotero/storage/` — user's Zotero PDF library (hashed subdirs)
- Build/test commands:
- Style/conventions:
- Dependency/tooling notes:
  - Zotero PDF storage: `~/Zotero/storage/<hash>/<author - year - title>.pdf`
- Known pitfalls:
- Project goal: Instance segmentation of bubbles in zero-gravity (microgravity) parabolic flight footage. Downstream use: bubble size distribution. False merges of touching bubbles are the worst failure mode.
- Data: gold_seed_v00 (124 annotated 384×384 patches, void fraction 0.3–16.3% mean 7.3%), seed_v04 (14 full-frame 1024×1024, annotated). Source is continuous video but annotated snapshots are temporally independent — treat each as independent sample. SAM2 temporal propagation is viable for future annotation batches only.
- Candidate models (ranked by effort): MicroSAM (Plan A, live), StarDist+RODARE (Plan B), BubMask/Mask R-CNN (Plan C).
- Key findings (lit review 2026-03-02): StarDist and Mask R-CNN are dominant architectures. RODARE dataset public at rodare.hzdr.de (~800 images). All methods degrade at void fraction >10-16%.
- Roadmap document: claude/hypotheses.md (6 hypotheses, Plan A/B/C, open questions, decision log).
(Keep short. Only stable, repo-wide truths.)
