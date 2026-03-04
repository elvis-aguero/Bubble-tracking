# ML Hypotheses & Research Roadmap
## Bubble Instance Segmentation — Zero-Gravity Bubbly Flow

*Created 2026-03-02. Update status fields and evidence notes in-place as results arrive. This document is intended to be self-contained: a new reader should be able to follow the reasoning without prior context.*

---

## Table of Contents

1. [Project Synopsis](#1-project-synopsis)
2. [Core Hypotheses](#2-core-hypotheses)
3. [Plan A / Plan B / Plan C Roadmap](#3-plan-a--plan-b--plan-c-roadmap)
4. [Open Questions](#4-open-questions)
5. [Decision Log](#5-decision-log)

---

## 1. Project Synopsis

### 1.1 Physical context

**Bubbly flow** is a multiphase flow regime in which gas bubbles are suspended and transported within a liquid. It appears in many engineering applications: chemical reactors, bioreactors, boiling heat exchangers, and — in this project — fluid management experiments under spaceflight conditions.

**Void fraction** (also called gas holdup) is the fraction of the total volume (or image area) occupied by the gas bubbles. It is *not* empty space — it is gas space. A void fraction of 10% means 10% of the fluid volume is occupied by bubbles; 90% is liquid. In the images in this dataset, void fraction is measured as the fraction of the in-view chamber area covered by annotated bubble masks. Our dataset spans 0.3%–16.3%, mean 7.3%.

**Zero-gravity (microgravity) conditions.** On Earth, buoyancy dominates bubble dynamics: light gas bubbles rise, deform into oblate caps, and cluster near the top of a vessel. In microgravity (aboard a parabolic flight aircraft or the International Space Station), buoyancy effectively vanishes. Without a preferred rise direction, bubbles:
- Stay more spherical (surface tension is the dominant shape force)
- Distribute more uniformly in space
- Move in response to inertia, drag, and surface tension rather than gravity

**ZeroG FlightDay data.** The footage in this project was recorded during parabolic flight campaigns — a standard technique to produce ~20 seconds of microgravity per parabola by flying an aircraft in a ballistic arc. The experimental setup is an octagonal transparent chamber filled with liquid, viewed through a window by a high-speed camera. Bubbles are introduced into the liquid and their behavior under microgravity is recorded.

**Why this matters for ML.** No published ML model for bubble segmentation has been validated on microgravity data. All 9 papers in our literature review (2016–2025) use Earth-gravity air-water systems. This means:
- We cannot directly trust published accuracy numbers to generalize to our data.
- Pre-trained models from Earth-gravity data may or may not transfer well (this is an open hypothesis).
- The sphericity of microgravity bubbles is a potential *advantage* for shape-prior-based models (see H4).

---

### 1.2 Task definition: instance segmentation

**Semantic segmentation** labels every pixel in an image as belonging to one of N classes (e.g., "gas" or "liquid"). It produces a single binary mask for the entire image. This is sufficient to measure total void fraction (gas area / total area) but cannot distinguish individual bubbles.

**Instance segmentation** assigns a separate, individually identified mask to each object instance. For bubble images, this means: bubble #1 gets mask #1, bubble #2 gets mask #2, and so on — even if they are touching or overlapping. This is necessary to measure per-bubble properties: equivalent diameter, aspect ratio, and shape.

**We need instance segmentation** because the primary downstream goal is **bubble size distribution** — the statistical distribution of individual bubble equivalent diameters. This is used to characterize the flow regime, validate CFD simulations, and compute mass transfer rates.

**The critical failure mode: false merges of touching bubbles.** When two bubbles touch, their boundaries appear merged in the image. If the model assigns a single mask to both, it produces one large "bubble" where there are actually two smaller ones. This distorts the size distribution — particularly its right tail — which is scientifically unacceptable. False merges of touching bubbles are therefore the worst failure mode in this project, more damaging than missed small bubbles.

**Annotation methodology.** Human annotators used **X-AnyLabeling** with **SAM2.1 human-in-the-loop**: the AI model proposes a mask for each bubble, the human corrects it if needed, and the corrected mask is saved. Touching bubbles were annotated as separate instances (the human corrected the SAM boundary to split them). Annotations are stored as **LabelMe polygon JSON** files — each shape entry is a list of `[x, y]` coordinates tracing one bubble's boundary.

---

### 1.3 Data assets

| Tier | Name | Format | Count | Notes |
|------|------|--------|-------|-------|
| Annotated patches | `gold_seed_v00` | LabelMe polygon JSON, 384×384 px | 124 labeled images | Tiles cropped from full frames; void fraction 0.3–16.3%, mean 7.3%; used for training |
| Full-frame workspace | `seed_v04` | LabelMe polygon JSON, 1024×1024 px | 14 full frames | 4–641 bubbles per image; annotated but not yet in gold (not yet used for training) |
| Augmented export | `dryrun_aug_20260214_1835` | MicroSAM-ready images + TIF masks | 496 images | 4× augmentation of `gold_seed_v00`: geometric (flip/rot/affine) + photometric (brightness/contrast/gamma/noise) + copy-paste |
| Raw video | ZeroG FlightDay footage | Continuous high-speed video | 33 source sequences | Full parabolic flight recordings; frames not yet tiled or annotated beyond the above |

**Where the data lives:**
- `bubbly_flows/annotations/gold/gold_seed_v00/labels_json/` — gold annotation JSONs
- `bubbly_flows/workspaces/seed_v04/` — full-frame images and labels
- `bubbly_flows/microsam/datasets/dryrun_aug_20260214_1835/` — augmented MicroSAM export
- `bubbly_flows/data/frames/` — raw tif frames (33 sequences)

**Important:** The source data is **continuous video** (frames are sequential within each sequence). This means temporal propagation (SAM2 video mode) is a viable strategy to cheaply multiply labeled data: annotate 1–2 keyframes per sequence, propagate to neighboring frames, human-review the propagated masks.

---

### 1.4 Candidate model landscape

Three models are under consideration, all providing instance-level segmentation masks, all with public code:

**MicroSAM** ([github.com/computational-cell-analytics/micro-sam](https://github.com/computational-cell-analytics/micro-sam))
- Based on Meta's Segment Anything Model (SAM), fine-tuned on scientific microscopy images (cells, organelles, tissues).
- Advantage: general-purpose foundation model; handles diverse object types; already fully integrated in this codebase (train.py, inference.py, Slurm submission via manage_bubbly.py).
- Disadvantage: not specifically trained on bubbly flows; may need substantial domain adaptation.

**StarDist** ([github.com/stardist/stardist](https://github.com/stardist/stardist))
- Specialized for instance segmentation of round and star-convex objects (originally developed for cell nuclei detection in biomedical microscopy).
- Represents each object as a set of radial distances from its center to its boundary at fixed angles — a "star-shaped" polygon. Objects that are not star-convex (e.g., highly elongated or concave shapes) are approximated.
- Advantage: strong inductive prior for nearly-spherical bubbles; well validated on bubbly flow data (Hessenkemper et al. 2022: AP@0.5 = 0.91); public dataset available for pre-training (RODARE).
- Disadvantage: star-convex assumption breaks down for very elongated or touching bubbles where the merge boundary is complex.

**Mask R-CNN / BubMask** ([github.com/ywflow/BubMask](https://github.com/ywflow/BubMask))
- Region-proposal-based instance segmentation (Faster R-CNN backbone → bounding box proposals → per-box mask prediction). Adapted for bubbles by Kim & Park (2021) with a size-weighted loss function to handle small bubbles and BubGAN synthetic data augmentation.
- Advantage: most flexible (no shape prior); highest reported AP50 (0.981, Kim & Park 2021); handles non-convex shapes; COCO pre-training provides broad visual feature knowledge.
- Disadvantage: most data-hungry; bounding-box anchors can merge nearby bubbles at the proposal stage; no shape regularization may hurt on small training sets.

**Excluded options and rationale:**
- **YOLO-based** (Yang et al. 2025, Nizovtseva et al. 2024): optimized for real-time speed (~20 fps). We use HPC Slurm jobs and prioritize accuracy — no need for real-time inference.
- **U-Net**: semantic segmentation only (single gas/liquid mask, no per-bubble identity). Insufficient for bubble size distribution.
- **Custom geometry-specific CNNs** (Cerqueira & Paladino 2021): no public code; designed for pipe geometry, not our octagonal chamber.
- **Traditional CV** (Fu & Liu 2016): no ML, requires manual parameter tuning per setup, degrades above 10% void fraction.

**Pre-training resource:**
- **RODARE dataset** ([rodare.hzdr.de](https://rodare.hzdr.de)): public air-water bubbly flow dataset from Helmholtz-Zentrum Dresden-Rossendorf (HZDR). ~800 annotated images, ~24,400 individual bubble instances. Earth-gravity, rectangular column, backlit high-speed video, void fraction 0.5–5%, bubble diameter 1–10 mm. This is the best publicly available bubbly flow training set and is the natural pre-training source for Plans B and C.

---

### 1.5 Evaluation metric

**AP@IoU[0.5:0.95]** (COCO-style mean Average Precision) is the primary metric.

- **IoU** (Intersection over Union): the ratio of the overlap area between predicted and ground-truth masks to their union. IoU = 1.0 means perfect mask match; IoU = 0.5 means the predicted mask overlaps half of the true mask area.
- **AP@IoU threshold T**: fraction of predicted bubbles that match a ground-truth bubble with IoU ≥ T, averaged over all recall levels.
- **AP@0.5**: lenient — any mask that overlaps ≥ 50% of a true bubble counts as correct.
- **AP@0.75**: strict — requires ≥ 75% overlap. More sensitive to boundary precision.
- **AP@[0.5:0.95]**: COCO standard — average of AP at IoU thresholds 0.5, 0.55, 0.60, ..., 0.95. This is the most informative single number because it rewards accurate boundaries (relevant for size distribution) not just rough detections.

**Why AP@[0.5:0.95] matters here:** Bubble equivalent diameter is computed from mask area (d_eq = 2√(A/π)). A mask that is 50% wrong in area gives a diameter error of ~30%. Tight mask boundaries (IoU > 0.75) are needed for reliable diameter estimates. AP@0.5 alone is insufficient.

**Additional metric: touching-bubble F1.** We will separately evaluate F1 on pairs of bubbles that are touching or nearly touching in the ground truth — since false merges on touching bubbles are the primary downstream failure mode.

---

## 2. Core Hypotheses

Each hypothesis is written as a falsifiable claim with numeric thresholds. Status is updated in place as experiments are completed.

---

### H1: MicroSAM fine-tuning on ~124 patches is sufficient for our distribution

**Statement:** Fine-tuning MicroSAM ViT-B on `gold_seed_v00` (124 images, 4× augmented to 496 samples) will produce AP@0.5 ≥ 0.70 on a held-out zero-G test set.

**Supporting evidence:**
- MicroSAM is pre-trained on diverse scientific microscopy images. Bubbles (circular blobs on a dark background with bright-halo or dark-interior appearance) share low-level edge characteristics with the cell/organelle images SAM was adapted for.
- The biomedical segmentation literature routinely fine-tunes SAM-based models on 50–200 images with augmentation and achieves usable accuracy (>0.70 AP) because the pre-trained encoder needs only modest adaptation.
- Cui et al. (2022) achieved mAP > 0.95 with Mask R-CNN on only 100 training images — a comparable dataset size — though in a different domain.
- The augmented export (`dryrun_aug_20260214_1835`) already provides 496 effective training samples with diverse photometric and geometric variations.

**Stress test / falsification criteria:**
- AP@0.5 < 0.60 after 100 epochs with augmentation → H1 rejected; data or architecture is insufficient.
- Training loss plateaus by epoch 10–15 with no validation improvement → insufficient data diversity, not model capacity.
- Gap between MicroSAM zero-shot AP (no fine-tuning) and fine-tuned AP < 0.05 → fine-tuning adds little; more data or a different model is needed.

**Status:** Untested

---

### H2: Transfer learning from Earth-gravity RODARE data improves over zero-G data alone

**Statement:** Pre-training on RODARE (~800 images, Hessenkemper 2022/2024) then fine-tuning on `gold_seed_v00` will yield AP@0.5 at least 0.05 higher than fine-tuning on `gold_seed_v00` alone, when evaluated on the zero-G test set.

**Supporting evidence:**
- Transfer learning from closer-domain checkpoints (bubbly flow, same optical contrast, similar bubble sizes) consistently reduces the data requirement for acceptable fine-tuning performance across biomedical and scientific imaging literature.
- RODARE bubble size range (1–10 mm) and imaging conditions (backlit, high-speed video, rectangular column) overlap with our data in key low-level visual features.
- StarDist trained on RODARE achieved AP@0.5 = 0.91 on RODARE data, showing that RODARE contains sufficient diversity for the model to learn generalizable bubble boundary features.

**Stress test / falsification criteria:**
- If RODARE pre-training → zero-G fine-tuning yields AP within ±0.03 of zero-G-only fine-tuning → RODARE pre-training adds no value; drop from pipeline.
- If RODARE pre-training *hurts* AP → distribution shift is damaging (Earth vs. microgravity bubble shape difference is a meaningful domain gap).
- Qualitative check: if the RODARE-pre-trained model systematically misses elongated or unusual-shaped bubbles specific to our data, this is evidence of a morphological domain gap rather than a feature-level gap.

**Status:** Untested

---

### H3: Local bubble appearance is transferable across gravity regimes

**Statement:** A model trained only on Earth-gravity RODARE data will detect > 60% of zero-G bubbles (recall > 0.60) without any fine-tuning on zero-G data — indicating that local bubble boundary features transfer across gravity regimes.

**Supporting evidence:**
- The optical appearance of a gas-liquid bubble boundary is governed by refraction and total internal reflection at the interface, which depend on the refractive indices of the fluids and the illumination geometry — not on gravity. A bubble boundary looks like a bubble boundary regardless of gravity level.
- Both RODARE and our dataset use backlit high-speed video of air-water systems in transparent chambers viewed from the side. The illumination geometry is similar.
- SAM and MicroSAM encode local patch features (via ViT self-attention at patch scale), not global flow statistics. Local bubble appearance is likely similar enough for feature transfer.

**Stress test / falsification criteria:**
- Run RODARE-trained StarDist (or BubMask) zero-shot on 2–3 held-out zero-G test images and compute recall. Recall < 0.50 → fundamental domain gap exists; transfer learning alone is insufficient and we may need to train from scratch.
- If false negatives cluster at high void fraction patches (>10%) but recall is good at low void fractions → density effect (H6) is the issue, not a gravity-regime domain gap.
- Visual comparison: sample 20 random 64×64 px crops of individual bubbles from zero-G data and from RODARE — if the appearance is visually indistinguishable to a human, H3 is plausible.

**Status:** Untested

---

### H4: StarDist's star-convex prior is better suited to microgravity bubbles than Mask R-CNN's bounding-box prior

**Statement:** On our zero-G test set, StarDist (fine-tuned) achieves AP@IoU[0.5:0.95] at least 0.05 higher than BubMask (Mask R-CNN, fine-tuned with equivalent training data), primarily due to better boundary accuracy on small and near-circular bubbles.

**Supporting evidence:**
- In microgravity, without buoyancy deformation, bubbles are expected to be more spherical in 3D and more circular in 2D projection than Earth-gravity bubbles. The star-convex prior is ideally matched to circular objects and acts as shape regularization — especially valuable for small training datasets.
- Hessenkemper et al. (2022) showed StarDist + UNet post-processing outperformed Mask R-CNN on Earth-gravity air-water data (AP@0.5 = 0.91 vs. lower). If microgravity bubbles are even more circular, the advantage of the star-convex prior should be equal or larger.
- Mask R-CNN's bounding-box region proposals can merge or split nearby bubbles at the anchor stage, before the mask head sees the image region. For densely packed bubble clouds, this is a documented failure mode.
- StarDist was designed for densely packed, nearly round instances in scientific microscopy — structurally matching our problem.

**Stress test / falsification criteria:**
- If > 20% of bubbles in `gold_seed_v00` have aspect ratio > 1.5 (significantly non-circular), the star-convex prior is too restrictive → compute eccentricity distribution from annotation JSONs before committing to Plan B.
- If touching / overlapping bubble failures dominate both StarDist and Mask R-CNN equally → the bottleneck is occlusion (H6), not architecture.
- If Mask R-CNN outperforms StarDist at high void fractions (>10%) but not at low void fractions (<5%) → Mask R-CNN's anchor diversity handles occlusion better in the dense regime — a conditional result worth capturing.

**Status:** Untested

---

### H5: 124 labeled patches is insufficient; we need 300–500 more for robust generalization

**Statement:** Adding 300–400 more labeled patches (from the existing unlabeled pool or from `seed_v04` full frames) will improve AP@IoU[0.5:0.95] by at least 0.08 compared to training on `gold_seed_v00` alone (with augmentation), indicating data quantity is the primary bottleneck.

**Supporting evidence:**
- The bubbly flow literature trains on 800–2,592 images. Even Cui et al. (2022) — the smallest dataset at 100 images — notes performance degradation at high void fractions, which is a data coverage problem (not enough dense-scene training examples).
- Our copy-paste augmentation is limited by the real bubble pool: with mean void fraction 7.3% across 124 patches (mostly 384×384), the number of unique bubble instances available for copy-paste is modest. Dense scenes (>10% void fraction) are underrepresented.
- The 14 full-frame `seed_v04` images (1024×1024 px, 4–641 bubbles each, already annotated) represent disproportionate value: the high-count frames contain dense interaction events not represented in the sparse `gold_seed_v00` patches.
- Empirical rule across scientific segmentation literature: learning curves typically show meaningful improvement up to ~500 real training images before plateauing.

**Stress test / falsification criteria:**
- Learning curve experiment: train on 25%, 50%, 75%, 100% of `gold_seed_v00` and plot AP@0.5 vs. training set size. A steep slope at 100% (curve not yet flat) confirms more data is needed; a flat slope indicates augmentation is compensating.
- If AP@0.5 plateaus above 0.75 with 124 images + augmentation → data quantity is not the bottleneck; focus shifts to architecture (H4) or domain adaptation (H2/H3).
- If incorporating `seed_v04` full-frame annotations yields a larger AP gain per hour of annotation effort than annotating more 384×384 patches → full-frame annotation is the preferred expansion strategy.

**Status:** Untested

---

### H6: High void fraction images (>10%) are qualitatively harder and require separate treatment

**Statement:** AP@0.5 on the subset of test images with void fraction > 10% will be at least 0.15 lower than AP@0.5 on the subset with void fraction < 5%, using the same model trained on the full dataset.

**Supporting evidence:**
- This pattern is reported universally across all 9 reviewed papers, across multiple architectures:
  - Fu & Liu (2016): number density error < 5% at void fraction ≤ 10%; degrades sharply above.
  - Cui et al. (2022): IoU > 0.9 at gas holdup < 16%, degrades above; best at 4.34%.
  - Hessenkemper et al. (2022): evaluated only at 0.5–5% void fraction.
  - Kim & Park (2021): 3 test conditions including dense, but AP degrades in dense scenes.
  - All methods fail because bubble boundaries merge in the image at high packing — an optical ambiguity that cannot be resolved from 2D alone.
- Our dataset spans 0.3–16.3% void fraction. The top quartile (>10%) is within the documented failure zone.
- The SAM2.1 annotation tool required significantly more human corrections on the dense images, which is indirect evidence that even the AI-assisted annotation process found them harder.
- Copy-paste augmentation in `dryrun_aug_20260214_1835` has a 20% overlap cap, which artificially limits training diversity for the >10% regime where real bubble overlaps exceed this cap.

**Stress test / falsification criteria:**
- If AP@0.5 gap between the low-void (<5%) and high-void (>10%) test subsets is < 0.10 → the void fraction effect is smaller than the literature suggests for our data (possibly due to better imaging contrast or larger bubble pixel size in our setup).
- If recall (not precision) drops sharply at high void fractions while precision stays stable → failure mode is missed detections (bubbles merged by the model), not hallucinations; fix via post-processing (watershed, shape-prior boundary recovery).
- If a specialist model trained *only* on high-void patches (>10%) outperforms the general model on high-void test images → a specialist model strategy is justified.

**Status:** Untested

---

## 3. Plan A / Plan B / Plan C Roadmap

Plans are ordered by increasing effort and decreasing reliance on existing infrastructure. Complete Plan A benchmarking before committing to Plan B or C. The decision criteria are explicit so the escalation decision is not subjective.

**Prerequisite (blocks all plans):** A fixed held-out test set must be defined and locked before any model is trained. See OQ3.

---

### Plan A: MicroSAM Fine-tuning (Current Infrastructure)

**Model:** MicroSAM ViT-B — already fully integrated.

**Training strategy:**
1. Lock a held-out test set: randomly reserve 20–25 images from `gold_seed_v00` before any augmentation export (stratified by void fraction so the test set covers the full range). These images are excluded from all training and augmentation from this point forward.
2. Export the remaining ~99 source images with 4× augmentation using the existing `dryrun_aug` pipeline → ~396 training images.
3. Train MicroSAM ViT-B for 100 epochs via `train.py` / Slurm. Use existing `manage_bubbly.py` submission interface.
4. Evaluate on the held-out test set using an evaluation script (to be written): AP@0.5, AP@0.75, AP@[0.5:0.95], touching-bubble F1, per-image void-fraction breakdown.
5. Qualitatively inspect false merges on touching bubbles — overlay predicted masks on source images.

**Pre-experiment diagnostic:** Run MicroSAM zero-shot (no fine-tuning) on 2–3 test images to establish a zero-shot baseline. This costs no GPU time and immediately tells us how much the pre-trained model already "knows" about bubble shapes.

**What success looks like:**
- AP@[0.5:0.95] ≥ 0.60 on the zero-G test set.
- AP@0.5 ≥ 0.75.
- Touching-bubble F1 ≥ 0.65.
- No systematic class of false positives (e.g., vessel wall artifacts mistaken for bubbles).

**Trigger to move to Plan B:**
- AP@0.5 < 0.70 after fine-tuning with augmentation, OR
- Touching-bubble F1 < 0.55, OR
- Qualitative analysis reveals consistent systematic failure mode not addressable by more augmentation alone.

**Estimated effort:** Low (infrastructure live). Main tasks: lock test split, run eval script, submit training job.

---

### Plan B: StarDist Pre-trained on RODARE, Fine-tuned on Zero-G Data

**Model:** StarDist 2D ([github.com/stardist/stardist](https://github.com/stardist/stardist)).

**Training strategy:**
1. Run RODARE-trained StarDist zero-shot on 2–3 test images first (diagnostic for H3 — is Earth-gravity transfer viable?).
2. Download RODARE dataset and HZDR StarDist code from [rodare.hzdr.de](https://rodare.hzdr.de).
3. Write `bubbly_flows/scripts/train_stardist.py` following the same `--dataset / --name / --epochs` interface as `train.py`. The Slurm submission logic in `manage_bubbly.py` can then reuse existing infrastructure.
4. Convert `gold_seed_v00` instance mask `.tif` files to StarDist format (distance transform + angle map, computed by StarDist's own `prepare_data` utilities). The instance-ID mask format already exported by `manage_bubbly.py` is the correct direct input.
5. Fine-tune for 50–100 epochs starting from RODARE checkpoint. Use the same held-out test split as Plan A for direct comparison.
6. Optionally: add UNet post-processing refinement step (per Hessenkemper 2022).

**What success looks like:**
- AP@[0.5:0.95] ≥ 0.68.
- AP@0.5 ≥ 0.82.
- Touching-bubble F1 ≥ 0.72.
- Clear improvement over Plan A (AP@0.5 gain ≥ 0.05).

**Trigger to move to Plan C:**
- AP@0.5 < 0.72 after fine-tuning, OR
- Aspect ratio analysis (OQ1) shows > 20% of bubbles have aspect ratio > 1.5 → star-convex prior is too restrictive, OR
- Touching-bubble F1 < 0.60 due to systematic star-convex approximation failures.

**Estimated effort:** Medium. New adapter script + RODARE download + training run + evaluation.

---

### Plan C: Mask R-CNN (BubMask) + Dataset Expansion

**Model:** BubMask ([github.com/ywflow/BubMask](https://github.com/ywflow/BubMask)) — Mask R-CNN with ResNet-101 backbone, size-weighted loss, COCO pre-trained init.

**Training strategy:**
1. Write `bubbly_flows/scripts/train_bubbask.py` following the same interface as `train.py`.
2. **Expand the dataset** before training: promote `seed_v04` full-frame annotations to gold (14 full-frame images, 4–641 bubbles each, already annotated in `seed_v04/labels/`). Tile the full frames into 384×384 patches using the existing patching pipeline to align with the `gold_seed_v00` format, or train on full-resolution images if GPU memory permits.
3. **Temporal propagation (future data expansion):** The existing annotated frames are temporally independent and cannot be used for propagation. However, for *new* annotation batches: choose a keyframe from each video sequence, annotate it, then use SAM2 video propagation to produce candidate masks for ±5–10 neighboring frames, human-review and accept/correct. This could multiply labeling throughput 3–5× on future batches.
4. Initialize from COCO pre-trained ResNet-101 weights (as Kim & Park 2021). Fine-tune on the expanded dataset using the same held-out test split.
5. Evaluate with same metrics as Plans A and B for direct comparison.

**What success looks like:**
- AP@[0.5:0.95] ≥ 0.72.
- AP@0.5 ≥ 0.85.
- AP@0.5 on void fraction > 10% images ≥ 0.70.
- Touching-bubble F1 ≥ 0.78.

**Escalation beyond Plan C:**
If AP@0.5 < 0.75 after dataset expansion and BubMask fine-tuning, the bottleneck is the fundamental 2D optical ambiguity at high void fractions — individual bubbles cannot be separated from their 2D projection alone. At this point, the right path forward is not more architectures but one of:
- Restrict scope to void fraction < 10% (all existing models work well there) and document the limitation.
- Pursue temporal consistency: use video frames together (SAM2 in video mode or optical-flow-guided mask propagation).
- Explore alternative imaging modalities (e.g., fluorescence tagging of one phase) that eliminate the occlusion ambiguity.

**Estimated effort:** High. Adapter script + `seed_v04` promotion + temporal propagation annotation + training + evaluation.

---

### Plan Comparison Matrix

| Criterion | Plan A (MicroSAM) | Plan B (StarDist + RODARE) | Plan C (BubMask + expansion) |
|-----------|------------------|---------------------------|------------------------------|
| Infrastructure ready | Yes (fully live) | Partial (adapter script needed) | No (adapter + data expansion needed) |
| Training data needed | 124 images (have) | 124 + RODARE (~800, download needed) | 124 + seed_v04 + temporal propagation |
| Key inductive prior | General (foundation model) | Star-convex (spherical objects) | None (flexible, data-driven) |
| Expected AP@0.5 ceiling | 0.75–0.85 | 0.82–0.91 | 0.85–0.98 |
| Expected AP@[0.5:0.95] ceiling | 0.55–0.70 | 0.65–0.80 | 0.70–0.88 |
| Touching-bubble risk | Moderate | Moderate (star-convex may merge) | Lower (flexible masks) |
| Hypotheses tested | H1, H5 | H2, H3, H4 | H4 (vs. B), H5, H6 |
| Time to first result | 1–2 weeks | 2–3 weeks | 3–5 weeks |

---

## 4. Open Questions

Most open questions are answerable from existing data or with a short diagnostic experiment. Record resolution date and answer when known.

---

**OQ1: What is the bubble shape distribution in gold_seed_v00?**
Specifically: distribution of per-bubble aspect ratio (major/minor axis of fitted ellipse) and circularity index (4π·area / perimeter²) across all annotated instances. If > 20% of bubbles have aspect ratio > 1.5, StarDist's star-convex prior is too restrictive. Answerable from existing LabelMe JSONs using OpenCV `fitEllipse` on the polygon points.
*Resolution: not yet measured*

---

**OQ2: How many annotated bubble instances are in gold_seed_v00 total?**
We know 124 images and void fraction 0.3–16.3%, but not the total instance count. This matters for (a) assessing copy-paste augmentation diversity, (b) comparing to literature minimums. Answerable by counting polygon entries across all JSONs in `gold_seed_v00/labels_json/`.
*Resolution: not yet measured*

---

**OQ3 (BLOCKING): No held-out test split is currently defined.**
A fixed test set (20–25 images from `gold_seed_v00`, stratified by void fraction) must be locked and excluded from all training and augmentation before any model is trained. Running experiments without this makes all reported accuracy numbers incomparable. **This must be done first, before Plan A begins.**
*Resolution: not yet done*

---

**OQ4: Do seed_v04 full-frame images cover the high-void-fraction end?**
The 14 full-frame images (4–641 bubbles each) are annotated but not quantified. Computing void fraction from the polygon areas in `seed_v04/labels/` JSONs will confirm whether they add high-density training diversity (critical for Plan C / H6). The computation is the same as was already performed for seed_v04 in this chat (script available).
*Resolution: 0.3%–16.3% confirmed (computed 2026-03-02); same range as gold_seed_v00*

---

**OQ5: What is the zero-shot performance of MicroSAM and RODARE-trained StarDist on zero-G test images?**
Running both models zero-shot (no fine-tuning on zero-G data) on 2–3 test images costs essentially no GPU time and immediately quantifies (a) the domain gap (H3) and (b) how much fine-tuning contributes. This diagnostic should be done before committing any GPU resources to Plan A or B training.
*Resolution: not yet done*

---

**OQ6: What is the imaging modality and contrast regime?**
The literature distinguishes backlit (silhouette: dark bubble interior, bright halo), front-lit, and fluorescence. RODARE uses backlit. If our zero-G footage uses a different modality or has lower contrast, the visual features that transfer from RODARE may be fundamentally different. From the images viewed (2026-03-02), the zero-G footage appears **backlit with dark bubble interiors and bright halos** — consistent with RODARE. Confirm from experimental documentation.
*Resolution: visually consistent with backlit; formal confirmation pending*

---

**OQ7 (RESOLVED): Is the source data continuous video, and are the annotated frames temporally correlated?**
The *source* data is continuous parabolic flight video. However, the annotated snapshots currently in the dataset (`gold_seed_v00`, `seed_v04`) were selected as independent frames — they are not consecutive frames from the same video sequence and have no temporal correlation with each other. Each annotated image should be treated as an independent sample. SAM2 temporal propagation remains a viable *future* strategy for generating new labeled data (annotate a keyframe, propagate to neighboring frames, human-review), but it does not apply retroactively to the existing labeled set.
*Resolved 2026-03-02*

---

**OQ8 (RESOLVED): What is the downstream use of the segmentation masks?**
Bubble size distribution (per-bubble equivalent diameter and shape). False merges of touching bubbles are the worst failure mode. Success criteria therefore prioritize AP@IoU[0.5:0.95] (not just AP@0.5) and explicitly track touching-bubble F1.
*Resolved 2026-03-02*

---

## 5. Decision Log

*Append entries as decisions are made. Include date, decision, rationale, and which hypothesis / plan it informs.*

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-03-02 | Established Plan A (MicroSAM) as first experiment | Full pipeline already live; lowest time-to-first-result; tests H1 and H5 | Defines AP baseline before any architecture comparison |
| 2026-03-02 | Flagged OQ3 (no test split defined) as prerequisite blocker | Without a fixed test set, all AP numbers are incomparable across runs | Must be resolved before Plan A begins |
| 2026-03-02 | Identified seed_v04 full-frame images as priority data for Plan C | 14 annotated full frames already in workspace; disproportionate value for high-void-fraction coverage (up to 641 bubbles/frame) | Avoids fresh annotation effort for Plan C dataset expansion |
| 2026-03-02 | Excluded YOLO, U-Net, and custom CNNs from candidate list | YOLO: speed-optimized, not accuracy-optimized. U-Net: semantic only. Custom CNNs: no public code, geometry-specific | Narrows scope to MicroSAM / StarDist / BubMask |
| 2026-03-02 | Added SAM2 temporal propagation as future annotation strategy (not retroactive) | Source data is continuous video but existing annotated frames are temporally independent; propagation applies only to new batches | Reduces annotation burden for future data collection rounds |
| 2026-03-02 | Set touching-bubble F1 as explicit evaluation metric | Downstream goal is bubble size distribution (OQ8 resolved); false merges are the worst failure mode | All plans must report touching-bubble F1 alongside AP |

---

*End of document. Next action: resolve OQ3 (lock test split), then run OQ5 diagnostic (MicroSAM + StarDist zero-shot), then execute Plan A.*
