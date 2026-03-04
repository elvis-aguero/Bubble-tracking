# Literature Review: ML Models for Bubble Segmentation

*Compiled 2026-03-02. Papers span 2016–2025. All 9 PDFs read from Zotero local library.*

---

## Summary Table

| Authors | Year | Model | Dataset Size | Key Metric | Score |
|---------|------|-------|-------------|------------|-------|
| Yang et al. | 2025 | YOLOv9 (segmentation head) | ~800 annotated images (augmented) | mAP@0.5 | ~0.91 (comparable to manual) |
| Nizovtseva et al. | 2024 | YOLOv9 | 500 frames (SOPAT high-speed video) | Agreement with manual annotation | ~93% |
| Maduabuchi | 2024 | U-Net CNN | Multi-fluid boiling HSV datasets (LN2, Ar, FC-72, HP water) | % relative error (dry area fraction) | <5% vs adaptive threshold |
| Hessenkemper et al. | 2024 | StarDist + SNN + LPT (3D tracking) | ~800 annotated images | MOTA | 0.812 |
| Hessenkemper et al. | 2022 | Double UNet / StarDist / Mask R-CNN (comparison) | ~800 images, ~24,400 bubbles | AP@IoU0.5 | 0.91 (StarDist+UNet) |
| Cui et al. | 2022 | Mask R-CNN + ResNet-101 + FPN + RDC | 100 images, >20,000 bubbles | mAP / IoU | mAP>0.95, IoU>0.9 (<16% holdup) |
| Kim & Park | 2021 | Mask R-CNN + ResNet-101 + custom loss + BubGAN | 2,592 images, ~38,080 bubbles | AP50 | 0.981 |
| Cerqueira & Paladino | 2021 | Custom CNN (anchor-point + ellipse fitting) | 2,400 images (augmented from 600) | Detection accuracy | 94.55% |
| Fu & Liu | 2016 | Traditional CV (watershed + skeleton + ellipse) | N/A (no ML) | Number density error | <5% (void fraction ≤10%) |

---

## Paper Summaries

### 1. A Deep Learning-Based Segmentation Method for Multi-Scale and Overlapped Bubbles in Gas–Liquid Bubble Columns — Yang et al. (2025)

**Model:** YOLOv9 with segmentation head for instance segmentation. Architecture uses Programmable Gradient Information (PGI) backbone. Data augmentation: brightness/contrast variation, flipping, Gaussian noise, random crop/scale.

**Dataset:** ~800 manually annotated images from gas-liquid bubble column experiments. Bubble radius ~0.13–0.16 cm, velocity ~15–30 cm/s. Not stated as publicly available.

**Assumptions:**
- Air-water gas-liquid bubble column, moderate gas fractions
- High-speed video with sufficient contrast
- Bubbles may overlap and span multiple scales simultaneously
- 2D segmentation only

**Accuracy:**
- mAP@0.5 comparable to or exceeding manual annotation quality
- Processing: ~20 fps on NVIDIA RTX 4060 Ti (~50 ms/frame)
- Quantitative bubble radius and velocity distributions closely matched manual measurements

**Code/Data links:** Not provided.

**Notes:** Specifically addresses the simultaneous multi-scale + heavily overlapping bubble challenge in dense columns. Validated against manual segmentation rather than independent ground truth. No cross-fluid/geometry generalization demonstrated.

---

### 2. Bubble Detection in Multiphase Flows Through Computer Vision and Deep Learning for Applied Modeling — Nizovtseva et al. (2024)

**Model:** YOLOv9 for detection and segmentation. Post-processing: superellipse boundary fitting for non-spherical inclusions. Targets downstream estimation of mass transfer rates (kLa) from bubble size distributions.

**Dataset:** 500 frames from SOPAT high-speed video system. Manual annotations from multiple human annotators. Industrial stirred tanks / bubble column context.

**Assumptions:**
- Industrial-scale gas-liquid reactors
- SOPAT proprietary high-speed video (backlit, high contrast)
- Bubbles approximately superellipsoidal
- 2D segmentation; 3D shape via superellipse fitting

**Accuracy:**
- ~93% agreement with manual annotator segmentation
- ~20 fps end-to-end (~50 ms/frame)
- Superellipse fitting reduces shape error vs. standard ellipse for irregular bubbles

**Code/Data links:** Not provided. References proprietary SOPAT system.

**Notes:** Unique in explicitly targeting kLa estimation downstream of detection — connects bubble sizing to applied chemical engineering quantities. Dataset of 500 frames may be too small for robust generalization.

---

### 3. Automated Segmentation and Analysis of High-Speed Video Phase-Detection Data for Boiling Heat Transfer — Maduabuchi (2024)

**Model:** U-Net CNN for semantic segmentation of high-speed video phase-detection images. Compared against adaptive thresholding baseline. Includes uncertainty quantification (UQ) framework.

**Dataset:** Multiple boiling fluid datasets: liquid nitrogen (LN2), liquid argon (Ar), FC-72, high-pressure water. Phase-detection images from HSV. Associated with an open-source repository (URL not captured in first 20 pages).

**Assumptions:**
- Pool boiling / flow boiling heat transfer (nucleate/transition boiling)
- Binary phase-detection contrast (liquid/vapor)
- Boiling metrics (dry area fraction, contact line density, nucleation site density) are target outputs

**Accuracy:**
- Weighted average % relative error for dry area fraction < 5% vs. adaptive thresholding
- Full AP/IoU results in later thesis chapters (not extracted from first 20 pages)

**Code/Data links:** Open-source repository referenced in thesis (URL not captured).

**Notes:** One of very few works applying DL to boiling heat transfer (distinct from bubbly flow columns). Rigorous UQ framework. Covers four distinct fluids. MIT Master's Thesis — peer-reviewed publication status unclear.

---

### 4. 3D Detection and Tracking of Deformable Bubbles in Swarms with the Aid of Deep Learning Models — Hessenkemper et al. (2024)

**Model:** Multi-stage 3D tracking pipeline:
1. **StarDist** — 2D star-convex polygon instance segmentation
2. **Siamese Neural Network (SNN)** — cross-view bubble matching using epipolar geometry + appearance features
3. **Lagrangian Particle Tracking (LPT)** — stereo triangulation + temporal linking + backward tracking

**Dataset:** ~800 manually annotated images. Octagonal tank, air-water, 3–4 synchronized cameras (300–1000 fps). Bubble diameter 2–8 mm, void fraction 0.5–5.4%. Dataset and code at [rodare.hzdr.de](https://rodare.hzdr.de).

**Assumptions:**
- Air-water bubbly flow, moderate gas fractions (0.5–5.4%)
- Multi-camera stereo setup (3–4 cameras), octagonal tank geometry, calibrated epipolar geometry
- Bubbles are deformable but approximately star-convex

**Accuracy:**
- MOTA (Multiple Object Tracking Accuracy): up to **0.812**
- Significant improvement from backward tracking (fewer identity switches and missed tracks)

**Code/Data links:**
- Dataset + code: [https://rodare.hzdr.de](https://rodare.hzdr.de)

**Notes:** One of very few works achieving validated 3D bubble tracking at moderate void fractions. SNN cross-view matching is a novel contribution. Requires high experimental complexity (multi-camera, calibrated stereo, octagonal tank). MOTA of 0.812 still reflects non-trivial errors in dense swarms.

---

### 5. Bubble Identification from Images with Machine Learning Methods — Hessenkemper et al. (2022)

**Model:** Comparative study of three approaches:
1. **Double UNet** — cascaded UNetL3 (coarse) + UNetL5 (fine boundary refinement)
2. **StarDist** — star-convex polygon instance segmentation, fine-tuned from biomedical pre-training
3. **Mask R-CNN** (BubMask from Kim & Park 2021) — region-proposal instance segmentation

Best: StarDist + UNet post-processing.

**Dataset:** ~800 annotated images, 512×512 px, ~24,400 annotated bubble instances. Air-water and water-glycerol, rectangular column, bubbles 1–10 mm, void fraction 0.5–5%. Data and code at [rodare.hzdr.de](https://rodare.hzdr.de).

**Assumptions:**
- Air-water / water-glycerol bubbly flow, moderate void fractions
- 2D backlit HSV, rectangular column
- Bubbles approximately star-convex
- In-distribution evaluation (same rig as training)

**Accuracy:**
- AP@IoU0.5:
  - **StarDist + UNet: ~0.91**
  - Double UNet: slightly lower
  - Mask R-CNN: competitive but lower than StarDist+UNet on this dataset
- All methods substantially outperform classical watershed segmentation

**Code/Data links:**
- Data: [https://rodare.hzdr.de](https://rodare.hzdr.de)
- Code: [https://rodare.hzdr.de](https://rodare.hzdr.de)
- StarDist: [https://github.com/stardist/stardist](https://github.com/stardist/stardist)
- BubMask: [https://github.com/ywflow/BubMask](https://github.com/ywflow/BubMask)

**Notes:** First systematic comparison of ML segmentation paradigms on identical bubbly flow datasets. Publicly releases both data and code. Establishes StarDist as a highly effective approach for bubbly flows. All methods evaluated on a single rig/fluid combination — cross-dataset generalization not assessed.

---

### 6. A Deep Learning-Based Image Processing Method for Bubble Detection, Segmentation, and Shape Reconstruction in High Gas Holdup Sub-Millimeter Bubbly Flows — Cui et al. (2022)

**Model:** Mask R-CNN + ResNet-101 backbone + Feature Pyramid Network (FPN). Custom Radial Distance Correction (RDC) post-processing module for occluded/partial bubble boundary reconstruction. Targets sub-millimeter bubbles (28–700 µm).

**Dataset:** 100 training images (70/30 train/val split), >20,000 annotated bubble instances. Media: tap water, SDS solution, diesel, diesel + catalyst particles. Gas holdups up to 20%. No public release.

**Assumptions:**
- High gas holdup (up to 20%) with sub-millimeter bubbles
- High-magnification / high-resolution camera, backlit
- Bubbles approximately ellipsoidal; RDC corrects occluded boundaries
- 2D imaging; 3D shape estimated as prolate spheroid for volume calculation

**Accuracy:**
- IoU > 0.9 at gas holdups < 16%
- Best results (Experiment 3, gas holdup 4.34%, tap water): Precision = 0.918, Recall = 0.993, mAP = 0.934, F1 = 0.954, IoU = 0.943
- All metrics > 0.95 at gas holdups < 14.67%
- Performance degrades at gas holdups > 16% due to severe occlusion

**Code/Data links:** Not provided.

**Notes:** Addresses the challenging sub-millimeter bubble regime at high gas holdups (relevant to Fischer-Tropsch synthesis etc.). RDC shape reconstruction provides physically meaningful bubble shapes. Validated across four media. Only 100 training images — very small dataset. No public code/data.

---

### 7. Deep Learning-Based Automated and Universal Bubble Detection and Mask Extraction in Complex Two-Phase Flows — Kim & Park (2021)

**Model:** Mask R-CNN + ResNet-101, customized with:
- **Size-weighted loss function** to upweight small bubble instances
- **BubGAN** (Bubble GAN) to synthesize additional training images
- Transfer learning from COCO pre-trained weights

**Dataset:** 2,592 training images (~38,080 annotated bubbles) from three sources: PIV images (1,588), shadowgraph images (854), BubGAN synthetic images (150). Test: 24 images (8 per condition).

**Assumptions:**
- General two-phase flows — not restricted to a single geometry or imaging modality
- Diverse imaging: PIV laser illumination, shadowgraphy, synthetic
- Small bubbles specifically addressed via custom loss

**Accuracy:**
- **AP50 = 0.981** on full combined test set
- AP50 = 0.997 on same-condition test images
- Small bubble AP_S improved ~4% over standard Mask R-CNN
- Speed: 4.4 s/image vs. 14 s for watershed (~3× speedup)

**Code/Data links:**
- Code: [https://github.com/ywflow/BubMask](https://github.com/ywflow/BubMask)

**Notes:** Claims universality across diverse imaging setups (one of few papers to test this). BubGAN synthetic data augmentation is a notable contribution for small datasets. Test set is small (24 images total). AP50 is a lenient threshold; AP@0.75 performance not emphasized. No public dataset.

---

### 8. Development of a Deep Learning-Based Image Processing Technique for Bubble Pattern Recognition and Shape Reconstruction in Dense Bubbly Flows — Cerqueira & Paladino (2021)

**Model:** Custom CNN with anchor-point region proposal strategy + parameterized ellipse fitting for shape estimation. Designed specifically for pipe flow bubbly flows (not a standard off-the-shelf architecture).

**Dataset:** 2,400 labeled images (augmented from 600 originals via rotation, flipping, brightness variation). Transparent pipe D = 26.2 mm, L = 2.0 m. Air-water and air-glycerol. Void fractions 1.41–9.03%. Imaging: 1024×1024 px at 400 fps.

**Assumptions:**
- Pipe bubbly flow, void fractions up to ~9%
- High-speed camera, front-lit transparent pipe
- Bubbles approximately ellipsoidal in 2D projection

**Accuracy:**
- Best detection accuracy: **94.55%**
- Recall ~0.95, Precision ~0.84 at low void fractions
- Precision degrades at higher void fractions (up to 9%)

**Code/Data links:** Not provided.

**Notes:** Geometry-specific custom architecture that produces physically interpretable ellipsoidal shapes directly, enabling equivalent diameter and velocity calculations. Precision drops (~0.84) at moderate void fractions. No public code/data.

---

### 9. Development of a Robust Image Processing Technique for Bubbly Flow Measurement in a Narrow Rectangular Channel — Fu & Liu (2016)

**Model:** Traditional (non-ML) computer vision pipeline:
1. Noise reduction + non-uniform brightness correction
2. Boundary curvature + intensity gradient breakpoint detection for occluded boundaries
3. Watershed + bubble skeleton + adaptive thresholding for touching bubble separation
4. Ellipse fitting for shape reconstruction
No trainable parameters.

**Dataset:** N/A (algorithm-based). Validated on air-water flow in 30 mm × 10 mm rectangular channel. Void fractions 2.4–9.1% (validated to 18% via synthetic test images). Bubble diameter 1–3.5 mm, 1000 fps.

**Assumptions:**
- Air-water bubbly flow in narrow rectangular channel (2D-like constraint)
- Backlit HSV with clear bubble boundaries
- Bubbles approximately ellipsoidal
- Clear intensity gradients at boundaries (degrades at very high void fractions)

**Accuracy:**
- Number density error < 5% for void fractions ≤ 10%
- Performance degrades at void fraction > 10% due to bubble occlusion

**Code/Data links:** Not provided.

**Notes:** Well-engineered traditional CV baseline. No training data required. Requires manual parameter tuning per experimental setup. Cannot automatically generalize to new geometries or imaging conditions.

---

## Observations & Recommendations

### Dominant Architectures

The field has converged on two primary paradigms for 2D bubble instance segmentation:

- **Mask R-CNN** (region-proposal): Used by Cui et al. (2022), Kim & Park (2021), benchmarked by Hessenkemper et al. (2022). Mature and accurate but slower and requires larger training sets.
- **StarDist** (star-convex polygon, distance-map): Used by Hessenkemper et al. (2022, 2024). Works well because most bubbles are approximately star-convex. Efficient; AP@0.5 ~ 0.91.
- **YOLO-based** (Yang et al. 2025, Nizovtseva et al. 2024): Real-time (~20 fps) at some accuracy cost. Good for online monitoring.
- **Custom CNN** (Cerqueira & Paladino 2021): Geometry-specific, not generalizable.
- **U-Net** (Maduabuchi 2024): Better for semantic (phase-level) segmentation in boiling, not instance segmentation in columns.

### Dataset Sizes and the Small-Data Challenge

Most studies train on 100–2,400 images — very small by general CV standards. Strategies used:
- **Synthetic data via BubGAN** (Kim & Park 2021) — most principled approach
- **Heavy augmentation** (Yang et al. 2025, Cerqueira & Paladino 2021)
- **Transfer learning** from COCO/ImageNet (Kim & Park 2021, Cui et al. 2022)
- The Hessenkemper group's public ~800-image RODARE dataset is the best available community resource

### Void Fraction is the Key Limiting Factor

All methods — ML and traditional alike — degrade at high void fractions (>10–16%) due to bubble occlusion. No current method reliably segments individual bubbles at void fractions > 20%. Potential directions: 3D imaging, physics-informed priors on bubble size distributions.

### Metrics Inconsistency

The field uses inconsistent evaluation metrics. Recommend adopting AP@IoU[0.5:0.95] (COCO-style mAP) alongside physical measurement errors (equivalent diameter error, number density error) to connect ML performance to engineering relevance.

### Code and Data Availability

| Paper | Code Public | Data Public |
|-------|------------|------------|
| Yang et al. 2025 | No | No |
| Nizovtseva et al. 2024 | No | No |
| Maduabuchi 2024 | Yes (link not captured) | Partial |
| Hessenkemper et al. 2024 | Yes (RODARE) | Yes (RODARE) |
| Hessenkemper et al. 2022 | Yes (RODARE) | Yes (RODARE) |
| Cui et al. 2022 | No | No |
| Kim & Park 2021 | Yes (GitHub) | No |
| Cerqueira & Paladino 2021 | No | No |
| Fu & Liu 2016 | No | No |

Only 3/9 papers provide public code; only 2/9 provide public data. The Hessenkemper group (HZDR) is the most open.

### Decision Guide for Model Selection

| Scenario | Recommended approach |
|----------|---------------------|
| Real-time processing (~20 fps) needed | YOLOv9 with segmentation head (Yang et al. 2025) |
| Maximum 2D accuracy, void fraction < 10% | StarDist ± UNet refinement (Hessenkemper et al. 2022); use RODARE dataset for pre-training |
| Sub-millimeter bubbles | Mask R-CNN + ResNet-101 + FPN + shape reconstruction (Cui et al. 2022) |
| 3D bubble tracking needed | StarDist + SNN + LPT pipeline (Hessenkemper et al. 2024), multi-camera stereo required |
| Boiling heat transfer (not column flow) | U-Net semantic segmentation (Maduabuchi 2024) |
| No annotated training data | Fu & Liu (2016) CV baseline, or BubGAN synthesis (Kim & Park 2021) |
