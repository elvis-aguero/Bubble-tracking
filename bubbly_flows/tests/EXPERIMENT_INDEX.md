# Hybrid Experiment Index

Current inventory of hybrid bubble-identification research artifacts under `bubbly_flows/tests/`.

## Pipeline Entrypoints
- `frst_point_backend_fb.py` (backend): `src/backends/frst_point_backend_fb.py`
- `frst_point_backend_hf.py` (backend): `src/backends/frst_point_backend_hf.py`
- `blackhat_mask.py` (deterministic): `src/deterministic/blackhat_mask.py`
  Label: Black-hat microbubble masks
  Status: active
  Components: blackhat, DoG, LoG
  Question: Can black-hat style filtering recover small spherical bubbles with acceptable false positives?
  Outputs: output/result19655_blackhat.png, output/result6001_blackhat.png, output/result8008_blackhat.png
  Notes: Focused on tiny, high-curvature bubbles that SAM3 often misses.
- `classical_test.py` (deterministic): `src/deterministic/classical_test.py`
  Label: FRST center detector
  Status: active
  Components: FRST
  Question: How reliable are FRST center proposals before any SAM3 masking?
  Outputs: output/result19655_frst.png, output/result6001_frst.png, output/result8008_frst.png
  Notes: Classical center proposal generator used by the hybrid pipeline.
- `detect_bubbles.py` (deterministic): `src/deterministic/detect_bubbles.py`
  Label: Adaptive threshold detector
  Status: tentative
  Components: adaptive-threshold, morphology
  Question: How far can a simple deterministic detector go without FRST or SAM3?
  Outputs: output/result19655.png, output/result6001.png, output/result8008.png
  Notes: Baseline classical detector kept as a comparison point.
- `bubble_frst_sam3_mask.py` (hybrid): `src/hybrid/bubble_frst_sam3_mask.py`
  Label: FRST + SAM3 + black-hat hybrid
  Status: active
  Components: FRST, SAM3, blackhat
  Question: Can classical proposals and SAM3 cover both small and large bubbles in one pass?
  Outputs: output/result6001.png, output/result6001_frst.png, output/result6001_blackhat.png, logs/bubble_frst_sam3_20260202_172755.log
  Notes: Primary hybrid workbench for combining classical proposals with SAM3 masks.
- `big_bubble_prompt_fb.py` (prompting): `src/prompting/big_bubble_prompt_fb.py`
  Label: SAM3 big-bubble prompt (Meta backend)
  Status: tentative
  Components: SAM3, text-prompt, facebook-backend
  Question: Do text prompts improve SAM3 recall on larger bubbles when using the Meta backend?
  Outputs: img6001_bubble_crop_1_masks.png, img6001_round object_crop_1_masks.png
  Notes: Prompting experiment for larger high-contrast bubbles using the Meta SAM3 stack.
- `big_bubble_prompt_hf.py` (prompting): `src/prompting/big_bubble_prompt_hf.py`
  Label: SAM3 big-bubble prompt (HF backend)
  Status: tentative
  Components: SAM3, text-prompt, hf-backend
  Question: Does the Hugging Face SAM3 path behave differently enough to justify a separate prompting workflow?
  Outputs: img6001_circular blob_crop_1_masks.png, img6001_tiny bubble_crop_0.51_masks.png
  Notes: Same prompting idea as the Meta backend variant, but through the Hugging Face stack.
- `bubble_sam3_mask.py` (sam3): `src/sam3/bubble_sam3_mask.py`
  Label: Automatic SAM3 segmentation
  Status: tentative
  Components: SAM3
  Question: How far does standalone SAM3 get before classical proposals are needed?
  Outputs: output/result.png, output/result.json
  Notes: Pure SAM3 pipeline without the hybrid classical branch.

## Helper Modules
- `hough/detect_bubbles.py`: `hough/detect_bubbles.py`
- `src/common/bubble_sam3/__init__.py`: `src/common/bubble_sam3/__init__.py`
- `src/common/bubble_sam3/backend.py`: `src/common/bubble_sam3/backend.py`
- `src/common/bubble_sam3/candidates.py`: `src/common/bubble_sam3/candidates.py`
- `src/common/bubble_sam3/config.py`: `src/common/bubble_sam3/config.py`
- `src/common/bubble_sam3/outputs.py`: `src/common/bubble_sam3/outputs.py`
- `src/common/bubble_sam3/pipeline.py`: `src/common/bubble_sam3/pipeline.py`
- `src/common/bubble_sam3/postprocess.py`: `src/common/bubble_sam3/postprocess.py`
- `src/common/bubble_sam3/preprocess.py`: `src/common/bubble_sam3/preprocess.py`
- `src/common/bubble_sam3/tiling.py`: `src/common/bubble_sam3/tiling.py`

## Sample Inputs
- `img19655.png`: `img19655.png`
- `img4966.png`: `img4966.png`
- `img6001.png`: `img6001.png`
- `img6001_bubble_crop_1_masks.png`: `img6001_bubble_crop_1_masks.png`
- `img6001_bubbles_crop_1_masks.png`: `img6001_bubbles_crop_1_masks.png`
- `img6001_circle_crop_0.51_masks.png`: `img6001_circle_crop_0.51_masks.png`
- `img6001_circular blob_crop_0.5_masks.png`: `img6001_circular blob_crop_0.5_masks.png`
- `img6001_circular blob_crop_1_masks.png`: `img6001_circular blob_crop_1_masks.png`
- `img6001_microbubble_crop_0.51_masks.png`: `img6001_microbubble_crop_0.51_masks.png`
- `img6001_round object_crop_1_masks.png`: `img6001_round object_crop_1_masks.png`
- `img6001_tiny bubble_crop_0.51_masks.png`: `img6001_tiny bubble_crop_0.51_masks.png`
- `img8008.png`: `img8008.png`

## Known Outputs
- `output/ZeroG_C1S0024_img011620_pred.png`: `output/ZeroG_C1S0024_img011620_pred.png`
- `output/ZeroG_C1S0024_img011620_pred_vis.png`: `output/ZeroG_C1S0024_img011620_pred_vis.png`
- `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`: `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`
- `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001_vis.png`: `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001_vis.png`
- `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`: `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`
- `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509_vis.png`: `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509_vis.png`
- `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`: `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`
- `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432_vis.png`: `output/eval_preds/microsam/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432_vis.png`
- `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`: `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`
- `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001_vis.png`: `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001_vis.png`
- `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`: `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`
- `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509_vis.png`: `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509_vis.png`
- `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`: `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`
- `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432_vis.png`: `output/eval_preds/microsam_seed_v04_run2/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432_vis.png`
- `output/eval_preds/microsam_seed_v04_run2/results.csv`: `output/eval_preds/microsam_seed_v04_run2/results.csv`
- `output/eval_preds/stardist/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`: `output/eval_preds/stardist/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`
- `output/eval_preds/stardist/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`: `output/eval_preds/stardist/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`
- `output/eval_preds/stardist/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`: `output/eval_preds/stardist/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`
- `output/eval_preds/stardist_seed_v04_run1_slurm/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`: `output/eval_preds/stardist_seed_v04_run1_slurm/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`
- `output/eval_preds/stardist_seed_v04_run1_slurm/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`: `output/eval_preds/stardist_seed_v04_run1_slurm/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`
- `output/eval_preds/stardist_seed_v04_run1_slurm/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`: `output/eval_preds/stardist_seed_v04_run1_slurm/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`
- `output/eval_preds/stardist_seed_v04_run1_slurm/results.csv`: `output/eval_preds/stardist_seed_v04_run1_slurm/results.csv`
- `output/eval_preds/yolov9/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`: `output/eval_preds/yolov9/ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png`
- `output/eval_preds/yolov9/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`: `output/eval_preds/yolov9/ZeroG_FlightDay_Test__C1S0004_IMG_S0001004509.png`
- `output/eval_preds/yolov9/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`: `output/eval_preds/yolov9/ZeroG_FlightDay_Test__C1S0010_IMG_S0001005432.png`
- `output/infer_microsam_C1S0024_img011620.png`: `output/infer_microsam_C1S0024_img011620.png`
- `output/infer_microsam_C1S0024_img011620_vis.png`: `output/infer_microsam_C1S0024_img011620_vis.png`
- `output/infer_stardist_C1S0024_img011620.png`: `output/infer_stardist_C1S0024_img011620.png`
- `output/infer_stardist_C1S0024_img011620_vis.png`: `output/infer_stardist_C1S0024_img011620_vis.png`
- `output/infer_yolov9_C1S0024_img011620.png`: `output/infer_yolov9_C1S0024_img011620.png`
- `output/infer_yolov9_C1S0024_img011620_vis.png`: `output/infer_yolov9_C1S0024_img011620_vis.png`
- `output/result.json`: `output/result.json`
- `output/result.png`: `output/result.png`
- `output/result19655.json`: `output/result19655.json`
- `output/result19655.png`: `output/result19655.png`
- `output/result19655_big.png`: `output/result19655_big.png`
- `output/result19655_blackhat.png`: `output/result19655_blackhat.png`
- `output/result19655_frst.png`: `output/result19655_frst.png`
- `output/result6001.json`: `output/result6001.json`
- `output/result6001.png`: `output/result6001.png`
- `output/result6001_big.png`: `output/result6001_big.png`
- `output/result6001_blackhat.png`: `output/result6001_blackhat.png`
- `output/result6001_frst.png`: `output/result6001_frst.png`
- `output/result8008.json`: `output/result8008.json`
- `output/result8008.png`: `output/result8008.png`
- `output/result8008_big.png`: `output/result8008_big.png`
- `output/result8008_blackhat.png`: `output/result8008_blackhat.png`
- `output/result8008_frst.png`: `output/result8008_frst.png`
- `output/result_big.png`: `output/result_big.png`
- `output/result_blackhat.png`: `output/result_blackhat.png`
- `output/result_frst.png`: `output/result_frst.png`

## Known Logs
- `logs/bubble_frst_sam3_20260201_154841.log`: `logs/bubble_frst_sam3_20260201_154841.log`
- `logs/bubble_frst_sam3_20260201_160516.log`: `logs/bubble_frst_sam3_20260201_160516.log`
- `logs/bubble_frst_sam3_20260201_163314.log`: `logs/bubble_frst_sam3_20260201_163314.log`
- `logs/bubble_frst_sam3_20260201_163718.log`: `logs/bubble_frst_sam3_20260201_163718.log`
- `logs/bubble_frst_sam3_20260201_171211.log`: `logs/bubble_frst_sam3_20260201_171211.log`
- `logs/bubble_frst_sam3_20260201_181912.log`: `logs/bubble_frst_sam3_20260201_181912.log`
- `logs/bubble_frst_sam3_20260201_195744.log`: `logs/bubble_frst_sam3_20260201_195744.log`
- `logs/bubble_frst_sam3_20260201_202451.log`: `logs/bubble_frst_sam3_20260201_202451.log`
- `logs/bubble_frst_sam3_20260201_204754.log`: `logs/bubble_frst_sam3_20260201_204754.log`
- `logs/bubble_frst_sam3_20260201_205242.log`: `logs/bubble_frst_sam3_20260201_205242.log`
- `logs/bubble_frst_sam3_20260201_210555.log`: `logs/bubble_frst_sam3_20260201_210555.log`
- `logs/bubble_frst_sam3_20260201_210834.log`: `logs/bubble_frst_sam3_20260201_210834.log`
- `logs/bubble_frst_sam3_20260201_211012.log`: `logs/bubble_frst_sam3_20260201_211012.log`
- `logs/bubble_frst_sam3_20260201_214052.log`: `logs/bubble_frst_sam3_20260201_214052.log`
- `logs/bubble_frst_sam3_20260201_215218.log`: `logs/bubble_frst_sam3_20260201_215218.log`
- `logs/bubble_frst_sam3_20260201_221841.log`: `logs/bubble_frst_sam3_20260201_221841.log`
- `logs/bubble_frst_sam3_20260202_145416.log`: `logs/bubble_frst_sam3_20260202_145416.log`
- `logs/bubble_frst_sam3_20260202_145439.log`: `logs/bubble_frst_sam3_20260202_145439.log`
- `logs/bubble_frst_sam3_20260202_160106.log`: `logs/bubble_frst_sam3_20260202_160106.log`
- `logs/bubble_frst_sam3_20260202_160916.log`: `logs/bubble_frst_sam3_20260202_160916.log`
- `logs/bubble_frst_sam3_20260202_161325.log`: `logs/bubble_frst_sam3_20260202_161325.log`
- `logs/bubble_frst_sam3_20260202_161718.log`: `logs/bubble_frst_sam3_20260202_161718.log`
- `logs/bubble_frst_sam3_20260202_162434.log`: `logs/bubble_frst_sam3_20260202_162434.log`
- `logs/bubble_frst_sam3_20260202_163601.log`: `logs/bubble_frst_sam3_20260202_163601.log`
- `logs/bubble_frst_sam3_20260202_164105.log`: `logs/bubble_frst_sam3_20260202_164105.log`
- `logs/bubble_frst_sam3_20260202_165618.log`: `logs/bubble_frst_sam3_20260202_165618.log`
- `logs/bubble_frst_sam3_20260202_165949.log`: `logs/bubble_frst_sam3_20260202_165949.log`
- `logs/bubble_frst_sam3_20260202_170658.log`: `logs/bubble_frst_sam3_20260202_170658.log`
- `logs/bubble_frst_sam3_20260202_171027.log`: `logs/bubble_frst_sam3_20260202_171027.log`
- `logs/bubble_frst_sam3_20260202_172045.log`: `logs/bubble_frst_sam3_20260202_172045.log`
- `logs/bubble_frst_sam3_20260202_172755.log`: `logs/bubble_frst_sam3_20260202_172755.log`
