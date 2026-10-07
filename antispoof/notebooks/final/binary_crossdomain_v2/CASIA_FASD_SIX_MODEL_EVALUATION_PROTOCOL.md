# CASIA-FASD six-checkpoint evaluation specification

## Question and hypothesis

Measure external CASIA Kaggle image-copy performance of frozen Clean, WBST and MAST-PAD checkpoints at 100K and full-scale. Spectral training may affect transfer, but no positive result is assumed. CASIA is not used for training, tuning, threshold fitting or model selection.

## Compared methods and controls

- 100K: `C_100K_crop15`, `P3_SF_100K_crop15`, `R7_SC_100K_crop15`.
- Full-scale: `C_FULL_crop15`, `P3_SF_FULL_crop15`, `R7_SC_FULL_crop15`.
- Treatment: existing trained checkpoint / training method. Scale groups are separate experiments; the full-scale schedule and source selection differ.
- Controls: same full extracted test-color manifest, ordered shared detections and crops; SCRFD-500M confidence 0.5, input sizes 640 → 480 → 320; largest finite positive box with confidence tie-break; exact notebook17/18 crop1.5, 80×80 BGR float [0,1]. No augmentation, jitter, gamma, normalization, spectral inference or auxiliary forward.
- No training budget. Best checkpoint and calibrated threshold are loaded unchanged from each source run. 100K uses source Val15K; full-scale uses source CelebA Test-as-Val.
- Inference uses each run's original AMP setting; convert logits to float32 before score subtraction. Evaluation seed 100. Deterministic ordering; cuDNN benchmarking disabled.

## Target dataset and labels

Use all JPEGs in `test_img/test_img/color`, with no frame sampling. Read explicit `real/fake` filename labels and cross-check the established video-code mapping. Stop on unknown names or label conflicts. Train names are inspected only for overlap, never for fitting.

The saved EDA inspected 2,408 test-color frames and reconstructed 360 video IDs; those are prior observed counts, not asserted results for a new mount. Notebook reports actual mounted counts. Train/test copy overlap is audited and disclosed. This extracted-image copy is not verified to implement the official subject-disjoint CASIA benchmark.

## Metrics and aggregation

- Binary score: `logit_real - logit_attack`; accept Real iff score ≥ the run's source-frozen threshold.
- Primary: mean valid-frame score per reconstructed video, minimum four successfully localized frames. Videos below this minimum remain in coverage and predictions with no decision; no resampling or replacement.
- Secondary: every valid extracted test frame, including frames from videos below the video minimum.
- AUC, descriptive ROC EER, TPR@FPR1%, APCER, BPCER, ACER, HTER and binary accuracy.
- EER / TPR@FPR1% describe the ROC and do not change the frozen decision threshold. ACER and HTER both equal `(APCER + BPCER)/2` for pooled binary attacks.
- Scores/metrics are fractions, not percentages. Undefined metrics remain null. Shared localization coverage and failed candidates are reported separately.
- Per attack type and quality diagnostics. No winner chosen on CASIA, no significance claim, no unpaired pooling of scales.

## Artifacts and execution

Notebook: `19_casia_fasd_C_P3SF_R7SC_100k_fullscale.ipynb`.
Mount CASIA, experiment17 runs, experiment18 runs, and the SCRFD ONNX detector. MiniFASNet architecture is embedded. No contract/readme/resource tree required.

`RUN_FILTER=None` evaluates six checkpoints sequentially on one shared crop cache. Any nonempty subset of the six IDs runs independently for parallel Kaggle execution. A single checkpoint is resident on GPU at a time. Progress is printed for setup, detection, each model and batches. Summaries are written after each completed model.

Outputs: CSV/JSON comparison summary, frame/video manifests, source provenance, effective evaluation protocol, full localization records, video coverage, per-run frame/video predictions and metrics, subgroup metrics, ROC PNGs, Markdown report and downloadable ZIP. Local shared-crop NPY is excluded from ZIP. Existing training outputs are read-only.

## Validation and limits

First validate JSON/nbformat, compile every cell, exercise parsing on saved EDA inventory, and exercise preprocessing/metrics/loading on synthetic inputs and existing local checkpoints. Do not run expensive source training. Real SCRFD/CUDA inference and actual CASIA metrics require Kaggle; local checks cannot establish those outcomes.

Filename video IDs are reconstructed rather than decoded from original videos. Largest-box localization is heuristic because the copy has no source bbox annotations. Excluded detections can bias conditional PAD metrics, so counts/coverage are required. Video averaging changes score distributions; source thresholds are transferred without video-specific calibration. Full-scale source Test-as-Val is reused selection data, not an untouched source Test.
