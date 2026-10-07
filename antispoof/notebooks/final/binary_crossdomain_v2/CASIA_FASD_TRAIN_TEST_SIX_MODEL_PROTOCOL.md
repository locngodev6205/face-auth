# CASIA-FASD train + test: six-model evaluation

## Question and compared methods

Measure external performance across the complete available CASIA Kaggle color-image copy for frozen Clean, WBST and MAST-PAD models. Evaluate both source-scale groups (100K / full-scale) separately. No improvement is assumed; no target model selection is performed.

Notebook: `20_casia_fasd_train_test_C_P3SF_R7SC_100k_fullscale.ipynb`.

This variant inherits the model, detector, crop, score and metric settings of notebook19. It changes only target evaluation populations and adds split-qualified joins and reports. Notebook19 remains the test-only version.

## Populations and overlap

- `train`: all `train_img/train_img/color` JPEGs.
- `test`: all `test_img/test_img/color` JPEGs.
- `combined`: union of reconstructed `(video_id, frame_index)` identities. Keep the test copy when the identity exists in both splits. Select this union before localization/inference, never based on model score or detection success.
- Train/test retain all their own images. Each split's videos are grouped independently using `(split, video_id)`.
- Combined merges unique frames into a single reconstructed video row, rather than pooling or averaging two split-level video predictions.
- Deduplication uses reconstructed identity keys, not byte hashes: repeated IDs are not evidence that JPEG bytes match. Conflicting subject/video labels fail validation. A failed preferred test copy is not replaced by its train duplicate.

Previously saved EDA inventory gives train 1,655 frames / 240 videos, test 2,408 frames / 360 videos, overlap 1,403 frame IDs and 240 video IDs, hence combined 2,660 unique frame IDs / 360 video IDs. These describe the saved copy only; current mounted counts are computed and exported.

## Frozen controls

Six unchanged best checkpoints: `C_100K_crop15`, `P3_SF_100K_crop15`, `R7_SC_100K_crop15`, `C_FULL_crop15`, `P3_SF_FULL_crop15`, `R7_SC_FULL_crop15`.

No training budget, tuning, adaptation, augmentation or threshold fitting on CASIA, including its train folder. Seed 100; deterministic ordering. One GPU checkpoint at a time, original per-run AMP setting.

SCRFD-500M, confidence 0.5, size order 640 → 480 → 320, largest finite positive box and confidence tie-break. No GT or whole-image fallback. Same detection records and exact notebook17/18 crop1.5 / 80×80 BGR [0,1] inputs across models. Binary score `logit(real)-logit(attack)`, source-frozen threshold per run. Spectral views and auxiliary heads are training-only.

The four input paths are copied from the user's updated notebook19 configuration. Output uses a distinct train/test directory. `RUN_FILTER=None` evaluates all six; a nonempty subset supports parallel Kaggle execution.

## Metrics and artifacts

For each model × population (18 summary rows with all models):

- Primary: reconstructed video mean valid-frame score, at least four valid frames.
- Secondary: all valid frames in the population.
- AUC, ROC EER, TPR@FPR1%, APCER, BPCER, ACER, HTER and accuracy. Frozen source thresholds produce decisions; EER never calibrates the model. Undefined metrics are null.
- Per attack-type and quality diagnostics, with split-qualified coverage denominators.

`comparison_summary.csv/json`, `runs/<run_id>/<train|test|combined>/` prediction CSVs and metric JSONs, source provenance, effective protocol, manifests, raw localization failures, coverage, per-population ROC PNGs, report and ZIP. Shared crop array is excluded from ZIP. Summary is saved after each completed model.

## Validation and limits

Validate notebook JSON and compilation, strict loading / original notebook forward parity for all six checkpoints, exact crop geometry, observed EDA manifest counts, and a synthetic overlapping train/test evaluation through metrics/report/ZIP. Synthetic tests must confirm no duplicate combined weighting and no detection-based duplicate replacement. Real CASIA CUDA/SCRFD inference runs on Kaggle, not in offline checks.

The CASIA train images are external evaluation data because these models were trained on CelebA. This does not establish the official CASIA subject-disjoint benchmark: original full videos are absent, reconstructed image identities overlap, and face localization is heuristic. Failed detections remain in coverage. Video aggregation transfers the unchanged frame-calibrated source threshold; no video calibration is performed. Source 100K selection uses Val15K; source full-scale selection uses CelebA Test-as-Val. Keep source-scale groups separate.
