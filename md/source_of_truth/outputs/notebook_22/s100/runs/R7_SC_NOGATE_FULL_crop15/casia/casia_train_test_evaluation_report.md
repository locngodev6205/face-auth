# CASIA-FASD Kaggle train/test image-copy evaluation

No CASIA weight training, adaptation or checkpoint selection. Optional thresholds are fitted only on train, then transferred unchanged to test and combined.

All available train and test color JPEGs were attempted; detection/crop failures remain in coverage CSVs. No GT/full-image bbox fallback.
The source-selected checkpoint uses SCRFD crops at crop1.5 and 80×80 BGR. Spectral selection, harmful gating and auxiliary heads are training-only; CASIA does not select checkpoints.

## Dataset and metric units

Raw train/test candidate frames: 4063; valid raw frames: 4063. Populations: {'train': {'candidate_frames': 1655, 'candidate_videos': 240}, 'test': {'candidate_frames': 2408, 'candidate_videos': 360}, 'combined': {'candidate_frames': 2660, 'candidate_videos': 360}}.
Train and test are evaluated separately. Combined is the unique union of reconstructed frame IDs, preferring test copies before detection; it does not count overlapping frame IDs twice.
Combined merges unique frames of the same reconstructed video and averages their valid-frame scores. The two split-wise video rows are not averaged or counted twice.
Primary metrics: mean valid-frame binary-logit score per reconstructed video; at least 4 valid frames. Secondary metrics: all valid frames.
The frozen source threshold is transferred unchanged to both units; video aggregation may change score distributions. Optional frame/video thresholds are calibrated separately on train.
AUC/EER/TPR@FPR1% describe the ROC. Baseline APCER/BPCER/ACER/HTER/accuracy use the frozen source threshold. EER does not set the prediction threshold.
Metrics are fractions (multiply by 100 for percentages); ACER = HTER = (APCER+BPCER)/2 for the pooled binary attack class.
AUC/EER/ACER/HTER need both classes; single-class subsets still report their applicable APCER or BPCER and accuracy. Undefined metrics remain null.

## Limitations

These are extracted-image Kaggle train and test splits, not a verified official subject-disjoint CASIA benchmark. No original full-video evaluation is claimed.
Filename real/fake labels are cross-checked against video-code mapping from the prior CASIA exploration. No label is guessed from a split folder.
Train/test image-copy overlap audit: {'subjects': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'video_ids': 240, 'frame_ids': 1403}. Models were trained on CelebA, not on the CASIA train images.
Face selection is heuristic: largest positive finite SCRFD box, confidence tie-break. Raw bbox records and failures are exported.
The full-scale source checkpoint and threshold were selected on CelebA Test-as-Val and frozen before this target evaluation.
No statistical significance or best-model selection on CASIA is claimed.

## Source-frozen baseline: selected runs and metrics

| run_id | split | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | scored_n | candidate_n | frame_auc | frame_acer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R7_SC_NOGATE_FULL_crop15 | train | 0.9033517425252333 | 0.2000095219958103 | 0.016853932584269662 | 0.5423728813559322 | 0.27961340697010095 | 0.27961340697010095 | 0.8523206751054853 | 237 | 240 | 0.8980983530007677 | 0.23957269827702193 |
| R7_SC_NOGATE_FULL_crop15 | test | 0.8910123966942148 | 0.19507575757575762 | 0.041666666666666664 | 0.5795454545454546 | 0.3106060606060606 | 0.3106060606060606 | 0.8238636363636364 | 352 | 360 | 0.8637990328231117 | 0.30497268232811564 |
| R7_SC_NOGATE_FULL_crop15 | combined | 0.9040630182421228 | 0.19145936981757872 | 0.03731343283582089 | 0.5666666666666667 | 0.30199004975124377 | 0.30199004975124377 | 0.8296089385474861 | 358 | 360 | 0.877354762267624 | 0.28116474072214703 |

Source provenance and thresholds: `model_provenance.json`; effective settings: `evaluation_protocol.json`.
Frame/video predictions and metrics: `<train|test|combined>/`.
Use comparison_summary.csv/json to merge independent RUN_FILTER notebooks. Check evaluation protocol and coverage match before merging.


## CASIA train-calibrated thresholds

Separate frame/video thresholds minimize ACER = HTER on train scores only; ties prefer the source threshold. Real is accepted iff score >= threshold.
Exactly the same fitted threshold is applied to train, test and combined. Test/combined labels are never used to choose it; no extra inference is performed.
Train metrics are fitted-set diagnostics; combined includes calibration data. Train/test identity overlap means this Kaggle test is not confirmed independent validation.
AUC/EER are unchanged. Lower test/combined ACER or HTER is not guaranteed. These thresholds require held-out webcam validation before deployment.

| run_id | split | unit | source_threshold | casia_calibrated_threshold | source_acer | acer | source_hter | hter | apcer | bpcer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R7_SC_NOGATE_FULL_crop15 | train | frame | 5.1689453125 | -1.0717773 | 0.23957269827702193 | 0.1877557360052552 | 0.23957269827702193 | 0.1877557360052552 | 0.2789768185451639 | 0.09653465346534654 |
| R7_SC_NOGATE_FULL_crop15 | test | frame | 5.1689453125 | -1.0717773 | 0.30497268232811564 | 0.22039731917116684 | 0.30497268232811564 | 0.22039731917116684 | 0.28343423225096315 | 0.15736040609137056 |
| R7_SC_NOGATE_FULL_crop15 | combined | frame | 5.1689453125 | -1.0717773 | 0.28116474072214703 | 0.21020287412480837 | 0.28116474072214703 | 0.21020287412480837 | 0.2794561933534743 | 0.14094955489614244 |
| R7_SC_NOGATE_FULL_crop15 | train | video | 5.1689453125 | -0.32524413 | 0.27961340697010095 | 0.18015616073128926 | 0.27961340697010095 | 0.18015616073128926 | 0.2247191011235955 | 0.13559322033898305 |
| R7_SC_NOGATE_FULL_crop15 | test | video | 5.1689453125 | -0.32524413 | 0.3106060606060606 | 0.19886363636363635 | 0.3106060606060606 | 0.19886363636363635 | 0.2159090909090909 | 0.18181818181818182 |
| R7_SC_NOGATE_FULL_crop15 | combined | video | 5.1689453125 | -0.32524413 | 0.30199004975124377 | 0.18781094527363185 | 0.30199004975124377 | 0.18781094527363185 | 0.208955223880597 | 0.16666666666666666 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
