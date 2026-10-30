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
| R7_SC_NOGATE_FULL_crop15 | train | 0.9162064368691677 | 0.18872595696057892 | 0.0056179775280898875 | 0.559322033898305 | 0.28247000571319747 | 0.28247000571319747 | 0.8565400843881856 | 237 | 240 | 0.9139905897064526 | 0.25256626382062664 |
| R7_SC_NOGATE_FULL_crop15 | test | 0.896522038567493 | 0.16856060606060608 | 0.0 | 0.5795454545454546 | 0.2897727272727273 | 0.2897727272727273 | 0.8551136363636364 | 352 | 360 | 0.8766737719619275 | 0.3009385880856398 |
| R7_SC_NOGATE_FULL_crop15 | combined | 0.9143864013266998 | 0.16728855721393038 | 0.0 | 0.5444444444444444 | 0.2722222222222222 | 0.2722222222222222 | 0.8631284916201117 | 358 | 360 | 0.8896186510320014 | 0.2773711230841409 |

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
| R7_SC_NOGATE_FULL_crop15 | train | frame | 5.2744140625 | -0.32629395 | 0.25256626382062664 | 0.1795909806808019 | 0.25256626382062664 | 0.1795909806808019 | 0.17106314948041568 | 0.18811881188118812 |
| R7_SC_NOGATE_FULL_crop15 | test | frame | 5.2744140625 | -0.32629395 | 0.3009385880856398 | 0.20827454935386513 | 0.3009385880856398 | 0.20827454935386513 | 0.16951018161805173 | 0.2470389170896785 |
| R7_SC_NOGATE_FULL_crop15 | combined | frame | 5.2744140625 | -0.32629395 | 0.2773711230841409 | 0.19312561819980217 | 0.2773711230841409 | 0.19312561819980217 | 0.16666666666666666 | 0.2195845697329377 |
| R7_SC_NOGATE_FULL_crop15 | train | video | 5.2744140625 | -2.0143433 | 0.28247000571319747 | 0.1688725956960579 | 0.28247000571319747 | 0.1688725956960579 | 0.21910112359550563 | 0.11864406779661017 |
| R7_SC_NOGATE_FULL_crop15 | test | video | 5.2744140625 | -2.0143433 | 0.2897727272727273 | 0.17234848484848486 | 0.2897727272727273 | 0.17234848484848486 | 0.20833333333333334 | 0.13636363636363635 |
| R7_SC_NOGATE_FULL_crop15 | combined | video | 5.2744140625 | -2.0143433 | 0.2722222222222222 | 0.16185737976782752 | 0.2722222222222222 | 0.16185737976782752 | 0.20149253731343283 | 0.12222222222222222 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
