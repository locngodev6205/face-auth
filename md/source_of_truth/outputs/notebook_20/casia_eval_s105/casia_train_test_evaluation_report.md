# CASIA-FASD Kaggle train/test image-copy evaluation

No CASIA weight training, adaptation or checkpoint selection. Optional thresholds are fitted only on train, then transferred unchanged to test and combined.

All available train and test color JPEGs were attempted; detection/crop failures remain in coverage CSVs. No GT/full-image bbox fallback.
The six checkpoints share the same ordered SCRFD detections, crop1.5 and 80×80 BGR tensors. Spectral views and auxiliary heads are training-only.

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
100K source selection/calibration: Val15K. Full-scale source selection/calibration: official CelebA Test-as-Val (reused source selection set). Scale groups remain separate.
No statistical significance or best-model selection on CASIA is claimed.

## Source-frozen baseline: selected runs and metrics

| run_id | split | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | scored_n | candidate_n | frame_auc | frame_acer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R7_SC_FULL_crop15 | train | 0.893163207008189 | 0.2000095219958103 | 0.02247191011235955 | 0.576271186440678 | 0.2993715482765188 | 0.2993715482765188 | 0.8396624472573839 | 237 | 240 | 0.8920279617889846 | 0.2634486470229757 |
| R7_SC_FULL_crop15 | test | 0.8725895316804407 | 0.21969696969696972 | 0.022727272727272728 | 0.6022727272727273 | 0.3125 | 0.3125 | 0.8323863636363636 | 352 | 360 | 0.8512930613020291 | 0.30991705522295077 |
| R7_SC_FULL_crop15 | combined | 0.8876865671641792 | 0.206301824212272 | 0.022388059701492536 | 0.5555555555555556 | 0.28897180762852404 | 0.28897180762852404 | 0.8435754189944135 | 358 | 360 | 0.8654942908968118 | 0.2860117259989063 |

Source provenance and thresholds: `model_provenance.json`; effective settings: `evaluation_protocol.json`.
Per-run frame/video predictions and metrics: `runs/<run_id>/<train|test|combined>/`.
Use comparison_summary.csv/json to merge independent RUN_FILTER notebooks. Check evaluation protocol and coverage match before merging.


## CASIA train-calibrated thresholds

Separate frame/video thresholds minimize ACER = HTER on train scores only; ties prefer the source threshold. Real is accepted iff score >= threshold.
Exactly the same fitted threshold is applied to train, test and combined. Test/combined labels are never used to choose it; no extra inference is performed.
Train metrics are fitted-set diagnostics; combined includes calibration data. Train/test identity overlap means this Kaggle test is not confirmed independent validation.
AUC/EER are unchanged. Lower test/combined ACER or HTER is not guaranteed. These thresholds require held-out webcam validation before deployment.

| run_id | split | unit | source_threshold | casia_calibrated_threshold | source_acer | acer | source_hter | hter | apcer | bpcer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R7_SC_FULL_crop15 | train | frame | 5.7451171875 | 0.8527832 | 0.2634486470229757 | 0.1975380091966031 | 0.2634486470229757 | 0.1975380091966031 | 0.20943245403677058 | 0.18564356435643564 |
| R7_SC_FULL_crop15 | test | frame | 5.7451171875 | 0.8527832 | 0.30991705522295077 | 0.23244046870736707 | 0.30991705522295077 | 0.23244046870736707 | 0.1992294991744634 | 0.2656514382402707 |
| R7_SC_FULL_crop15 | combined | frame | 5.7451171875 | 0.8527832 | 0.2860117259989063 | 0.21787527529501766 | 0.2860117259989063 | 0.21787527529501766 | 0.19687814702920442 | 0.23887240356083086 |
| R7_SC_FULL_crop15 | train | video | 5.7451171875 | 0.20256348 | 0.2993715482765188 | 0.18863073700247573 | 0.2993715482765188 | 0.18863073700247573 | 0.2247191011235955 | 0.15254237288135594 |
| R7_SC_FULL_crop15 | test | video | 5.7451171875 | 0.20256348 | 0.3125 | 0.23484848484848486 | 0.3125 | 0.23484848484848486 | 0.20833333333333334 | 0.26136363636363635 |
| R7_SC_FULL_crop15 | combined | video | 5.7451171875 | 0.20256348 | 0.28897180762852404 | 0.22296849087893864 | 0.28897180762852404 | 0.22296849087893864 | 0.20149253731343283 | 0.24444444444444444 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
