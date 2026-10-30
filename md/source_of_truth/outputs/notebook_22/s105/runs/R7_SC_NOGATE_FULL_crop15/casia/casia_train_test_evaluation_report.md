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
| R7_SC_NOGATE_FULL_crop15 | train | 0.8882117691868215 | 0.22252904208722152 | 0.028089887640449437 | 0.6101694915254238 | 0.3191296895829366 | 0.3191296895829366 | 0.8270042194092827 | 237 | 240 | 0.8841827528076549 | 0.2872721624680454 |
| R7_SC_NOGATE_FULL_crop15 | test | 0.8792613636363636 | 0.19886363636363635 | 0.011363636363636364 | 0.6704545454545454 | 0.3409090909090909 | 0.3409090909090909 | 0.8238636363636364 | 352 | 360 | 0.8526182035243381 | 0.32251661549550353 |
| R7_SC_NOGATE_FULL_crop15 | combined | 0.8914179104477613 | 0.1858623548922056 | 0.011194029850746268 | 0.6555555555555556 | 0.3333747927031509 | 0.3333747927031509 | 0.8268156424581006 | 358 | 360 | 0.865916758556184 | 0.3009658111229646 |

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
| R7_SC_NOGATE_FULL_crop15 | train | frame | 5.4794921875 | 0.30647278 | 0.2872721624680454 | 0.19938207849561934 | 0.2872721624680454 | 0.19938207849561934 | 0.22302158273381295 | 0.17574257425742573 |
| R7_SC_NOGATE_FULL_crop15 | test | frame | 5.4794921875 | 0.30647278 | 0.32251661549550353 | 0.2319720593343372 | 0.32251661549550353 | 0.2319720593343372 | 0.1915244909190974 | 0.272419627749577 |
| R7_SC_NOGATE_FULL_crop15 | combined | frame | 5.4794921875 | 0.30647278 | 0.3009658111229646 | 0.22010079458285148 | 0.3009658111229646 | 0.22010079458285148 | 0.19687814702920442 | 0.2433234421364985 |
| R7_SC_NOGATE_FULL_crop15 | train | video | 5.4794921875 | 1.0867056 | 0.3191296895829366 | 0.19443915444677204 | 0.3191296895829366 | 0.19443915444677204 | 0.16853932584269662 | 0.22033898305084745 |
| R7_SC_NOGATE_FULL_crop15 | test | video | 5.4794921875 | 1.0867056 | 0.3409090909090909 | 0.20454545454545456 | 0.3409090909090909 | 0.20454545454545456 | 0.11363636363636363 | 0.29545454545454547 |
| R7_SC_NOGATE_FULL_crop15 | combined | video | 5.4794921875 | 1.0867056 | 0.3333747927031509 | 0.1892620232172471 | 0.3333747927031509 | 0.1892620232172471 | 0.10074626865671642 | 0.2777777777777778 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
