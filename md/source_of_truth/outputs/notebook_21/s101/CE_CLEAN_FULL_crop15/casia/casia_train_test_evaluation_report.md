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
| CE_CLEAN_FULL_crop15 | train | 0.887545229480099 | 0.2056274995239002 | 0.02247191011235955 | 0.5932203389830508 | 0.3078461245477052 | 0.3078461245477052 | 0.8354430379746836 | 237 | 240 | 0.8822318382917428 | 0.3133008444729365 |
| CE_CLEAN_FULL_crop15 | test | 0.8623019972451791 | 0.24621212121212122 | 0.03787878787878788 | 0.5454545454545454 | 0.29166666666666663 | 0.29166666666666663 | 0.8352272727272727 | 352 | 360 | 0.8388834722264904 | 0.32209802700012197 |
| CE_CLEAN_FULL_crop15 | combined | 0.8750829187396352 | 0.2304726368159204 | 0.033582089552238806 | 0.5555555555555556 | 0.2945688225538972 | 0.2945688225538972 | 0.835195530726257 | 358 | 360 | 0.850303011286722 | 0.31292937804991017 |

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
| CE_CLEAN_FULL_crop15 | train | frame | 4.6982421875 | -0.9724121 | 0.3133008444729365 | 0.19955026078147384 | 0.3133008444729365 | 0.19955026078147384 | 0.23820943245403678 | 0.1608910891089109 |
| CE_CLEAN_FULL_crop15 | test | frame | 4.6982421875 | -0.9724121 | 0.32209802700012197 | 0.24666782139355048 | 0.32209802700012197 | 0.24666782139355048 | 0.2344523940561365 | 0.25888324873096447 |
| CE_CLEAN_FULL_crop15 | combined | frame | 4.6982421875 | -0.9724121 | 0.31292937804991017 | 0.23528721824283336 | 0.31292937804991017 | 0.23528721824283336 | 0.23615307150050352 | 0.2344213649851632 |
| CE_CLEAN_FULL_crop15 | train | video | 4.6982421875 | -1.0883967 | 0.3078461245477052 | 0.18577413825937916 | 0.3078461245477052 | 0.18577413825937916 | 0.23595505617977527 | 0.13559322033898305 |
| CE_CLEAN_FULL_crop15 | test | video | 4.6982421875 | -1.0883967 | 0.29166666666666663 | 0.23106060606060608 | 0.29166666666666663 | 0.23106060606060608 | 0.21212121212121213 | 0.25 |
| CE_CLEAN_FULL_crop15 | combined | video | 4.6982421875 | -1.0883967 | 0.2945688225538972 | 0.22665837479270312 | 0.2945688225538972 | 0.22665837479270312 | 0.19776119402985073 | 0.25555555555555554 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
