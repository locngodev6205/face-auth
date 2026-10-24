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
| CE_CLEAN_FULL_crop15 | train | 0.9103027994667683 | 0.19434393448866882 | 0.0 | 0.6610169491525424 | 0.3305084745762712 | 0.3305084745762712 | 0.8354430379746836 | 237 | 240 | 0.898557391710394 | 0.3146928002152733 |
| CE_CLEAN_FULL_crop15 | test | 0.9014290633608816 | 0.20265151515151514 | 0.003787878787878788 | 0.6704545454545454 | 0.3371212121212121 | 0.3371212121212121 | 0.8295454545454546 | 352 | 360 | 0.8760624185754582 | 0.33454533094565614 |
| CE_CLEAN_FULL_crop15 | combined | 0.913681592039801 | 0.2007462686567164 | 0.0037313432835820895 | 0.6222222222222222 | 0.31297678275290214 | 0.31297678275290214 | 0.840782122905028 | 358 | 360 | 0.8843966369930762 | 0.31863549296111354 |

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
| CE_CLEAN_FULL_crop15 | train | frame | 4.9775390625 | -1.4594727 | 0.3146928002152733 | 0.18517562187873465 | 0.3146928002152733 | 0.18517562187873465 | 0.23421262989608313 | 0.13613861386138615 |
| CE_CLEAN_FULL_crop15 | test | frame | 4.9775390625 | -1.4594727 | 0.33454533094565614 | 0.21792350306887293 | 0.33454533094565614 | 0.21792350306887293 | 0.23280132085855806 | 0.20304568527918782 |
| CE_CLEAN_FULL_crop15 | combined | frame | 4.9775390625 | -1.4594727 | 0.31863549296111354 | 0.21305891985739941 | 0.31863549296111354 | 0.21305891985739941 | 0.2391742195367573 | 0.18694362017804153 |
| CE_CLEAN_FULL_crop15 | train | video | 4.9775390625 | -1.6680734 | 0.3305084745762712 | 0.16882498571700627 | 0.3305084745762712 | 0.16882498571700627 | 0.23595505617977527 | 0.1016949152542373 |
| CE_CLEAN_FULL_crop15 | test | video | 4.9775390625 | -1.6680734 | 0.3371212121212121 | 0.20265151515151514 | 0.3371212121212121 | 0.20265151515151514 | 0.21212121212121213 | 0.19318181818181818 |
| CE_CLEAN_FULL_crop15 | combined | video | 4.9775390625 | -1.6680734 | 0.31297678275290214 | 0.19709784411276948 | 0.31297678275290214 | 0.19709784411276948 | 0.21641791044776118 | 0.17777777777777778 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
