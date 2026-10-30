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
| CE_CLEAN_FULL_crop15 | train | 0.8829746714911445 | 0.1972005332317654 | 0.011235955056179775 | 0.576271186440678 | 0.2937535707484289 | 0.2937535707484289 | 0.8481012658227848 | 237 | 240 | 0.8816807939786786 | 0.29113936573513466 |
| CE_CLEAN_FULL_crop15 | test | 0.8650568181818181 | 0.20454545454545459 | 0.026515151515151516 | 0.6477272727272727 | 0.3371212121212121 | 0.3371212121212121 | 0.8181818181818182 | 352 | 360 | 0.8428537771209492 | 0.33512502246595655 |
| CE_CLEAN_FULL_crop15 | combined | 0.8771973466003317 | 0.2007462686567164 | 0.026119402985074626 | 0.6333333333333333 | 0.329726368159204 | 0.329726368159204 | 0.8212290502793296 | 358 | 360 | 0.855509710406077 | 0.3133791137368105 |

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
| CE_CLEAN_FULL_crop15 | train | frame | 4.8642578125 | 0.08685303 | 0.29113936573513466 | 0.20866277275209535 | 0.29113936573513466 | 0.20866277275209535 | 0.18465227817745802 | 0.23267326732673269 |
| CE_CLEAN_FULL_crop15 | test | frame | 4.8642578125 | 0.08685303 | 0.33512502246595655 | 0.2373145336346798 | 0.33512502246595655 | 0.2373145336346798 | 0.17006053935057788 | 0.30456852791878175 |
| CE_CLEAN_FULL_crop15 | combined | frame | 4.8642578125 | 0.08685303 | 0.3133791137368105 | 0.22558204165060466 | 0.3133791137368105 | 0.22558204165060466 | 0.17371601208459214 | 0.2774480712166172 |
| CE_CLEAN_FULL_crop15 | train | video | 4.8642578125 | -1.6458008 | 0.2937535707484289 | 0.1942963245096172 | 0.2937535707484289 | 0.1942963245096172 | 0.21910112359550563 | 0.1694915254237288 |
| CE_CLEAN_FULL_crop15 | test | video | 4.8642578125 | -1.6458008 | 0.3371212121212121 | 0.2121212121212121 | 0.3371212121212121 | 0.2121212121212121 | 0.23106060606060605 | 0.19318181818181818 |
| CE_CLEAN_FULL_crop15 | combined | video | 4.8642578125 | -1.6458008 | 0.329726368159204 | 0.20456053067993368 | 0.329726368159204 | 0.20456053067993368 | 0.23134328358208955 | 0.17777777777777778 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
