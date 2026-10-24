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
| C_FULL_crop15_s101 | train | 0.8966863454580081 | 0.1943915444677205 | 0.0056179775280898875 | 0.6101694915254238 | 0.30789373452675683 | 0.30789373452675683 | 0.8438818565400844 | 237 | 240 | 0.9009238154031229 | 0.2756428520549897 |
| C_FULL_crop15_s101 | test | 0.8910554407713498 | 0.2272727272727273 | 0.0 | 0.6136363636363636 | 0.3068181818181818 | 0.3068181818181818 | 0.8465909090909091 | 352 | 360 | 0.8670629987325941 | 0.3187255726374427 |
| C_FULL_crop15_s101 | combined | 0.906301824212272 | 0.2137645107794362 | 0.0 | 0.5888888888888889 | 0.29444444444444445 | 0.29444444444444445 | 0.8519553072625698 | 358 | 360 | 0.8802171580888176 | 0.2961554322393251 |
| C_FULL_crop15_s102 | train | 0.8979242049133498 | 0.18591696819653403 | 0.011235955056179775 | 0.559322033898305 | 0.2852789944772424 | 0.2852789944772424 | 0.8523206751054853 | 237 | 240 | 0.8940313096057807 | 0.2663868904876099 |
| C_FULL_crop15_s102 | test | 0.8911845730027548 | 0.21590909090909088 | 0.003787878787878788 | 0.5681818181818182 | 0.2859848484848485 | 0.2859848484848485 | 0.8551136363636364 | 352 | 360 | 0.8619998938396253 | 0.314647710521145 |
| C_FULL_crop15_s102 | combined | 0.9013266998341627 | 0.1951492537313433 | 0.0037313432835820895 | 0.5444444444444444 | 0.2740878938640132 | 0.2740878938640132 | 0.8603351955307262 | 358 | 360 | 0.8737807829883368 | 0.29328220391404525 |

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
| C_FULL_crop15_s101 | train | frame | 5.23046875 | -0.34155273 | 0.2756428520549897 | 0.19598875355161416 | 0.2756428520549897 | 0.19598875355161416 | 0.1518784972022382 | 0.2400990099009901 |
| C_FULL_crop15_s101 | test | frame | 5.23046875 | -0.34155273 | 0.3187255726374427 | 0.2388748117748618 | 0.3187255726374427 | 0.2388748117748618 | 0.15795266923500276 | 0.3197969543147208 |
| C_FULL_crop15_s101 | combined | frame | 5.23046875 | -0.34155273 | 0.2961554322393251 | 0.22168458138721794 | 0.2961554322393251 | 0.22168458138721794 | 0.15256797583081572 | 0.29080118694362017 |
| C_FULL_crop15_s101 | train | video | 5.23046875 | -0.99026054 | 0.30789373452675683 | 0.18882117691868217 | 0.30789373452675683 | 0.18882117691868217 | 0.15730337078651685 | 0.22033898305084745 |
| C_FULL_crop15_s101 | test | video | 5.23046875 | -0.99026054 | 0.3068181818181818 | 0.2253787878787879 | 0.3068181818181818 | 0.2253787878787879 | 0.14393939393939395 | 0.3068181818181818 |
| C_FULL_crop15_s101 | combined | video | 5.23046875 | -0.99026054 | 0.29444444444444445 | 0.21902985074626866 | 0.29444444444444445 | 0.21902985074626866 | 0.13805970149253732 | 0.3 |
| C_FULL_crop15_s102 | train | frame | 5.0888671875 | 2.8095703 | 0.2663868904876099 | 0.19188708439189242 | 0.2663868904876099 | 0.19188708439189242 | 0.0991207034372502 | 0.28465346534653463 |
| C_FULL_crop15_s102 | test | frame | 5.0888671875 | 2.8095703 | 0.314647710521145 | 0.2604877603606473 | 0.314647710521145 | 0.2604877603606473 | 0.09796367638965328 | 0.4230118443316413 |
| C_FULL_crop15_s102 | combined | frame | 5.0888671875 | 2.8095703 | 0.29328220391404525 | 0.24072662943273537 | 0.29328220391404525 | 0.24072662943273537 | 0.09718026183282981 | 0.38427299703264095 |
| C_FULL_crop15_s102 | train | video | 5.0888671875 | 0.18251953 | 0.2852789944772424 | 0.17739478194629593 | 0.2852789944772424 | 0.17739478194629593 | 0.20224719101123595 | 0.15254237288135594 |
| C_FULL_crop15_s102 | test | video | 5.0888671875 | 0.18251953 | 0.2859848484848485 | 0.2140151515151515 | 0.2859848484848485 | 0.2140151515151515 | 0.20075757575757575 | 0.22727272727272727 |
| C_FULL_crop15_s102 | combined | video | 5.0888671875 | 0.18251953 | 0.2740878938640132 | 0.20626036484245439 | 0.2740878938640132 | 0.20626036484245439 | 0.19029850746268656 | 0.2222222222222222 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
