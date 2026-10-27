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
| P3_SF_FULL_crop15_s101 | train | 0.9223957341458769 | 0.1718244143972577 | 0.011235955056179775 | 0.559322033898305 | 0.2852789944772424 | 0.2852789944772424 | 0.8523206751054853 | 237 | 240 | 0.9187244263994745 | 0.2612296301572603 |
| P3_SF_FULL_crop15_s101 | test | 0.896952479338843 | 0.17992424242424238 | 0.0 | 0.625 | 0.3125 | 0.3125 | 0.84375 | 352 | 360 | 0.8789836913452289 | 0.31589183561531575 |
| P3_SF_FULL_crop15_s101 | combined | 0.9128524046434494 | 0.1784411276948591 | 0.0 | 0.6111111111111112 | 0.3055555555555556 | 0.3055555555555556 | 0.8463687150837989 | 358 | 360 | 0.8915546809864899 | 0.29195615600001196 |
| P3_SF_FULL_crop15_s102 | train | 0.885355170443725 | 0.2197200533231765 | 0.02247191011235955 | 0.5423728813559322 | 0.28242239573414585 | 0.28242239573414585 | 0.8481012658227848 | 237 | 240 | 0.8866639361777905 | 0.2511495753891936 |
| P3_SF_FULL_crop15_s102 | test | 0.8779269972451791 | 0.21780303030303033 | 0.030303030303030304 | 0.6477272727272727 | 0.3390151515151515 | 0.3390151515151515 | 0.8153409090909091 | 352 | 360 | 0.8509359340762698 | 0.322862102329289 |
| P3_SF_FULL_crop15_s102 | combined | 0.8922470978441127 | 0.21189883913764512 | 0.026119402985074626 | 0.6 | 0.3130597014925373 | 0.3130597014925373 | 0.8296089385474861 | 358 | 360 | 0.8656332457768176 | 0.29789460944713886 |
| R7_SC_FULL_crop15_s101 | train | 0.9152542372881356 | 0.19434393448866882 | 0.0 | 0.6101694915254238 | 0.3050847457627119 | 0.3050847457627119 | 0.8481012658227848 | 237 | 240 | 0.9128974048483984 | 0.28110877634526044 |
| R7_SC_FULL_crop15_s101 | test | 0.8909263085399448 | 0.20643939393939392 | 0.0 | 0.6363636363636364 | 0.3181818181818182 | 0.3181818181818182 | 0.8409090909090909 | 352 | 360 | 0.8690181189685309 | 0.32155930965956975 |
| R7_SC_FULL_crop15_s101 | combined | 0.9074626865671641 | 0.19705638474295195 | 0.0 | 0.5888888888888889 | 0.29444444444444445 | 0.29444444444444445 | 0.8519553072625698 | 358 | 360 | 0.8832166411168983 | 0.2986192666170613 |
| R7_SC_FULL_crop15_s102 | train | 0.8847838506951058 | 0.2000095219958103 | 0.02247191011235955 | 0.6101694915254238 | 0.31632070081889163 | 0.31632070081889163 | 0.8312236286919831 | 237 | 240 | 0.8812890281833938 | 0.2827985136643161 |
| R7_SC_FULL_crop15_s102 | test | 0.8680268595041323 | 0.2121212121212121 | 0.011363636363636364 | 0.625 | 0.3181818181818182 | 0.3181818181818182 | 0.8352272727272727 | 352 | 360 | 0.840765490800831 | 0.33146574884504026 |
| R7_SC_FULL_crop15_s102 | combined | 0.8808872305140962 | 0.2137645107794362 | 0.011194029850746268 | 0.6111111111111112 | 0.3111525704809287 | 0.3111525704809287 | 0.8379888268156425 | 358 | 360 | 0.8559516018658802 | 0.30916639025104514 |

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
| P3_SF_FULL_crop15_s101 | train | frame | 5.14599609375 | 0.24615479 | 0.2612296301572603 | 0.17249863475556187 | 0.2612296301572603 | 0.17249863475556187 | 0.12470023980815348 | 0.2202970297029703 |
| P3_SF_FULL_crop15_s101 | test | frame | 5.14599609375 | 0.24615479 | 0.31589183561531575 | 0.21946888150732835 | 0.31589183561531575 | 0.21946888150732835 | 0.1309851403412218 | 0.3079526226734349 |
| P3_SF_FULL_crop15_s101 | combined | frame | 5.14599609375 | 0.24615479 | 0.29195615600001196 | 0.20191638203328344 | 0.29195615600001196 | 0.20191638203328344 | 0.12638469284994966 | 0.2774480712166172 |
| P3_SF_FULL_crop15_s101 | train | video | 5.14599609375 | -1.3437996 | 0.2852789944772424 | 0.16334983812607123 | 0.2852789944772424 | 0.16334983812607123 | 0.17415730337078653 | 0.15254237288135594 |
| P3_SF_FULL_crop15_s101 | test | video | 5.14599609375 | -1.3437996 | 0.3125 | 0.1856060606060606 | 0.3125 | 0.1856060606060606 | 0.1553030303030303 | 0.2159090909090909 |
| P3_SF_FULL_crop15_s101 | combined | video | 5.14599609375 | -1.3437996 | 0.3055555555555556 | 0.17649253731343284 | 0.3055555555555556 | 0.17649253731343284 | 0.15298507462686567 | 0.2 |
| P3_SF_FULL_crop15_s102 | train | frame | 5.2490234375 | 0.03652954 | 0.2511495753891936 | 0.19722934523668195 | 0.2511495753891936 | 0.19722934523668195 | 0.22861710631494805 | 0.16584158415841585 |
| P3_SF_FULL_crop15_s102 | test | frame | 5.2490234375 | 0.03652954 | 0.322862102329289 | 0.24168247431896722 | 0.322862102329289 | 0.24168247431896722 | 0.24986241056686848 | 0.233502538071066 |
| P3_SF_FULL_crop15_s102 | combined | frame | 5.2490234375 | 0.03652954 | 0.29789460944713886 | 0.2264523773237589 | 0.29789460944713886 | 0.2264523773237589 | 0.24370594159113795 | 0.20919881305637983 |
| P3_SF_FULL_crop15_s102 | train | video | 5.2490234375 | 1.2649286 | 0.28242239573414585 | 0.1916777756617787 | 0.28242239573414585 | 0.1916777756617787 | 0.14606741573033707 | 0.23728813559322035 |
| P3_SF_FULL_crop15_s102 | test | video | 5.2490234375 | 1.2649286 | 0.3390151515151515 | 0.21969696969696972 | 0.3390151515151515 | 0.21969696969696972 | 0.1553030303030303 | 0.2840909090909091 |
| P3_SF_FULL_crop15_s102 | combined | video | 5.2490234375 | 1.2649286 | 0.3130597014925373 | 0.21538142620232173 | 0.3130597014925373 | 0.21538142620232173 | 0.15298507462686567 | 0.2777777777777778 |
| R7_SC_FULL_crop15_s101 | train | frame | 5.8095703125 | -0.90405273 | 0.28110877634526044 | 0.1717022421666627 | 0.28110877634526044 | 0.1717022421666627 | 0.19984012789768185 | 0.14356435643564355 |
| R7_SC_FULL_crop15_s101 | test | frame | 5.8095703125 | -0.90405273 | 0.32155930965956975 | 0.2085380878281543 | 0.32155930965956975 | 0.2085380878281543 | 0.19372592184920198 | 0.2233502538071066 |
| R7_SC_FULL_crop15_s101 | combined | frame | 5.8095703125 | -0.90405273 | 0.2986192666170613 | 0.1938174043228415 | 0.2986192666170613 | 0.1938174043228415 | 0.18882175226586104 | 0.19881305637982197 |
| R7_SC_FULL_crop15_s101 | train | video | 5.8095703125 | -2.2544045 | 0.3050847457627119 | 0.16596838697390973 | 0.3050847457627119 | 0.16596838697390973 | 0.24719101123595505 | 0.0847457627118644 |
| R7_SC_FULL_crop15_s101 | test | video | 5.8095703125 | -2.2544045 | 0.3181818181818182 | 0.1893939393939394 | 0.3181818181818182 | 0.1893939393939394 | 0.23106060606060605 | 0.14772727272727273 |
| R7_SC_FULL_crop15_s101 | combined | video | 5.8095703125 | -2.2544045 | 0.29444444444444445 | 0.1804726368159204 | 0.29444444444444445 | 0.1804726368159204 | 0.22761194029850745 | 0.13333333333333333 |
| R7_SC_FULL_crop15_s102 | train | frame | 5.3798828125 | 0.18395996 | 0.2827985136643161 | 0.20856087407301882 | 0.2827985136643161 | 0.20856087407301882 | 0.21662669864108713 | 0.2004950495049505 |
| R7_SC_FULL_crop15_s102 | test | frame | 5.3798828125 | 0.18395996 | 0.33146574884504026 | 0.2427948301759934 | 0.33146574884504026 | 0.2427948301759934 | 0.22839845899834893 | 0.2571912013536379 |
| R7_SC_FULL_crop15_s102 | combined | frame | 5.3798828125 | 0.18395996 | 0.30916639025104514 | 0.22952357899958464 | 0.30916639025104514 | 0.22952357899958464 | 0.22759315206445116 | 0.2314540059347181 |
| R7_SC_FULL_crop15_s102 | train | video | 5.3798828125 | -0.77854 | 0.31632070081889163 | 0.19139211578746906 | 0.31632070081889163 | 0.19139211578746906 | 0.24719101123595505 | 0.13559322033898305 |
| R7_SC_FULL_crop15_s102 | test | video | 5.3798828125 | -0.77854 | 0.3181818181818182 | 0.23295454545454547 | 0.3181818181818182 | 0.23295454545454547 | 0.26136363636363635 | 0.20454545454545456 |
| R7_SC_FULL_crop15_s102 | combined | video | 5.3798828125 | -0.77854 | 0.3111525704809287 | 0.21944444444444444 | 0.3111525704809287 | 0.21944444444444444 | 0.25 | 0.18888888888888888 |

Exports: `casia_calibrated_threshold_summary.csv/json`; per-run frame/video calibration JSON and train threshold curves; per-split `*_metrics_casia_calibrated.json` and `*_predictions_casia_calibrated.csv`.
Original source-frozen summaries, predictions and metrics are retained. Runtime configs are not overwritten.
