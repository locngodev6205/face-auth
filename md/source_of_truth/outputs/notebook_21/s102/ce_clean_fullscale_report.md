# CE-clean-only full-scale baseline

Loss = CE_clean only (weight1). No spectral selection, CE_worst, auxiliary forward or L_aux.
Historical C_FULL used CE_clean + L_aux. Auxiliary modules are frozen compatibility placeholders; not optimized.
Matched notebook18: full eligible Train, crop1.5 BGR80, pretrained trunk/seed101, SGD .005/.9/5e-4, batch256 AMP, max12 epochs, milestones[4,8], gamma.2, phase-aware patience9/3. Checkpoints eligible from epoch1.
CelebA full Test-as-Val is reused for selection and calibration, as in notebook18: not independent held-out Test evidence.
LCC uses the source-frozen threshold. CASIA source and optional train-only calibrated thresholds are separate; video and frame units are separate.
CASIA train and combined calibrated metrics include fitting data; Kaggle train/test copy overlap limits independence. No webcam validation or significance claim.

| Run | Dataset | Unit | Threshold policy | AUC | EER | APCER | BPCER | ACER | HTER | Scored | Candidates |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CE_CLEAN_FULL_crop15 | test | frame | source-calibrated | 0.9917187942179668 | 0.044512944931911325 | 0.04850048371493067 | 0.04022351993556182 | 0.044362001825246244 | 0.044362001825246244 | 66379 | 67170 |
| CE_CLEAN_FULL_crop15 | lcc_training | frame | source-calibrated | 0.8544641725102572 | 0.23109060603749737 | 0.053391870839824385 | 0.5540098199672667 | 0.3037008454035456 | 0.3037008454035456 | 8283 | 8299 |
| CE_CLEAN_FULL_crop15 | lcc_development | frame | source-calibrated | 0.8277364960029528 | 0.2337276956998961 | 0.0924468922108576 | 0.5530864197530864 | 0.32276665598197196 | 0.32276665598197196 | 2947 | 2948 |
| CE_CLEAN_FULL_crop15 | lcc_evaluation | frame | source-calibrated | 0.8056755985064792 | 0.2608491104766088 | 0.08496551724137931 | 0.5509554140127388 | 0.3179604656270591 | 0.3179604656270591 | 7564 | 7580 |
| CE_CLEAN_FULL_crop15 | lcc_combined | frame | source-calibrated | 0.8323860415210191 | 0.24165303315425052 | 0.07286536521687534 | 0.5533230293663061 | 0.3130941972915907 | 0.3130941972915907 | 18794 | 18827 |
| CE_CLEAN_FULL_crop15 | casia_train | video | source-calibrated | 0.9103027994667683 | 0.19434393448866882 | 0.0 | 0.6610169491525424 | 0.3305084745762712 | 0.3305084745762712 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | source-calibrated | 0.898557391710394 | 0.19892996493893988 | 0.030375699440447643 | 0.599009900990099 | 0.3146928002152733 | 0.3146928002152733 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | video | source-calibrated | 0.9014290633608816 | 0.20265151515151514 | 0.003787878787878788 | 0.6704545454545454 | 0.3371212121212121 | 0.3371212121212121 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | source-calibrated | 0.8760624185754582 | 0.21588596885776096 | 0.015960374243258118 | 0.6531302876480541 | 0.33454533094565614 | 0.33454533094565614 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | source-calibrated | 0.913681592039801 | 0.2007462686567164 | 0.0037313432835820895 | 0.6222222222222222 | 0.31297678275290214 | 0.31297678275290214 | 358 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | source-calibrated | 0.8843966369930762 | 0.21033958779707207 | 0.015609264853977844 | 0.6216617210682492 | 0.31863549296111354 | 0.31863549296111354 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | CASIA train-calibrated | 0.898557391710394 | 0.19892996493893988 | 0.23421262989608313 | 0.13613861386138615 | 0.18517562187873465 | 0.18517562187873465 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | CASIA train-calibrated | 0.8760624185754582 | 0.21588596885776096 | 0.23280132085855806 | 0.20304568527918782 | 0.21792350306887293 | 0.21792350306887293 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | CASIA train-calibrated | 0.8843966369930762 | 0.21033958779707207 | 0.2391742195367573 | 0.18694362017804153 | 0.21305891985739941 | 0.21305891985739941 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | video | CASIA train-calibrated | 0.9103027994667683 | 0.19434393448866882 | 0.23595505617977527 | 0.1016949152542373 | 0.16882498571700627 | 0.16882498571700627 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_test | video | CASIA train-calibrated | 0.9014290633608816 | 0.20265151515151514 | 0.21212121212121213 | 0.19318181818181818 | 0.20265151515151514 | 0.20265151515151514 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | CASIA train-calibrated | 0.913681592039801 | 0.2007462686567164 | 0.21641791044776118 | 0.17777777777777778 | 0.19709784411276948 | 0.19709784411276948 | 358 | 360 |

Metrics are fractions. ACER=HTER=(APCER+BPCER)/2; Real iff logit(real)-logit(attack) >= threshold.
Source and LCC predictions/metrics: runs/<run_id>/predictions and metrics. CASIA predictions, threshold curves, coverage and report: runs/<run_id>/casia.
One new model, one seed. No new-model results exist until training/evaluation completes.
