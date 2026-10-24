# CE-clean-only full-scale baseline

Loss = CE_clean only (weight1). No spectral selection, CE_worst, auxiliary forward or L_aux.
Historical C_FULL used CE_clean + L_aux. Auxiliary modules are frozen compatibility placeholders; not optimized.
Matched notebook18: full eligible Train, crop1.5 BGR80, pretrained trunk/seed101, SGD .005/.9/5e-4, batch256 AMP, max12 epochs, milestones[4,8], gamma.2, phase-aware patience9/3. Checkpoints eligible from epoch1.
CelebA full Test-as-Val is reused for selection and calibration, as in notebook18: not independent held-out Test evidence.
LCC uses the source-frozen threshold. CASIA source and optional train-only calibrated thresholds are separate; video and frame units are separate.
CASIA train and combined calibrated metrics include fitting data; Kaggle train/test copy overlap limits independence. No webcam validation or significance claim.

| Run | Dataset | Unit | Threshold policy | AUC | EER | APCER | BPCER | ACER | HTER | Scored | Candidates |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CE_CLEAN_FULL_crop15 | test | frame | source-calibrated | 0.9920274344095152 | 0.044631455306381185 | 0.0509943029130388 | 0.0376057188884414 | 0.0443000109007401 | 0.0443000109007401 | 66379 | 67170 |
| CE_CLEAN_FULL_crop15 | lcc_training | frame | source-calibrated | 0.8556738206756136 | 0.23254624014115016 | 0.05976490582070528 | 0.5212765957446809 | 0.29052075078269307 | 0.29052075078269307 | 8283 | 8299 |
| CE_CLEAN_FULL_crop15 | lcc_development | frame | source-calibrated | 0.8173184330409613 | 0.27112898369127053 | 0.11408339889850512 | 0.5037037037037037 | 0.3088935513011044 | 0.3088935513011044 | 2947 | 2948 |
| CE_CLEAN_FULL_crop15 | lcc_evaluation | frame | source-calibrated | 0.8138640456841643 | 0.25400922468701964 | 0.07544827586206897 | 0.5254777070063694 | 0.3004629914342192 | 0.3004629914342192 | 7564 | 7580 |
| CE_CLEAN_FULL_crop15 | lcc_combined | frame | source-calibrated | 0.8340786483161531 | 0.24730511337650013 | 0.07470480033228505 | 0.5182895414734673 | 0.2964971709028762 | 0.2964971709028762 | 18794 | 18827 |
| CE_CLEAN_FULL_crop15 | casia_train | video | source-calibrated | 0.887545229480099 | 0.2056274995239002 | 0.02247191011235955 | 0.5932203389830508 | 0.3078461245477052 | 0.3078461245477052 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | source-calibrated | 0.8822318382917428 | 0.2078772625463985 | 0.03996802557953637 | 0.5866336633663366 | 0.3133008444729365 | 0.3133008444729365 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | video | source-calibrated | 0.8623019972451791 | 0.24621212121212122 | 0.03787878787878788 | 0.5454545454545454 | 0.29166666666666663 | 0.29166666666666663 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | source-calibrated | 0.8388834722264904 | 0.2501427112056001 | 0.041827187671986794 | 0.6023688663282571 | 0.32209802700012197 | 0.32209802700012197 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | source-calibrated | 0.8750829187396352 | 0.2304726368159204 | 0.033582089552238806 | 0.5555555555555556 | 0.2945688225538972 | 0.2945688225538972 | 358 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | source-calibrated | 0.850303011286722 | 0.23602905800544466 | 0.041289023162134945 | 0.5845697329376854 | 0.31292937804991017 | 0.31292937804991017 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | CASIA train-calibrated | 0.8822318382917428 | 0.2078772625463985 | 0.23820943245403678 | 0.1608910891089109 | 0.19955026078147384 | 0.19955026078147384 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | CASIA train-calibrated | 0.8388834722264904 | 0.2501427112056001 | 0.2344523940561365 | 0.25888324873096447 | 0.24666782139355048 | 0.24666782139355048 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | CASIA train-calibrated | 0.850303011286722 | 0.23602905800544466 | 0.23615307150050352 | 0.2344213649851632 | 0.23528721824283336 | 0.23528721824283336 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | video | CASIA train-calibrated | 0.887545229480099 | 0.2056274995239002 | 0.23595505617977527 | 0.13559322033898305 | 0.18577413825937916 | 0.18577413825937916 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_test | video | CASIA train-calibrated | 0.8623019972451791 | 0.24621212121212122 | 0.21212121212121213 | 0.25 | 0.23106060606060608 | 0.23106060606060608 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | CASIA train-calibrated | 0.8750829187396352 | 0.2304726368159204 | 0.19776119402985073 | 0.25555555555555554 | 0.22665837479270312 | 0.22665837479270312 | 358 | 360 |

Metrics are fractions. ACER=HTER=(APCER+BPCER)/2; Real iff logit(real)-logit(attack) >= threshold.
Source and LCC predictions/metrics: runs/<run_id>/predictions and metrics. CASIA predictions, threshold curves, coverage and report: runs/<run_id>/casia.
One new model, one seed. No new-model results exist until training/evaluation completes.
