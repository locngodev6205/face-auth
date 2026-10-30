# CE-clean-only full-scale baseline

Loss = CE_clean only (weight1). No spectral selection, CE_worst, auxiliary forward or L_aux.
Historical C_FULL used CE_clean + L_aux. Auxiliary modules are frozen compatibility placeholders; not optimized.
Matched notebook18: full eligible Train, crop1.5 BGR80, pretrained trunk/seed105, SGD .005/.9/5e-4, batch256 AMP, max12 epochs, milestones[4,8], gamma.2, phase-aware patience9/3. Checkpoints eligible from epoch1.
CelebA full Test-as-Val is reused for selection and calibration, as in notebook18: not independent held-out Test evidence.
LCC uses the source-frozen threshold. CASIA source and optional train-only calibrated thresholds are separate; video and frame units are separate.
CASIA train and combined calibrated metrics include fitting data; Kaggle train/test copy overlap limits independence. No webcam validation or significance claim.

| Run | Dataset | Unit | Threshold policy | AUC | EER | APCER | BPCER | ACER | HTER | Scored | Candidates |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CE_CLEAN_FULL_crop15 | test | frame | source-calibrated | 0.9922898519780795 | 0.0443729361160784 | 0.05006986993442975 | 0.038058799838904554 | 0.04406433488666715 | 0.04406433488666715 | 66379 | 67170 |
| CE_CLEAN_FULL_crop15 | lcc_training | frame | source-calibrated | 0.8714290896422593 | 0.2128834744039027 | 0.0566491998300524 | 0.5057283142389526 | 0.2811887570345025 | 0.2811887570345025 | 8283 | 8299 |
| CE_CLEAN_FULL_crop15 | lcc_development | frame | source-calibrated | 0.8168424784606269 | 0.259645365270857 | 0.10700236034618411 | 0.49135802469135803 | 0.2991801925187711 | 0.2991801925187711 | 2947 | 2948 |
| CE_CLEAN_FULL_crop15 | lcc_evaluation | frame | source-calibrated | 0.8161076213485614 | 0.258567098616297 | 0.08179310344827587 | 0.5063694267515924 | 0.2940812650999341 | 0.2940812650999341 | 7564 | 7580 |
| CE_CLEAN_FULL_crop15 | lcc_combined | frame | source-calibrated | 0.8444270643082059 | 0.2400791913027499 | 0.07506082003204177 | 0.502833590932509 | 0.28894720548227537 | 0.28894720548227537 | 18794 | 18827 |
| CE_CLEAN_FULL_crop15 | casia_train | video | source-calibrated | 0.8829746714911445 | 0.1972005332317654 | 0.011235955056179775 | 0.576271186440678 | 0.2937535707484289 | 0.2937535707484289 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | source-calibrated | 0.8816807939786786 | 0.21315027186171853 | 0.032773780975219824 | 0.5495049504950495 | 0.29113936573513466 | 0.29113936573513466 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | video | source-calibrated | 0.8650568181818181 | 0.20454545454545459 | 0.026515151515151516 | 0.6477272727272727 | 0.3371212121212121 | 0.3371212121212121 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | source-calibrated | 0.8428537771209492 | 0.23120519031109638 | 0.03742432581177765 | 0.6328257191201354 | 0.33512502246595655 | 0.33512502246595655 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | source-calibrated | 0.8771973466003317 | 0.2007462686567164 | 0.026119402985074626 | 0.6333333333333333 | 0.329726368159204 | 0.329726368159204 | 358 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | source-calibrated | 0.855509710406077 | 0.21631464763731878 | 0.03625377643504532 | 0.5905044510385756 | 0.3133791137368105 | 0.3133791137368105 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | CASIA train-calibrated | 0.8816807939786786 | 0.21315027186171853 | 0.18465227817745802 | 0.23267326732673269 | 0.20866277275209535 | 0.20866277275209535 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | CASIA train-calibrated | 0.8428537771209492 | 0.23120519031109638 | 0.17006053935057788 | 0.30456852791878175 | 0.2373145336346798 | 0.2373145336346798 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | CASIA train-calibrated | 0.855509710406077 | 0.21631464763731878 | 0.17371601208459214 | 0.2774480712166172 | 0.22558204165060466 | 0.22558204165060466 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | video | CASIA train-calibrated | 0.8829746714911445 | 0.1972005332317654 | 0.21910112359550563 | 0.1694915254237288 | 0.1942963245096172 | 0.1942963245096172 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_test | video | CASIA train-calibrated | 0.8650568181818181 | 0.20454545454545459 | 0.23106060606060605 | 0.19318181818181818 | 0.2121212121212121 | 0.2121212121212121 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | CASIA train-calibrated | 0.8771973466003317 | 0.2007462686567164 | 0.23134328358208955 | 0.17777777777777778 | 0.20456053067993368 | 0.20456053067993368 | 358 | 360 |

Metrics are fractions. ACER=HTER=(APCER+BPCER)/2; Real iff logit(real)-logit(attack) >= threshold.
Source and LCC predictions/metrics: runs/<run_id>/predictions and metrics. CASIA predictions, threshold curves, coverage and report: runs/<run_id>/casia.
One new model, one seed. No new-model results exist until training/evaluation completes.
