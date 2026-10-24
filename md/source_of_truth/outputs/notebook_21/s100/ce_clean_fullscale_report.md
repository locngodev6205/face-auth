# CE-clean-only full-scale baseline

Loss = CE_clean only (weight1). No spectral selection, CE_worst, auxiliary forward or L_aux.
Historical C_FULL used CE_clean + L_aux. Auxiliary modules are frozen compatibility placeholders; not optimized.
Matched notebook18: full eligible Train, crop1.5 BGR80, pretrained trunk/seed100, SGD .005/.9/5e-4, batch256 AMP, max12 epochs, milestones[4,8], gamma.2, phase-aware patience9/3. Checkpoints eligible from epoch1.
CelebA full Test-as-Val is reused for selection and calibration, as in notebook18: not independent held-out Test evidence.
LCC uses the source-frozen threshold. CASIA source and optional train-only calibrated thresholds are separate; video and frame units are separate.
CASIA train and combined calibrated metrics include fitting data; Kaggle train/test copy overlap limits independence. No webcam validation or significance claim.

| Run | Dataset | Unit | Threshold policy | AUC | EER | APCER | BPCER | ACER | HTER | Scored | Candidates |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CE_CLEAN_FULL_crop15 | test | frame | source-calibrated | 0.9921439799017713 | 0.04475704217898091 | 0.05073632161668279 | 0.038260169150221505 | 0.04449824538345215 | 0.04449824538345215 | 66379 | 67170 |
| CE_CLEAN_FULL_crop15 | lcc_training | frame | source-calibrated | 0.8702520657603567 | 0.22532317742673103 | 0.056365953830902135 | 0.5090016366612111 | 0.28268379524605663 | 0.28268379524605663 | 8283 | 8299 |
| CE_CLEAN_FULL_crop15 | lcc_development | frame | source-calibrated | 0.8176146904838224 | 0.2696977202746938 | 0.0979543666404406 | 0.5555555555555556 | 0.32675496109799806 | 0.32675496109799806 | 2947 | 2948 |
| CE_CLEAN_FULL_crop15 | lcc_evaluation | frame | source-calibrated | 0.8126872391829563 | 0.27335646826268395 | 0.08055172413793103 | 0.5 | 0.2902758620689655 | 0.2902758620689655 | 7564 | 7580 |
| CE_CLEAN_FULL_crop15 | lcc_combined | frame | source-calibrated | 0.8439879397180328 | 0.2446227070073732 | 0.0730433750667537 | 0.5172591447707368 | 0.29515125991874525 | 0.29515125991874525 | 18794 | 18827 |
| CE_CLEAN_FULL_crop15 | casia_train | video | source-calibrated | 0.9123976385450391 | 0.19153494572462382 | 0.0056179775280898875 | 0.6779661016949152 | 0.34179203961150256 | 0.34179203961150256 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | source-calibrated | 0.9047000023743381 | 0.19082061083806223 | 0.011191047162270184 | 0.6237623762376238 | 0.31747671169994696 | 0.31747671169994696 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | video | source-calibrated | 0.8836949035812672 | 0.17992424242424238 | 0.003787878787878788 | 0.7386363636363636 | 0.3712121212121212 | 0.3712121212121212 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | source-calibrated | 0.8620939482067744 | 0.2167115054565501 | 0.011557512383048982 | 0.7174280879864636 | 0.3644928001847563 | 0.3644928001847563 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | source-calibrated | 0.8983001658374792 | 0.1654228855721393 | 0.0037313432835820895 | 0.7333333333333333 | 0.36853233830845766 | 0.36853233830845766 | 358 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | source-calibrated | 0.8743309994890047 | 0.20184690459328053 | 0.011077542799597181 | 0.685459940652819 | 0.3482687417262081 | 0.3482687417262081 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | CASIA train-calibrated | 0.9047000023743381 | 0.19082061083806223 | 0.1342925659472422 | 0.21534653465346534 | 0.1748195503003538 | 0.1748195503003538 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | CASIA train-calibrated | 0.8620939482067744 | 0.2167115054565501 | 0.16730875068794718 | 0.2622673434856176 | 0.2147880470867824 | 0.2147880470867824 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | CASIA train-calibrated | 0.8743309994890047 | 0.20184690459328053 | 0.16112789526686808 | 0.23887240356083086 | 0.20000014941384947 | 0.20000014941384947 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | video | CASIA train-calibrated | 0.9123976385450391 | 0.19153494572462382 | 0.29213483146067415 | 0.05084745762711865 | 0.1714911445438964 | 0.1714911445438964 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_test | video | CASIA train-calibrated | 0.8836949035812672 | 0.17992424242424238 | 0.3143939393939394 | 0.14772727272727273 | 0.23106060606060608 | 0.23106060606060608 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | CASIA train-calibrated | 0.8983001658374792 | 0.1654228855721393 | 0.31716417910447764 | 0.12222222222222222 | 0.21969320066334994 | 0.21969320066334994 | 358 | 360 |

Metrics are fractions. ACER=HTER=(APCER+BPCER)/2; Real iff logit(real)-logit(attack) >= threshold.
Source and LCC predictions/metrics: runs/<run_id>/predictions and metrics. CASIA predictions, threshold curves, coverage and report: runs/<run_id>/casia.
One new model, one seed. No new-model results exist until training/evaluation completes.
