# Fullscale C / P3_SF / R7_SC — crop1.5

Full eligible official Train → full official Test-as-Val → full LCC train/dev/eval/Combined. Test-as-Val is a selection set, not an untouched independent Test.
The full CelebA-Spoof training set is approximately 500K samples. Therefore the full-scale training budget uses 12 epochs with milestones [4,8], while preserving the same three-phase SGD schedule structure.
New phase-aware patience: patience_start_epoch=9, early_stop_patience=3, earliest stop11, max12; validation and best selection are not delayed. This differs from older historical scale100K training protocols.
C eligible epoch1; P3/R7 epoch1 clean warmup excluded. All methods retain SGD .005/.9/5e-4, milestones[4,8], gamma.2, batch256 AMP, TRAIN_SEED101, DATA_SEED43.
Frozen P3_SF scaled centers and sigma; R7_SC same centers and sigma.10; clean/spectral .75/.25, clean-only auxiliary losses. No mechanism tuning or fullscale survival gate.
Reload best and re-infer full usable Test; freeze its calibrated threshold/checkpoint SHA before LCC. Reference metrics use the same logits and require no additional inference.

## Dataset accounting



## Project metrics (source-calibrated threshold)

| Run | selected_epoch | test_auc | test_eer | test_acer | lcc_evaluation_auc | lcc_evaluation_eer | lcc_evaluation_hter | lcc_combined_auc | lcc_combined_eer | lcc_combined_hter | lcc_combined_tpr_at_fpr_1pct | lcc_combined_detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | 9.000000 | 0.989987 | 0.049093 | 0.048975 | 0.822530 | 0.254078 | 0.279781 | 0.854042 | 0.231863 | 0.272004 | 0.246023 | 0.998247 |
| P3_SF_FULL_crop15 | 9.000000 | 0.992498 | 0.044631 | 0.044390 | 0.849955 | 0.224017 | 0.255193 | 0.876375 | 0.207216 | 0.248546 | 0.218444 | 0.998247 |
| R7_SC_FULL_crop15 | 10.000000 | 0.991916 | 0.045188 | 0.044986 | 0.833133 | 0.238806 | 0.293704 | 0.861868 | 0.221638 | 0.289721 | 0.184441 | 0.998247 |

## Full Test and full LCC comparison with the published reference

Same full split scope as confirmed by the user; preprocessing differs. Our full Test numbers reuse the checkpoint-selection set. Published benchmark values are copied from the pinned README, not rerun here.
Reference APCER variable means real→attack; BPCER means attack→real. This table preserves those names. Reference EER=min(FPR,FNR) at nearest crossing; project EER averages them. Reference rates use binary argmax, independently of our calibrated threshold.
Params are deployment parameters (millions); our GFLOPs cover profiler-supported Conv/Linear operations at 80x80. Published reference input is 128x128 and counting conventions may differ. FFT/selection is training-only.
Full LCC comparison uses Combined. Physical Evaluation metrics are additionally reported above.

| Dataset | Model | AUC | EER % | APCER % | BPCER % | ACER % | Params M | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CelebA | MN3_large (published) | 0.998000 | 2.260000 | 0.690000 | 6.920000 | 3.800000 | 3.020000 | 0.150000 |
| CelebA | AENET (published) | 0.999000 | 1.120000 | 0.230000 | 6.270000 | 3.250000 | 11.220000 | 3.640000 |
| CelebA | MN3_large_075 (published) | 0.997000 | 2.790000 | 1.210000 | 5.440000 | 3.320000 | 1.860000 | 0.100000 |
| CelebA | MN3_small (published) | 0.994000 | 3.840000 | 1.470000 | 8.630000 | 5.050000 | 1.000000 | 0.040000 |
| CelebA | MN3_small_075 (published) | 0.991000 | 4.740000 | 1.620000 | 10.550000 | 6.090000 | 0.600000 | 0.030000 |
| LCC | MN3_large (published) | 0.921000 | 16.130000 | 17.260000 | 15.400000 | 16.330000 | 3.020000 | 0.150000 |
| LCC | AENET (published) | 0.868000 | 20.910000 | 12.520000 | 32.700000 | 22.610000 | 11.220000 | 3.640000 |
| LCC | MN3_large_075 (published) | 0.892000 | 19.420000 | 28.340000 | 12.180000 | 20.260000 | 1.860000 | 0.100000 |
| LCC | MN3_small (published) | 0.889000 | 18.700000 | 14.790000 | 24.600000 | 19.690000 | 1.000000 | 0.040000 |
| LCC | MN3_small_075 (published) | 0.879000 | 21.070000 | 22.770000 | 19.300000 | 21.040000 | 0.600000 | 0.030000 |
| CelebA Test-as-Val | C_FULL_crop15 | 0.989987 | 4.908377 | 0.468184 | 22.646458 | 11.557321 | 0.434434 | 0.081622 |
| CelebA Test-as-Val | P3_SF_FULL_crop15 | 0.992498 | 4.460927 | 0.417841 | 22.904439 | 11.661140 | 0.434434 | 0.081622 |
| CelebA Test-as-Val | R7_SC_FULL_crop15 | 0.991916 | 4.516823 | 0.327225 | 24.641513 | 12.484369 | 0.434434 | 0.081622 |
| LCC Combined | C_FULL_crop15 | 0.854042 | 23.183926 | 18.650180 | 28.339168 | 23.494674 | 0.434434 | 0.081622 |
| LCC Combined | P3_SF_FULL_crop15 | 0.876375 | 20.710974 | 12.570840 | 30.101466 | 21.336153 | 0.434434 | 0.081622 |
| LCC Combined | R7_SC_FULL_crop15 | 0.861868 | 22.153529 | 12.158681 | 34.124488 | 23.141585 | 0.434434 | 0.081622 |

## Physical LCC metrics and coverage

### test (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | 0.989987 | 0.049093 | 0.054907 | 0.043043 | 0.048975 | 0.048975 | 0.948643 | 67170.000000 | 66379.000000 | 0.988224 |
| P3_SF_FULL_crop15 | 0.992498 | 0.044631 | 0.049059 | 0.039720 | 0.044390 | 0.044390 | 0.953735 | 67170.000000 | 66379.000000 | 0.988224 |
| R7_SC_FULL_crop15 | 0.991916 | 0.045188 | 0.045319 | 0.044654 | 0.044986 | 0.044986 | 0.954880 | 67170.000000 | 66379.000000 | 0.988224 |

### lcc_training (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | 0.874517 | 0.215386 | 0.047727 | 0.500000 | 0.273863 | 0.273863 | 0.885549 | 8299.000000 | 8283.000000 | 0.998072 |
| P3_SF_FULL_crop15 | 0.892770 | 0.198422 | 0.054808 | 0.448445 | 0.251627 | 0.251627 | 0.887118 | 8299.000000 | 8283.000000 | 0.998072 |
| R7_SC_FULL_crop15 | 0.883276 | 0.206424 | 0.044611 | 0.543372 | 0.293991 | 0.293991 | 0.881806 | 8299.000000 | 8283.000000 | 0.998072 |

### lcc_development (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | 0.842881 | 0.227270 | 0.112903 | 0.402469 | 0.257686 | 0.257686 | 0.847302 | 2948.000000 | 2947.000000 | 0.999661 |
| P3_SF_FULL_crop15 | 0.864092 | 0.221851 | 0.126672 | 0.340741 | 0.233706 | 0.233706 | 0.843909 | 2948.000000 | 2947.000000 | 0.999661 |
| R7_SC_FULL_crop15 | 0.838059 | 0.247375 | 0.121951 | 0.424691 | 0.273321 | 0.273321 | 0.836444 | 2948.000000 | 2947.000000 | 0.999661 |

### lcc_evaluation (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | 0.822530 | 0.254078 | 0.065931 | 0.493631 | 0.279781 | 0.279781 | 0.916314 | 7580.000000 | 7564.000000 | 0.997889 |
| P3_SF_FULL_crop15 | 0.849955 | 0.224017 | 0.070897 | 0.439490 | 0.255193 | 0.255193 | 0.913802 | 7580.000000 | 7564.000000 | 0.997889 |
| R7_SC_FULL_crop15 | 0.833133 | 0.238806 | 0.061931 | 0.525478 | 0.293704 | 0.293704 | 0.918826 | 7580.000000 | 7564.000000 | 0.997889 |

### lcc_combined (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | 0.854042 | 0.231863 | 0.065389 | 0.478619 | 0.272004 | 0.272004 | 0.891934 | 18827.000000 | 18794.000000 | 0.998247 |
| P3_SF_FULL_crop15 | 0.876375 | 0.207216 | 0.072569 | 0.424523 | 0.248546 | 0.248546 | 0.891082 | 18827.000000 | 18794.000000 | 0.998247 |
| R7_SC_FULL_crop15 | 0.861868 | 0.221638 | 0.063728 | 0.515714 | 0.289721 | 0.289721 | 0.889592 | 18827.000000 | 18794.000000 | 0.998247 |

## Descriptive fullscale vs 100K context

Pairwise Combined deltas (first minus second): {"P3_minus_C": {"auc": 0.022332715908477074, "eer": -0.02464744007437336, "hter": -0.023458048140796744, "tpr_at_fpr_1pct": -0.02757856774858322}, "R7_minus_C": {"auc": 0.007825845532266174, "eer": -0.010225264235185993, "hter": 0.017716428016384234, "tpr_at_fpr_1pct": -0.061581658938691414}, "R7_minus_P3": {"auc": -0.0145068703762109, "eer": 0.014422175839187368, "hter": 0.04117447615718098, "tpr_at_fpr_1pct": -0.03400309119010819}}.
Completed fullscale Combined-AUC order: ['P3_SF_FULL_crop15', 'R7_SC_FULL_crop15', 'C_FULL_crop15']. Full ordering pending until all three complete.
Read-only100K AUC: {"C": 0.848273, "P3_SF": 0.851978, "R7_SC": 0.867844}; ordering ['R7_SC', 'P3_SF', 'C']. No PASS/FAIL gate and no statistical superiority claim.
Training diagnostics remain in each run/training_history.csv and spectral_diagnostics.csv (P3/R7): clean/worst/harmful CE, harmful counts/fraction, margin/drop, flip, gains, selected bands.

## Limitations

- One seed; descriptive comparison, no significance or multi-seed claim.
- CelebA full Test is used for selection and threshold calibration; its final metrics are not independent Test evidence.
- LCC has been repeatedly inspected during method development. This is a confirmatory experiment with a frozen design, not a fresh independent benchmark.
- Official identity overlap is disclosed and retained; no new population filtering or split construction.
- Reference and project preprocessing/localization differ; the size of their effect has not been measured. Scored coverage is explicit.
- Published reference values, including imported AENET, are not reproduced runs; the metric audit and complexity scope document the conventions.
- Multi-seed confirmation and additional independent validation remain future work.

## Artifacts

Dataset counts, manifest copies/hashes, init equality, config, best/last checkpoint, evaluation_freeze.json, both metric layers, complexity, histories and predictions are retained.
