# Fullscale C / P3_SF / R7_SC — crop1.5

Full eligible official Train → full official Test-as-Val → full LCC train/dev/eval/Combined. Test-as-Val is a selection set, not an untouched independent Test.
The full CelebA-Spoof training set is approximately 500K samples. Therefore the full-scale training budget uses 12 epochs with milestones [4,8], while preserving the same three-phase SGD schedule structure.
New phase-aware patience: patience_start_epoch=9, early_stop_patience=3, earliest stop11, max12; validation and best selection are not delayed. This differs from older historical scale100K training protocols.
C eligible epoch1; P3/R7 epoch1 clean warmup excluded. All methods retain SGD .005/.9/5e-4, milestones[4,8], gamma.2, batch256 AMP, TRAIN_SEED101, DATA_SEED43.
Frozen P3_SF scaled centers and sigma; R7_SC same centers and sigma.10; clean/spectral .75/.25, clean-only auxiliary losses. No mechanism tuning or fullscale survival gate.
Reload best and re-infer full usable Test; freeze its calibrated threshold/checkpoint SHA before LCC. Reference metrics use the same logits and require no additional inference.

## Dataset accounting

```json
{
  "train_candidates": {
    "n": 494405,
    "three_class_counts": {
      "0": 162462,
      "1": 226715,
      "2": 105228
    },
    "real_n": 162462,
    "attack_n": 331943,
    "spoof_type_counts": {
      "0": 162462,
      "1": 35547,
      "2": 31221,
      "3": 31776,
      "4": 33647,
      "5": 30167,
      "6": 33285,
      "7": 31072,
      "8": 33085,
      "9": 31527,
      "10": 40616
    }
  },
  "train_usable": {
    "n": 484561,
    "three_class_counts": {
      "0": 160073,
      "1": 222731,
      "2": 101757
    },
    "real_n": 160073,
    "attack_n": 324488,
    "spoof_type_counts": {
      "0": 160073,
      "1": 35381,
      "2": 29815,
      "3": 31381,
      "4": 33239,
      "5": 29113,
      "6": 33211,
      "7": 30591,
      "8": 32861,
      "9": 28647,
      "10": 40249
    }
  },
  "train_excluded": 9844,
  "train_exclusion_reasons": {
    "DET_FAIL": 4580,
    "TOO_SMALL": 4484,
    "SUSPICIOUS": 779,
    "INVALID_BBOX": 1
  },
  "test_candidates": {
    "n": 67170,
    "three_class_counts": {
      "0": 19923,
      "1": 35495,
      "2": 11752
    },
    "real_n": 19923,
    "attack_n": 47247,
    "spoof_type_counts": {
      "0": 19923,
      "1": 3600,
      "2": 5421,
      "3": 6083,
      "4": 4287,
      "5": 6097,
      "6": 3530,
      "7": 6477,
      "8": 3659,
      "9": 4483,
      "10": 3610
    }
  },
  "test_usable": {
    "n": 66379,
    "three_class_counts": {
      "0": 19864,
      "1": 34834,
      "2": 11681
    },
    "real_n": 19864,
    "attack_n": 46515,
    "spoof_type_counts": {
      "0": 19864,
      "1": 3592,
      "2": 5286,
      "3": 5657,
      "4": 4243,
      "5": 6070,
      "6": 3509,
      "7": 6477,
      "8": 3655,
      "9": 4458,
      "10": 3568
    }
  },
  "test_unscored": 791,
  "train_test_identity_overlap": [
    "5028",
    "7332",
    "9735"
  ],
  "identity_overlap_policy": "Official membership retained; no subject removal or new split",
  "sampling": "None; full official eligible Train and full official Test-as-Val",
  "data_seed": 43
}
```

## Project metrics (source-calibrated threshold)

| Run | selected_epoch | test_auc | test_eer | test_acer | lcc_evaluation_auc | lcc_evaluation_eer | lcc_evaluation_hter | lcc_combined_auc | lcc_combined_eer | lcc_combined_hter | lcc_combined_tpr_at_fpr_1pct | lcc_combined_detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| P3_SF_FULL_crop15 | 9.000000 | 0.992498 | 0.044631 | 0.044390 | 0.849955 | 0.224017 | 0.255193 | 0.876375 | 0.207216 | 0.248546 | 0.218444 | 0.998247 |
| R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

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
| CelebA Test-as-Val | C_FULL_crop15 | pending | pending | pending | pending | pending | 0.434434 | 0.081622 |
| CelebA Test-as-Val | P3_SF_FULL_crop15 | 0.992498 | 4.460927 | 0.417841 | 22.904439 | 11.661140 | 0.434434 | 0.081622 |
| CelebA Test-as-Val | R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | 0.434434 | 0.081622 |
| LCC Combined | C_FULL_crop15 | pending | pending | pending | pending | pending | 0.434434 | 0.081622 |
| LCC Combined | P3_SF_FULL_crop15 | 0.876375 | 20.710974 | 12.570840 | 30.101466 | 21.336153 | 0.434434 | 0.081622 |
| LCC Combined | R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | 0.434434 | 0.081622 |

## Physical LCC metrics and coverage

### test (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| P3_SF_FULL_crop15 | 0.992498 | 0.044631 | 0.049059 | 0.039720 | 0.044390 | 0.044390 | 0.953735 | 67170.000000 | 66379.000000 | 0.988224 |
| R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

### lcc_training (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| P3_SF_FULL_crop15 | 0.892770 | 0.198422 | 0.054808 | 0.448445 | 0.251627 | 0.251627 | 0.887118 | 8299.000000 | 8283.000000 | 0.998072 |
| R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

### lcc_development (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| P3_SF_FULL_crop15 | 0.864092 | 0.221851 | 0.126672 | 0.340741 | 0.233706 | 0.233706 | 0.843909 | 2948.000000 | 2947.000000 | 0.999661 |
| R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

### lcc_evaluation (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| P3_SF_FULL_crop15 | 0.849955 | 0.224017 | 0.070897 | 0.439490 | 0.255193 | 0.255193 | 0.913802 | 7580.000000 | 7564.000000 | 0.997889 |
| R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

### lcc_combined (project)

| Run | auc | eer | apcer | bpcer | acer | hter | binary_accuracy | candidate_n | scored_n | detector_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| P3_SF_FULL_crop15 | 0.876375 | 0.207216 | 0.072569 | 0.424523 | 0.248546 | 0.248546 | 0.891082 | 18827.000000 | 18794.000000 | 0.998247 |
| R7_SC_FULL_crop15 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Descriptive fullscale vs 100K context

Pairwise Combined deltas (first minus second): {"P3_minus_C": null, "R7_minus_C": null, "R7_minus_P3": null}.
Completed fullscale Combined-AUC order: ['P3_SF_FULL_crop15']. Full ordering pending until all three complete.
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
