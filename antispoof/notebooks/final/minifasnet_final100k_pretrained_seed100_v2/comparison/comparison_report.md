# MiniFASNetV2 paired final experiment

## Objective
Measure the frozen CSMR-v1 effect against Vanilla with identical initialization and source resources.

## Initialization
MiniFASNetV2 used the released PAD-pretrained feature weights, while the final 3-class task classifier was reinitialized.

## Frozen resources
Train 100000, Val 15000, Test 5000, LCC candidates 3766 and valid 3762.
Resource tree SHA: `f668bec7ba152124e0c2ec733286abde4dfc4dbe7e4918cd64448cddabb74022`. Other file hashes: `{'bundle_tree_sha256': 'f668bec7ba152124e0c2ec733286abde4dfc4dbe7e4918cd64448cddabb74022', 'train_manifest_sha256': '3ec46d1cc6b9bb384f244ba4fcc3657fd1e6f050b13b00b37818a913b590c8b6', 'val_manifest_sha256': '5adf32016920e2e886e148eaedd6abdc73f092525c070ced2ca476c24326d4a9', 'test_manifest_sha256': 'c22b25293b640b49159750399d1913330b55836dc36d83ec75a1ed4e3f048586', 'lcc_manifest_sha256': 'b0d6927e0e509cbdacf452f9a8101aa963eb146f6070ff9496914cbec711ba50', 'celeba_bbox_sha256': 'e15411af09c64581a7567c98e62b97df95bcfde2b3b5affa4af0761f8908ca28', 'lcc_bbox_sha256': 'd6508bde1c7f947ba7c995340c166bde9dee96b77794152a8bb51cbc56fcfbcf', 'minifasnet_source_sha256': 'e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7', 'official_checkpoint_sha256': 'a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0'}`.

## Model and preprocessing
MiniFASNetV2, 128-D embedding, 3 logits, conv6 5x5, 434560 parameters. Face localization cache then 2.7x crop, direct 80x80 BGR float [0,1].

## Hyperparameters selected by Notebook 00b-v2
Frozen selection SHA256: `bfc8652b237d7a8156bf455667bcbc00794baa9b0f0283c4a449406da6caeb0a`. Scratch LR: 0.001/0.001; pretrained LR: 0.0002/0.001; CSMR lambda 0.4, gain range [0.65, 0.9]. Selection used Tune-Val for optimizer choice and a Tune-Val regression guard plus LCC external-development ranking for the regime-specific CSMR recipe. The 5K/2K single-seed screen selects these values; it does not set the final notebook epoch budget.

## Training configuration
AdamW backbone LR 0.0002, head LR 0.001; weight decay 0.0001; label smoothing 0.1; grad clip 5.0; batch 128; AMP True; epochs 12; patience 3. CSMR warmup 1, lambda 0.4, centers [0.15, 0.45, 0.75], sigma 0.1, gains [0.65, 0.9], preserve DC True.

## Vanilla and CSMR checkpoints
Vanilla best epoch 5, best Val ACER 0.009653, best Val AUC 0.998464, checkpoint SHA `e91120a89db7a2594f418bf261e6e6923fa0b63eeb7af28fc11305b2a47a8cbb`, locked Val threshold -0.905557.
CSMR best epoch 6, best Val ACER 0.011061, best Val AUC 0.997759, checkpoint SHA `ced6af1a626deb750617ff29e07889f3d05f958452e25183cec30f2957a0f089`, locked Val threshold -8.833043. Both started from `318dbe2a0fa2c80cc36d4f881aa3967b226255af71a45049529daa5430b93adc`.

## Val selection, Test evaluation and LCC external stress
Checkpoint policy: lowest Val ACER, ties within 1e-12 resolved by highest Val AUC. Test role: fixed in-domain evaluation under final 100K protocol.
| Split | Metric | Vanilla | CSMR | Delta CSMR minus Vanilla |
|---|---|---:|---:|---:|
| val | apcer | 0.008531 | 0.009721 | +0.001190 |
| val | bpcer | 0.010775 | 0.012401 | +0.001626 |
| val | acer | 0.009653 | 0.011061 | +0.001408 |
| val | auc | 0.998464 | 0.997759 | -0.000705 |
| test | apcer | 0.335798 | 0.322434 | -0.013364 |
| test | bpcer | 0.012138 | 0.016183 | +0.004046 |
| test | acer | 0.173968 | 0.169309 | -0.004659 |
| test | auc | 0.928305 | 0.881835 | -0.046471 |
| lcc | apcer | 0.470205 | 0.438482 | -0.031723 |
| lcc | bpcer | 0.215938 | 0.149100 | -0.066838 |
| lcc | acer | 0.343071 | 0.293791 | -0.049280 |
| lcc | auc | 0.741400 | 0.795833 | +0.054433 |
| lcc | hter | 0.343071 | 0.293791 | -0.049280 |

## Class-wise analysis
See `class_metrics.csv` and Val/Test confusion JSON files.

## CSMR diagnostics
See `csmr/csmr_diagnostics.csv` and `csmr/class_diagnostics.csv`; Test-5K was not used for tuning. LCC-FASD was used as an external-development CSMR selection signal in Notebook 00b-v2 and is not an untouched external test. CASIA-FASD remains untouched.

## Efficiency
Vanilla inference architecture == CSMR inference architecture. CSMR is training-time only.
```
 branch  parameters  trainable_parameters input_resolution  pytorch_checkpoint_bytes  onnx_model_bytes            inference_graph             provider  batch_size  warmup_runs  timed_runs  mean_ms  median_ms   p95_ms        fps
vanilla      434560                434560            80x80                   1875959           1744241 ordinary MiniFASNetV2 only CPUExecutionProvider           1           20         100  4.27237   4.183547 5.190295 234.062110
   csmr      434560                434560            80x80                   1875959           1744241 ordinary MiniFASNetV2 only CPUExecutionProvider           1           20         100  4.16369   4.114181 4.574378 240.171558
```

## Face_auth runtime sidecars
`vanilla/runtime_config.json` and `csmr/runtime_config.json` sit beside their parity-checked ONNX files. Each sidecar records the ONNX SHA, its CelebA Val threshold, BGR 80x80 [0,1] input, mean/std disabled, the 2.7x MiniFASNet training crop, gamma off, and crop smoothing off. Set `PAD_RUNTIME_CONFIG_PATH` to the selected sidecar and keep `best.onnx` in the same folder. Fields present in the sidecar take priority over matching `PAD_*` environment values; environment values are used for fields absent from the sidecar.


## Interpretation and limitations
One paired seed is descriptive, not a significance test. LCC AUC delta is +0.054433; LCC HTER delta is -0.049280. Val ACER delta is +0.001408.
This notebook evaluated Test-5K and LCC only after both Val checkpoints and thresholds were frozen. Test-5K did not influence selection; LCC already served as external-development input during Notebook 00b-v2. CPU ONNX timing is an offline proxy and does not establish camera performance.
