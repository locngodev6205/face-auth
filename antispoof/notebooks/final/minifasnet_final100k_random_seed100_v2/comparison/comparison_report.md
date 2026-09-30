# MiniFASNetV2 paired final experiment

## Objective
Measure the frozen CSMR-v1 effect against Vanilla with identical initialization and source resources.

## Initialization
MiniFASNetV2 was trained from random initialization. No PAD-pretrained checkpoint was loaded.

## Frozen resources
Train 100000, Val 15000, Test 5000, LCC candidates 3766 and valid 3762.
Resource tree SHA: `f668bec7ba152124e0c2ec733286abde4dfc4dbe7e4918cd64448cddabb74022`. Other file hashes: `{'bundle_tree_sha256': 'f668bec7ba152124e0c2ec733286abde4dfc4dbe7e4918cd64448cddabb74022', 'train_manifest_sha256': '3ec46d1cc6b9bb384f244ba4fcc3657fd1e6f050b13b00b37818a913b590c8b6', 'val_manifest_sha256': '5adf32016920e2e886e148eaedd6abdc73f092525c070ced2ca476c24326d4a9', 'test_manifest_sha256': 'c22b25293b640b49159750399d1913330b55836dc36d83ec75a1ed4e3f048586', 'lcc_manifest_sha256': 'b0d6927e0e509cbdacf452f9a8101aa963eb146f6070ff9496914cbec711ba50', 'celeba_bbox_sha256': 'e15411af09c64581a7567c98e62b97df95bcfde2b3b5affa4af0761f8908ca28', 'lcc_bbox_sha256': 'd6508bde1c7f947ba7c995340c166bde9dee96b77794152a8bb51cbc56fcfbcf', 'minifasnet_source_sha256': 'e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7'}`.

## Model and preprocessing
MiniFASNetV2, 128-D embedding, 3 logits, conv6 5x5, 434560 parameters. Face localization cache then 2.7x crop, direct 80x80 BGR float [0,1].

## Hyperparameters selected by Notebook 00b-v2
Frozen selection SHA256: `bfc8652b237d7a8156bf455667bcbc00794baa9b0f0283c4a449406da6caeb0a`. Scratch LR: 0.001/0.001; pretrained LR: 0.0002/0.001; CSMR lambda 0.2, gain range [0.55, 0.85]. Selection used Tune-Val for optimizer choice and a Tune-Val regression guard plus LCC external-development ranking for the regime-specific CSMR recipe. The 5K/2K single-seed screen selects these values; it does not set the final notebook epoch budget.

## Training configuration
AdamW backbone LR 0.001, head LR 0.001; weight decay 0.0001; label smoothing 0.1; grad clip 5.0; batch 128; AMP True; epochs 12; patience 3. CSMR warmup 1, lambda 0.2, centers [0.15, 0.45, 0.75], sigma 0.1, gains [0.55, 0.85], preserve DC True.

## Vanilla and CSMR checkpoints
Vanilla best epoch 12, best Val ACER 0.019278, best Val AUC 0.997166, checkpoint SHA `7d562b8a74fae0e0d336cd625b8c61f96f04a5b68fa4e1203a7053d333383e1a`, locked Val threshold -0.383404.
CSMR best epoch 11, best Val ACER 0.017409, best Val AUC 0.997670, checkpoint SHA `096041f0548d6195ee159c863eb93d85f715398dfd173d84bf508cff96dda568`, locked Val threshold 0.156794. Both started from `f134adaf0d46b5fb50325aefe05220d2ec6c6b50dcfe34b70d0a58b82c711828`.

## Val selection, Test evaluation and LCC external stress
Checkpoint policy: lowest Val ACER, ties within 1e-12 resolved by highest Val AUC. Test role: fixed in-domain evaluation under final 100K protocol.
| Split | Metric | Vanilla | CSMR | Delta CSMR minus Vanilla |
|---|---|---:|---:|---:|
| val | apcer | 0.019244 | 0.018351 | -0.000893 |
| val | bpcer | 0.019313 | 0.016467 | -0.002846 |
| val | acer | 0.019278 | 0.017409 | -0.001869 |
| val | auc | 0.997166 | 0.997670 | +0.000504 |
| test | apcer | 0.371908 | 0.359682 | -0.012226 |
| test | bpcer | 0.032367 | 0.021578 | -0.010789 |
| test | acer | 0.202137 | 0.190630 | -0.011508 |
| test | auc | 0.888879 | 0.926548 | +0.037669 |
| lcc | apcer | 0.537800 | 0.512007 | -0.025793 |
| lcc | bpcer | 0.210797 | 0.213368 | +0.002571 |
| lcc | acer | 0.374299 | 0.362687 | -0.011611 |
| lcc | auc | 0.687602 | 0.696461 | +0.008859 |
| lcc | hter | 0.374299 | 0.362687 | -0.011611 |

## Class-wise analysis
See `class_metrics.csv` and Val/Test confusion JSON files.

## CSMR diagnostics
See `csmr/csmr_diagnostics.csv` and `csmr/class_diagnostics.csv`; Test-5K was not used for tuning. LCC-FASD was used as an external-development CSMR selection signal in Notebook 00b-v2 and is not an untouched external test. CASIA-FASD remains untouched.

## Efficiency
Vanilla inference architecture == CSMR inference architecture. CSMR is training-time only.
```
 branch  parameters  trainable_parameters input_resolution  pytorch_checkpoint_bytes  onnx_model_bytes            inference_graph             provider  batch_size  warmup_runs  timed_runs  mean_ms  median_ms   p95_ms        fps
vanilla      434560                434560            80x80                   1875959           1744241 ordinary MiniFASNetV2 only CPUExecutionProvider           1           20         100 5.661653   5.649090 6.168106 176.626859
   csmr      434560                434560            80x80                   1875959           1744241 ordinary MiniFASNetV2 only CPUExecutionProvider           1           20         100 4.359218   4.323017 4.848280 229.398946
```

## Face_auth runtime sidecars
`vanilla/runtime_config.json` and `csmr/runtime_config.json` sit beside their parity-checked ONNX files. Each sidecar records the ONNX SHA, its CelebA Val threshold, BGR 80x80 [0,1] input, mean/std disabled, the 2.7x MiniFASNet training crop, gamma off, and crop smoothing off. Set `PAD_RUNTIME_CONFIG_PATH` to the selected sidecar and keep `best.onnx` in the same folder. Fields present in the sidecar take priority over matching `PAD_*` environment values; environment values are used for fields absent from the sidecar.


## Interpretation and limitations
One paired seed is descriptive, not a significance test. LCC AUC delta is +0.008859; LCC HTER delta is -0.011611. Val ACER delta is -0.001869.
This notebook evaluated Test-5K and LCC only after both Val checkpoints and thresholds were frozen. Test-5K did not influence selection; LCC already served as external-development input during Notebook 00b-v2. CPU ONNX timing is an offline proxy and does not establish camera performance.
