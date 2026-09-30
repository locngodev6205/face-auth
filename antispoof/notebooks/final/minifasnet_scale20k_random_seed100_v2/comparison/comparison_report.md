# MiniFASNetV2 20K scale sanity experiment

## Objective
Measure the frozen CSMR-v1 effect against Vanilla with identical initialization and source resources.

## Initialization
MiniFASNetV2 was trained from random initialization. No PAD-pretrained checkpoint was loaded.

## Frozen resources
Train 20000, Val 5000, Test 5000, LCC candidates 3766 and valid 3762.
Resource tree SHA: `f668bec7ba152124e0c2ec733286abde4dfc4dbe7e4918cd64448cddabb74022`. Other file hashes: `{'bundle_tree_sha256': 'f668bec7ba152124e0c2ec733286abde4dfc4dbe7e4918cd64448cddabb74022', 'train_manifest_sha256': '93f679b1dc974febecf1eb68f5e2215899e346a9afd6475d90b785b534fbbf8d', 'val_manifest_sha256': '5905191a69e89c7c5fc4f87a7dae56548a7f31e8c298ad10b5bd91936b32d62a', 'test_manifest_sha256': 'c22b25293b640b49159750399d1913330b55836dc36d83ec75a1ed4e3f048586', 'lcc_manifest_sha256': 'b0d6927e0e509cbdacf452f9a8101aa963eb146f6070ff9496914cbec711ba50', 'celeba_bbox_sha256': 'e15411af09c64581a7567c98e62b97df95bcfde2b3b5affa4af0761f8908ca28', 'lcc_bbox_sha256': 'd6508bde1c7f947ba7c995340c166bde9dee96b77794152a8bb51cbc56fcfbcf', 'minifasnet_source_sha256': 'e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7'}`.

## Model and preprocessing
MiniFASNetV2, 128-D embedding, 3 logits, conv6 5x5, 434560 parameters. Face localization cache then 2.7x crop, direct 80x80 BGR float [0,1].

## Hyperparameters selected by Notebook 00b-v2
Frozen selection SHA256: `bfc8652b237d7a8156bf455667bcbc00794baa9b0f0283c4a449406da6caeb0a`. Scratch LR: 0.001/0.001; pretrained LR: 0.0002/0.001; CSMR lambda 0.2, gain range [0.55, 0.85]. Selection used Tune-Val for optimizer choice and a Tune-Val regression guard plus LCC external-development ranking for the regime-specific CSMR recipe. The 5K/2K single-seed screen selects these values; it does not set the final notebook epoch budget.

## Training configuration
AdamW backbone LR 0.001, head LR 0.001; weight decay 0.0001; label smoothing 0.1; grad clip 5.0; batch 128; AMP True; epochs 8; patience 2. CSMR warmup 1, lambda 0.2, centers [0.15, 0.45, 0.75], sigma 0.1, gains [0.55, 0.85], preserve DC True.

## Vanilla and CSMR checkpoints
Vanilla best epoch 7, best Val ACER 0.049514, best Val AUC 0.985864, checkpoint SHA `1865bb9f5e16c70593240763bc0bbc10e5abc50e89ef8162f6f162ff87b98118`, locked Val threshold 0.073400.
CSMR best epoch 6, best Val ACER 0.048951, best Val AUC 0.986491, checkpoint SHA `656cc75b4fd44897ba42c1023c9dfd62bae1e4766711be3ab32d8d5999d368d5`, locked Val threshold -0.298274. Both started from `f134adaf0d46b5fb50325aefe05220d2ec6c6b50dcfe34b70d0a58b82c711828`.

## Val selection, Test evaluation and LCC external stress
Checkpoint policy: lowest Val ACER, ties within 1e-12 resolved by highest Val AUC. Test role: fixed in-domain descriptive evaluation during scale sanity.
| Split | Metric | Vanilla | CSMR | Delta CSMR minus Vanilla |
|---|---|---:|---:|---:|
| val | apcer | 0.039881 | 0.048512 | +0.008631 |
| val | bpcer | 0.059146 | 0.049390 | -0.009756 |
| val | acer | 0.049514 | 0.048951 | -0.000563 |
| val | auc | 0.985864 | 0.986491 | +0.000627 |
| test | apcer | 0.338072 | 0.362241 | +0.024168 |
| test | bpcer | 0.105192 | 0.093729 | -0.011463 |
| test | acer | 0.221632 | 0.227985 | +0.006353 |
| test | auc | 0.874082 | 0.868893 | -0.005188 |
| lcc | apcer | 0.271865 | 0.260302 | -0.011562 |
| lcc | bpcer | 0.637532 | 0.604113 | -0.033419 |
| lcc | acer | 0.454698 | 0.432208 | -0.022491 |
| lcc | auc | 0.577192 | 0.613921 | +0.036729 |
| lcc | hter | 0.454698 | 0.432208 | -0.022491 |

## Class-wise analysis
See `class_metrics.csv` and Val/Test confusion JSON files.

## CSMR diagnostics
See `csmr/csmr_diagnostics.csv` and `csmr/class_diagnostics.csv`; Test-5K was not used for tuning. LCC-FASD was used as an external-development CSMR selection signal in Notebook 00b-v2 and is not an untouched external test. CASIA-FASD remains untouched.

## Efficiency
Vanilla inference architecture == CSMR inference architecture. CSMR is training-time only.
Model parameter count: 434560. Vanilla and CSMR checkpoints share the same ordinary MiniFASNetV2 inference architecture; CSMR affects training only.

## ONNX exports and parity
`vanilla/best.onnx` and `csmr/best.onnx` are exported at opset 17 with dynamic batch and fixed 3x80x80 input. Each export is checked against PyTorch on the same eight deterministic Val tensors (rtol=1e-4, atol=1e-5); export stops if parity fails. See each branch `onnx_validation.json` and `comparison/efficiency_metrics.csv`.
```
 branch  parameters  trainable_parameters input_resolution  pytorch_checkpoint_bytes  onnx_model_bytes            inference_graph             provider  batch_size  warmup_runs  timed_runs  mean_ms  median_ms   p95_ms        fps
vanilla      434560                434560            80x80                   1875959           1744241 ordinary MiniFASNetV2 only CPUExecutionProvider           1           20         100 4.890497   4.868738 5.289918 204.478199
   csmr      434560                434560            80x80                   1875959           1744241 ordinary MiniFASNetV2 only CPUExecutionProvider           1           20         100 4.743145   4.730202 5.053214 210.830574
```

## Face_auth runtime sidecars
`vanilla/runtime_config.json` and `csmr/runtime_config.json` sit beside their parity-checked ONNX files. Each sidecar records the ONNX SHA, its CelebA Val threshold, BGR 80x80 [0,1] input, mean/std disabled, the 2.7x MiniFASNet training crop, gamma off, and crop smoothing off. Set `PAD_RUNTIME_CONFIG_PATH` to the selected sidecar and keep `best.onnx` in the same folder. Fields present in the sidecar take priority over matching `PAD_*` environment values; environment values are used for fields absent from the sidecar. This 20K/5K scale model is for immediate smoke testing; final 100K models are evaluated separately.

## Interpretation and limitations
Test-5K is reported descriptively in this scale experiment. It does not influence checkpoint selection, threshold calibration, hyperparameter tuning, or CSMR configuration.
One paired seed is descriptive, not a significance test. LCC AUC delta is +0.036729; LCC HTER delta is -0.022491. Val ACER delta is -0.000563.
This notebook evaluated Test-5K and LCC only after both Val checkpoints and thresholds were frozen. Test-5K did not influence selection; LCC already served as external-development input during Notebook 00b-v2. CPU ONNX timing is an offline proxy and does not establish camera performance.
