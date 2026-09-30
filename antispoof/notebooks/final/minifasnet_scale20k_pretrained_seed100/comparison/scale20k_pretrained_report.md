# MiniFASNetV2 pretrained 20K scale confirmation

## Objective
Compare pretrained Vanilla with the pretrained-specific CSMR recipe frozen by Notebook 00b-v2. No learning rate, lambda, gain, threshold, or checkpoint is selected from Test-5K or LCC here.

## Initialization
Official PAD-pretrained feature weights plus deterministic fresh 3-class classifier. Loading audit: `pretrained_loading_audit.json`. Both branches start from canonical state SHA256 `318dbe2a0fa2c80cc36d4f881aa3967b226255af71a45049529daa5430b93adc`.

## Frozen resources
Train 20000, Val 5000, Test 5000, LCC candidates 3766 and valid 3762. Resource hashes: `{'bundle_tree_sha256': 'f668bec7ba152124e0c2ec733286abde4dfc4dbe7e4918cd64448cddabb74022', 'train_manifest_sha256': '93f679b1dc974febecf1eb68f5e2215899e346a9afd6475d90b785b534fbbf8d', 'val_manifest_sha256': '5905191a69e89c7c5fc4f87a7dae56548a7f31e8c298ad10b5bd91936b32d62a', 'test_manifest_sha256': 'c22b25293b640b49159750399d1913330b55836dc36d83ec75a1ed4e3f048586', 'lcc_manifest_sha256': 'b0d6927e0e509cbdacf452f9a8101aa963eb146f6070ff9496914cbec711ba50', 'celeba_bbox_sha256': 'e15411af09c64581a7567c98e62b97df95bcfde2b3b5affa4af0761f8908ca28', 'lcc_bbox_sha256': 'd6508bde1c7f947ba7c995340c166bde9dee96b77794152a8bb51cbc56fcfbcf', 'minifasnet_source_sha256': 'e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7', 'official_checkpoint_sha256': 'a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0'}`.

## Model and preprocessing
MiniFASNetV2, conv6 5x5, 3 logits, 434560 trainable parameters; 2.7x face crop, 80x80 BGR float [0,1].

## Frozen v2 recipe
Selection SHA256 `bfc8652b237d7a8156bf455667bcbc00794baa9b0f0283c4a449406da6caeb0a`. AdamW backbone/head LR 0.0002/0.001; CSMR lambda 0.4, gains [0.65, 0.9], centers [0.15, 0.45, 0.75], sigma 0.1, warmup 1, preserve DC True.

## Training budget
Seed 100; batch 128; AMP True; maximum 8 epochs; patience 2; earliest stop epoch 4. Both branches use the same optimizer recipe and canonical initial state.

## Frozen Val decisions
Vanilla epoch 2, Val ACER 0.015041, Val AUC 0.998053, threshold -0.607246, checkpoint `9d252237915552b79153a40bcd32410cc1ce5093a6eb1499e4ded7452f45055f`.
CSMR epoch 1, Val ACER 0.019309, Val AUC 0.998173, threshold -0.094344, checkpoint `3bdf07d57e77e494d9807c50adc2090418e96e83390bea437e1f16b094806017`.

## Primary paired metrics
| Split | Metric | Vanilla | CSMR | Delta CSMR minus Vanilla |
|---|---|---:|---:|---:|
| val | acer | 0.015041 | 0.019309 | +0.004268 |
| val | auc | 0.998053 | 0.998173 | +0.000120 |
| test | acer | 0.207019 | 0.204554 | -0.002465 |
| test | auc | 0.942841 | 0.940973 | -0.001867 |
| lcc | auc | 0.798377 | 0.794560 | -0.003817 |
| lcc | hter | 0.348090 | 0.317912 | -0.030178 |

## Diagnostics
Class metrics and confusion matrices are saved under comparison/ and each branch. CSMR training diagnostics are saved under csmr/. Inference uses the ordinary MiniFASNetV2 graph; FFT and CSMR training logic are absent from ONNX.

## ONNX and runtime configs
Each branch exports `best.onnx` at opset 17 with dynamic batch and fixed 3x80x80 BGR input. PyTorch/ONNX Runtime logits are checked on eight deterministic Val tensors (rtol=1e-4, atol=1e-5); export stops on parity failure. The ZIP includes `onnx_validation.json` and a complete `runtime_config.json` beside each model. Runtime sidecars record model/checkpoint SHA, crop mode `minifasnet_train_v1` at 2.7x, BGR [0,1] preprocessing, gamma/normalization settings, and the branch-specific CelebA Val threshold. Set `PAD_RUNTIME_CONFIG_PATH` to the selected branch sidecar.

## Evaluation roles
Test-5K is fixed in-domain descriptive evaluation. It did not influence hyperparameters, checkpoint selection, or threshold calibration. LCC-FASD is external-development stress and was used in 00b-v2 for CSMR selection, so it is not an untouched final external test. CASIA-FASD remains untouched.

## Limitations
This single-seed 20K comparison confirms scale behavior; it does not estimate multi-seed uncertainty.
