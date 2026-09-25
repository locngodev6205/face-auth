# E2 — Frequency-Only DCT Diagnostic Protocol

## 1. Research role

E2 is a diagnostic ablation answering:

> **Do explicit frequency-domain cues have discriminative power on their own for Face Presentation Attack Detection?**

E2 is not intended to replace E1 or E3.

The experiment sequence is:

```text
E1  MobileNetV3-Small spatial-only
        ↓
E2  DCT frequency-only diagnostic
        ↓
E3  MobileNetV3-Small + DCT Frequency + Concat
        ↓
Does frequency add complementary information?
```

The primary scientific comparison later remains **E1 vs E3**. E2 is used to understand the frequency branch before fusion.

---

## 2. Fair-comparison contract inherited from E1

E2 does not recreate the dataset split or rerun the face detector.

It must reuse:

```text
E1 split manifest
E1 SCRFD bbox cache
E1 face crop policy
E1 TRAIN-only filtering result
E1 augmentation policy
seed = 42
```

Frozen preprocessing:

```text
SCRFD valid bbox         → 1.55× square crop
CelebA fallback bbox     → 1.50× square crop
TRAIN min face           → 48 px, already reflected in E1 train_keys
Validation/Test          → no small-face filtering
gamma                    → OFF
input spatial size       → 224×224
```

The E2 notebook fails loudly if the E1 manifest/cache does not match the expected v5.3 policy.

If E1 artifacts are preliminary, E2 is preliminary. Do not mix an official manifest with a preliminary cache or vice versa.

---

## 3. E2 input transform

Starting from the exact E1 RGB face crop:

```text
RGB face
   ↓
same TRAIN augmentation as E1
   ↓
resize / reflect-pad to 224×224
   ↓
luminance
   ↓
2D DCT
   ↓
signed-log compression
   ↓
per-sample standardization
   ↓
1×224×224 frequency map
```

Luminance:

```text
Y = 0.299 R + 0.587 G + 0.114 B
```

DCT dynamic-range compression:

```text
C = DCT(Y)
C_log = sign(C) × log(1 + |C|)
```

Final normalization:

```text
C_norm = (C_log - mean(C_log)) / std(C_log)
```

with a small epsilon guard.

No handcrafted low/high-frequency mask is used in E2 v1. The full DCT map is presented to the CNN.

This deliberately avoids assuming that "high frequency = spoof". The model must learn any useful frequency pattern.

---

## 4. Frequency-only architecture

```text
1×224×224 DCT map
        ↓
Conv 3×3, 1→32, stride 2
BN + ReLU
        ↓
Depthwise 3×3, stride 2
Pointwise 1×1, 32→32
BN + ReLU
        ↓
Depthwise 3×3, stride 2
Pointwise 1×1, 32→64
BN + ReLU
        ↓
Global Average Pool
        ↓
Linear 64→64 + ReLU
        ↓
64-D Frequency Feature
        ↓
Linear 64→128
Hardswish
Dropout 0.2
        ↓
Linear 128→3
```

Classes:

```text
0 Real
1 Physical Spoof
2 Digital Spoof
```

The `FrequencyBranch` module is designed to be reused unchanged in E3.

---

## 5. PAD decision and calibration

Three-class training is retained for consistency with E1.

Binary PAD score:

```text
d = real_logit - logsumexp(
    physical_spoof_logit,
    digital_spoof_logit
)
```

Decision:

```text
REAL iff d >= locked_logit_threshold
```

Checkpoint selection and threshold calibration are performed on Validation only.

Metrics:

```text
Accuracy
3-class Accuracy
APCER
BPCER
ACER
AUC
```

Held-out Test remains disabled until the E2 architecture/training protocol is frozen.

---

## 6. Training recipe

To keep E2 aligned with E1:

```text
batch size                = 128
optimizer                 = AdamW
weight decay              = 1e-4
max epochs                = 24
warmup                    = 2 epochs
scheduler                 = cosine
early-stop metric         = Validation ACER
early-stop patience       = 4
earliest early stop       = epoch 10
gradient clip             = 5.0
dropout                   = 0.20
label smoothing           = 0.10
class weights             = [1, 1, 1]
seed                      = 42
```

E2 has no pretrained spatial backbone. All E2 trainable parameters therefore use:

```text
LR = 1e-4
```

which corresponds to the E1 new-head LR.

TRAIN augmentation is the same as E1:

```text
HorizontalFlip            p=0.50
Brightness/Contrast       ±10%, p=0.30
bbox jitter               p=0.20
bbox scale                0.95–1.05
bbox center translation   ±5%
```

The augmentation is applied to the face crop before the luminance/DCT transform.

---

## 7. E2 artifacts

The notebook saves:

```text
e2_freq_only_<mode>_dct_v1_best.pth
e2_freq_only_<mode>_dct_v1_checkpoint_last.pth
e2_freq_only_<mode>_dct_v1_best_meta.json
e2_freq_only_<mode>_dct_v1_training_history.json
e2_freq_only_<mode>_dct_v1_training_history.png
e2_freq_only_<mode>_dct_v1_run_config.json

e2_freq_only_<mode>_dct_v1_frequency_branch_best.pth
e2_freq_only_<mode>_dct_v1_frequency_branch_spec.json

e2_freq_only_<mode>_dct_v1_best.onnx
e2_freq_only_<mode>_dct_v1_runtime_config.json
```

If held-out Test is enabled after freeze:

```text
e2_freq_only_<mode>_dct_v1_heldout_test_results.json
```

### Frequency branch artifact

`*_frequency_branch_best.pth` contains only `FrequencyBranch` weights.

This artifact is saved because E3 uses the same branch architecture.

Whether E3 initializes this branch from E2 weights or starts the branch randomly is a separate protocol choice. Decide it before E3 training and apply it consistently; do not silently preload E2 weights.

---

## 8. ONNX scope

E2 ONNX input is:

```text
[N, 1, 224, 224]
```

representing the already computed normalized DCT map.

The E2 ONNX therefore benchmarks the **frequency model**, not total preprocessing cost.

For edge efficiency reporting, separate:

```text
crop + luminance + DCT preprocessing
frequency model inference
```

E3 deployment can later decide whether DCT stays external or is integrated into a broader runtime graph.

---

## 9. Interpretation

Possible outcomes:

### E2 performs well

This supports the statement that frequency-domain information itself carries useful PAD signal.

It still does not prove that E3 will improve over E1. Complementarity must be tested by E1 vs E3.

### E2 is weaker than E1 but clearly above chance

This is still a useful result. A weak standalone cue can add complementary information after fusion.

### E2 is near chance / unstable

Do not jump directly to gated fusion. First inspect:

```text
DCT preprocessing
frequency normalization
attack-type breakdown
cross-domain behavior
```

The project should not assume frequency cues are beneficial before E3 ablation confirms it.

---

## 10. Next step after E2

Once E2 is frozen:

```text
E3 =
MobileNetV3-Small spatial branch
+
the exact E2 FrequencyBranch architecture
+
Concat(256-D, 64-D)
+
128-D fusion classifier
```

E3 must continue using the exact same E1 split/cache/crop/evaluation protocol.
