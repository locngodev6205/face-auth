# MAST-PAD Architecture Specification
## Full-scale lightweight spatial Face PAD with margin-aware training-time spectral intervention

_Last updated: 2026-10-07_

> **Official method:** **MAST-PAD — Margin-Aware Spectral Training for Face Presentation Attack Detection**
>
> **Primary implementation:** `R7_SC_FULL_crop15`
>
> **Primary paper evidence:** full-scale CelebA source training; LCC Combined + reconstructed CASIA Combined cross-domain AUC.

---

# 0. Frozen status

```text
backbone      = MiniFASNetV2
primary run   = R7_SC_FULL_crop15
paper name    = MAST-PAD
input         = 80×80 BGR float [0,1]
crop factor   = 1.5
PAD output    = binary Real / Attack
spectral use  = training only
params        = 434,434 deployment
```

No new inference branch is allowed in the course-paper method.

---

# 1. High-level architecture

## Training

```text
raw frame
→ frozen SCRFD bbox
→ crop1.5
→ 80×80 BGR [0,1]
→ base augmentation
    ├── clean
    ├── LOW attenuation
    ├── MID attenuation
    └── HIGH attenuation
         ↓
     shared MiniFASNetV2
         ↓
 clean + candidate aligned PAD margins
         ↓
 select minimum-margin candidate
         ↓
 compare m_worst with m_clean
         ↓
 harmful-only spectral CE
         +
 clean-only auxiliary losses
```

## Inference

```text
frame → SCRFD → crop1.5 → 80×80 → MiniFASNetV2 → binary PAD score
```

Removed at inference:

```text
FFT
spectral masks
LOW/MID/HIGH candidate generation
worst-view selector
harmful gate
training-only auxiliary heads
```

---

# 2. Input and crop contract

- frozen SCRFD localization;
- boundary-aware crop factor `1.5`;
- resize `80×80`;
- BGR;
- float range `[0,1]`.

All Clean / WBST / MAST-PAD checkpoints use the same crop/input contract.

---

# 3. Shared base augmentation

```text
horizontal flip       p=0.5
ISONoise              p=0.2
BrightnessContrast    p=0.3
MotionBlur(kernel=5)  p=0.2
bbox jitter           p=0.20
scale                  0.95–1.05
center shift           ±0.05 face size
```

Spectral candidates are generated after the shared base augmentation so clean/LOW/MID/HIGH originate from the same base view.

---

# 4. MiniFASNetV2 training model

## Feature path

```text
80×80×3
→ MiniFASNetV2 convolutional trunk
→ 128-D embedding
→ BatchNorm
→ Dropout(p=0.2)
```

## Heads

```text
PAD:         128 → 2 logits
spoof type:  128 → 11 classes
lighting:    128 → 5 classes
attributes:  128 → 40 binary labels
```

Attribute BCE is Real-only.

---

# 5. Parameter accounting

```text
deployment PAD parameters = 434,434
full training model params = 441,658
trainable params           = 441,658
MACs                       = 40,810,892
reported GFLOPs            = 0.081621784
input                      = [1,3,80,80]
```

The GFLOP profiler does not guarantee identical counting convention to external papers.

---

# 6. Binary score and aligned margin

```text
d = z_real - z_attack
s(y) = +1 for Real, -1 for Attack
m(x,y) = s(y) * d(x)
```

Higher aligned margin means a safer decision for the ground-truth class.

---

# 7. Spectral counterfactual generator

For each channel:

```text
FFT2
→ center zero frequency
→ preserve phase
→ attenuate amplitude with radial Gaussian band
→ inverse FFT
→ real component
→ clamp [0,1]
```

Normalized radial coordinate `r ∈ [0,1]`.

Final centers:

```text
LOW  = 0.0833333333
MID  = 0.25
HIGH = 0.4166666667
sigma = 0.10
gain ~ Uniform(0.65, 0.90)
```

```text
M_k(r) = exp(-(r-c_k)^2 / (2 sigma^2))
G_k(r) = 1 - (1-g) M_k(r)
X'_k = X * G_k
```

Constraints:

- attenuation only;
- no phase randomization;
- DC preserved;
- same gain across LOW/MID/HIGH for one sample;
- same radial field across channels;
- no hard frequency deletion.

---

# 8. BN-safe candidate selection

```text
save train/eval state
model.eval()
with no_grad:
    forward clean
    forward LOW
    forward MID
    forward HIGH
    compute aligned margins
    k* = argmin(m_low, m_mid, m_high)
restore train state
```

Selected worst view is then forwarded in train mode when gradients are required.

---

# 9. Harmful gate

```text
harmful = (m_worst < m_clean).detach()
```

The gate reacts to margin degradation even without a prediction flip.

---

# 10. Loss

## Clean auxiliary loss

```text
L_aux = 0.1 CE_spoof + 0.1 CE_light + 1.0 BCE_attributes_real_only
```

## Epoch 1 warmup

```text
L = CE_PAD(clean) + L_aux
```

## Epoch 2+

```text
L_clean = CE_PAD(clean)
L_harm  = mean CE_PAD(selected_worst) over harmful samples only
L_total = 0.75 L_clean + 0.25 L_harm + L_aux
```

If no sample is harmful, `L_harm` is differentiable zero and the clean coefficient remains `0.75`.

---

# 11. Clean and WBST controls

## Clean

```text
CE_PAD(clean) + L_aux
```

## WBST

Primary run: `P3_SF_FULL_crop15`

```text
centers = [0.0833333333, 0.25, 0.4166666667]
sigma   = 0.0555555556
gain    = Uniform(0.65, 0.90)
L = 0.75 CE(clean) + 0.25 CE(selected_worst) + L_aux
```

All samples receive worst-view CE.

## MAST-PAD

Primary run: `R7_SC_FULL_crop15`

```text
sigma = 0.10
selected-worst CE is harmful-only
```

---

# 12. Primary full-scale training contract

```text
source train      = full eligible CelebA Train
usable train      = 484,561
selection set     = full usable CelebA Test-as-Val
crop              = 1.5
TRAIN_SEED        = 100
DATA_SEED         = 43
batch             = 256
AMP               = on
SGD lr            = 0.005
momentum          = 0.9
weight_decay      = 5e-4
max_epoch         = 12
milestones        = [4,8]
gamma             = 0.2
patience_start    = 9
patience          = 3
```

Selected best epochs:

```text
Clean     = 9
WBST      = 6
MAST-PAD  = 7
```

MAST-PAD stops at epoch 11.

---

# 13. Evaluation contract

## Source selection

```text
full CelebA Test-as-Val
→ select checkpoint
→ calibrate source threshold
→ freeze model + threshold
```

Because the paper headline metric is AUC, threshold-dependent target metrics are not used in the main comparison tables.

## LCC

```text
frozen model
→ LCC training
→ LCC development
→ LCC evaluation
→ pooled LCC Combined AUC
```

No target-domain refitting.

## CASIA

```text
frozen model
→ extracted-image CASIA train/test copy
→ SCRFD/crop1.5/80×80
→ frame scores
→ reconstructed-video mean score
→ unique Combined population
→ CASIA Combined AUC
```

No CASIA training, adaptation, threshold fitting, or checkpoint selection.

Limit: extracted-image reconstruction is not claimed as the official subject-disjoint full-video protocol.

---

# 14. Primary full-scale metrics

| Method | LCC Combined AUC | CASIA Combined AUC |
|---|---:|---:|
| Clean | 0.846011 | 0.920398 |
| WBST | **0.860831** | 0.922264 |
| **MAST-PAD** | 0.860272 | **0.935199** |

Paper-facing:

```text
MAST-PAD LCC Combined   = 0.860
MAST-PAD CASIA Combined = 0.935
Params                  = 0.434M
```

---

# 15. Full-scale MAST-PAD diagnostics

At last executed epoch:

```text
harmful fraction = 0.832135
margin drop      = 0.391421
flip rate        = 0.003077
LOW              = 0.426452
MID              = 0.273035
HIGH             = 0.300513
clean CE         = 0.012028
worst CE         = 0.019878
harmful CE       = 0.017835
```

These are optimization diagnostics, not proof of mechanism.

---

# 16. Supporting matched100K contract

The earlier `R7_SC_100K_crop15` study remains development provenance. Its LCC Combined AUC was `0.867844`, but it is no longer the primary paper checkpoint.

Do not mix full-scale and 100K threshold-dependent metrics as if they came from the same selection contract.

---

# 17. Deployment contract

```text
detector → crop → 80×80 spatial PAD model
```

Training-only modules:

```text
FFT
radial spectral masks
LOW/MID/HIGH generation
candidate selection
harmful gate
auxiliary losses
```

Core claim:

> **MAST-PAD has no additional spectral branch or model parameters at inference relative to the matched clean baseline.**

---

# 18. Export path

```text
PyTorch checkpoint
→ strip training-only logic / auxiliary heads as needed
→ ONNX
→ ONNX Runtime
→ edge target
```

Optional future INT8 quantization is a deployment optimization, not part of MAST-PAD novelty.

---

# 19. Architecture freeze checklist

- [x] MiniFASNetV2 selected
- [x] crop1.5 selected
- [x] binary PAD head selected
- [x] auxiliary heads frozen
- [x] spectral centers frozen
- [x] sigma frozen
- [x] gain range frozen
- [x] harmful-gated objective frozen
- [x] full-scale source training complete
- [x] LCC evaluation complete
- [x] CASIA reconstructed evaluation complete
- [x] deployment complexity measured

---

# 20. Final architecture summary

```text
MAST-PAD TRAINING
SCRFD → crop1.5 → 80×80 BGR
→ base augmentation
→ clean + LOW/MID/HIGH counterfactuals
→ shared MiniFASNetV2
→ label-aligned margins
→ worst-band selection
→ harmful gate
→ 0.75 clean CE + 0.25 harmful-only worst CE
→ clean-view auxiliary supervision

MAST-PAD INFERENCE
SCRFD → crop1.5 → 80×80 BGR → MiniFASNetV2 → binary PAD score
```
