# MAST-PAD Architecture Specification
## Lightweight spatial Face PAD with margin-aware training-time spectral intervention

_Last updated: 2026-10-05_

> **Official method name:** **MAST-PAD — Margin-Aware Spectral Training for Face Presentation Attack Detection**
>
> **Primary implementation:** `R7_SC_100K_crop15`
>
> This specification supersedes the previous CSMR / 3-logit / crop2.7 architecture documents. The final course-paper method uses the binary PAD + auxiliary-head architecture from the `binary_crossdomain_v2` experiment family, crop factor `1.5`, and harmful-only selected-worst spectral CE.

---

# 0. Frozen status

```text
backbone     = MiniFASNetV2
primary run  = R7_SC_100K_crop15
paper name   = MAST-PAD
input        = 80×80 BGR float [0,1]
crop factor  = 1.5
PAD output   = binary Real / Attack
spectral use = training only
```

No new inference branch is allowed in the course-paper method.

---

# 1. High-level architecture

## 1.1 Training

```text
raw frame
→ frozen SCRFD bbox
→ crop1.5
→ 80×80 BGR [0,1]
→ base augmentation
        │
        ├──────────── clean ───────────────────┐
        │                                      │
        ├─ LOW spectral attenuation ────────┐  │
        ├─ MID spectral attenuation ─────┐  │  │
        └─ HIGH spectral attenuation ─┐  │  │  │
                                     ▼  ▼  ▼  ▼
                                  shared model
                                     │
                     clean + candidate PAD margins
                                     │
                         choose minimum-margin view
                                     │
                         compare m_worst vs m_clean
                                     │
                        harmful-only spectral CE
                                     │
                         clean-only auxiliary loss
```

## 1.2 Inference

```text
frame
→ SCRFD
→ crop1.5
→ resize80×80
→ MiniFASNetV2
→ PAD head
→ Real / Attack score
```

Removed at inference:

```text
FFT
spectral masks
LOW/MID/HIGH candidates
worst-view selector
harmful gate
auxiliary heads if deployment export strips them
```

---

# 2. Input and crop contract

## 2.1 Localization

Face localization is performed by a frozen SCRFD detector.

The experiment uses frozen bbox/crop preparation so C / WBST / MAST-PAD see the same candidate population.

## 2.2 Crop

Final crop factor:

```text
crop_factor = 1.5
```

This replaces the historical `2.7×` context recipe used in earlier development runs.

The crop helper is boundary-aware: when the requested context does not fit inside the image, the crop is limited/translated inward rather than padded with artificial pixels.

## 2.3 Resize and color convention

```text
resize = 80×80
channel order = BGR
range = [0,1]
```

Spectral intervention operates on this float image before the backbone consumes the view.

---

# 3. Shared base augmentation

Training-time base augmentation is shared across C / WBST / MAST-PAD.

Current recipe:

```text
horizontal flip       p=0.5
ISONoise              p=0.2
BrightnessContrast    p=0.3
MotionBlur(kernel=5)  p=0.2
```

BBox jitter:

```text
probability = 0.20
scale       = 0.95–1.05
center shift = ±0.05 face size
```

The spectral counterfactual is generated after the base augmentation so the clean and LOW/MID/HIGH views originate from the same base sample.

---

# 4. MiniFASNetV2 training model

## 4.1 Initialization

Use the official PAD-pretrained MiniFASNetV2-compatible trunk.

All matched primary runs share the same initial-state SHA and the same fresh project heads.

Do not initialize MAST-PAD from the earlier 100K checkpoint when rerunning the primary protocol.

## 4.2 Feature path

Conceptually:

```text
80×80×3
→ MiniFASNetV2 convolutional trunk
→ embedding 128
→ BatchNorm
→ Dropout(p=0.2)
```

## 4.3 Heads

### PAD head

```text
128 → 2 logits
```

Semantics:

```text
0 / one logit = Real
1 / other logit = Attack
```

The exact internal index ordering must follow the saved implementation; paper equations use semantic names `z_real` and `z_attack`.

### Spoof-type auxiliary head

```text
128 → 11 classes
```

### Lighting auxiliary head

```text
128 → 5 classes
```

### Attribute auxiliary head

```text
128 → 40 binary attributes
```

Attribute BCE is applied to Real samples only.

---

# 5. Parameter accounting

Saved complexity artifacts report:

```text
deployment PAD parameters = 434,434
full training model params = 441,658
trainable params           = 441,658
MACs                        = 40,810,892
GFLOPs                      = 0.081621784
input                       = [1,3,80,80]
```

Interpretation:

- deployment parameters refer to the PAD inference graph;
- training-only auxiliary heads account for the difference to total training parameters;
- reported GFLOPs cover profiler-supported Conv/Linear operations;
- normalization, activation, pooling, detector cost, and training FFT cost are not guaranteed to be included;
- do not compare FLOPs across papers without a counting-convention qualifier.

---

# 6. Binary PAD score and aligned margin

For PAD logits:

```text
z_real
z_attack
```

define:

```text
d = z_real - z_attack
```

Higher `d` means more Real.

Label-aligned sign:

```text
s(y) =
  +1 for Real
  -1 for Attack
```

Aligned margin:

```text
m(x,y) = s(y) * d(x)
```

Interpretation:

```text
larger m = safer correct binary PAD decision
smaller m = more fragile / more attack-favoring for Real
            or more real-favoring for Attack
```

MAST-PAD uses the aligned margin for both candidate selection and harmful gating.

---

# 7. Spectral counterfactual generator

## 7.1 Frequency transform

For each channel:

```text
FFT2
→ shift zero frequency to center
→ preserve phase
→ attenuate amplitude using one radial band mask
→ inverse FFT
→ real component
→ clamp to [0,1]
```

No phase randomization is used.

## 7.2 Normalized radial coordinate

Use:

```text
r ∈ [0,1]
```

relative to the maximum radial frequency.

## 7.3 Final centers

Historical base centers at crop2.7:

```text
[0.15, 0.45, 0.75]
```

MAST-PAD scales centers by:

```text
1.5 / 2.7 = 0.555555...
```

Final centers:

```text
LOW  = 0.0833333333
MID  = 0.25
HIGH = 0.4166666667
```

## 7.4 Final sigma

Unlike the P3-SF intermediate configuration, MAST-PAD retains:

```text
sigma = 0.10
```

This historical development choice was called `SC` = scale centers only.

## 7.5 Gain

For each sample:

```text
g ~ Uniform(0.65, 0.90)
```

One sampled gain is shared across LOW/MID/HIGH candidates for that sample.

## 7.6 Gaussian attenuation

For band `k`:

```text
M_k(r) = exp(-(r-c_k)^2 / (2 sigma^2))
G_k(r) = 1 - (1-g) * M_k(r)
```

Then:

```text
X'_k = X * G_k
```

in the frequency domain.

Constraints:

- `G_k <= 1`;
- no amplification;
- DC preserved;
- same radial gain field across channels;
- phase unchanged;
- no hard band deletion.

---

# 8. BN-safe candidate selection

Running LOW/MID/HIGH forward passes in training mode would update BatchNorm statistics multiple times.

The selection procedure is therefore:

```text
save current training state
model.eval()
with no_grad:
    forward LOW
    forward MID
    forward HIGH
    compute aligned margins
    select k* = argmin margin
restore train state
```

The selected view is then forwarded again in train mode when gradient is required.

This isolates candidate selection from BN state updates.

---

# 9. Worst-view selection

For each sample:

```text
m_low
m_mid
m_high
```

Choose:

```text
k* = argmin(m_low, m_mid, m_high)
```

The selected view is:

```text
x_worst = x_k*
```

This is a discrete selection; gradients do not flow through the argmin decision.

---

# 10. Harmful gate

During the BN-safe selection phase also compute:

```text
m_clean
m_worst
```

Then:

```text
harmful = (m_worst < m_clean).detach()
```

A view is harmful if it reduces the sample’s aligned margin relative to the clean view.

This does **not** require a prediction flip.

Therefore MAST-PAD can react to a degradation before it becomes a classification error.

---

# 11. Loss

## 11.1 Clean auxiliary loss

Always computed on the clean view:

```text
L_aux =
    0.1 * CE_spoof_type
  + 0.1 * CE_lighting
  + 1.0 * BCE_attributes_real_only
```

No auxiliary head is supervised on spectral views.

## 11.2 Epoch 1 warmup

```text
L = CE_PAD(clean) + L_aux
```

The warmup epoch is excluded from best-checkpoint eligibility for MAST-PAD.

## 11.3 Epoch 2+

Clean PAD term:

```text
L_clean = CE_PAD(clean)
```

Selected worst PAD CE per sample:

```text
ce_i = CE_PAD(x_worst_i)
```

Harmful-only average:

```text
if harmful_count > 0:
    L_harm = sum(harmful_i * ce_i) / harmful_count
else:
    L_harm = differentiable_zero
```

Final:

```text
L_total =
    0.75 * L_clean
  + 0.25 * L_harm
  + L_aux
```

Important:

> when `harmful_count = 0`, the clean coefficient remains `0.75`; the implementation does not renormalize it to `1.0`.

---

# 12. Clean baseline and WBST definitions

These are needed for the main ablation.

## Clean baseline

```text
L_total = CE_PAD(clean) + L_aux
```

No spectral candidate generation is used for optimization.

## WBST

Development ID: `P3_SF_100K_crop15`.

Spectral geometry:

```text
centers = [0.0833333333, 0.25, 0.4166666667]
sigma   = 0.0555555556
gain    = Uniform(0.65,0.90)
```

Active loss:

```text
L_total =
    0.75 * CE(clean)
  + 0.25 * CE(selected_worst)
  + L_aux
```

All samples receive selected-worst spectral CE.

## MAST-PAD

Development ID: `R7_SC_100K_crop15`.

Difference:

```text
sigma = 0.10
selected-worst CE is harmful-only
```

Because both geometry and gating differ between the final WBST and MAST-PAD primary runs, the main C→WBST→MAST-PAD comparison is a method progression, not a perfectly isolated one-factor ablation.

If a formal causal ablation is required for a later journal paper, rerun matched geometry with only the harmful gate changed.

---

# 13. Primary training contract

```text
Train100K
Val15K
crop1.5
seed100
DATA_SEED43
batch256
AMP
SGD lr=.005
momentum=.9
weight_decay=5e-4
max20
milestones=[6,14]
gamma=.2
patience_start=15
patience=3
```

Primary selected epoch:

```text
MAST-PAD = epoch 11
```

Training stopped:

```text
epoch 17
```

---

# 14. Threshold and evaluation contract

## 14.1 Primary 100K

```text
Val15K
→ find min-ACER PAD threshold
→ freeze threshold
→ full CelebA Test
→ LCC Training
→ LCC Development
→ LCC Evaluation
→ pooled LCC Combined
```

No target-domain threshold fitting.

MAST-PAD frozen source threshold in the saved 100K run:

```text
-0.373016
```

## 14.2 Full-scale study

Different contract:

```text
full Test-as-Val
→ checkpoint + min-ACER threshold
→ freeze
→ LCC
```

Do not merge full-scale and 100K threshold-dependent metrics as if they came from the same protocol.

---

# 15. Primary 100K metrics

MAST-PAD:

```text
Val15K:
AUC   = 0.999606
EER   = 0.8412%
ACER  = 0.8266%

Full CelebA Test:
AUC   = 0.984804
EER   = 6.0734%
ACER  = 11.3518%
(threshold frozen from Val15K)

LCC Training:
AUC = 0.889963

LCC Development:
AUC = 0.853790

LCC Evaluation:
AUC  = 0.825330
EER  = 25.8015%
HTER = 25.7182%

LCC Combined:
AUC  = 0.867844
EER  = 21.5867%
HTER = 21.5738%
TPR@FPR1% = 0.270479
```

Paper-facing:

```text
LCC Combined AUC  = 0.868
Combined EER      = 21.59%
Combined HTER     = 21.57%
Params            = 0.434M
```

---

# 16. Diagnostics contract

Log per epoch:

```text
clean CE
selected worst CE
harmful CE
harmful count
harmful fraction
clean aligned margin
worst aligned margin
margin drop
spectral flip rate
sampled gain mean
LOW selection fraction
MID selection fraction
HIGH selection fraction
auxiliary losses
learning rate
Val AUC / ACER
best epoch
```

For the final 100K MAST-PAD run, the last executed epoch records:

```text
harmful fraction = 0.847790
margin drop      = 0.417364
flip rate        = 0.004370

LOW  = 0.427060
MID  = 0.254670
HIGH = 0.318270
```

These are diagnostic observations, not causal proof.

---

# 17. Deployment contract

Deployment is intentionally simple:

```text
detector
→ crop
→ 80×80 spatial PAD model
```

Training-only modules:

```text
FFT
radial spectral masks
candidate LOW/MID/HIGH generation
candidate selection
harmful gate
auxiliary losses
```

Core claim:

> **MAST-PAD has no additional spectral branch or model parameters at inference relative to the matched clean baseline.**

---

# 18. Export path

Recommended:

```text
PyTorch best checkpoint
→ strip training-only logic / auxiliary heads as needed
→ ONNX
→ ONNX Runtime
→ edge target
```

Optional later:

```text
INT8 quantization
```

Quantization is a deployment optimization, not part of the MAST-PAD contribution.

---

# 19. Architecture freeze checklist

Completed:

- [x] MiniFASNetV2 selected;
- [x] crop1.5 selected;
- [x] binary PAD head selected;
- [x] auxiliary training heads frozen;
- [x] spectral centers frozen;
- [x] sigma frozen;
- [x] gain range frozen;
- [x] harmful-gated loss frozen;
- [x] matched100K run complete;
- [x] full-scale support complete;
- [x] deployment complexity measured.

Do not introduce a new backbone or inference frequency branch into the current course-paper method.

---

# 20. Final architecture summary

```text
MAST-PAD TRAINING

SCRFD bbox
→ crop1.5
→ 80×80 BGR
→ base augmentation
→ clean + LOW/MID/HIGH counterfactuals
→ shared MiniFASNetV2
→ label-aligned binary PAD margins
→ worst-band selection
→ harmful gate
→ 0.75 clean CE + 0.25 harmful-only worst CE
→ clean-view auxiliary supervision


MAST-PAD INFERENCE

SCRFD bbox
→ crop1.5
→ 80×80 BGR
→ MiniFASNetV2
→ binary PAD score
```

This is the architecture contract to use in the paper, figures, README, and future code comments.
