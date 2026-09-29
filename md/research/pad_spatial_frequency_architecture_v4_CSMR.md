# CSMR-PAD Architecture Specification
## Lightweight Spatial Inference + Training-Time Spectral Robustness

_Last updated: 2026-09-29_

> This document supersedes the old dual-branch spatial–frequency architecture specification.
>
> Historical DCT branches M0–M4 remain part of the research motivation, but the current proposed method is **spatial-only at inference**.

---

# 0. Architecture status

Frozen method:

```text
CSMR-v1 = FROZEN
```

Final backbone:

```text
PENDING one fair MNV3 2.7× / 12-epoch comparison
```

Candidates:

```text
CASE A: MiniFASNetV2
CASE B: MobileNetV3-Small
```

---

# 1. High-level concept

## Training

```text
base image
   │
   ├────────────── clean ───────────────┐
   │                                    │
   ├─ LOW spectral attenuation ───────┐ │
   ├─ MID spectral attenuation ─────┐ │ │
   └─ HIGH spectral attenuation ──┐ │ │ │
                                  ▼ ▼ ▼ ▼
                             shared PAD model
                                  │
                     label-aligned PAD margins
                                  │
                     select worst spectral view
                                  │
                clean/worst CE + harmful margin loss
```

## Inference

```text
image
→ face crop
→ selected spatial backbone
→ 3 logits
```

No FFT at inference.

---

# 2. Output semantics

Three classes:

```text
0 = Real
1 = Physical Spoof
2 = Digital Spoof
```

Binary PAD score derived from logits:

```text
d = z_real - logsumexp(z_physical, z_digital)
```

Interpretation:

```text
larger d → more Real
smaller d → more Attack
```

This score is used for:

- ROC AUC;
- Val threshold calibration;
- APCER/BPCER/ACER/HTER;
- CSMR aligned margin.

---

# 3. Label-aligned PAD margin

Define:

```text
sign(y) =
  +1, y = Real
  -1, y = Physical/Digital
```

Then:

```text
m(z,y) = sign(y) * d(z)
```

Higher:

> model is more correct/confident on the binary PAD decision.

Lower:

> counterfactual moved decision in a harmful direction.

Worst view:

```text
k* = argmin_k m_k
```

---

# 4. Spectral counterfactual generator

Input:

```text
image float in [0,1]
```

before backbone normalization.

For an image of spatial size `H×W`:

1. compute shifted 2D FFT per channel;
2. retain phase;
3. apply smooth radial amplitude attenuation;
4. inverse FFT;
5. keep valid image range.

Normalized radial coordinate:

```text
r ∈ [0,1]
```

Band centers:

```text
LOW  = 0.15
MID  = 0.45
HIGH = 0.75
```

Gaussian mask:

```text
M_k(r) = exp(-(r-c_k)^2 / (2 sigma^2))
sigma = 0.10
```

Sample:

```text
g ~ Uniform(0.65, 0.90)
```

Shared across all three views of a sample.

Gain:

```text
G_k = 1 - (1-g) M_k
```

Constraints:

- `G_k <= 1`;
- no amplification;
- preserve DC;
- same gain field across channels;
- phase unchanged;
- smooth mask;
- no hard zero band deletion.

---

# 5. CSMR training objective

## Epoch 1

```text
L = CE(clean)
```

Purpose:

> clean warm-up.

## Epoch 2+

Classification:

```text
L_cls =
    0.5 CE(clean)
  + 0.5 CE(worst)
```

Margin target:

```text
target = m_clean.detach()
```

Harmful mask:

```text
harmful = (m_worst < target)
```

Margin robustness:

```text
L_margin =
    mean(
        SmoothL1(m_worst, target)
        * harmful
    )
```

Final:

```text
L =
    L_cls
  + 0.20 L_margin
```

Frozen values:

```text
warmup = 1
lambda = 0.20
sigma = 0.10
centers = [0.15, 0.45, 0.75]
gain = [0.65, 0.90]
```

---

# 6. BatchNorm-safe worst-view selection

Both candidate backbones contain normalization state that should not be corrupted by multiple selection passes.

Selection procedure:

```text
temporarily eval()
+ no_grad()
→ forward LOW/MID/HIGH
→ compute aligned margins
→ choose k*
→ restore train()
```

Optimization:

```text
train-mode forward
clean + selected worst view
→ loss
→ backward
```

The selection forwards must not triple-update BN statistics.

---

# 7. CASE A — MiniFASNetV2 final architecture

## 7.1. Preprocessing

```text
raw image
→ frozen SCRFD bbox coordinates
→ 2.7× face-context crop
→ direct resize 80×80
→ channel order matching official checkpoint
→ float [0,1]
```

Do not apply MNV3 CelebA mean/std.

## 7.2. Model

Official source:

```text
MiniFASNetV2
embedding_size = 128
conv6_kernel = (5,5)
drop_p = 0.2
num_classes = 3
img_channel = 3
```

Important:

> `80×80` checkpoint requires `conv6_kernel=(5,5)`, not the source default `(7,7)`.

Architecture flow:

```text
80×80
→ MiniFAS depthwise/residual trunk
→ 512 flatten
→ Linear 512→128
→ BN
→ Dropout
→ Linear 128→3
```

Current trainable parameter count:

```text
~434,560
```

## 7.3. Initialization

Use:

```text
official 2.7_80x80_MiniFASNetV2.pth
```

Policy:

```text
load compatible feature/embedding weights
discard original prob.weight
reinitialize project 128→3 classifier
```

Reason:

> official 3-class semantics are not assumed to match Real/Physical/Digital.

## 7.4. CSMR

CSMR operates on:

```text
80×80 float image in [0,1]
```

Masks are generated at 80×80 but normalized centers/sigma/gain remain unchanged.

## 7.5. Inference

```text
SCRFD
→ 2.7× crop
→ 80×80
→ MiniFASNetV2
→ logits
```

No FFT.

---

# 8. CASE B — MobileNetV3-Small final architecture

## 8.1. Preprocessing

Final candidate recipe:

```text
raw image
→ frozen SCRFD bbox coordinates
→ 2.7× face-context crop
→ existing resize + REFLECT_101 letterbox
→ 224×224 RGB
→ CelebA normalization
```

Normalization:

```text
mean = [0.5931, 0.4690, 0.4229]
std  = [0.2471, 0.2214, 0.2157]
```

Gamma:

```text
OFF
```

## 8.2. Model

```text
MobileNetV3-Small ImageNet
→ GAP
→ 576→256
→ BN
→ Hardswish
→ Dropout 0.2
→ 256→128
→ Hardswish
→ Dropout 0.2
→ 3 logits
```

Current micro parameter count:

```text
~1,108,515
```

## 8.3. CSMR

Intervention is applied to:

```text
224×224 RGB float [0,1]
```

before CelebA normalization.

Masks are 224×224, but normalized spectral parameters are identical to MiniFAS case.

## 8.4. Inference

```text
SCRFD
→ 2.7× crop
→ 224×224 RGB
→ MobileNetV3-Small
→ project PAD head
→ logits
```

No FFT.

---

# 9. Shared augmentation

For paired Vanilla vs CSMR within the selected backbone:

- same base image;
- same mild augmentation;
- only CSMR adds the spectral counterfactual after base augmentation.

Do not let Vanilla and CSMR differ in:

- crop;
- color order;
- spatial augmentation;
- optimizer schedule;
- batch size;
- data membership.

---

# 10. Historical explicit-frequency architectures

These are not final candidates anymore.

## M0

```text
Global DCT
→ signed-log
→ z-score
→ Tiny CNN
→ GAP
→ 64-D
```

## M1

```text
Global DCT
→ CNN
→ Pool4×4
→ Linear
→ 64-D
```

## M2

```text
DCT + coordinate channels
→ CNN
→ Pool4×4
→ 64-D
```

## M3

```text
LOW/MID/HIGH masked DCT channels
→ CNN
→ Pool4×4
→ 64-D
```

## M4

```text
8×8 block DCT
→ coefficient identities as channels
→ spatial grid CNN
→ 64-D
```

Role now:

```text
historical diagnosis
+
motivation for spectral-shortcut interpretation
```

Do not add these branches to the final inference graph unless a new research cycle is explicitly started.

---

# 11. Why CSMR is different from M3

M3:

```text
frequency bands
→ explicit learned feature
→ used at inference
```

CSMR:

```text
frequency bands
→ controlled perturbations
→ used to stress current spatial decision during training
→ no frequency feature at inference
```

The LOW/MID/HIGH concept is shared only at a high level.

The mechanism and deployment graph are fundamentally different.

---

# 12. Backbone decision experiment

Current unfair comparison:

```text
MNV3:
1.55× / 224 / ImageNet / 6ep

MiniFAS:
2.7× / 80 / PAD pretrained / 12ep
```

Required fairer MNV3 run:

```text
crop = 2.7×
input = 224
seed = 100
max epochs = 12
patience = 3
earliest stop = 6
Vanilla + CSMR
```

After this run choose CASE A or CASE B.

Do not reopen architecture search afterward.

---

# 13. Current evidence table

| Recipe | Params | Val ACER | LCC HTER | LCC AUC |
|---|---:|---:|---:|---:|
| MNV3 old Vanilla | ~1.11M | 2.1062% | 34.0274% | 0.690181 |
| MNV3 old + CSMR | ~1.11M | 2.1357% | 33.1171% | 0.714238 |
| MiniFAS Vanilla | ~0.435M | 1.6909% | 31.8242% | 0.819484 |
| MiniFAS + CSMR | ~0.435M | 2.1691% | 28.7548% | 0.846264 |

Do not interpret this table as pure architecture isolation.

---

# 14. Training scale contracts

## Micro development

```text
Train = 5K
Val = 2K
```

Purpose:

- CSMR freeze;
- backbone development.

## Scale sanity

```text
Train = 20K
```

Purpose:

- verify gain survives more data;
- choose final epoch budget.

## Final

```text
Train = 100K
Val = 15K
```

Purpose:

- main paper checkpoints.

---

# 15. Final optimization — CASE A

MiniFASNetV2:

```text
optimizer = fixed fine-tuning policy
same for Vanilla/CSMR
```

Current transfer benchmark used a conservative PAD-pretrained fine-tuning regime.

For final 100K:

- pre-register after 20K trajectory;
- default max 12 if plateau observed;
- do not tune on LCC.

---

# 16. Final optimization — CASE B

MNV3-S:

Historical 100K E1 v5.3 budget:

```text
max epochs = 24
patience = 4
earliest stop = 10
backbone LR = 1e-5
head LR = 1e-4
weight decay = 1e-4
```

If MNV3 wins, preserve this as default final optimization budget unless the 20K pre-registered protocol explicitly supersedes it.

Same schedule for Vanilla and CSMR.

---

# 17. Metrics

Source/development:

```text
APCER
BPCER
ACER
AUC
```

Cross-domain:

```text
HTER
AUC
APCER
BPCER
```

Efficiency:

```text
Params
MACs/FLOPs
model size
peak memory
latency
FPS
```

---

# 18. Threshold protocol

Always:

```text
Train
→ Val checkpoint selection
→ Val threshold calibration
→ LOCK
→ LCC / Test / CASIA
```

Never:

```text
target dataset
→ threshold sweep
```

---

# 19. Dataset roles

```text
CelebA development Train/Val
→ architecture/method development

LCC
→ external development stress

fixed Test-2000
→ already-opened descriptive benchmark

100K/15K
→ final training/checkpoint protocol

CASIA
→ untouched final external confirmation
```

---

# 20. CSMR diagnostics

Log:

```text
LOW worst fraction
MID worst fraction
HIGH worst fraction
harmful fraction
mean harmful margin drop
mean clean aligned margin
mean worst aligned margin
mean sampled gain
clip fraction
classification loss
margin loss
total loss
```

Class-wise:

```text
Real
Physical
Digital
```

Warning conditions:

- one band ≈ 100%;
- non-finite loss;
- excessive clipping;
- broken phase/DC sanity.

Warnings do not automatically change hyperparameters.

---

# 21. Mechanism ablation architecture

All variants use the final frozen backbone.

```text
A0 Vanilla

A1 Random spectral augmentation
   random LOW/MID/HIGH
   no worst selection
   no margin loss

A2 Generic consistency
   spectral view
   + KL/logit consistency

A3 Worst-band CE
   worst selection
   lambda = 0

A4 Full CSMR
   worst selection
   harmful margin
```

This is the core novelty proof.

---

# 22. Edge deployment

CSMR deployment is intentionally boring:

```text
detector
→ crop
→ spatial PAD model
```

Training-only components removed:

```text
FFT
LOW/MID/HIGH generator
worst selector
margin loss
```

Therefore within the same backbone:

```text
Vanilla inference cost
≈
CSMR inference cost
```

This must be verified empirically with latency/FPS.

---

# 23. Export / deployment path

After final model:

```text
PyTorch
→ ONNX
→ ONNX Runtime
→ Raspberry Pi
```

Optional later:

```text
INT8 quantization
```

Quantization is deployment optimization, not core method contribution.

---

# 24. Architecture freeze checklist

Before declaring final architecture:

- [x] CSMR-v1 mechanism frozen;
- [x] MNV3 CSMR positive development evidence;
- [x] MiniFAS CSMR transfer evidence;
- [ ] MNV3 2.7× / 12ep fair check;
- [ ] CASE A or CASE B selected;
- [ ] final preprocessing contract frozen;
- [ ] final 20K sanity completed;
- [ ] final 100K optimization budget pre-registered.

After these items:

> no new backbone, crop or frequency module may be introduced into the main paper experiment.

---

# 25. Final architecture templates

## CASE A — MiniFASNetV2

```text
TRAINING
80×80 2.7× crop
→ clean + spectral counterfactuals
→ MiniFASNetV2
→ CSMR loss

INFERENCE
80×80 2.7× crop
→ MiniFASNetV2
→ PAD
```

## CASE B — MNV3-S

```text
TRAINING
224×224 2.7× crop
→ clean + spectral counterfactuals
→ MobileNetV3-Small
→ CSMR loss

INFERENCE
224×224 2.7× crop
→ MobileNetV3-Small
→ PAD
```

The paper method should be described at a backbone-agnostic level first, then instantiate the selected case in the Experimental Setup.
