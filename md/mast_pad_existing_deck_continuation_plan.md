# MAST-PAD Continuation Deck Plan
## Designed to continue directly after the existing “MAST-PAD” title slide

_Last updated: 2026-10-05_

---

# 0. Integration rule

This deck is **not a standalone replacement** for the existing “Secure Face Authentication System” deck.

It should be inserted **after the current MAST-PAD divider slide**.

Current deck structure:

```text
Slide 1  — Secure Face Authentication System title
Slide 2  — AI pipeline
Slide 3  — Face Detection
Slide 4  — Detection filtering
Slide 5  — Face Tracking
Slide 6  — Anti-Spoofing overview
Slide 7  — Old model candidates / MobileNetV3-style plan
Slide 8  — CelebA-Spoof dataset
Slide 9  — Old LogSumExp decision formula
Slide 10 — Temporal smoothing
Slide 11 — Alignment
Slide 12 — Face embedding & identity matching
Slide 13 — Vector similarity search
Slide 14 — MAST-PAD divider
Slide 15 — Old Face PAD positioning
Slide 16 — Old evaluation framework
Slide 17 — Thank you
```

New rule:

```text
Keep slides 1–14 as system/background context.
Replace slides 15–16 with the new MAST-PAD continuation.
Move "Thank you" to the final slide.
```

Recommended final sequence:

```text
Slides 1–14: existing deck, lightly patched where necessary
Slides 15–26: new MAST-PAD research section
Slide 27: Thank you / Q&A
```

If presentation time is limited, slides 24–25 can be merged, resulting in 10–11 new slides.

---

# 1. Minimal patches before the MAST-PAD section

## Slide 7 must be corrected or softened

Current slide 7 says or implies:

```text
MobileNetV3-Small
224×224 crop
3-class classification
REAL / SPOOF Print / Screen
```

This is outdated for the final method.

Patch it to say:

```text
Early model options / historical exploration
```

or replace the lower flow with:

```text
SCRFD crop1.5
→ 80×80 BGR
→ MiniFASNetV2
→ binary Real / Attack
```

Keep the table of model families only as background, not as the final choice.

## Slide 9 should be marked as historical or removed

The LogSumExp slide belongs to the older 3-class PAD formulation.

Patch option:

```text
Historical decision rule for 3-class experiments.
Final MAST-PAD uses a binary Real/Attack margin.
```

Better option: move it to backup.

## Slide 15 and 16 should be replaced

Slide 15 currently frames the method as explicit frequency branch / fusion. The final method is not that.

Slide 16 has a useful metric framework but the references should be updated to the final comparison set.

---

# 2. New continuation section after Slide 14

## Slide 14 — keep as divider

### Existing title
```text
MAST-PAD
```

### Patch it slightly
Add subtitle:

```text
Margin-Aware Spectral Training for Face Presentation Attack Detection
```

Add 3 small badges:

```text
0.434M params
LCC AUC 0.868
No FFT at inference
```

Speaker message:

> “From here, we zoom into the research contribution inside Stage 3: Anti-Spoofing.”

---

# Slide 15 — Research problem after system context

## Title
**Why Stage 3 still needs research**

## Purpose
Connect the system deck to MAST-PAD.

The previous slides already explained why PAD protects the authentication pipeline. This slide explains why a normal PAD model is not enough.

## Content

```text
In-domain PAD accuracy can look very high.
But cross-domain deployment changes:
- camera sensor;
- lighting;
- print/display media;
- attack device;
- spoof texture statistics.
```

## Key sentence
```text
A lightweight model must generalize, not just fit source-domain spoof texture.
```

## Suggested visual
Left side:

```text
CelebA-Spoof source
high validation AUC
```

Right side:

```text
LCC-FASD target
domain shift
```

Bottom:

```text
Goal: robust PAD without increasing inference cost
```

## Speaker note
“Our system can already detect and track faces. The hard research problem is making the PAD decision robust when the target camera and attack media differ.”

---

# Slide 16 — Final method positioning

## Title
**MAST-PAD positioning**

## Purpose
Replace old slide 15 with the final positioning.

## Content table

| Direction | Typical idea | Limitation | MAST-PAD choice |
|---|---|---|---|
| Lightweight CNN | MiniFAS / Mobile CNN | domain-sensitive texture | keep 0.434M inference |
| Frequency-aware PAD | FFT/DCT branch or fusion | extra inference path | frequency only during training |
| Transformer / ViT | strong representation | heavy | avoid large backbone |
| Domain generalization | robust training | often complex | margin-aware spectral stress |

## Bottom takeaway
```text
MAST-PAD = lightweight spatial inference + training-only spectral robustness
```

## Speaker note
“The method is frequency-aware, but it is not a frequency-branch model at deployment.”

---

# Slide 17 — Motivation: frequency as a stress test

## Title
**Why use frequency during training?**

## Content

Attack media can introduce:

```text
Moiré patterns
pixel-grid artifacts
resampling traces
micro-texture shifts
print/display spectral changes
```

But frequency can also be a shortcut:

```text
high-frequency ≠ always spoof
frequency cues may change across domains
```

## Key idea

```text
Do not trust frequency as a permanent feature.
Use it to create controlled counterfactual views that expose fragile decisions.
```

## Visual
Use figure:

```text
fig_frequency_counterfactual_examples.png
```

Expected layout:

```text
Real row:   clean | LOW | MID | HIGH | FFT magnitude
Attack row: clean | LOW | MID | HIGH | FFT magnitude
```

## Speaker note
“The spectral views are not used at test time. They are only probes during training.”

---

# Slide 18 — MAST-PAD method overview

## Title
**MAST-PAD in one diagram**

## Purpose
This is the only essential method diagram. The presenter can explain it verbally.

## Diagram contents

```text
Clean crop
├── LOW attenuation
├── MID attenuation
└── HIGH attenuation
      ↓
shared MiniFASNetV2 PAD
      ↓
label-aligned margins
      ↓
select worst band
      ↓
harmful if m_worst < m_clean
      ↓
apply spectral CE only if harmful
```

## Use figure
```text
fig_mast_pad_method_flow.png
```

## Minimal text on slide
```text
1. Generate LOW/MID/HIGH counterfactuals.
2. Select the view with the lowest PAD margin.
3. Train on it only if it reduces the clean margin.
4. Remove all spectral operations at inference.
```

## Speaker note
“The key is not just selecting the worst band. The key is the harmful gate.”

---

# Slide 19 — Margin, gate, and loss

## Title
**Margin-aware harmful gating**

## Purpose
Give the technical definition without overloading the audience.

## Content

Binary PAD score:

```text
d = z_real - z_attack
```

Label-aligned margin:

```text
Real   → m = +d
Attack → m = -d
```

Worst spectral view:

```text
k* = argmin_k m_k
```

Harmful gate:

```text
h = 1[m_worst < m_clean]
```

Final loss:

```text
L = 0.75 CE_clean + 0.25 CE_harmful + L_aux
```

Auxiliary loss:

```text
L_aux = 0.1 CE_spoof + 0.1 CE_light + BCE_attr(real only)
```

## Speaker note
“A sample can be harmful even when the class prediction is still correct. MAST-PAD trains on degradation before it becomes an error.”

---

# Slide 20 — Training protocol

## Title
**Controlled 100K protocol**

## Purpose
Make it clear that the main result is a controlled experiment, not an accidental best run.

## Table

| Component | Setting |
|---|---|
| Source train | CelebA-Spoof Train100K |
| Source validation | CelebA-Spoof Val15K |
| Source test | full usable CelebA-Spoof Test |
| Target evaluation | full LCC-FASD train/dev/eval + pooled Combined |
| Crop / input | crop1.5, 80×80 BGR |
| Backbone | MiniFASNetV2 |
| Optimizer | SGD, lr 0.005, momentum 0.9, wd 5e-4 |
| Batch | 256 |
| Schedule | max20, milestones [6,14], gamma 0.2 |
| Early stop | patience starts epoch 15, patience 3 |
| Threshold | calibrated on Val15K, frozen before LCC |

## Bottom note
```text
Clean, WBST, and MAST-PAD share the same manifests, initialization, crop, optimizer, and deployed architecture.
```

## Speaker note
“LCC is used only after checkpoint and threshold are frozen.”

---

# Slide 21 — Main result: Clean vs WBST vs MAST-PAD

## Title
**Main ablation: harmful gating matters**

## Main table

| Method | Test AUC | LCC Eval AUC | LCC Combined AUC | Combined EER | Combined HTER | TPR@FPR1% |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.982413 | 0.798517 | 0.848273 | 23.5328% | 24.0514% | 0.199776 |
| WBST | 0.982573 | 0.812841 | 0.851978 | 23.8822% | 23.8322% | 0.214838 |
| **MAST-PAD** | **0.984804** | **0.825330** | **0.867844** | **21.5867%** | **21.5738%** | **0.270479** |

## Paper-facing highlight

```text
Clean:     0.848
WBST:      0.852
MAST-PAD:  0.868
```

## Main delta

```text
MAST-PAD vs Clean:
+1.96 pp LCC Combined AUC
-1.95 pp Combined EER
-2.48 pp Combined HTER
```

## Figure
Use:

```text
fig_main_ablation_lcc_combined_auc.png
```

---

# Slide 22 — Training diagnostics

## Title
**What does the harmful gate select?**

## Key diagnostics from MAST-PAD 100K

```text
selected checkpoint epoch = 11
training stopped epoch = 17

harmful fraction = 0.847790
margin drop      = 0.417364
flip rate        = 0.004370

LOW  = 0.427060
MID  = 0.254670
HIGH = 0.318270
```

## Suggested layout
Left:

```text
line chart:
harmful fraction
margin drop
spectral flip rate
```

Right:

```text
band selection distribution:
LOW / MID / HIGH
```

## Interpretation bullets
```text
Most selected spectral views reduce margin.
Only a small fraction flip the prediction.
The gate catches fragile decisions before outright error.
No single band dominates all samples.
```

## Figure
Use:

```text
fig_mast_pad_training_diagnostics.png
```

---

# Slide 23 — Contextual comparison with literature

## Title
**Contextual LCC-FASD comparison**

## Table

| Method | LCC setting | AUC ↑ | EER ↓ | HTER / ACER ↓ | Params |
|---|---|---:|---:|---:|---:|
| Graph for Transformer Feature | LCC intra | 0.833 | 12.23% | — | ~10.84M |
| AENet | benchmark-reported LCC | 0.868 | 20.91% | ACER 22.61% | 11.22M |
| ViT-B/16 | SiW-M → LCC | — | 30.98% | HTER 34.62% | ~86.6M |
| ViT-B/16 + LoRA | SiW-M → LCC | — | 28.95% | HTER 29.33% | ~86.6M |
| LBP-GBM | official LCC | — | 14.60% | — | classical |
| **MAST-PAD** | **CelebA100K → pooled LCC** | **0.868** | **21.59%** | **HTER 21.57%** | **0.434M** |

## Highlight
```text
MAST-PAD matches AENet's reported AUC after rounding
with about 25.8× fewer parameters.
```

## Caveat
```text
Protocols differ; this is contextual comparison, not a strict leaderboard.
```

## Optional figure
Use:

```text
fig_lcc_auc_vs_params_context.png
```

---

# Slide 24 — Efficiency and deployment

## Title
**Training-only spectral robustness, spatial-only inference**

## Content

Deployment path:

```text
SCRFD bbox
→ crop1.5
→ 80×80 BGR
→ MiniFASNetV2
→ Real / Attack
```

Training-only modules:

```text
FFT
LOW/MID/HIGH masks
worst-view selection
harmful gate
spectral CE
```

Complexity:

```text
deployment parameters = 434,434
training model params = 441,658
MACs = 40,810,892
GFLOPs = 0.081621784
```

## Main claim
```text
MAST-PAD adds no spectral branch or model parameters at inference relative to the clean baseline.
```

## Speaker note
“Training is heavier; inference is not.”

---

# Slide 25 — Full-scale check as supporting evidence

## Title
**Scale-up check: full CelebA-Spoof train**

## Purpose
Show transparency without changing the main result.

## Table

| Method | Full-scale Test AUC | Test ACER | LCC Eval AUC | LCC Combined AUC |
|---|---:|---:|---:|---:|
| Clean | 0.990736 | 4.6248% | 0.811622 | 0.846011 |
| WBST | 0.992242 | 4.3548% | 0.819687 | **0.860831** |
| MAST-PAD | **0.992280** | **4.3459% | **0.840060** | 0.860272 |

## Message
```text
MAST-PAD remains effective at full scale and has the best LCC Eval AUC.
The 100K run remains the primary course-paper protocol.
```

## Reason
```text
Full-scale changes both data scale and selection set, so it is a separate scale-up study.
```

## Figure
Use:

```text
fig_fullscale_scaleup_summary.png
```

---

# Slide 26 — Takeaways and limitations

## Title
**Takeaways**

## Main takeaways

```text
1. MAST-PAD uses frequency as a training-time stressor, not an inference branch.
2. The harmful gate improves over always-on worst-band training.
3. LCC Combined AUC improves from 0.848 to 0.868.
4. The deployed model remains lightweight: 0.434M parameters.
```

## Limitations

```text
one seed only
literature protocols differ
LCC Combined pools multiple LCC splits
training cost increases due to spectral candidate forwards
evidence is MiniFASNetV2-specific
```

## Future work

```text
multi-seed validation
independent cross-domain dataset
Raspberry Pi / edge latency
ONNX / INT8 deployment
```

## Closing line
```text
Frequency helps most when it exposes fragile decisions,
not when it is blindly added as a permanent inference branch.
```

---

# 3. Speaker narrative across the appended section

Use this storyline:

```text
Slide 15:
The system needs PAD, but the research difficulty is cross-domain robustness.

Slide 16:
Existing directions either stay lightweight but fragile, or become robust but heavier.

Slide 17:
Frequency contains attack cues, but can also be a shortcut.

Slide 18:
MAST-PAD turns frequency into a training-time stress test.

Slide 19:
The margin decides which spectral view is harmful.

Slide 20:
We compare methods under a matched 100K protocol.

Slide 21:
MAST-PAD improves over Clean and WBST on LCC.

Slide 22:
Diagnostics show the gate captures margin degradation before class flips.

Slide 23:
The result is competitive with selected LCC references while much smaller.

Slide 24:
Deployment stays spatial-only and lightweight.

Slide 25:
Full-scale supports viability but is secondary.

Slide 26:
Clear takeaway + limitations.
```

---

# 4. Checklist before editing the actual deck

- [ ] Current slide 14 becomes the divider into the MAST-PAD research section.
- [ ] Old slide 15 is replaced by the new positioning slide.
- [ ] Old slide 16 is replaced by the new evaluation / comparison content.
- [ ] Old slide 17 “Thank you” is moved to the very end.
- [ ] No new slide says MobileNetV3 is the final model.
- [ ] No new slide says 224×224 input.
- [ ] No new slide says 3-class final classifier.
- [ ] No new slide uses LogSumExp as the current decision rule.
- [ ] MAST-PAD is defined once and used consistently.
- [ ] `R7_SC_100K_crop15` appears only as provenance / internal ID if needed.
- [ ] P3-SF appears as WBST, not as a second proposed method.
- [ ] Full-scale appears only as “secondary scale-up evidence.”
- [ ] Literature table says contextual comparison, not SOTA.
