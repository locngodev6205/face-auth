# MAST-PAD Paper Roadmap & Experimental Design Plan
## Course-paper final story centered on the R7-SC 100K configuration

_Last updated: 2026-10-05_

> **Official title candidate:**  
> **MAST-PAD: Margin-Aware Spectral Training for Lightweight Cross-Domain Face Anti-Spoofing**
>
> **Primary method:** MAST-PAD  
> **Development ID:** `R7_SC_100K_crop15`
>
> This document supersedes the earlier CSMR-centered paper plan. Historical CSMR pilots remain experiment provenance only; the final paper method is the harmful-selective R7-SC formulation.

---

# 1. One-sentence paper idea

> **MAST-PAD uses the current Face PAD decision margin to identify a sample-specific harmful spectral counterfactual during training and applies worst-view spectral supervision only when that intervention degrades the clean margin, while deployment remains the same 0.434M-parameter spatial MiniFASNetV2.**

---

# 2. Final paper scope

This is a **course paper**, not a final publication-grade benchmark study.

The goal is to present one clear method and one clear controlled experiment.

Primary protocol:

```text
CelebA-Spoof Train100K
→ Val15K checkpoint selection + threshold calibration
→ full CelebA Test
→ full LCC-FASD cross-domain evaluation
```

Primary proposed method:

```text
MAST-PAD = R7_SC_100K_crop15
```

Secondary evidence:

```text
full-scale C / WBST / MAST-PAD experiment
```

The full-scale run is a scale-up study and should not replace the 100K headline result.

---

# 3. Problem statement

Lightweight RGB Face PAD models can perform extremely well on source validation while degrading under domain shift.

Frequency-domain cues are attractive because print/replay attacks can alter:

- fine texture;
- resampling patterns;
- display-grid / Moiré behavior;
- camera-display transfer statistics;
- low-level spectral energy.

However, spectral cues can also become source-specific shortcuts.

The paper therefore asks:

> **Instead of trusting frequency as an inference feature, can we use controlled spectral interventions to identify and reduce fragile PAD decisions during training?**

---

# 4. Method progression used in the paper

Use only three conceptual stages.

## 4.1 Clean baseline

A lightweight multi-task MiniFASNetV2 model trained only on the clean/base image.

Role:

> matched spatial baseline.

## 4.2 WBST — Worst-Band Spectral Training

Development origin: `P3_SF`.

For every sample:

```text
generate LOW / MID / HIGH views
→ select the view with minimum aligned PAD margin
→ always train on that selected worst view
```

Role:

> shows the value of adaptive worst-view spectral training.

## 4.3 MAST-PAD

Development origin: `R7_SC`.

MAST-PAD adds the key selective rule:

```text
if m_worst < m_clean:
    spectral supervision is active
else:
    no spectral CE is applied to that sample
```

Role:

> final proposed method.

The paper-facing progression is therefore:

```text
Clean baseline
→ WBST
→ MAST-PAD
```

Do not expose all R0–R9 development IDs in the main method section.

---

# 5. Final model

## 5.1 Shared inference architecture

```text
SCRFD face localization
→ crop factor 1.5
→ 80×80 BGR [0,1]
→ MiniFASNetV2 trunk
→ 128-D embedding
→ binary PAD head
```

Training additionally uses auxiliary heads:

```text
spoof type: 11 classes
lighting:   5 classes
attributes: 40 binary attributes, Real-only
```

Deployment complexity:

```text
deployment params = 434,434
training model total params = 441,658
MACs = 40,810,892
GFLOPs = 0.081621784
```

All C / WBST / MAST-PAD variants have the same deployed PAD architecture.

---

# 6. MAST-PAD method

## 6.1 PAD score

For binary logits:

```text
z_real, z_attack
d = z_real - z_attack
```

Label-aligned margin:

```text
m = +d for Real
m = -d for Attack
```

Higher margin means a safer binary PAD decision.

## 6.2 Spectral counterfactuals

For each base-augmented sample:

```text
LOW
MID
HIGH
```

with:

```text
centers = [0.0833333333, 0.25, 0.4166666667]
sigma = 0.10
gain ~ Uniform(0.65, 0.90)
```

Radial attenuation:

```text
M_k(r) = exp(-(r-c_k)^2 / (2 sigma^2))
G_k(r) = 1 - (1-g) M_k(r)
```

Constraints:

- no amplification;
- DC preserved;
- phase unchanged;
- shared gain across three candidate bands for one sample;
- same mask across image channels.

## 6.3 Worst-view selection

```text
k* = argmin_k m_k
```

Selection is performed BN-safely with temporary `eval()` + `no_grad()`.

## 6.4 Harmful gate

```text
harmful = (m_worst < m_clean)
```

This is more informative than a misclassification-only gate: a view can be harmful before it actually flips the predicted class.

## 6.5 Objective

Epoch 1:

```text
clean PAD CE + auxiliary loss
```

Epoch 2+:

```text
L_PAD =
    0.75 * CE(clean)
  + 0.25 * CE_harmful(selected_worst)

L_aux =
    0.1 * CE(spoof_type)
  + 0.1 * CE(lighting)
  + 1.0 * BCE(attributes on Real)

L_total = L_PAD + L_aux
```

`CE_harmful` is averaged only over harmful samples and becomes differentiable zero when no sample is harmful.

---

# 7. Primary experimental protocol

## 7.1 Dataset roles

| Data | Role |
|---|---|
| CelebA-Spoof Train100K | source training |
| CelebA-Spoof Val15K | checkpoint selection + min-ACER threshold calibration |
| Full usable CelebA-Spoof Test | held-out source evaluation after freeze |
| LCC-FASD Training | cross-domain target evaluation |
| LCC-FASD Development | cross-domain target evaluation |
| LCC-FASD Evaluation | cross-domain target evaluation |
| LCC-FASD Combined | pooled headline cross-domain result |

LCC is not used to refit the threshold in the primary matched100K run.

## 7.2 Fairness controls

All three primary methods use:

```text
same Train100K manifest
same Val15K manifest
same initial state
same seed
same crop
same base augmentation
same optimizer schedule
same inference architecture
```

The intended causal contrast is therefore the spectral training rule.

## 7.3 Optimization

```text
SGD
lr = 0.005
momentum = 0.9
weight_decay = 5e-4
batch = 256
AMP = on
max_epoch = 20
milestones = [6, 14]
gamma = 0.2
patience_start = 15
patience = 3
```

---

# 8. Main results

## 8.1 Source and target performance

| Method | Val15K AUC | Val ACER | Test AUC | LCC Eval AUC | LCC Combined AUC | Combined EER | Combined HTER |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.999523 | 0.8271% | 0.982413 | 0.798517 | 0.848273 | 23.5328% | 24.0514% |
| WBST | 0.999587 | 0.8463% | 0.982573 | 0.812841 | 0.851978 | 23.8822% | 23.8322% |
| **MAST-PAD** | **0.999606** | **0.8266%** | **0.984804** | **0.825330** | **0.867844** | **21.5867%** | **21.5738%** |

Paper-facing rounding:

```text
Clean     LCC AUC = 0.848
WBST      LCC AUC = 0.852
MAST-PAD  LCC AUC = 0.868
```

## 8.2 Main paired effect

Against Clean:

```text
Combined AUC:  +1.9571 pp
Combined EER:  -1.9461 pp
Combined HTER: -2.4776 pp
TPR@FPR1%:     +0.070703
```

Against WBST:

```text
Combined AUC: +0.015866
```

This supports the specific idea that **harmful selectivity matters beyond always-on selected-worst training**.

Do not call the one-seed difference statistically significant.

---

# 9. Main literature comparison

Use one compact contextual comparison table.

| Method | LCC protocol | AUC ↑ | EER ↓ | HTER / ACER ↓ | Params |
|---|---|---:|---:|---:|---:|
| Graph for Transformer Feature | LCC intra | 0.833 | 12.23% | — | ~10.84M |
| AENet | benchmark-reported LCC result | 0.868 | 20.91% | ACER 22.61% | 11.22M |
| ViT-B/16 | SiW-M → LCC | — | 30.98% | HTER 34.62% | ~86.6M |
| ViT-B/16 + LoRA | SiW-M → LCC | — | 28.95% | HTER 29.33% | ~86.6M backbone |
| LBP-GBM | official LCC | — | 14.60% | — | classical |
| **MAST-PAD** | **CelebA100K → pooled LCC** | **0.868** | **21.59%** | **HTER 21.57%** | **0.434M** |

Interpretation:

> MAST-PAD is not claimed as SOTA. Its main advantage is the combination of a very small deployment model, no spectral inference branch, and a numerically competitive LCC result under a cross-domain source-to-target setup.

Strong course-paper sentence:

> **With only 0.434M deployment parameters, MAST-PAD reaches 0.868 pooled LCC AUC, numerically exceeding the 0.833 Graph Transformer result and matching the 0.868 AENet reference after rounding, while using substantially fewer parameters.**

Immediately follow with:

> **The compared studies use different LCC training/evaluation protocols, so the table is contextual rather than a strict common-protocol leaderboard.**

---

# 10. Full-scale result as a scale-up study

Do not replace the primary 100K table with full-scale.

Add a short subsection:

## Scale-up behavior

| Method | Full-scale Test-as-Val AUC | LCC Eval AUC | LCC Combined AUC |
|---|---:|---:|---:|
| Clean | 0.990736 | 0.811622 | 0.846011 |
| WBST | 0.992242 | 0.819687 | **0.860831** |
| MAST-PAD | **0.992280** | **0.840060** | 0.860272 |

Suggested discussion:

> The frozen spectral strategies remain effective when source training is expanded to the full eligible CelebA-Spoof training set. MAST-PAD retains approximately 0.860 pooled LCC AUC and achieves the strongest LCC Evaluation AUC, while WBST is marginally higher on the pooled Combined AUC. Because the full-scale experiment changes both data scale and checkpoint/threshold selection protocol, it is treated as a separate scale-up study.

---

# 11. Main contributions

## C1 — lightweight matched baseline

> **We establish a compact MiniFASNetV2-based binary PAD baseline with auxiliary supervision and a 0.434M-parameter deployment graph.**

Do not claim a novel MiniFASNet architecture.

## C2 — MAST-PAD

> **We introduce Margin-Aware Spectral Training, which selects the LOW/MID/HIGH spectral intervention that most reduces the sample’s label-aligned PAD margin and applies worst-view spectral supervision only when that intervention is harmful.**

## C3 — cross-domain evidence

> **Under the matched 100K CelebA→LCC protocol, MAST-PAD improves pooled LCC AUC from 0.848 to 0.868 over the clean baseline and improves both EER and HTER.**

## C4 — deployment

> **The spectral mechanism is training-only, so MAST-PAD preserves the same lightweight spatial inference graph as the clean baseline.**

---

# 12. Recommended paper structure

## 1. Introduction

- Face PAD and domain shift;
- lightweight deployment requirement;
- frequency cues are useful but can be fragile;
- frequency-as-feature vs frequency-as-training-stressor;
- MAST-PAD contributions.

## 2. Related Work

### 2.1 Lightweight Face PAD
### 2.2 Frequency-aware Face PAD
### 2.3 Cross-domain / generalizable Face PAD
### 2.4 Spectral shortcut and robustness training

## 3. Method

### 3.1 Lightweight multi-task PAD backbone
### 3.2 Spectral counterfactual generator
### 3.3 Label-aligned PAD margin
### 3.4 Worst-band selection
### 3.5 Harmful-only spectral training
### 3.6 Inference complexity

## 4. Experimental Setup

- CelebA Train100K / Val15K;
- full CelebA Test;
- full LCC splits and pooled Combined;
- threshold calibration;
- crop1.5;
- optimizer;
- metrics;
- literature-comparison protocol caveat.

## 5. Results

### 5.1 Clean vs WBST vs MAST-PAD
### 5.2 Comparison with selected LCC references
### 5.3 Scale-up study
### 5.4 Training diagnostics
### 5.5 Efficiency

## 6. Discussion

Must discuss:

- why harmful gating can outperform always-on worst-view CE;
- why pooled and Evaluation rankings can differ;
- why full-scale is not directly interchangeable with 100K;
- protocol heterogeneity in literature comparison;
- one-seed limitation.

## 7. Conclusion

Keep bounded to evidence.

---

# 13. Recommended figures

## Figure 1 — method overview

```text
clean image
├── LOW attenuation
├── MID attenuation
└── HIGH attenuation
        ↓
   shared PAD model
        ↓
 label-aligned margins
        ↓
  select worst band
        ↓
compare with clean margin
        ↓
 harmful?
   yes → spectral CE
   no  → no spectral CE
```

## Figure 2 — training vs inference

Training:

```text
RGB + spectral counterfactual generator + margin-aware gate
```

Inference:

```text
RGB → MiniFASNetV2 → PAD
```

## Figure 3 — LCC Combined AUC

```text
Clean      0.848
WBST       0.852
MAST-PAD   0.868
```

## Figure 4 — selected literature context

Use parameter count vs AUC where both are available:

```text
Graph Transformer: ~10.84M, AUC 0.833
AENet:            11.22M, AUC 0.868
MAST-PAD:          0.434M, AUC 0.868
```

Caption must state protocol differences.

## Figure 5 — training diagnostics

Optional:

- harmful fraction;
- LOW/MID/HIGH selection fractions;
- clean vs worst margin;
- training/validation curve.

---

# 14. Reporting precision

For paper-facing literature comparison:

```text
AUC: 3 decimals
rates: 2 decimals
params: 2–3 significant decimals
```

Therefore:

```text
0.867844 → 0.868
21.5867% → 21.59%
21.5738% → 21.57%
0.434434M → 0.434M
```

Keep full precision in CSV/JSON and internal appendices.

---

# 15. Limitations

State explicitly:

1. primary matched100K results use one training seed;
2. literature comparison protocols differ;
3. LCC Combined pools training/development/evaluation populations and is not the same as the official LCC evaluation-only protocol;
4. the full-scale experiment uses Test-as-Val for checkpoint/threshold selection;
5. MAST-PAD increases training compute because multiple spectral candidate forwards are required;
6. the current evidence is specific to MiniFASNetV2 and does not prove architecture independence;
7. parameter/GFLOP counting conventions differ across sources.

---

# 16. Claims to avoid

Do not claim:

> MAST-PAD is SOTA.

Do not claim:

> MAST-PAD beats AENet under identical conditions.

Do not claim:

> MAST-PAD is statistically superior.

Do not claim:

> frequency is domain invariant.

Do not claim:

> harmful gating always improves every split.

Do not claim:

> full-scale confirms the exact 100K effect size.

---

# 17. Reference links for the main comparison

Graph Transformer:
- https://doi.org/10.14428/esann/2023.ES2023-14
- https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf

AENet LCC benchmark:
- https://github.com/kprokofi/light-weight-face-anti-spoofing

Original CelebA-Spoof / AENet context:
- https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php

ViT / LoRA:
- https://github.com/ozogxyz/spoof

Original LCC-FASD:
- https://doi.org/10.1109/CSITechnol.2019.8895208
- https://pureportal.spbu.ru/files/51584200/CSIT2019_Grishkin_Timoshenko.pdf

---

# 18. Final paper decision

For the course paper:

```text
PROPOSED METHOD:
MAST-PAD

PRIMARY CHECKPOINT:
R7_SC_100K_crop15

PRIMARY TARGET RESULT:
LCC Combined AUC = 0.868

MAIN CONTROL:
Clean baseline = 0.848

INTERMEDIATE ABLATION:
WBST = 0.852

FULL-SCALE:
secondary scale-up study
```

No new method search is required for the current paper.
